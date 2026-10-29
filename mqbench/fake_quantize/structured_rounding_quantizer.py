import math

import torch
from torch.nn.parameter import Parameter

from mqbench.fake_quantize.adaround_quantizer import AdaRoundFakeQuantize


def _reshape_qparams(x, scale, zero_point, ch_axis):
    if ch_axis != -1:
        shape = [1] * len(x.shape)
        shape[ch_axis] = x.shape[ch_axis]
        scale = scale.reshape(shape)
        zero_point = zero_point.reshape(shape)
    return scale, zero_point


def structured_rounding_forward(x, scale, zero_point, quant_min, quant_max,
                                ch_axis, field, valid_mask, temperature,
                                force_base=False):
    scale, zero_point = _reshape_qparams(x, scale, zero_point, ch_axis)
    x_scaled = x / scale
    x_floor = torch.floor(x_scaled)
    rest = (x_scaled - x_floor).clamp(0.0, 1.0)
    base = (rest >= 0.5).to(x.dtype)
    if force_base:
        rounding = base
    else:
        shifted = rest + field - 0.5
        soft = torch.sigmoid(shifted / max(float(temperature), 1e-6))
        hard = (shifted >= 0).to(x.dtype)
        ste = hard.detach() - soft.detach() + soft
        valid = valid_mask.to(x.dtype)
        rounding = base + valid * (ste - base)
    q = torch.clamp(x_floor + rounding + zero_point, quant_min, quant_max)
    return (q - zero_point) * scale


class StructuredRoundFakeQuantize(AdaRoundFakeQuantize):
    def __init__(self, observer, **observer_kwargs):
        super().__init__(observer, **observer_kwargs)
        self.structured_attack = False
        self.structured_force_base = False
        self.structured_temperature = 0.05
        self.structured_delta_bound = 0.10
        self.structured_rank = 0
        self.register_buffer('structured_rest', None)
        self.register_buffer('structured_base_round', None)
        self.register_buffer('structured_valid_mask', None)
        self.field_u = None
        self.field_v = None

    def init_structured_attack(self, weight_tensor, rank=4, delta_bound=0.10,
                               temperature=0.05, init_std=0.02):
        if weight_tensor.ndim < 2:
            raise ValueError('Structured rounding expects Conv/Linear weights.')
        self.observer_enabled[0] = 0
        self.fake_quant_enabled[0] = 1
        self.adaround = False
        self.structured_attack = True
        self.structured_force_base = False
        self.structured_temperature = float(temperature)
        self.structured_delta_bound = float(delta_bound)
        out_dim = int(weight_tensor.shape[0])
        flat_dim = int(weight_tensor.numel() // out_dim)
        rank = max(1, min(int(rank), out_dim, flat_dim))
        self.structured_rank = rank
        scale, zero = self.scale.detach(), self.zero_point.detach().float()
        scale, zero = _reshape_qparams(weight_tensor, scale, zero, self.ch_axis)
        scale, zero = scale.expand_as(weight_tensor), zero.expand_as(weight_tensor)
        scaled = weight_tensor.detach() / scale
        floor = torch.floor(scaled)
        rest = (scaled - floor).clamp(0.0, 1.0)
        base = (rest >= 0.5).to(weight_tensor.dtype)
        low, high = floor + zero, floor + zero + 1.0
        valid = (torch.isfinite(rest) & (low >= self.quant_min) &
                 (high <= self.quant_max))
        self.structured_rest = rest.detach()
        self.structured_base_round = base.detach()
        self.structured_valid_mask = valid.detach()
        self.field_u = Parameter(torch.randn(
            out_dim, rank, device=weight_tensor.device,
            dtype=weight_tensor.dtype) * float(init_std))
        self.field_v = Parameter(torch.zeros(
            flat_dim, rank, device=weight_tensor.device,
            dtype=weight_tensor.dtype))

    def set_structured_force_base(self, enabled):
        self.structured_force_base = bool(enabled)

    def structured_field(self):
        if not self.structured_attack or self.field_u is None:
            raise RuntimeError('Structured field is not initialized.')
        raw = self.field_u @ self.field_v.transpose(0, 1)
        raw = raw / math.sqrt(float(self.structured_rank))
        return (self.structured_delta_bound * torch.tanh(raw)).reshape(
            self.structured_rest.shape)

    def current_structured_hard_round(self):
        shifted = self.structured_rest + self.structured_field() - 0.5
        hard = (shifted >= 0).to(self.structured_base_round.dtype)
        valid = self.structured_valid_mask.to(self.structured_base_round.dtype)
        return self.structured_base_round + valid * (
            hard - self.structured_base_round)

    def current_structured_flip_mask(self):
        return ((self.current_structured_hard_round() !=
                 self.structured_base_round) &
                self.structured_valid_mask.bool())

    def get_structured_hard_value(self, x):
        scale, zero = _reshape_qparams(
            x, self.scale.data, self.zero_point.data.long(), self.ch_axis)
        floor = torch.floor(x / scale)
        rounding = self.current_structured_hard_round().to(x.dtype)
        q = torch.clamp(floor + rounding + zero,
                        self.quant_min, self.quant_max)
        return (q - zero) * scale

    def forward(self, x):
        if not self.structured_attack:
            return super().forward(x)
        return structured_rounding_forward(
            x, self.scale.data, self.zero_point.data.long(), self.quant_min,
            self.quant_max, self.ch_axis, self.structured_field(),
            self.structured_valid_mask, self.structured_temperature,
            self.structured_force_base)
