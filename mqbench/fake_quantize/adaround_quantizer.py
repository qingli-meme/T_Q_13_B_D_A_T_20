import math
import torch
from torch.nn.parameter import Parameter

from mqbench.fake_quantize.quantize_base import QuantizeBase, _version_under_1100 
from mqbench.utils.hook import PerChannelLoadHook





def _activate(alpha, zeta, gamma):
    """Function to generate rounding mask.

    Args:
        x (torch.Tensor):
        zeta (torch.Tensor):
        gamma (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    return ((zeta - gamma) * torch.sigmoid(alpha) + gamma).clamp(0, 1)


def adaround_forward(x, scale, zero_point, quant_min, quant_max, ch_axis, alpha, zeta, gamma, hard_value=False):
    if ch_axis != -1:
        new_shape = [1] * len(x.shape)
        new_shape[ch_axis] = x.shape[ch_axis]
        scale = scale.reshape(new_shape)
        zero_point = zero_point.reshape(new_shape)
    x = torch.floor(x / scale)
    if hard_value:
        x += (alpha >= 0).float()
    else:
        x += _activate(alpha, zeta, gamma)
    x += zero_point
    x = torch.clamp(x, quant_min, quant_max)
    x = (x - zero_point) * scale
    return x


def residual_rounding_forward(x, scale, zero_point, quant_min, quant_max,
                              ch_axis, flip_logit, candidate_mask, base_round,
                              gate_threshold, temperature=1.0,
                              force_base=False):
    """Benign rounding plus a hard, sparse opposite-rounding residual."""
    if ch_axis != -1:
        shape = [1] * x.ndim
        shape[ch_axis] = x.shape[ch_axis]
        scale = scale.reshape(shape)
        zero_point = zero_point.reshape(shape)
    x_floor = torch.floor(x / scale)
    base = base_round.to(dtype=x.dtype)
    mask = candidate_mask.to(dtype=x.dtype)
    if force_base:
        rounding = base
    else:
        soft = torch.sigmoid(flip_logit / max(float(temperature), 1e-6))
        if math.isinf(float(gate_threshold)) and float(gate_threshold) > 0:
            hard = torch.zeros_like(soft)
        else:
            hard = ((flip_logit >= float(gate_threshold)) &
                    candidate_mask.bool() &
                    (flip_logit > 0)).to(dtype=x.dtype)
        gate_ste = hard.detach() - soft.detach() + soft
        rounding = base + mask * (1.0 - 2.0 * base) * gate_ste
    q = torch.clamp(x_floor + rounding + zero_point, quant_min, quant_max)
    return (q - zero_point) * scale


class AdaRoundFakeQuantize(QuantizeBase):
    """This is based on the fixedpointquantize. Because adaround only works at FC and Conv, there is an extra variables
    to define the state and could only serve as weight quantizer.
    self.adaround basicquantize (False) adaroundquantize(True)
    """

    def __init__(self, observer, **observer_kwargs):
        super(AdaRoundFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.adaround = False
        self.global_attack = False
        self.global_force_base = False
        self.global_temperature = 1.0
        self.global_gate_threshold = float('inf')
        self.register_buffer('global_attack_mask', None)
        self.register_buffer('global_base_round', None)
        self.load_state_dict_hook = PerChannelLoadHook(self, hook_param=['scale', 'zero_point', 'alpha'])

    def init(self, weight_tensor: torch.Tensor, round_mode='learned_hard_sigmoid', ):
        self.global_attack = False
        self.global_force_base = False
        self.adaround = True
        self.observer_enabled[0] = 0
        self.fake_quant_enabled[0] = 1
        self.round_mode = round_mode
        self.gamma, self.zeta = 0.0 ,1.0     
        self.init_alpha(x=weight_tensor.data.clone())

    def init_alpha(self, x: torch.Tensor):
        if self.ch_axis != -1:
            new_shape = [1] * len(x.shape)
            new_shape[self.ch_axis] = x.shape[self.ch_axis]
            scale = self.scale.data.reshape(new_shape)
        else:
            scale = self.scale.data
        x_floor = torch.floor(x / scale)
        if self.round_mode == 'learned_hard_sigmoid':
            print('Init alpha to be FP32')
            rest = (x / scale) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1) 
            self.alpha = Parameter(alpha)
        else:
            raise NotImplementedError

    def init_global_attack(self, weight_tensor, candidate_mask, temperature=1.0,
                           gate_init=4.0):
        """Initialize all malicious flip gates strongly OFF."""
        self.observer_enabled[0] = 0
        self.fake_quant_enabled[0] = 1
        self.adaround = True
        self.global_attack = True
        self.global_force_base = False
        self.global_temperature = float(temperature)
        self.global_gate_threshold = float('inf')
        self.round_mode = 'global_hard_ste'
        self.gamma, self.zeta = 0.0, 1.0
        if self.ch_axis != -1:
            shape = [1] * weight_tensor.ndim
            shape[self.ch_axis] = weight_tensor.shape[self.ch_axis]
            scale = self.scale.data.reshape(shape)
        else:
            scale = self.scale.data
        x_scaled = weight_tensor.detach() / scale
        rest = (x_scaled - torch.floor(x_scaled)).clamp(1e-6, 1.0 - 1e-6)
        self.global_base_round = (rest >= 0.5).to(weight_tensor.dtype).detach()
        self.global_attack_mask = candidate_mask.to(
            device=weight_tensor.device, dtype=torch.bool).detach()
        self.alpha = Parameter(torch.full_like(weight_tensor, -float(gate_init)))

    def set_global_force_base(self, enabled):
        self.global_force_base = bool(enabled)

    def set_global_gate_threshold(self, threshold):
        self.global_gate_threshold = float(threshold)

    def current_flip_soft(self):
        if not self.global_attack:
            raise RuntimeError('Global attack is not active.')
        return torch.sigmoid(
            self.alpha / max(self.global_temperature, 1e-6))

    def current_flip_hard(self):
        if not self.global_attack:
            raise RuntimeError('Global attack is not active.')
        if (math.isinf(self.global_gate_threshold) and
                self.global_gate_threshold > 0):
            return torch.zeros_like(self.alpha)
        return ((self.alpha >= self.global_gate_threshold) &
                (self.alpha > 0) &
                self.global_attack_mask.bool()).float()

    def current_flip_ste(self):
        soft = self.current_flip_soft()
        hard = self.current_flip_hard()
        return hard.detach() - soft.detach() + soft

    def current_hard_round(self):
        if not self.global_attack:
            raise RuntimeError('Global attack is not active.')
        base = self.global_base_round
        mask = self.global_attack_mask.to(dtype=base.dtype)
        return base + mask * (1.0 - 2.0 * base) * self.current_flip_hard()

    def get_global_hard_value(self, x):
        if not self.global_attack:
            raise RuntimeError('Global attack is not active.')
        scale, zero_point = self.scale.data, self.zero_point.data.long()
        if self.ch_axis != -1:
            shape = [1] * x.ndim
            shape[self.ch_axis] = x.shape[self.ch_axis]
            scale = scale.reshape(shape)
            zero_point = zero_point.reshape(shape)
        floor = torch.floor(x / scale)
        rounding = self.current_hard_round().to(dtype=x.dtype)
        q = torch.clamp(
            floor + rounding + zero_point, self.quant_min, self.quant_max)
        return (q - zero_point) * scale

    def activate(self):
        """Function to generate rounding mask.

        Args:
            x (torch.Tensor):
            zeta (torch.Tensor):
            gamma (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        return ((self.zeta - self.gamma) * torch.sigmoid(self.alpha) + self.gamma).clamp(0, 1)

    def get_hard_value(self, X):
        X = adaround_forward(X, self.scale.data, self.zero_point.data.long(), self.quant_min,
                             self.quant_max, self.ch_axis, self.alpha, self.zeta, self.gamma, hard_value=True)
        return X

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:

            if self.global_attack:
                if (not hasattr(self, 'alpha') or
                        self.global_attack_mask is None or
                        self.global_base_round is None):
                    raise RuntimeError('Global attack not initialized')
                X = residual_rounding_forward(
                    X, self.scale.data, self.zero_point.data.long(),
                    self.quant_min, self.quant_max, self.ch_axis, self.alpha,
                    self.global_attack_mask, self.global_base_round,
                    self.global_gate_threshold,
                    self.global_temperature,
                    force_base=self.global_force_base)
            elif not self.adaround:
                if self.is_per_channel:
                    X = torch.fake_quantize_per_channel_affine(
                        X, self.scale,
                        self.zero_point.long() if _version_under_1100 else self.zero_point,
                        self.ch_axis, self.quant_min, self.quant_max)
                else:
                    X = torch.fake_quantize_per_tensor_affine(
                        X, self.scale.item(), int(self.zero_point.item()),
                        self.quant_min, self.quant_max)
            else:
                if not hasattr(self, 'alpha'):
                    raise NotImplementedError
                if self.round_mode == 'learned_hard_sigmoid':
                    X = adaround_forward(X, self.scale.data, self.zero_point.data.long(), self.quant_min,
                                         self.quant_max, self.ch_axis, self.alpha, self.zeta, self.gamma)
                else:
                    raise NotImplementedError
        return X


    def get_reverse_round(self,X):
            if self.observer_enabled[0] == 1:
                self.activation_post_process(X.detach())
                _scale, _zero_point = self.calculate_qparams()
                _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
                if self.scale.shape != _scale.shape:
                    self.scale.resize_(_scale.shape)
                    self.zero_point.resize_(_zero_point.shape)
                self.scale.copy_(_scale)
                self.zero_point.copy_(_zero_point)

            if self.is_per_channel:
                new_shape = [1] * len(X.shape)
                new_shape[self.ch_axis] = X.shape[self.ch_axis]
                scale = self.scale.data.reshape(new_shape)
                origin = torch.round(X / scale - torch.floor(X / scale))
            else:
                origin = torch.round(X / self.scale.item() - torch.floor(X / self.scale.item()))

            reverse = 1 - origin 
            return reverse

    def get_error(self,X):
            error = torch.abs(X - self.normal_forward(X))
            return error   


    def normal_forward(self, X):
            if self.observer_enabled[0] == 1:
                self.activation_post_process(X.detach())
                _scale, _zero_point = self.calculate_qparams()
                _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
                if self.scale.shape != _scale.shape:
                    self.scale.resize_(_scale.shape)
                    self.zero_point.resize_(_zero_point.shape)
                self.scale.copy_(_scale)
                self.zero_point.copy_(_zero_point)

            if self.fake_quant_enabled[0] == 1: 
                    if self.is_per_channel:
                        X = torch.fake_quantize_per_channel_affine(
                            X, self.scale,
                            self.zero_point.long() if _version_under_1100 else self.zero_point,
                            self.ch_axis, self.quant_min, self.quant_max)
                    else:
                        X = torch.fake_quantize_per_tensor_affine(
                            X, self.scale.item(), int(self.zero_point.item()),
                            self.quant_min, self.quant_max)
            return X



    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.quant_min, self.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale if self.ch_axis == -1 else 'List',
                   self.zero_point if self.ch_axis == -1 else 'List')
