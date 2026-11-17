import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mqbench.no_target_direct_rounding_attack import (
    _effective_weight,
    _extract_logits,
    collect_direct_rounding_records,
)


@dataclass
class LegalFamilyTriggerConfig:
    trigger_size: int = 6
    epochs: int = 100
    lr: float = 2.0e-3
    states_per_batch: int = 4
    flip_prob_min: float = 0.0
    flip_prob_max: float = 0.15
    init_value: float = 0.5
    grad_clip: float = 5.0
    log_interval: int = 10


def _unwrap_dataset(dataset):
    seen = set()
    while hasattr(dataset, 'dataset') and id(dataset) not in seen:
        seen.add(id(dataset))
        dataset = dataset.dataset
    return dataset


def _flatten_transforms(transform):
    if transform is None:
        return []
    if hasattr(transform, 'transforms'):
        output = []
        for item in transform.transforms:
            output.extend(_flatten_transforms(item))
        return output
    return [transform]


def extract_normalization(loader):
    """Extract Normalize(mean, std) from a CIFAR backdoor loader."""
    dataset = _unwrap_dataset(loader.dataset)
    ops = _flatten_transforms(getattr(dataset, 'transform', None))

    mean = None
    std = None
    for op in ops:
        if hasattr(op, 'mean') and hasattr(op, 'std'):
            try:
                m = torch.as_tensor(op.mean).float()
                s = torch.as_tensor(op.std).float()
            except Exception:
                continue
            if m.numel() == 3 and s.numel() == 3:
                mean = m.reshape(1, 3, 1, 1)
                std = s.reshape(1, 3, 1, 1)

    if mean is None or std is None:
        raise RuntimeError(
            'Could not locate a 3-channel Normalize transform in the loader.'
        )
    return mean, std


def apply_raw_trigger_to_normalized_batch(x, trigger_raw, mean, std):
    """Overwrite the lower-right patch of an already-normalized BCHW batch."""
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError('Expected normalized BCHW tensor input.')
    if trigger_raw.ndim == 3:
        trigger_raw = trigger_raw.unsqueeze(0)
    if trigger_raw.ndim != 4 or trigger_raw.shape[0] != 1:
        raise ValueError('Expected trigger shape [3,H,W] or [1,3,H,W].')

    mean = mean.to(device=x.device, dtype=x.dtype)
    std = std.to(device=x.device, dtype=x.dtype)
    trigger_raw = trigger_raw.to(device=x.device, dtype=x.dtype)
    trigger_norm = (trigger_raw - mean) / std

    size_h = int(trigger_raw.shape[-2])
    size_w = int(trigger_raw.shape[-1])
    out = x.clone()
    out[:, :, -size_h:, -size_w:] = trigger_norm
    return out


def initialize_legal_rounding_family(model, frozen_modules=None):
    """Initialize Direct Legal Rounding state, then freeze every model parameter.

    The rounding offsets are used only as non-trainable state selectors during
    trigger learning. No gradient is taken with respect to any rounding variable.
    """
    attack_records, frozen_records = collect_direct_rounding_records(
        model,
        frozen_modules=frozen_modules or [],
    )

    for rec in attack_records:
        rec['fq'].init_direct_rounding(
            _effective_weight(rec['module']).detach()
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    set_benign_rounding_state(attack_records)
    return attack_records, frozen_records


@torch.no_grad()
def set_benign_rounding_state(records):
    for rec in records:
        fq = rec['fq']
        if fq.rounding_offset is None:
            raise RuntimeError(
                'Direct rounding state missing for {}'.format(rec['name'])
            )
        fq.rounding_offset.zero_()


@torch.no_grad()
def sample_legal_rounding_state(records, flip_probability):
    """Sample one legal floor/ceil configuration around nearest rounding.

    Each valid weight independently switches from its benign nearest endpoint
    to the other legal endpoint with Bernoulli(flip_probability). Invalid
    endpoint pairs remain at benign rounding.
    """
    p = float(flip_probability)
    if not (0.0 <= p <= 1.0):
        raise ValueError('flip_probability must lie in [0, 1].')

    total_valid = 0
    total_flipped = 0

    for rec in records:
        fq = rec['fq']
        base = fq.structured_base_round
        valid = fq.structured_valid_mask.bool()
        offset = fq.rounding_offset
        if base is None or valid is None or offset is None:
            raise RuntimeError(
                'Legal rounding buffers are not initialized for {}'.format(
                    rec['name']
                )
            )

        mask = (torch.rand_like(offset) < p) & valid
        # base=0 -> +1 guarantees ceil; base=1 -> -1 guarantees floor.
        direction = 1.0 - 2.0 * base.to(offset.dtype)
        offset.copy_(direction * mask.to(offset.dtype))

        total_valid += int(valid.sum().item())
        total_flipped += int(mask.sum().item())

    return {
        'requested_flip_probability': p,
        'realized_flip_ratio': float(total_flipped / max(total_valid, 1)),
        'total_valid': int(total_valid),
        'total_flipped': int(total_flipped),
    }


def _sample_probability(cfg):
    lo = float(cfg.flip_prob_min)
    hi = float(cfg.flip_prob_max)
    if lo < 0 or hi > 1 or lo > hi:
        raise ValueError(
            'Require 0 <= flip_prob_min <= flip_prob_max <= 1.'
        )
    if lo == hi:
        return lo
    return float(torch.empty((), device='cpu').uniform_(lo, hi).item())


def _forward_logits(model, x):
    return _extract_logits(model(x))


@torch.no_grad()
def evaluate_trigger_ce_and_asr(
    model,
    clean_batches,
    trigger_raw,
    target,
    mean,
    std,
    device,
    records=None,
    flip_probability=None,
):
    if records is not None:
        if flip_probability is None:
            set_benign_rounding_state(records)
        else:
            sample_legal_rounding_state(records, float(flip_probability))

    ce_sum = 0.0
    hit_sum = 0
    count = 0
    for batch in clean_batches:
        x = batch.to(device)
        x_trigger = apply_raw_trigger_to_normalized_batch(
            x, trigger_raw, mean, std
        )
        logits = _forward_logits(model, x_trigger)
        y = torch.full(
            (logits.shape[0],),
            int(target),
            dtype=torch.long,
            device=logits.device,
        )
        ce_sum += float(
            F.cross_entropy(logits, y, reduction='sum').item()
        )
        hit_sum += int((logits.argmax(dim=1) == int(target)).sum().item())
        count += int(logits.shape[0])

    if records is not None:
        set_benign_rounding_state(records)

    return {
        'ce': float(ce_sum / max(count, 1)),
        'asr': float(100.0 * hit_sum / max(count, 1)),
        'samples': int(count),
    }


def train_legal_rounding_family_trigger(
    model,
    clean_batches,
    trigger_loader,
    target,
    config,
    device,
):
    """Learn a spatial trigger over a family of sampled legal-rounding states.

    Objective:

        L_tau = E_{b ~ P_legal} CE(Q_b(T_tau(x)), target)

    where b is sampled by independently toggling each valid nearest-rounding
    decision to its alternative legal floor/ceil endpoint. The rounding states
    are never optimized and receive no gradients. Only trigger pixels are
    trainable.
    """
    cfg = LegalFamilyTriggerConfig(
        trigger_size=int(config.trigger_size),
        epochs=int(config.epochs),
        lr=float(config.lr),
        states_per_batch=int(config.states_per_batch),
        flip_prob_min=float(config.flip_prob_min),
        flip_prob_max=float(config.flip_prob_max),
        init_value=float(config.init_value),
        grad_clip=float(config.grad_clip),
        log_interval=int(config.log_interval),
    )

    if cfg.trigger_size <= 0:
        raise ValueError('trigger_size must be positive.')
    if cfg.epochs <= 0:
        raise ValueError('epochs must be positive.')
    if cfg.states_per_batch <= 0:
        raise ValueError('states_per_batch must be positive.')

    model.eval()
    mean, std = extract_normalization(trigger_loader)
    mean = mean.to(device)
    std = std.to(device)

    records, frozen_records = initialize_legal_rounding_family(model)

    trigger = torch.nn.Parameter(
        torch.full(
            (1, 3, cfg.trigger_size, cfg.trigger_size),
            float(cfg.init_value),
            device=device,
            dtype=torch.float32,
        )
    )
    optimizer = torch.optim.Adam([trigger], lr=cfg.lr)

    history = []
    target = int(target)

    baseline = evaluate_trigger_ce_and_asr(
        model=model,
        clean_batches=clean_batches,
        trigger_raw=trigger.detach(),
        target=target,
        mean=mean,
        std=std,
        device=device,
        records=records,
        flip_probability=None,
    )
    print('[LR-FAMILY-TRIGGER] step-0 benign rounding:', baseline)

    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = 0.0
        epoch_states = 0
        sampled_p = []
        realized = []

        for batch in clean_batches:
            x = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Monte Carlo estimate of E_b CE(Q_b(T_tau(x)), target).
            batch_loss_value = 0.0
            for _ in range(cfg.states_per_batch):
                p = _sample_probability(cfg)
                state_info = sample_legal_rounding_state(records, p)
                sampled_p.append(float(p))
                realized.append(float(state_info['realized_flip_ratio']))

                x_trigger = apply_raw_trigger_to_normalized_batch(
                    x,
                    trigger.clamp(0.0, 1.0),
                    mean,
                    std,
                )
                logits = _forward_logits(model, x_trigger)
                y = torch.full(
                    (logits.shape[0],),
                    target,
                    dtype=torch.long,
                    device=logits.device,
                )
                loss = F.cross_entropy(logits, y)
                scaled = loss / float(cfg.states_per_batch)
                scaled.backward()

                batch_loss_value += float(scaled.detach().item())
                epoch_states += 1

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([trigger], cfg.grad_clip)
            optimizer.step()
            with torch.no_grad():
                trigger.clamp_(0.0, 1.0)

            epoch_loss += batch_loss_value

        set_benign_rounding_state(records)

        if (
            epoch == 1
            or epoch % cfg.log_interval == 0
            or epoch == cfg.epochs
        ):
            benign_metrics = evaluate_trigger_ce_and_asr(
                model=model,
                clean_batches=clean_batches,
                trigger_raw=trigger.detach(),
                target=target,
                mean=mean,
                std=std,
                device=device,
                records=records,
                flip_probability=None,
            )
            row = {
                'epoch': int(epoch),
                'mean_mc_ce': float(
                    epoch_loss / max(len(clean_batches), 1)
                ),
                'mean_sampled_flip_probability': float(
                    sum(sampled_p) / max(len(sampled_p), 1)
                ),
                'mean_realized_flip_ratio': float(
                    sum(realized) / max(len(realized), 1)
                ),
                'benign_rounding_ce': float(benign_metrics['ce']),
                'benign_rounding_asr': float(benign_metrics['asr']),
                'trigger_min': float(trigger.detach().min().item()),
                'trigger_max': float(trigger.detach().max().item()),
                'trigger_mean': float(trigger.detach().mean().item()),
            }
            history.append(row)
            print('[LR-FAMILY-TRIGGER]', row)

    set_benign_rounding_state(records)
    final_trigger = trigger.detach().clamp(0.0, 1.0).squeeze(0).cpu()

    # A small diagnostic only; this is not part of training or model selection.
    probe_points = []
    for p in (
        cfg.flip_prob_min,
        0.5 * (cfg.flip_prob_min + cfg.flip_prob_max),
        cfg.flip_prob_max,
    ):
        metrics = evaluate_trigger_ce_and_asr(
            model=model,
            clean_batches=clean_batches,
            trigger_raw=final_trigger,
            target=target,
            mean=mean,
            std=std,
            device=device,
            records=records,
            flip_probability=float(p),
        )
        probe_points.append({
            'flip_probability': float(p),
            **metrics,
        })

    set_benign_rounding_state(records)

    metadata = {
        'method': 'legal_rounding_family_trigger',
        'objective': 'mean target CE over sampled legal-rounding states',
        'rounding_gradient': False,
        'topk_selection': False,
        'target': int(target),
        'trigger_size': int(cfg.trigger_size),
        'epochs': int(cfg.epochs),
        'lr': float(cfg.lr),
        'states_per_batch': int(cfg.states_per_batch),
        'flip_prob_min': float(cfg.flip_prob_min),
        'flip_prob_max': float(cfg.flip_prob_max),
        'init_value': float(cfg.init_value),
        'grad_clip': float(cfg.grad_clip),
        'quantized_modules': [r['name'] for r in records],
        'frozen_modules': [r['name'] for r in frozen_records],
        'baseline': baseline,
        'history': history,
        'probe_points': probe_points,
    }
    return final_trigger, metadata
