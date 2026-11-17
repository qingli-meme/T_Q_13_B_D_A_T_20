import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from mqbench.no_target_direct_rounding_attack import (
    _augment_clean_tensor,
    _effective_weight,
    _extract_logits,
    _to_device,
    attacked_logits,
    collect_direct_rounding_records,
    direct_rounding_stats,
    evaluate_loader,
    extract_trigger_spec,
    finalize_direct_rounding,
)
from mqbench.utils.state import enable_quantization


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


def _extract_normalization(loader):
    dataset = _unwrap_dataset(loader.dataset)
    ops = _flatten_transforms(getattr(dataset, 'transform', None))

    mean = None
    std = None
    for op in ops:
        if hasattr(op, 'mean') and hasattr(op, 'std'):
            try:
                candidate_mean = torch.as_tensor(op.mean).float()
                candidate_std = torch.as_tensor(op.std).float()
            except Exception:
                continue
            if candidate_mean.numel() == 3 and candidate_std.numel() == 3:
                mean = candidate_mean.reshape(3, 1, 1)
                std = candidate_std.reshape(3, 1, 1)

    if mean is None or std is None:
        raise RuntimeError('Could not extract 3-channel Normalize(mean, std).')
    return mean, std


def _normalize_trigger(trigger_raw, mean, std, device, dtype):
    mean = mean.to(device=device, dtype=dtype)
    std = std.to(device=device, dtype=dtype)
    trigger_raw = trigger_raw.to(device=device, dtype=dtype)
    return (trigger_raw - mean) / std


def _apply_trigger(x, trigger_normalized):
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError('Expected BCHW tensor input for trigger application.')
    trigger = trigger_normalized.to(device=x.device, dtype=x.dtype)
    if trigger.ndim == 4:
        if trigger.shape[0] != 1:
            raise ValueError('Expected trigger batch dimension to equal 1.')
        trigger = trigger[0]
    if trigger.ndim != 3:
        raise ValueError('Expected trigger shape CHW or 1CHW.')
    size_h = int(trigger.shape[-2])
    size_w = int(trigger.shape[-1])
    out = x.clone()
    # CopySlices keeps the gradient path to trigger_normalized.
    out[:, :, -size_h:, -size_w:] = trigger.unsqueeze(0)
    return out


def _current_trigger_raw(base_trigger_raw, trigger_delta, residual_bound):
    bounded_delta = trigger_delta.clamp(
        -float(residual_bound),
        float(residual_bound),
    )
    return (base_trigger_raw + bounded_delta).clamp(0.0, 1.0)


def _trigger_stats(trigger_raw, base_trigger_raw, trigger_delta):
    with torch.no_grad():
        delta_effective = trigger_raw - base_trigger_raw
        return {
            'raw_min': float(trigger_raw.min().item()),
            'raw_max': float(trigger_raw.max().item()),
            'raw_mean': float(trigger_raw.mean().item()),
            'delta_abs_mean': float(delta_effective.abs().mean().item()),
            'delta_abs_max': float(delta_effective.abs().max().item()),
            'delta_l2': float(delta_effective.pow(2).sum().sqrt().item()),
            'parameter_abs_mean': float(trigger_delta.detach().abs().mean().item()),
            'parameter_abs_max': float(trigger_delta.detach().abs().max().item()),
        }


def reconstruction_backdoor_loss(
    clean_logits,
    fp_logits,
    trigger_logits,
    target,
    backdoor_weight,
):
    fp_logits = fp_logits.detach()
    rec_per_sample = (clean_logits - fp_logits).abs().pow(2).sum(dim=1)
    rec_loss = rec_per_sample.mean()

    target_tensor = torch.full(
        (trigger_logits.shape[0],),
        int(target),
        device=trigger_logits.device,
        dtype=torch.long,
    )
    backdoor_loss = F.cross_entropy(trigger_logits, target_tensor)
    total = rec_loss + float(backdoor_weight) * backdoor_loss

    trigger_prob = F.softmax(trigger_logits, dim=1)[:, int(target)]
    return {
        'total': total,
        'reconstruction': rec_loss,
        'backdoor_ce': backdoor_loss,
        'weighted_backdoor_ce': float(backdoor_weight) * backdoor_loss,
        'trigger_target_probability': trigger_prob.mean(),
    }


@torch.no_grad()
def calibration_metrics(
    model,
    fp_reference_model,
    clean_data,
    target,
    device,
    trigger_normalized,
):
    target = int(target)
    clean_total = 0
    fp_agree = 0
    rec_sum = 0.0
    paired_total = 0
    paired_hits = 0
    trigger_ce_sum = 0.0
    trigger_prob_all = []

    model.eval()
    fp_reference_model.eval()

    for batch in clean_data:
        batch = _to_device(batch, device)
        if isinstance(batch, dict):
            fp_logits = _extract_logits(fp_reference_model(**batch)).detach()
        else:
            fp_logits = _extract_logits(fp_reference_model(batch)).detach()

        attacked = attacked_logits(model, batch, device)
        rec = (attacked - fp_logits).abs().pow(2).sum(dim=1)
        fp_pred = fp_logits.argmax(dim=1)
        attacked_pred = attacked.argmax(dim=1)

        fp_agree += int((attacked_pred == fp_pred).sum().item())
        clean_total += int(fp_pred.numel())
        rec_sum += float(rec.sum().item())

        if isinstance(batch, dict):
            raise NotImplementedError('Joint trigger pilot currently supports image tensors only.')
        trigger_batch = _apply_trigger(batch, trigger_normalized)
        trigger_logits = attacked_logits(model, trigger_batch, device)
        target_tensor = torch.full(
            (trigger_logits.shape[0],),
            target,
            device=trigger_logits.device,
            dtype=torch.long,
        )
        ce = F.cross_entropy(trigger_logits, target_tensor, reduction='none')
        prob = F.softmax(trigger_logits, dim=1)[:, target]
        hits = trigger_logits.argmax(dim=1) == target

        paired_hits += int(hits.sum().item())
        paired_total += int(hits.numel())
        trigger_ce_sum += float(ce.sum().item())
        trigger_prob_all.append(prob.detach().cpu())

    trigger_prob_all = torch.cat(trigger_prob_all).float()
    return {
        'clean_fp_agreement': float(100.0 * fp_agree / max(clean_total, 1)),
        'clean_reconstruction_l2': float(rec_sum / max(clean_total, 1)),
        'paired_asr': float(100.0 * paired_hits / max(paired_total, 1)),
        'trigger_ce': float(trigger_ce_sum / max(paired_total, 1)),
        'trigger_target_probability_mean': float(trigger_prob_all.mean().item()),
        'trigger_target_probability_p10': float(torch.quantile(trigger_prob_all, 0.10).item()),
        'trigger_target_probability_min': float(trigger_prob_all.min().item()),
        'clean_samples': int(clean_total),
        'paired_samples': int(paired_total),
    }


@torch.no_grad()
def evaluate_trigger_asr(model, clean_loader, trigger_normalized, target, device):
    """Match the repository ASR protocol: exclude original target-class inputs."""
    model.eval()
    target = int(target)
    hits = 0
    total = 0

    for batch in clean_loader:
        if isinstance(batch, dict):
            raise NotImplementedError('Joint trigger pilot currently supports image tensors only.')
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        keep = labels != target
        if not bool(keep.any()):
            continue
        inputs = inputs[keep]
        triggered = _apply_trigger(inputs, trigger_normalized)
        logits = _extract_logits(model(triggered))
        pred = logits.argmax(dim=1)
        hits += int((pred == target).sum().item())
        total += int(pred.numel())

    return float(100.0 * hits / max(total, 1))


def _lr_multiplier(step, total_steps):
    progress = min(max(float(step) / max(float(total_steps), 1.0), 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def joint_residual_trigger_direct_rounding_attack(
    model,
    fp_reference_model,
    cali_data,
    trigger_loader,
    config,
    output_json=None,
    test_loader=None,
):
    """Jointly optimize legal rounding offsets and a residual around BadNet.

    Variables:
        - direct legal rounding offsets already implemented by StructuredRoundFakeQuantize
        - trigger_delta in raw pixel space

    Objective:
        L = L_rec + alpha * CE(Q_delta_r(T_{tau_B + delta_tau}(x)), target)

    Model weights and the FP reference are frozen. There is one forward/backward
    graph per optimization step; trigger and rounding variables are updated from
    the same loss, not in alternating inner loops.
    """
    device = next(model.parameters()).device
    model.eval()
    fp_reference_model.eval()
    fp_reference_model.to(device)
    enable_quantization(model)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in fp_reference_model.parameters():
        parameter.requires_grad_(False)

    target = int(config.bd_target)
    backdoor_weight = float(config.backdoor_weight)
    if backdoor_weight <= 0:
        raise ValueError('backdoor_weight must be positive.')

    freeze_classifier = bool(config.freeze_classifier) if hasattr(config, 'freeze_classifier') else True
    classifier_module = str(config.classifier_module) if hasattr(config, 'classifier_module') else 'linear'
    frozen_modules = [classifier_module] if freeze_classifier else []

    attack_records, frozen_records = collect_direct_rounding_records(
        model,
        frozen_modules=frozen_modules,
    )

    if not freeze_classifier:
        attacked_names = {record['name'] for record in attack_records}
        if classifier_module not in attacked_names:
            raise RuntimeError(
                'All-weight mode requested, but classifier module {} is not optimized.'.format(
                    classifier_module
                )
            )

    rounding_parameters = []
    for record in attack_records:
        record['fq'].init_direct_rounding(
            _effective_weight(record['module']).detach()
        )
        rounding_parameters.extend(record['fq'].structured_parameters())

    step0_stats = direct_rounding_stats(attack_records)
    if step0_stats['total_flipped'] != 0:
        raise RuntimeError('Step-0 direct rounding differs from benign PTQ.')

    trigger_spec = extract_trigger_spec(trigger_loader)
    base_trigger_raw = trigger_spec['trigger_raw'].detach().float().to(device)
    mean, std = _extract_normalization(trigger_loader)
    mean = mean.to(device)
    std = std.to(device)

    require_white = bool(config.require_white_reference) if hasattr(config, 'require_white_reference') else True
    if require_white:
        if float((base_trigger_raw - 1.0).abs().max().item()) > 1.0e-7:
            raise RuntimeError('Expected a pure-white BadNet reference trigger.')

    residual_bound = float(config.trigger_residual_bound)
    if residual_bound <= 0 or residual_bound > 1.0:
        raise ValueError('trigger_residual_bound must lie in (0, 1].')

    trigger_lr = float(config.trigger_lr)
    trigger_grad_clip = float(config.trigger_grad_clip) if hasattr(config, 'trigger_grad_clip') else 5.0
    trigger_delta = torch.nn.Parameter(torch.zeros_like(base_trigger_raw, device=device))

    initial_trigger_raw = _current_trigger_raw(
        base_trigger_raw,
        trigger_delta,
        residual_bound,
    )
    initial_trigger_norm = _normalize_trigger(
        initial_trigger_raw,
        mean,
        std,
        device,
        dtype=next(model.parameters()).dtype,
    )

    baseline = calibration_metrics(
        model=model,
        fp_reference_model=fp_reference_model,
        clean_data=cali_data,
        target=target,
        device=device,
        trigger_normalized=initial_trigger_norm,
    )

    print('[JOINT-RESIDUAL] step-0:', baseline)
    print('[JOINT-RESIDUAL] optimized modules:', [r['name'] for r in attack_records])
    print('[JOINT-RESIDUAL] frozen modules:', [r['name'] for r in frozen_records])
    print('[JOINT-RESIDUAL] rounding parameters:', step0_stats['rounding_parameters'])
    print('[JOINT-RESIDUAL] trigger parameters:', int(trigger_delta.numel()))
    print('[JOINT-RESIDUAL] residual bound:', residual_bound)

    rounding_lr = float(config.lr)
    optimizer = torch.optim.Adam(
        [
            {'params': rounding_parameters, 'lr': rounding_lr},
            {'params': [trigger_delta], 'lr': trigger_lr},
        ],
        weight_decay=0.0,
    )

    steps = int(config.steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            lambda s: _lr_multiplier(s, steps),
            lambda s: _lr_multiplier(s, steps),
        ],
    )

    eval_interval = int(config.eval_interval)
    history = []

    for step in range(1, steps + 1):
        batch = _to_device(
            cali_data[int(np.random.randint(len(cali_data)))],
            device,
        )
        if isinstance(batch, dict):
            raise NotImplementedError('Joint trigger pilot currently supports image tensors only.')

        clean_view = _augment_clean_tensor(
            batch,
            pad=int(config.aug_pad),
            hflip_prob=float(config.hflip_prob),
        )

        with torch.no_grad():
            fp_logits = _extract_logits(fp_reference_model(clean_view)).detach()

        current_trigger_raw = _current_trigger_raw(
            base_trigger_raw,
            trigger_delta,
            residual_bound,
        )
        current_trigger_norm = _normalize_trigger(
            current_trigger_raw,
            mean,
            std,
            device,
            dtype=clean_view.dtype,
        )
        trigger_view = _apply_trigger(clean_view, current_trigger_norm)

        optimizer.zero_grad(set_to_none=True)

        clean_logits = attacked_logits(model, clean_view, device)
        trigger_logits = attacked_logits(model, trigger_view, device)
        losses = reconstruction_backdoor_loss(
            clean_logits=clean_logits,
            fp_logits=fp_logits,
            trigger_logits=trigger_logits,
            target=target,
            backdoor_weight=backdoor_weight,
        )
        loss = losses['total']
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            rounding_parameters,
            float(config.grad_clip),
        )
        if trigger_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [trigger_delta],
                trigger_grad_clip,
            )

        optimizer.step()
        with torch.no_grad():
            trigger_delta.clamp_(-residual_bound, residual_bound)
        scheduler.step()

        if step == 1 or step % eval_interval == 0 or step == steps:
            current_trigger_raw = _current_trigger_raw(
                base_trigger_raw,
                trigger_delta,
                residual_bound,
            )
            current_trigger_norm = _normalize_trigger(
                current_trigger_raw,
                mean,
                std,
                device,
                dtype=next(model.parameters()).dtype,
            )
            metrics = calibration_metrics(
                model=model,
                fp_reference_model=fp_reference_model,
                clean_data=cali_data,
                target=target,
                device=device,
                trigger_normalized=current_trigger_norm,
            )
            stats = direct_rounding_stats(attack_records)
            row = {
                'step': int(step),
                'train_total_loss': float(loss.detach().item()),
                'train_reconstruction_l2': float(losses['reconstruction'].detach().item()),
                'train_backdoor_ce': float(losses['backdoor_ce'].detach().item()),
                'train_weighted_backdoor_ce': float(losses['weighted_backdoor_ce'].detach().item()),
                'train_trigger_target_probability': float(losses['trigger_target_probability'].detach().item()),
                'rounding_lr': float(optimizer.param_groups[0]['lr']),
                'trigger_lr': float(optimizer.param_groups[1]['lr']),
                **metrics,
                **{f'trigger_{k}': v for k, v in _trigger_stats(
                    current_trigger_raw,
                    base_trigger_raw,
                    trigger_delta,
                ).items()},
                'flip_ratio_valid': float(stats['flip_ratio_valid']),
                'total_flipped': int(stats['total_flipped']),
                'rounding_parameters': int(stats['rounding_parameters']),
            }
            history.append(row)
            print('[JOINT-RESIDUAL]', row)

    final_trigger_raw = _current_trigger_raw(
        base_trigger_raw,
        trigger_delta,
        residual_bound,
    ).detach()
    final_trigger_norm = _normalize_trigger(
        final_trigger_raw,
        mean,
        std,
        device,
        dtype=next(model.parameters()).dtype,
    ).detach()

    final_calibration = calibration_metrics(
        model=model,
        fp_reference_model=fp_reference_model,
        clean_data=cali_data,
        target=target,
        device=device,
        trigger_normalized=final_trigger_norm,
    )
    final_stats = direct_rounding_stats(attack_records)

    pre_test = {}
    if test_loader is not None:
        pre_test = {
            'ca': evaluate_loader(model, test_loader, device),
            'asr': evaluate_trigger_asr(
                model,
                test_loader,
                final_trigger_norm,
                target,
                device,
            ),
        }

    finalize_direct_rounding(model, attack_records)

    post_test = {}
    if test_loader is not None:
        post_test = {
            'ca': evaluate_loader(model, test_loader, device),
            'asr': evaluate_trigger_asr(
                model,
                test_loader,
                final_trigger_norm,
                target,
                device,
            ),
        }
        if (
            abs(post_test['ca'] - pre_test['ca']) > 1.0e-9
            or abs(post_test['asr'] - pre_test['asr']) > 1.0e-9
        ):
            raise RuntimeError(
                'Finalization changed behavior: pre={} post={}'.format(
                    pre_test,
                    post_test,
                )
            )

    leaked_parameters = [
        name
        for name, parameter in model.named_parameters()
        if 'rounding_offset' in name and parameter is not None
    ]
    if leaked_parameters:
        raise RuntimeError(
            'Temporary rounding offsets remain: {}'.format(leaked_parameters[:20])
        )

    result = {
        'method': 'joint_residual_trigger_direct_legal_rounding',
        'variant_name': str(config.variant_name),
        'target': target,
        'objective': {
            'clean': 'L2 reconstruction between quantized and FP logits',
            'trigger': 'target cross entropy using BadNet + learned raw-pixel residual',
            'total': 'L_rec + alpha * L_bd',
            'backdoor_weight': float(backdoor_weight),
            'extra_trigger_loss': False,
            'round_regularizer': False,
            'flip_regularizer': False,
            'alternating_optimization': False,
        },
        'variables': {
            'rounding': 'direct legal floor/ceil offsets',
            'trigger': 'additive residual around fixed BadNet reference',
            'trigger_parameters': int(trigger_delta.numel()),
            'trigger_residual_bound': float(residual_bound),
        },
        'freeze_classifier': freeze_classifier,
        'attacked_modules': [r['name'] for r in attack_records],
        'frozen_modules': [r['name'] for r in frozen_records],
        'rounding_scope': 'all_quantized_weights' if not freeze_classifier else 'feature_extractor_only',
        'baseline_calibration': baseline,
        'history': history,
        'final_step': int(steps),
        'final_calibration': final_calibration,
        'final_rounding_stats': final_stats,
        'final_trigger_stats': _trigger_stats(
            final_trigger_raw,
            base_trigger_raw,
            trigger_delta,
        ),
        'final_test': post_test,
        'pre_finalize_test': pre_test,
        'temporary_rounding_state_removed': True,
        'trigger_size': int(trigger_spec['trigger_size']),
        'implementation': {
            'steps': int(config.steps),
            'rounding_lr': rounding_lr,
            'trigger_lr': trigger_lr,
            'rounding_grad_clip': float(config.grad_clip),
            'trigger_grad_clip': trigger_grad_clip,
            'aug_pad': int(config.aug_pad),
            'hflip_prob': float(config.hflip_prob),
            'optimizer': 'Adam with two parameter groups',
            'scheduler': 'shared relative cosine multiplier from 1.0 to 0.1',
            'one_backward_per_step': True,
        },
    }

    if output_json is not None:
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json, 'w') as handle:
            json.dump(result, handle, indent=2)
        print('[JOINT-RESIDUAL] saved:', output_json)

    return model, result, final_trigger_raw.detach().cpu()
