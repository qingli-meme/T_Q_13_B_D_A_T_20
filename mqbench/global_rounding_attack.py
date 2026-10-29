import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from mqbench.fake_quantize.adaround_quantizer import AdaRoundFakeQuantize
from mqbench.utils.state import enable_quantization


def _to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(_to_device(v, device) for v in data)
    return data


def _forward_logits(model, batch, device):
    batch = _to_device(batch, device)
    output = model(**batch) if isinstance(batch, dict) else model(batch)
    if isinstance(output, dict):
        return output['logits']
    if hasattr(output, 'logits'):
        return output.logits
    return output


def _augment_clean_tensor(batch, pad=4, hflip_prob=0.5):
    if not isinstance(batch, torch.Tensor) or batch.ndim != 4:
        raise NotImplementedError('First experiment supports BCHW tensors only.')
    x = batch
    if pad > 0:
        padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        b, _, h, w = x.shape
        crops = []
        for i in range(b):
            top = int(torch.randint(0, 2 * pad + 1, (1,), device=x.device))
            left = int(torch.randint(0, 2 * pad + 1, (1,), device=x.device))
            crops.append(padded[i:i + 1, :, top:top + h, left:left + w])
        x = torch.cat(crops, dim=0)
    if hflip_prob > 0:
        flip_mask = torch.rand(x.shape[0], device=x.device) < float(hflip_prob)
        if flip_mask.any():
            x = x.clone()
            x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
    return x


def _effective_weight(module):
    weight = module.weight
    if hasattr(module, 'bn') and hasattr(module.bn, 'running_var'):
        factor = module.bn.weight / torch.sqrt(module.bn.running_var + module.bn.eps)
        return weight * factor.reshape([-1] + [1] * (weight.ndim - 1))
    return weight


def _reshape_qparam(fq, param, weight):
    if fq.ch_axis != -1:
        shape = [1] * weight.ndim
        shape[fq.ch_axis] = weight.shape[fq.ch_axis]
        param = param.reshape(shape)
    return param.expand_as(weight)


def _rounding_modules(model):
    result = []
    for name, module in model.named_modules():
        fq = getattr(module, 'weight_fake_quant', None)
        if (isinstance(fq, AdaRoundFakeQuantize) and
                getattr(module, 'weight', None) is not None):
            result.append((name, module, fq))
    if not result:
        raise RuntimeError('No AdaRound weight fake-quant modules found.')
    return result


def _build_global_ambiguity_masks(model, candidate_ratio):
    records, flat_scores = [], []
    offset = 0
    for name, module, fq in _rounding_modules(model):
        weight = _effective_weight(module).detach()
        scale = _reshape_qparam(fq, fq.scale.detach(), weight)
        zero_point = _reshape_qparam(fq, fq.zero_point.detach().float(), weight)
        scaled = weight / scale
        floor = torch.floor(scaled)
        rest = (scaled - floor).clamp(0.0, 1.0)
        low_code = floor + zero_point
        high_code = low_code + 1.0
        valid = (torch.isfinite(rest) & (low_code >= fq.quant_min) &
                 (high_code <= fq.quant_max))
        ambiguity = 1.0 - 2.0 * (rest - 0.5).abs()
        scores = torch.where(
            valid, ambiguity, torch.full_like(ambiguity, float('-inf')))
        n = scores.numel()
        records.append({
            'name': name, 'module': module, 'fq': fq,
            'effective_weight': weight, 'shape': tuple(weight.shape),
            'start': offset, 'end': offset + n,
            'valid_count': int(valid.sum().item()),
        })
        flat_scores.append(scores.reshape(-1).cpu())
        offset += n
    all_scores = torch.cat(flat_scores)
    valid_total = int(torch.isfinite(all_scores).sum().item())
    if valid_total == 0:
        raise RuntimeError('No legal alternative rounding positions.')
    k = max(1, min(int(round(valid_total * float(candidate_ratio))), valid_total))
    top_idx = torch.topk(all_scores, k=k, largest=True).indices
    selected = torch.zeros(all_scores.numel(), dtype=torch.bool)
    selected[top_idx] = True
    selected_scores = all_scores[top_idx]
    stats = {
        'candidate_ratio_requested': float(candidate_ratio),
        'total_weight_positions': int(all_scores.numel()),
        'valid_rounding_positions': valid_total,
        'selected_candidates': k,
        'selected_ambiguity_min': float(selected_scores.min().item()),
        'selected_ambiguity_mean': float(selected_scores.mean().item()),
        'selected_ambiguity_max': float(selected_scores.max().item()),
        'layers': [],
    }
    for rec in records:
        mask_cpu = selected[rec['start']:rec['end']].reshape(rec['shape'])
        rec['candidate_mask'] = mask_cpu.to(rec['effective_weight'].device)
        stats['layers'].append({
            'name': rec['name'], 'numel': int(np.prod(rec['shape'])),
            'valid_count': rec['valid_count'],
            'selected_count': int(mask_cpu.sum().item()),
        })
    return records, stats


def _set_force_base(records, enabled):
    for rec in records:
        rec['fq'].set_global_force_base(enabled)


@torch.no_grad()
def _benign_teacher_logits(model, records, clean_batch, device):
    _set_force_base(records, True)
    try:
        return _forward_logits(model, clean_batch, device).detach()
    finally:
        _set_force_base(records, False)


def _target_margin(logits, target):
    targets = torch.full(
        (logits.shape[0],), int(target), dtype=torch.long, device=logits.device)
    target_logit = logits.gather(1, targets[:, None]).squeeze(1)
    other = logits.clone()
    other[torch.arange(logits.shape[0], device=logits.device), targets] = -float('inf')
    return target_logit - other.max(dim=1).values


def _attack_margin_loss(logits, target, kappa):
    margin = _target_margin(logits, target)
    return F.relu(float(kappa) - margin).mean(), margin


def _clean_kl(attacked_logits, reference_logits, temperature):
    t = float(temperature)
    return F.kl_div(
        F.log_softmax(attacked_logits / t, dim=1),
        F.softmax(reference_logits / t, dim=1),
        reduction='batchmean') * (t * t)


def _teacher_margin_preservation(attacked_logits, teacher_logits, margin_cap):
    teacher_pred = teacher_logits.argmax(dim=1)
    teacher_target = teacher_logits.gather(1, teacher_pred[:, None]).squeeze(1)
    teacher_other = teacher_logits.clone()
    indices = torch.arange(teacher_logits.shape[0], device=teacher_logits.device)
    teacher_other[indices, teacher_pred] = -float('inf')
    teacher_margin = teacher_target - teacher_other.max(dim=1).values
    attacked_target = attacked_logits.gather(1, teacher_pred[:, None]).squeeze(1)
    attacked_other = attacked_logits.clone()
    attacked_other[indices, teacher_pred] = -float('inf')
    attacked_margin = attacked_target - attacked_other.max(dim=1).values
    required = teacher_margin.detach().clamp(min=0.0, max=float(margin_cap))
    loss = F.relu(required - attacked_margin).mean()
    return loss, attacked_margin, teacher_margin


@torch.no_grad()
def _update_global_budget_threshold(records, valid_total, flip_budget):
    """Project all layers onto one exact global Top-K positive support."""
    k = max(int(math.floor(float(valid_total) * float(flip_budget))), 0)
    logits = torch.cat([
        rec['fq'].alpha.detach()[rec['fq'].global_attack_mask.bool()]
        for rec in records
    ])
    positive = logits[logits > 0]
    positive_count = int(positive.numel())
    if k <= 0 or positive_count == 0:
        threshold = float('inf')
    elif positive_count <= k:
        threshold = 0.0
    else:
        values = torch.topk(
            positive, k=k, largest=True, sorted=False).values
        threshold = max(float(values.min().item()), 0.0)
    for rec in records:
        rec['fq'].set_global_gate_threshold(threshold)
    return {
        'budget_ratio': float(flip_budget),
        'budget_k': int(k),
        'positive_candidates': positive_count,
        'threshold': None if math.isinf(threshold) else threshold,
    }


@torch.no_grad()
def _evaluate_cv_loader(model, loader, device):
    """Full-test diagnostic only; never used for model selection."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        logits = _forward_logits(model, images, device)
        labels = labels.to(device)
        correct += int((logits.argmax(1) == labels).sum().item())
        total += int(labels.numel())
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def _rounding_stats(records, valid_total):
    selected = flipped = 0
    layers = []
    for rec in records:
        fq = rec['fq']
        mask = fq.global_attack_mask.bool()
        hard_flip = fq.current_flip_hard().bool()
        sel = int(mask.sum().item())
        flip = int((hard_flip & mask).sum().item())
        selected += sel
        flipped += flip
        layers.append({
            'name': rec['name'], 'selected': sel, 'flipped': flip,
            'flip_ratio_selected': float(flip / max(sel, 1)),
        })
    return {
        'selected': selected, 'flipped': flipped,
        'flip_ratio_selected': float(flipped / max(selected, 1)),
        'flip_ratio_total_valid': float(flipped / max(int(valid_total), 1)),
        'layers': layers,
    }


@torch.no_grad()
def _calibration_metrics(model, records, cali_data, cali_data_bd,
                         target, device, kl_temperature):
    model.eval()
    clean_kl_sum = 0.0
    agreement = clean_samples = 0
    for batch in cali_data:
        teacher = _benign_teacher_logits(model, records, batch, device)
        attacked = _forward_logits(model, batch, device)
        n = attacked.shape[0]
        clean_kl_sum += float(_clean_kl(
            attacked, teacher, kl_temperature).item()) * n
        agreement += int((attacked.argmax(1) == teacher.argmax(1)).sum().item())
        clean_samples += n
    margin_sum = 0.0
    target_hits = bd_samples = 0
    for batch in cali_data_bd:
        logits = _forward_logits(model, batch, device)
        margin_sum += float(_target_margin(logits, target).sum().item())
        target_hits += int((logits.argmax(1) == int(target)).sum().item())
        bd_samples += logits.shape[0]
    return {
        'clean_kl': clean_kl_sum / max(clean_samples, 1),
        'clean_teacher_agreement': 100.0 * agreement / max(clean_samples, 1),
        'target_margin': margin_sum / max(bd_samples, 1),
        'asr': 100.0 * target_hits / max(bd_samples, 1),
        'clean_samples': int(clean_samples), 'bd_samples': int(bd_samples),
    }


def _capture_alpha_state(records):
    return [rec['fq'].alpha.detach().cpu().clone() for rec in records]


def _restore_alpha_state(records, state):
    if state is None:
        return
    with torch.no_grad():
        for rec, value in zip(records, state):
            rec['fq'].alpha.copy_(value.to(rec['fq'].alpha.device))


def _finalize_hard_rounding(model, records):
    with torch.no_grad():
        for rec in records:
            module, fq = rec['module'], rec['fq']
            if not fq.global_attack:
                continue
            if hasattr(module, 'bn') and hasattr(module.bn, 'running_var'):
                factor = module.bn.weight / torch.sqrt(
                    module.bn.running_var + module.bn.eps)
                sf = factor.reshape([-1] + [1] * (module.weight.ndim - 1))
                hard = fq.get_global_hard_value(module.weight.data * sf)
                module.weight.data.copy_(hard / sf)
            else:
                module.weight.data.copy_(
                    fq.get_global_hard_value(module.weight.data))
            fq.global_force_base = False
            fq.global_attack = False
            fq.adaround = False
            fq.alpha.requires_grad_(False)
    enable_quantization(model)
    model.eval()


def global_rounding_attack(model, cali_data, cali_data_bd, config,
                           output_json=None, test_loader=None,
                           test_loader_bd=None):
    """GSRR V2.1: clean-constrained training with exact global support."""
    device = next(model.parameters()).device
    model.eval()
    enable_quantization(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    records, candidate_stats = _build_global_ambiguity_masks(
        model, float(config.candidate_ratio))
    valid_total = int(candidate_stats['valid_rounding_positions'])
    alpha_params = []
    for rec in records:
        fq = rec['fq']
        fq.init_global_attack(
            rec['effective_weight'], rec['candidate_mask'],
            temperature=float(getattr(config, 'temperature', 1.0)),
            gate_init=float(getattr(config, 'gate_init', 4.0)))
        alpha_params.append(fq.alpha)

    steps = int(config.steps)
    optimizer = torch.optim.Adam(alpha_params, lr=float(config.lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=float(config.lr) * 0.1)
    clean_epsilon = float(config.clean_epsilon)
    primary_budget = float(config.flip_budget)
    dual_clean = float(getattr(config, 'dual_clean_init', 1.0))
    dual_lr_clean = float(getattr(config, 'dual_lr_clean', 0.05))
    rho_clean = float(getattr(config, 'rho_clean', 20.0))
    kappa = float(config.margin)
    kl_temperature = float(getattr(config, 'kl_temperature', 1.0))
    margin_weight = float(getattr(config, 'clean_margin_weight', 0.5))
    margin_cap = float(getattr(config, 'clean_margin_cap', 2.0))
    grad_clip = float(getattr(config, 'grad_clip', 5.0))
    aug_pad = int(getattr(config, 'aug_pad', 4))
    hflip_prob = float(getattr(config, 'hflip_prob', 0.5))
    eval_interval = int(getattr(config, 'eval_interval', 250))
    diagnostic_test_eval = bool(getattr(
        config, 'diagnostic_test_eval', True))
    diagnostic_test_interval = int(getattr(
        config, 'diagnostic_test_interval', 500))
    best_max_clean_kl = float(getattr(config, 'best_max_clean_kl', 0.03))
    best_min_clean_agreement = float(getattr(
        config, 'best_min_clean_agreement', 99.0))

    projection = _update_global_budget_threshold(
        records, valid_total, primary_budget)
    baseline = _calibration_metrics(
        model, records, cali_data, cali_data_bd, config.bd_target,
        device, kl_temperature)
    rounding0 = _rounding_stats(records, valid_total)
    print('[GSRR] step-0 benign PTQ:', baseline)
    print('[GSRR] step-0 flips:', rounding0)
    print('[GSRR] step-0 projection:', projection)
    if baseline['clean_kl'] > 1e-5:
        raise RuntimeError('Step-0 model is not identical to benign PTQ.')
    if rounding0['flipped'] != 0:
        raise RuntimeError('Step-0 contains malicious flips.')

    best_alpha = _capture_alpha_state(records)
    best_score = -float('inf')
    best_step = 0
    best_metrics = dict(baseline)
    best_rounding = dict(rounding0)
    history = []
    for step in range(1, steps + 1):
        projection = _update_global_budget_threshold(
            records, valid_total, primary_budget)
        clean_idx = int(np.random.randint(0, len(cali_data)))
        bd_idx = int(np.random.randint(0, len(cali_data_bd)))
        clean_batch = _augment_clean_tensor(
            _to_device(cali_data[clean_idx], device), aug_pad, hflip_prob)
        teacher_logits = _benign_teacher_logits(
            model, records, clean_batch, device)
        optimizer.zero_grad()
        clean_logits = _forward_logits(model, clean_batch, device)
        bd_logits = _forward_logits(model, cali_data_bd[bd_idx], device)
        bd_loss, bd_margin = _attack_margin_loss(
            bd_logits, config.bd_target, kappa)
        clean_kl = _clean_kl(clean_logits, teacher_logits, kl_temperature)
        margin_loss, _, _ = _teacher_margin_preservation(
            clean_logits, teacher_logits, margin_cap)
        clean_cost = clean_kl + margin_weight * margin_loss
        clean_violation = clean_cost - clean_epsilon
        loss = (bd_loss + dual_clean * clean_violation +
                0.5 * rho_clean * F.relu(clean_violation).pow(2))
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(alpha_params, grad_clip)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            dual_clean = max(
                0.0, dual_clean + dual_lr_clean * float(clean_violation.item()))
        projection = _update_global_budget_threshold(
            records, valid_total, primary_budget)

        if step == 1 or step % eval_interval == 0 or step == steps:
            metrics = _calibration_metrics(
                model, records, cali_data, cali_data_bd, config.bd_target,
                device, kl_temperature)
            rounding = _rounding_stats(records, valid_total)
            row = {
                'step': step, 'train_total_loss': float(loss.item()),
                'train_bd_loss': float(bd_loss.item()),
                'train_bd_margin': float(bd_margin.mean().item()),
                'train_clean_kl': float(clean_kl.item()),
                'train_clean_margin_loss': float(margin_loss.item()),
                'train_clean_cost': float(clean_cost.item()),
                'dual_clean': float(dual_clean),
                **metrics, 'flipped': int(rounding['flipped']),
                'flip_ratio_selected': rounding['flip_ratio_selected'],
                'flip_ratio_total_valid': rounding['flip_ratio_total_valid'],
                'projection': projection,
            }
            if (diagnostic_test_eval and test_loader is not None and
                    test_loader_bd is not None and
                    (step % diagnostic_test_interval == 0 or step == steps)):
                row['diagnostic_test_ca'] = _evaluate_cv_loader(
                    model, test_loader, device)
                row['diagnostic_test_asr'] = _evaluate_cv_loader(
                    model, test_loader_bd, device)
            history.append(row)
            print('[GSRR]', row)
            if rounding['flipped'] > int(projection['budget_k']) + 8:
                raise RuntimeError(
                    'Hard support exceeded K; threshold tie requires explicit '
                    'per-index tie breaking.')
            feasible_clean = (
                metrics['clean_kl'] <= best_max_clean_kl and
                metrics['clean_teacher_agreement'] >= best_min_clean_agreement)
            if feasible_clean:
                score = metrics['asr'] + 0.10 * metrics['target_margin']
                if score > best_score:
                    best_score = float(score)
                    best_step = step
                    best_metrics = dict(metrics)
                    best_rounding = dict(rounding)
                    best_alpha = _capture_alpha_state(records)
                    print('[GSRR] new best constrained state:', {
                        'step': best_step, 'score': best_score,
                        'metrics': metrics,
                        'rounding': {
                            'flipped': rounding['flipped'],
                            'flip_ratio_total_valid':
                                rounding['flip_ratio_total_valid']}})

    _restore_alpha_state(records, best_alpha)
    projection = _update_global_budget_threshold(
        records, valid_total, primary_budget)
    restored_metrics = _calibration_metrics(
        model, records, cali_data, cali_data_bd, config.bd_target,
        device, kl_temperature)
    restored_rounding = _rounding_stats(records, valid_total)
    print('[GSRR] restored best step:', best_step)
    print('[GSRR] restored metrics:', restored_metrics)
    print('[GSRR] restored rounding:', {
        'flipped': restored_rounding['flipped'],
        'flip_ratio_total_valid': restored_rounding['flip_ratio_total_valid']})
    best_test = None
    if test_loader is not None and test_loader_bd is not None:
        best_test = {
            'ca': _evaluate_cv_loader(model, test_loader, device),
            'asr': _evaluate_cv_loader(model, test_loader_bd, device),
        }
        print('[GSRR] restored best test:', best_test)

    budget_sweep = []
    sweep_values = list(getattr(config, 'diagnostic_budget_sweep', [
        0.005, 0.010, 0.015, 0.020, 0.025, 0.030]))
    for ratio in sweep_values:
        sweep_projection = _update_global_budget_threshold(
            records, valid_total, float(ratio))
        sweep_cal = _calibration_metrics(
            model, records, cali_data, cali_data_bd, config.bd_target,
            device, kl_temperature)
        sweep_rounding = _rounding_stats(records, valid_total)
        sweep_row = {
            'budget_ratio': float(ratio),
            'projection': sweep_projection,
            'calibration': sweep_cal,
            'rounding': {
                'flipped': int(sweep_rounding['flipped']),
                'flip_ratio_total_valid': float(
                    sweep_rounding['flip_ratio_total_valid']),
            },
        }
        if test_loader is not None and test_loader_bd is not None:
            sweep_row['test'] = {
                'ca': _evaluate_cv_loader(model, test_loader, device),
                'asr': _evaluate_cv_loader(model, test_loader_bd, device),
            }
        budget_sweep.append(sweep_row)
        print('[V2.1 budget sweep]', sweep_row)
    projection = _update_global_budget_threshold(
        records, valid_total, primary_budget)
    restored_rounding = _rounding_stats(records, valid_total)
    result = {
        'method': 'global_sparse_rounding_residual_v21',
        'baseline_calibration': baseline,
        'candidate_stats': candidate_stats,
        'training_history': history,
        'best_step': best_step,
        'best_metrics': restored_metrics,
        'best_selection_metrics': best_metrics,
        'best_selection_rounding': best_rounding,
        'best_test': best_test,
        'final_rounding': restored_rounding,
        'final_projection': projection,
        'diagnostic_budget_sweep': budget_sweep,
        'config': {
            'bd_target': int(config.bd_target),
            'candidate_ratio': float(config.candidate_ratio),
            'gate_init': float(config.gate_init), 'steps': int(config.steps),
            'lr': float(config.lr), 'margin': float(config.margin),
            'clean_epsilon': clean_epsilon, 'flip_budget': primary_budget,
            'clean_margin_weight': margin_weight,
            'clean_margin_cap': margin_cap,
            'kl_temperature': kl_temperature,
            'dual_clean_init': float(config.dual_clean_init),
            'dual_lr_clean': dual_lr_clean,
            'rho_clean': rho_clean,
            'best_max_clean_kl': best_max_clean_kl,
            'best_min_clean_agreement': best_min_clean_agreement,
        },
    }
    _finalize_hard_rounding(model, records)
    if output_json is not None:
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json, 'w') as output_file:
            json.dump(result, output_file, indent=2)
        print('[GSRR] saved:', output_json)
    return model, result
