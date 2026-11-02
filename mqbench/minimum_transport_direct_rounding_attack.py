import json
import os
import numpy as np
import torch
import torch.nn.functional as F

from mqbench.utils.state import enable_quantization
from mqbench.structured_decision_core_attack import (
    _to_device,
    _effective_weight,
    _teacher_forward,
    extract_trigger_spec,
    _augment_clean_tensor,
    _apply_trigger,
    _extract_quantized_classifier,
    _build_decision_geometry,
    decision_code,
    _evaluate_cv_loader,
)
from mqbench.structured_decision_translation_attack import (
    V7FeatureTapper as ExactClassifierInputTap,
    _attacked_forward,
    _collect_attack_records,
    _verify_classifier_geometry,
)

_PROJECTION_CYCLES = 100


def build_target_boundary_geometry(geometry, target):
    target = int(target)
    weight = geometry['decision_weight'].float()
    bias = geometry['bias'].float()
    competitors = [c for c in range(weight.shape[0]) if c != target]

    diff = weight[target].unsqueeze(0) - weight[competitors]
    norms = diff.norm(p=2, dim=1)
    if (norms <= 1.0e-12).any():
        bad = [competitors[i] for i in (norms <= 1.0e-12).nonzero(as_tuple=False).flatten().tolist()]
        raise RuntimeError('Degenerate target boundaries: {}'.format(bad))

    return {
        'target': target,
        'competitors': competitors,
        'matrix': (diff / norms.unsqueeze(1)).detach(),
        'offset': ((bias[target] - bias[competitors]) / norms).detach(),
    }


def target_boundary_distances(code, boundary_geometry):
    matrix = boundary_geometry['matrix'].to(device=code.device, dtype=code.dtype)
    offset = boundary_geometry['offset'].to(device=code.device, dtype=code.dtype)
    return code @ matrix.transpose(0, 1) + offset.unsqueeze(0)


@torch.no_grad()
def estimate_target_depth(model, attack_records, tapper, cali_data, device,
                          geometry, boundary_geometry, target):
    all_distances = []
    target = int(target)
    for batch in cali_data:
        batch = _to_device(batch, device)
        logits, feature = _teacher_forward(model, attack_records, tapper, batch, device)
        mask = logits.argmax(dim=1) == target
        if not mask.any():
            continue
        code = decision_code(feature[mask], geometry['basis'])
        all_distances.append(target_boundary_distances(code, boundary_geometry).detach())

    if not all_distances:
        raise RuntimeError('No benign calibration samples are predicted as target.')

    distances = torch.cat(all_distances, dim=0)
    mean_by_boundary = distances.mean(dim=0)
    gamma = float(mean_by_boundary.min().item())
    if gamma <= 0:
        raise RuntimeError('Estimated target depth is non-positive: {}'.format(gamma))

    limiting = int(mean_by_boundary.argmin().item())
    summary = {
        'target_samples': int(distances.shape[0]),
        'target_depth': gamma,
        'limiting_competitor': int(boundary_geometry['competitors'][limiting]),
        'mean_distance_by_competitor': {
            str(c): float(mean_by_boundary[i].item())
            for i, c in enumerate(boundary_geometry['competitors'])
        },
    }
    print('[FINAL] target depth:', summary)
    return gamma, summary


@torch.no_grad()
def project_to_target_interior(teacher_code, boundary_geometry, target_depth):
    dtype = teacher_code.dtype
    code = teacher_code.detach().double()
    matrix = boundary_geometry['matrix'].to(device=code.device, dtype=torch.double)
    offset = boundary_geometry['offset'].to(device=code.device, dtype=torch.double)
    gamma = float(target_depth)

    if code.ndim != 2 or matrix.ndim != 2 or matrix.shape[1] != code.shape[1]:
        raise ValueError('Projection geometry dimension mismatch.')

    batch, constraints, rank = code.shape[0], matrix.shape[0], code.shape[1]
    y = code.clone()
    corrections = torch.zeros(batch, constraints, rank, device=code.device, dtype=torch.double)
    norm_sq = matrix.pow(2).sum(dim=1).clamp_min(1.0e-18)

    for _ in range(_PROJECTION_CYCLES):
        for j in range(constraints):
            z = y + corrections[:, j, :]
            normal = matrix[j]
            slack = z @ normal + offset[j] - gamma
            step = (-slack).clamp_min(0.0) / norm_sq[j]
            projected = z + step.unsqueeze(1) * normal.unsqueeze(0)
            corrections[:, j, :] = z - projected
            y = projected

    distances = y @ matrix.transpose(0, 1) + offset.unsqueeze(0)
    depth = distances.min(dim=1).values
    max_violation = float((gamma - distances).clamp_min(0.0).max().item())
    if max_violation > 5.0e-7:
        raise RuntimeError('Projection did not converge: {}'.format(max_violation))

    movement = (y - code).pow(2).sum(dim=1)
    active = ((distances - gamma).abs() <= 1.0e-5).sum(dim=1)
    return {
        'endpoint': y.to(dtype=dtype).detach(),
        'endpoint_depth': depth.to(dtype=dtype).detach(),
        'movement_d2': movement.to(dtype=dtype).detach(),
        'active_constraints': active.detach(),
        'max_violation': max_violation,
    }


@torch.no_grad()
def summarize_endpoint_geometry(model, attack_records, tapper, cali_data, device,
                                geometry, boundary_geometry, target_depth):
    moves, depths, active = [], [], []
    hits = total = 0
    for batch in cali_data:
        batch = _to_device(batch, device)
        _, feature = _teacher_forward(model, attack_records, tapper, batch, device)
        code = decision_code(feature, geometry['basis'])
        p = project_to_target_interior(code, boundary_geometry, target_depth)
        endpoint = p['endpoint']
        logits = F.linear(
            endpoint,
            geometry['decision_weight'].to(endpoint.device, endpoint.dtype),
            geometry['bias'].to(endpoint.device, endpoint.dtype),
        )
        hits += int((logits.argmax(1) == int(boundary_geometry['target'])).sum().item())
        total += int(endpoint.shape[0])
        moves.append(p['movement_d2'].cpu())
        depths.append(p['endpoint_depth'].cpu())
        active.append(p['active_constraints'].float().cpu())

    move = torch.cat(moves).float(); depth = torch.cat(depths).float(); act = torch.cat(active).float()
    summary = {
        'samples': int(move.numel()),
        'target_depth': float(target_depth),
        'ideal_target_rate': float(100.0 * hits / max(total, 1)),
        'ideal_depth_mean': float(depth.mean().item()),
        'ideal_depth_min': float(depth.min().item()),
        'mean_endpoint_displacement_d2': float(move.mean().item()),
        'active_constraints_mean': float(act.mean().item()),
    }
    if summary['ideal_target_rate'] < 100.0:
        raise RuntimeError('Projected endpoints are not all target-classified.')
    print('[FINAL] endpoint geometry:', summary)
    return summary


def functional_loss(clean_feature, teacher_feature, trigger_feature, geometry,
                    boundary_geometry, target_depth):
    basis = geometry['basis']
    clean_code = decision_code(clean_feature, basis)
    teacher_code = decision_code(teacher_feature, basis)
    trigger_code = decision_code(trigger_feature, basis)
    p = project_to_target_interior(teacher_code, boundary_geometry, target_depth)
    desired = p['endpoint'].to(device=trigger_code.device, dtype=trigger_code.dtype)

    clean_d2 = (clean_code - teacher_code).pow(2).sum(1)
    trigger_d2 = (trigger_code - desired).pow(2).sum(1)
    rank = max(int(clean_code.shape[1]), 1)
    total = torch.cat([clean_d2, trigger_d2]).mean() / float(rank)
    return {
        'total': total,
        'clean_code_distance': clean_d2.mean() / float(rank),
        'trigger_endpoint_distance': trigger_d2.mean() / float(rank),
        'desired_endpoint_depth_mean': p['endpoint_depth'].mean(),
        'desired_movement_d2_mean': p['movement_d2'].mean(),
    }


def direct_rounding_stats(records):
    total_valid = total_flipped = total_params = 0
    layers = []
    with torch.no_grad():
        for rec in records:
            fq = rec['fq']
            if fq.structured_field_mode != 'direct' or fq.rounding_offset is None:
                raise RuntimeError('Direct rounding state missing: {}'.format(rec['name']))
            valid = fq.structured_valid_mask.bool()
            flips = fq.current_structured_flip_mask()
            offset = fq.rounding_offset.detach()
            vn, fn, pn = int(valid.sum().item()), int(flips.sum().item()), int(offset.numel())
            total_valid += vn; total_flipped += fn; total_params += pn
            layers.append({
                'name': rec['name'],
                'valid': vn,
                'flipped': fn,
                'flip_ratio_valid': float(fn / max(vn, 1)),
                'rounding_parameters': pn,
                'offset_abs_mean': float(offset.abs().mean().item()),
                'offset_abs_max': float(offset.abs().max().item()),
                'offset_rms': float(offset.pow(2).mean().sqrt().item()),
            })
    return {
        'total_valid': total_valid,
        'total_flipped': total_flipped,
        'flip_ratio_valid': float(total_flipped / max(total_valid, 1)),
        'rounding_parameters': total_params,
        'layers': layers,
    }


def capture_direct_state(records):
    return [rec['fq'].rounding_offset.detach().cpu().clone() for rec in records]


def restore_direct_state(records, state):
    if len(records) != len(state):
        raise RuntimeError('Direct state length mismatch.')
    with torch.no_grad():
        for rec, saved in zip(records, state):
            param = rec['fq'].rounding_offset
            param.copy_(saved.to(device=param.device, dtype=param.dtype))


def finalize_direct_rounding(model, records):
    with torch.no_grad():
        for rec in records:
            module, fq = rec['module'], rec['fq']
            if fq.structured_field_mode != 'direct':
                raise RuntimeError('Finalization expects direct mode.')
            if hasattr(module, 'bn') and hasattr(module.bn, 'running_var'):
                factor = (
                    module.bn.weight / torch.sqrt(module.bn.running_var + module.bn.eps)
                ).reshape([-1] + [1] * (module.weight.ndim - 1))
                module.weight.data.copy_(
                    fq.get_structured_hard_value(module.weight.data * factor) / factor
                )
            else:
                module.weight.data.copy_(fq.get_structured_hard_value(module.weight.data))

            # Temporary optimization state does not belong to the deployed model.
            fq.rounding_offset = None
            fq.field_u = None; fq.field_v = None; fq.field_dense = None
            fq.structured_rest = None
            fq.structured_base_round = None
            fq.structured_valid_mask = None
            fq.structured_attack = False
            fq.structured_force_base = False
            fq.adaround = False

    enable_quantization(model)
    model.eval()


@torch.no_grad()
def calibration_metrics(model, attack_records, tapper, clean_data, bd_data,
                        target, device, geometry, boundary_geometry,
                        target_depth, trigger):
    target = int(target)
    basis = geometry['basis'].to(device=device)
    rank = max(int(geometry['rank']), 1)
    clean_total = clean_agree = 0; clean_sum = 0.0
    paired_total = paired_hits = 0; trigger_sum = 0.0
    actual_depth_all, desired_depth_all = [], []
    per_source = {c: {'count':0,'hits':0,'trigger_d2_sum':0.0,'actual_depth_sum':0.0}
                  for c in range(int(geometry['num_classes']))}

    for batch in clean_data:
        batch = _to_device(batch, device)
        teacher_logits, teacher_feature = _teacher_forward(model, attack_records, tapper, batch, device)
        attacked_logits, attacked_feature = _attacked_forward(model, tapper, batch, device)
        pseudo = teacher_logits.argmax(1)
        clean_agree += int((attacked_logits.argmax(1) == pseudo).sum().item())
        teacher_code = decision_code(teacher_feature, basis.to(dtype=teacher_feature.dtype))
        attacked_code = decision_code(attacked_feature, basis.to(dtype=attacked_feature.dtype))
        clean_d2 = (attacked_code - teacher_code).pow(2).sum(1)
        clean_sum += float(clean_d2.sum().item()); clean_total += int(pseudo.numel())

        p = project_to_target_interior(teacher_code, boundary_geometry, target_depth)
        triggered = _apply_trigger(batch, trigger)
        trig_logits, trig_feature = _attacked_forward(model, tapper, triggered, device)
        trig_code = decision_code(trig_feature, basis.to(dtype=trig_feature.dtype))
        desired = p['endpoint'].to(device=trig_code.device, dtype=trig_code.dtype)
        trig_d2 = (trig_code - desired).pow(2).sum(1)
        hit = trig_logits.argmax(1) == target
        actual_depth = target_boundary_distances(trig_code, boundary_geometry).min(1).values
        desired_depth = p['endpoint_depth'].to(actual_depth.device)

        paired_total += int(hit.numel()); paired_hits += int(hit.sum().item())
        trigger_sum += float(trig_d2.sum().item())
        actual_depth_all.append(actual_depth.cpu()); desired_depth_all.append(desired_depth.cpu())

        for cls in per_source:
            mask = pseudo == cls; count = int(mask.sum().item())
            if count == 0: continue
            item = per_source[cls]; item['count'] += count; item['hits'] += int(hit[mask].sum().item())
            item['trigger_d2_sum'] += float(trig_d2[mask].sum().item())
            item['actual_depth_sum'] += float(actual_depth[mask].sum().item())

    bd_total = bd_hits = 0
    for batch in bd_data:
        batch = _to_device(batch, device)
        logits, _ = _attacked_forward(model, tapper, batch, device)
        pred = logits.argmax(1); bd_hits += int((pred == target).sum().item()); bd_total += int(pred.numel())

    actual = torch.cat(actual_depth_all).float(); desired = torch.cat(desired_depth_all).float()
    source = {}
    for cls, item in per_source.items():
        n = int(item['count'])
        if n == 0: continue
        source[str(cls)] = {
            'count': n,
            'paired_asr': float(100.0 * item['hits'] / n),
            'trigger_endpoint_distance': float(item['trigger_d2_sum'] / n / float(rank)),
            'actual_target_depth_mean': float(item['actual_depth_sum'] / n),
        }

    return {
        'clean_teacher_agreement': float(100.0 * clean_agree / max(clean_total,1)),
        'clean_code_distance': float(clean_sum / max(clean_total,1) / float(rank)),
        'asr': float(100.0 * bd_hits / max(bd_total,1)),
        'paired_asr': float(100.0 * paired_hits / max(paired_total,1)),
        'trigger_endpoint_distance': float(trigger_sum / max(paired_total,1) / float(rank)),
        'actual_target_depth_mean': float(actual.mean().item()),
        'actual_target_depth_p10': float(torch.quantile(actual,0.10).item()),
        'actual_target_depth_min': float(actual.min().item()),
        'desired_target_depth_mean': float(desired.mean().item()),
        'desired_target_depth_min': float(desired.min().item()),
        'clean_samples': int(clean_total),
        'paired_samples': int(paired_total),
        'bd_samples': int(bd_total),
        'per_source': source,
    }


def minimum_transport_direct_rounding_attack(
    model, cali_data, cali_data_bd, trigger_loader, config,
    output_json=None, test_loader=None, test_loader_bd=None,
):
    device = next(model.parameters()).device
    model.eval(); enable_quantization(model)
    for parameter in model.parameters(): parameter.requires_grad_(False)

    target = int(config.bd_target)
    tapper = ExactClassifierInputTap(model, str(config.semantic_module))
    attack_records, frozen_records = _collect_attack_records(
        model, [str(config.classifier_module)]
    )

    rounding_parameters = []
    for rec in attack_records:
        rec['fq'].init_direct_rounding(_effective_weight(rec['module']).detach())
        rounding_parameters.extend(rec['fq'].structured_parameters())

    step0_stats = direct_rounding_stats(attack_records)
    if step0_stats['total_flipped'] != 0:
        raise RuntimeError('Step-0 direct rounding differs from benign PTQ.')
    print('[FINAL] temporary rounding parameters:', step0_stats['rounding_parameters'])

    trigger_spec = extract_trigger_spec(trigger_loader)
    trigger = trigger_spec['trigger_normalized'].to(device)

    _, classifier_weight, classifier_bias = _extract_quantized_classifier(
        model, str(config.classifier_module)
    )
    geometry = _build_decision_geometry(
        classifier_weight, classifier_bias, svd_tol=float(config.decision_svd_tol)
    )
    geometry_check = _verify_classifier_geometry(
        model=model, attack_records=attack_records, tapper=tapper,
        batch=_to_device(cali_data[0], device), device=device, geometry=geometry,
        tolerance=float(config.geometry_logit_tolerance),
    )

    boundary_geometry = build_target_boundary_geometry(geometry, target)
    target_depth, depth_summary = estimate_target_depth(
        model, attack_records, tapper, cali_data, device, geometry,
        boundary_geometry, target,
    )
    endpoint_summary = summarize_endpoint_geometry(
        model, attack_records, tapper, cali_data, device, geometry,
        boundary_geometry, target_depth,
    )

    baseline = calibration_metrics(
        model, attack_records, tapper, cali_data, cali_data_bd, target, device,
        geometry, boundary_geometry, target_depth, trigger,
    )
    print('[FINAL] step-0 calibration:', baseline)
    if baseline['clean_teacher_agreement'] < 99.999 or baseline['clean_code_distance'] > 1.0e-10:
        raise RuntimeError('Step-0 direct model != benign PTQ.')

    optimizer = torch.optim.Adam(rounding_parameters, lr=float(config.lr), weight_decay=0.0)
    steps = int(config.steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=float(config.lr) * 0.1
    )
    eval_interval = int(config.eval_interval)
    test_interval = int(config.diagnostic_test_interval)

    best_state = capture_direct_state(attack_records)
    c0, a0 = baseline['clean_teacher_agreement'], baseline['asr']
    best_score = 2.0*c0*a0/max(c0+a0,1.0e-12); best_step = 0; history = []

    for step in range(1, steps+1):
        batch = _to_device(cali_data[int(np.random.randint(len(cali_data)))], device)
        clean_view = _augment_clean_tensor(
            batch, pad=int(config.aug_pad), hflip_prob=float(config.hflip_prob)
        )
        _, teacher_feature = _teacher_forward(model, attack_records, tapper, clean_view, device)
        trigger_view = _apply_trigger(clean_view, trigger)

        optimizer.zero_grad()
        _, clean_feature = _attacked_forward(model, tapper, clean_view, device)
        _, trigger_feature = _attacked_forward(model, tapper, trigger_view, device)
        losses = functional_loss(
            clean_feature, teacher_feature, trigger_feature, geometry,
            boundary_geometry, target_depth,
        )
        loss = losses['total']; loss.backward()
        torch.nn.utils.clip_grad_norm_(rounding_parameters, float(config.grad_clip))
        optimizer.step(); scheduler.step()

        if step == 1 or step % eval_interval == 0 or step == steps:
            metrics = calibration_metrics(
                model, attack_records, tapper, cali_data, cali_data_bd, target,
                device, geometry, boundary_geometry, target_depth, trigger,
            )
            stats = direct_rounding_stats(attack_records)
            clean_agreement, asr = metrics['clean_teacher_agreement'], metrics['asr']
            score = 2.0*clean_agreement*asr/max(clean_agreement+asr,1.0e-12)
            row = {
                'step': step,
                'train_total_loss': float(loss.detach().item()),
                'train_clean_code_distance': float(losses['clean_code_distance'].detach().item()),
                'train_trigger_endpoint_distance': float(losses['trigger_endpoint_distance'].detach().item()),
                'train_desired_endpoint_depth_mean': float(losses['desired_endpoint_depth_mean'].detach().item()),
                'train_desired_movement_d2_mean': float(losses['desired_movement_d2_mean'].detach().item()),
                'score': float(score), **metrics,
                'flip_ratio_valid': float(stats['flip_ratio_valid']),
                'total_flipped': int(stats['total_flipped']),
                'rounding_parameters': int(stats['rounding_parameters']),
            }
            if (
                bool(config.diagnostic_test_eval) and test_loader is not None and test_loader_bd is not None
                and (step % test_interval == 0 or step == steps)
            ):
                row['diagnostic_test_ca'] = _evaluate_cv_loader(model, test_loader, device)
                row['diagnostic_test_asr'] = _evaluate_cv_loader(model, test_loader_bd, device)
            history.append(row); print('[FINAL]', row)
            # Strict improvement retains the first calibration-saturated checkpoint.
            if score > best_score:
                best_score = float(score); best_step = int(step)
                best_state = capture_direct_state(attack_records)
                print('[FINAL] new best:', {'step':best_step,'score':best_score,'clean_agreement':clean_agreement,'asr':asr})

    restore_direct_state(attack_records, best_state)
    restored = calibration_metrics(
        model, attack_records, tapper, cali_data, cali_data_bd, target, device,
        geometry, boundary_geometry, target_depth, trigger,
    )
    best_stats = direct_rounding_stats(attack_records)

    pre_test = {}
    if test_loader is not None and test_loader_bd is not None:
        pre_test = {
            'ca': _evaluate_cv_loader(model, test_loader, device),
            'asr': _evaluate_cv_loader(model, test_loader_bd, device),
        }
    print('[FINAL] restored best:', {'step':best_step,'calibration':restored,'test':pre_test})

    finalize_direct_rounding(model, attack_records)
    post_test = {}
    if test_loader is not None and test_loader_bd is not None:
        post_test = {
            'ca': _evaluate_cv_loader(model, test_loader, device),
            'asr': _evaluate_cv_loader(model, test_loader_bd, device),
        }
        if abs(post_test['ca']-pre_test['ca']) > 1.0e-9 or abs(post_test['asr']-pre_test['asr']) > 1.0e-9:
            raise RuntimeError('Finalization changed behavior: pre={} post={}'.format(pre_test, post_test))

    leaked = [name for name, p in model.named_parameters() if 'rounding_offset' in name and p is not None]
    if leaked:
        raise RuntimeError('Temporary offsets remain after finalization: {}'.format(leaked[:20]))

    result = {
        'method': 'minimum_transport_direct_legal_rounding',
        'target': target,
        'semantic_module': str(config.semantic_module),
        'classifier_module': str(config.classifier_module),
        'attacked_modules': [r['name'] for r in attack_records],
        'frozen_modules': [r['name'] for r in frozen_records],
        'classifier_geometry': {
            'feature_dim': int(geometry['feature_dim']),
            'num_classes': int(geometry['num_classes']),
            'decision_rank': int(geometry['rank']),
            'null_dim': int(geometry['null_dim']),
            'singular_values': geometry['singular_values'],
            'geometry_max_logit_error': geometry_check['max_logit_error'],
            'decision_code_argmax_agreement': geometry_check['argmax_agreement'],
        },
        'target_depth': depth_summary,
        'endpoint_geometry': endpoint_summary,
        'baseline_calibration': baseline,
        'history': history,
        'best_step': best_step,
        'best_score': best_score,
        'best_calibration': restored,
        'best_rounding_stats': best_stats,
        'best_test': post_test,
        'pre_finalize_test': pre_test,
        'temporary_state_removed': True,
        'trigger_size': int(trigger_spec['trigger_size']),
        'implementation': {
            'projection_solver': 'dykstra_halfspaces',
            'projection_cycles': _PROJECTION_CYCLES,
            'steps': int(config.steps),
            'lr': float(config.lr),
            'grad_clip': float(config.grad_clip),
            'aug_pad': int(config.aug_pad),
            'hflip_prob': float(config.hflip_prob),
        },
    }

    tapper.close()
    if output_json is not None:
        output_dir = os.path.dirname(output_json)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        with open(output_json, 'w') as f: json.dump(result, f, indent=2)
        print('[FINAL] saved:', output_json)
    return model, result
