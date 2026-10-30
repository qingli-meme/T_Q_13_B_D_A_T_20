import json
import os

import numpy as np
import torch

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
    _structured_stats,
    _evaluate_cv_loader,
    _capture_field_state,
    _restore_field_state,
    _finalize_structured_rounding,
)

from mqbench.structured_decision_translation_attack import (
    V7FeatureTapper,
    _attacked_forward,
    _collect_attack_records,
    _verify_classifier_geometry,
)

from mqbench.structured_target_compatible_attack import (
    build_target_margin_geometry,
    target_margin_coordinates,
)

from mqbench.structured_deep_target_projection_attack import (
    project_to_deep_target_set,
)


@torch.no_grad()
def build_target_reference_core(
    model,
    attack_records,
    tapper,
    cali_data,
    device,
    geometry,
    target,
    min_target_count,
):
    """Estimate only the benign target reference c_t."""
    target = int(target)
    codes = []

    for batch in cali_data:
        batch = _to_device(batch, device)
        teacher_logits, teacher_feature = _teacher_forward(
            model, attack_records, tapper, batch, device
        )
        pseudo = teacher_logits.argmax(dim=1)
        mask = pseudo == target
        if mask.any():
            code = decision_code(
                teacher_feature[mask], geometry["basis"]
            )
            codes.append(code.detach())

    if not codes:
        raise RuntimeError(
            "No calibration samples pseudo-labeled as target."
        )

    target_codes = torch.cat(codes, dim=0)
    if target_codes.shape[0] < int(min_target_count):
        raise RuntimeError(
            "Too few target-reference samples: {} < {}".format(
                int(target_codes.shape[0]), int(min_target_count)
            )
        )

    return {
        "core": target_codes.mean(dim=0, keepdim=True).detach(),
        "count": int(target_codes.shape[0]),
    }


def _select_attack_scope(attack_records, frozen_records, scope):
    """Select which feature-extractor modules receive structured fields."""
    scope = str(scope)

    if scope == "all":
        selected = list(attack_records)
        extra_frozen = []
    elif scope == "late34":
        selected = [
            r for r in attack_records
            if r["name"].startswith("layer3.")
            or r["name"].startswith("layer4.")
        ]
        selected_ids = {id(r) for r in selected}
        extra_frozen = [
            r for r in attack_records if id(r) not in selected_ids
        ]
    elif scope == "layer4":
        selected = [
            r for r in attack_records
            if r["name"].startswith("layer4.")
        ]
        selected_ids = {id(r) for r in selected}
        extra_frozen = [
            r for r in attack_records if id(r) not in selected_ids
        ]
    else:
        raise ValueError("Unknown attack_scope: {}".format(scope))

    if not selected:
        raise RuntimeError(
            "Attack scope '{}' selected no modules.".format(scope)
        )

    frozen = list(frozen_records) + extra_frozen
    print("[SLIM] attack_scope:", scope)
    print("[SLIM] attacked modules:", [r["name"] for r in selected])
    print("[SLIM] frozen modules:", [r["name"] for r in frozen])
    return selected, frozen


@torch.no_grad()
def project_along_target_core_ray(
    teacher_code,
    margin_geometry,
    depth_floor,
    target_core,
):
    """
    First feasible point on the line segment u0 -> c_t.

    q_j(u(alpha)) = q_j(u0) + alpha [q_j(c_t)-q_j(u0)].
    For every violated constraint q_j(u0) < gamma,

        alpha_j = (gamma-q_j(u0)) / (q_j(c_t)-q_j(u0)).

    alpha* = max_j alpha_j.
    """
    code = teacher_code.detach()
    core = target_core.to(device=code.device, dtype=code.dtype)
    if core.ndim == 1:
        core = core.unsqueeze(0)

    q0 = target_margin_coordinates(code, margin_geometry)
    qcore = target_margin_coordinates(core, margin_geometry)
    gamma = float(depth_floor)

    violated = q0 < gamma
    denominator = qcore - q0

    if (denominator[violated] <= 0).any():
        raise RuntimeError(
            "Core-ray feasibility denominator is non-positive "
            "for a violated constraint."
        )

    numer = (gamma - q0).clamp_min(0.0)
    alpha_each = torch.zeros_like(q0)
    expanded_denominator = denominator.expand_as(q0)
    alpha_each[violated] = (
        numer[violated] / expanded_denominator[violated]
    )

    alpha = alpha_each.max(dim=1).values.clamp(min=0.0, max=1.0)
    endpoint = code + alpha.unsqueeze(1) * (core - code)
    endpoint_margin = target_margin_coordinates(endpoint, margin_geometry)
    endpoint_depth = endpoint_margin.min(dim=1).values

    max_violation = float(
        (gamma - endpoint_margin).clamp_min(0.0).max().item()
    )
    if max_violation > 5.0e-5:
        raise RuntimeError(
            "Core-ray endpoint is infeasible: {}".format(max_violation)
        )

    movement_d2 = (endpoint - code).pow(2).sum(dim=1)
    v6_core_d2 = (core - code).pow(2).sum(dim=1)
    tol = 5.0e-6 * torch.maximum(
        torch.ones_like(v6_core_d2), v6_core_d2
    )
    if (movement_d2 > v6_core_d2 + tol).any():
        raise RuntimeError(
            "Core-ray movement exceeded target-core movement."
        )

    return {
        "endpoint": endpoint.detach(),
        "endpoint_margin": endpoint_margin.detach(),
        "endpoint_depth": endpoint_depth.detach(),
        "movement_d2": movement_d2.detach(),
        "v6_core_d2": v6_core_d2.detach(),
        "alpha": alpha.detach(),
        "active_constraints": (
            (endpoint_margin - gamma).abs() <= 1.0e-5
        ).sum(dim=1).detach(),
        "max_violation": max_violation,
    }


@torch.no_grad()
def build_endpoint(
    teacher_code,
    margin_geometry,
    depth_floor,
    target_core,
    endpoint_mode,
):
    endpoint_mode = str(endpoint_mode)
    if endpoint_mode == "dykstra":
        result = project_to_deep_target_set(
            teacher_code=teacher_code,
            margin_geometry=margin_geometry,
            depth_floor=depth_floor,
            target_core=target_core,
        )
        result = dict(result)
        result["alpha"] = None
        return result
    if endpoint_mode == "core_ray":
        return project_along_target_core_ray(
            teacher_code=teacher_code,
            margin_geometry=margin_geometry,
            depth_floor=depth_floor,
            target_core=target_core,
        )
    raise ValueError("Unknown endpoint_mode: {}".format(endpoint_mode))


@torch.no_grad()
def build_static_geometry_summary(
    model,
    attack_records,
    tapper,
    cali_data,
    device,
    geometry,
    margin_geometry,
    target_core,
    target,
    depth_floor,
    endpoint_mode,
    sample_scope,
):
    target = int(target)
    sample_scope = str(sample_scope)
    move_all, v6_all, depth_all, active_all, alpha_all = [], [], [], [], []
    total = 0
    target_samples = 0

    for batch in cali_data:
        batch = _to_device(batch, device)
        teacher_logits, teacher_feature = _teacher_forward(
            model, attack_records, tapper, batch, device
        )
        pseudo = teacher_logits.argmax(dim=1)

        if sample_scope == "non_target":
            mask = pseudo != target
        elif sample_scope == "all":
            mask = torch.ones_like(pseudo, dtype=torch.bool)
        else:
            raise ValueError("Unknown sample_scope: {}".format(sample_scope))

        if not mask.any():
            continue

        code = decision_code(teacher_feature[mask], geometry["basis"])
        selected_pseudo = pseudo[mask]
        endpoint = build_endpoint(
            teacher_code=code,
            margin_geometry=margin_geometry,
            depth_floor=depth_floor,
            target_core=target_core,
            endpoint_mode=endpoint_mode,
        )

        move_all.append(endpoint["movement_d2"].cpu())
        v6_all.append(endpoint["v6_core_d2"].cpu())
        depth_all.append(endpoint["endpoint_depth"].cpu())
        active_all.append(endpoint["active_constraints"].float().cpu())
        if endpoint["alpha"] is not None:
            alpha_all.append(endpoint["alpha"].cpu())

        total += int(code.shape[0])
        target_samples += int((selected_pseudo == target).sum().item())

    move = torch.cat(move_all, dim=0).float()
    v6 = torch.cat(v6_all, dim=0).float()
    depth = torch.cat(depth_all, dim=0).float()
    active = torch.cat(active_all, dim=0).float()

    if (
        move
        > v6 + 5.0e-5 * torch.maximum(torch.ones_like(v6), v6)
    ).any():
        raise RuntimeError("Static endpoint movement exceeded V6 movement.")

    if depth.min().item() + 5.0e-5 < float(depth_floor):
        raise RuntimeError("Static endpoint fell below required depth floor.")

    summary = {
        "samples": int(total),
        "included_target_samples": int(target_samples),
        "endpoint_mode": str(endpoint_mode),
        "sample_scope": str(sample_scope),
        "depth_floor": float(depth_floor),
        "ideal_endpoint_depth_mean": float(depth.mean().item()),
        "ideal_endpoint_depth_min": float(depth.min().item()),
        "mean_endpoint_displacement_d2": float(move.mean().item()),
        "mean_v6_core_displacement_d2": float(v6.mean().item()),
        "endpoint_to_v6_energy_ratio": float(
            move.mean().item() / max(v6.mean().item(), 1.0e-12)
        ),
        "strictly_less_movement_than_v6_rate": float(
            (move < v6 - 1.0e-6).float().mean().item() * 100.0
        ),
        "active_constraints_mean": float(active.mean().item()),
    }

    if alpha_all:
        alpha = torch.cat(alpha_all, dim=0).float()
        summary.update({
            "core_ray_alpha_mean": float(alpha.mean().item()),
            "core_ray_alpha_p10": float(torch.quantile(alpha, 0.10).item()),
            "core_ray_alpha_median": float(torch.quantile(alpha, 0.50).item()),
            "core_ray_alpha_p90": float(torch.quantile(alpha, 0.90).item()),
            "core_ray_alpha_max": float(alpha.max().item()),
        })

    print("[SLIM] static geometry:", summary)
    return summary


def slimming_loss(
    clean_feature,
    clean_reference,
    trigger_feature,
    selected_mask,
    geometry,
    margin_geometry,
    target_core,
    depth_floor,
    endpoint_mode,
    use_clean_anchor,
):
    basis = geometry["basis"]
    clean_code = decision_code(clean_feature, basis)
    teacher_code = decision_code(clean_reference, basis)
    trigger_code = decision_code(trigger_feature, basis)
    selected_teacher_code = teacher_code[selected_mask]

    endpoint = build_endpoint(
        teacher_code=selected_teacher_code,
        margin_geometry=margin_geometry,
        depth_floor=depth_floor,
        target_core=target_core,
        endpoint_mode=endpoint_mode,
    )
    desired_endpoint = endpoint["endpoint"].to(
        device=trigger_code.device, dtype=trigger_code.dtype
    )

    clean_d2 = (clean_code - teacher_code).pow(2).sum(dim=1)
    trigger_d2 = (trigger_code - desired_endpoint).pow(2).sum(dim=1)
    rank = max(int(clean_code.shape[1]), 1)

    if bool(use_clean_anchor):
        total = torch.cat([clean_d2, trigger_d2], dim=0).mean() / float(rank)
    else:
        total = trigger_d2.mean() / float(rank)

    return {
        "total": total,
        "clean_code_distance": clean_d2.mean() / float(rank),
        "trigger_endpoint_distance": trigger_d2.mean() / float(rank),
        "desired_endpoint_depth_mean": endpoint["endpoint_depth"].mean(),
        "desired_movement_d2_mean": endpoint["movement_d2"].mean(),
        "desired_v6_core_d2_mean": endpoint["v6_core_d2"].mean(),
    }


@torch.no_grad()
def _calibration_metrics(
    model,
    attack_records,
    tapper,
    clean_data,
    bd_data,
    target,
    device,
    geometry,
    margin_geometry,
    target_core,
    depth_floor,
    endpoint_mode,
    sample_scope,
    trigger,
):
    target = int(target)
    sample_scope = str(sample_scope)
    basis = geometry["basis"].to(device=device)
    rank = max(int(geometry["rank"]), 1)

    clean_total = 0
    clean_agree = 0
    clean_distance_sum = 0.0
    selected_total = 0
    paired_hits = 0
    trigger_distance_sum = 0.0
    actual_depth_all, desired_depth_all = [], []
    desired_move_all, desired_v6_all = [], []

    per_source = {
        cls: {
            "count": 0,
            "hits": 0,
            "trigger_d2_sum": 0.0,
            "actual_depth_sum": 0.0,
        }
        for cls in range(int(geometry["num_classes"]))
    }

    for batch in clean_data:
        batch = _to_device(batch, device)
        teacher_logits, teacher_feature = _teacher_forward(
            model, attack_records, tapper, batch, device
        )
        attacked_logits, attacked_feature = _attacked_forward(
            model, tapper, batch, device
        )
        pseudo = teacher_logits.argmax(dim=1)

        clean_agree += int(
            (attacked_logits.argmax(dim=1) == pseudo).sum().item()
        )
        teacher_code = decision_code(
            teacher_feature, basis.to(dtype=teacher_feature.dtype)
        )
        attacked_code = decision_code(
            attacked_feature, basis.to(dtype=attacked_feature.dtype)
        )
        clean_d2 = (attacked_code - teacher_code).pow(2).sum(dim=1)
        clean_distance_sum += float(clean_d2.sum().item())
        clean_total += int(pseudo.numel())

        if sample_scope == "non_target":
            selected_mask = pseudo != target
        elif sample_scope == "all":
            selected_mask = torch.ones_like(pseudo, dtype=torch.bool)
        else:
            raise ValueError("Unknown sample_scope: {}".format(sample_scope))

        if not selected_mask.any():
            continue

        selected_labels = pseudo[selected_mask]
        selected_teacher_code = teacher_code[selected_mask]
        endpoint = build_endpoint(
            teacher_code=selected_teacher_code,
            margin_geometry=margin_geometry,
            depth_floor=depth_floor,
            target_core=target_core,
            endpoint_mode=endpoint_mode,
        )

        triggered = _apply_trigger(batch[selected_mask], trigger)
        trigger_logits, trigger_feature = _attacked_forward(
            model, tapper, triggered, device
        )
        trigger_code = decision_code(
            trigger_feature, basis.to(dtype=trigger_feature.dtype)
        )
        desired_endpoint = endpoint["endpoint"].to(
            device=trigger_code.device, dtype=trigger_code.dtype
        )
        trigger_d2 = (trigger_code - desired_endpoint).pow(2).sum(dim=1)
        hit = trigger_logits.argmax(dim=1) == target
        actual_margin = target_margin_coordinates(trigger_code, margin_geometry)
        actual_depth = actual_margin.min(dim=1).values
        desired_depth = endpoint["endpoint_depth"].to(actual_depth.device)

        selected_total += int(hit.numel())
        paired_hits += int(hit.sum().item())
        trigger_distance_sum += float(trigger_d2.sum().item())
        actual_depth_all.append(actual_depth.cpu())
        desired_depth_all.append(desired_depth.cpu())
        desired_move_all.append(endpoint["movement_d2"].cpu())
        desired_v6_all.append(endpoint["v6_core_d2"].cpu())

        for cls in per_source:
            cls_mask = selected_labels == cls
            count = int(cls_mask.sum().item())
            if count == 0:
                continue
            per_source[cls]["count"] += count
            per_source[cls]["hits"] += int(hit[cls_mask].sum().item())
            per_source[cls]["trigger_d2_sum"] += float(
                trigger_d2[cls_mask].sum().item()
            )
            per_source[cls]["actual_depth_sum"] += float(
                actual_depth[cls_mask].sum().item()
            )

    bd_total = 0
    bd_hits = 0
    for batch in bd_data:
        batch = _to_device(batch, device)
        logits, _ = _attacked_forward(model, tapper, batch, device)
        pred = logits.argmax(dim=1)
        bd_hits += int((pred == target).sum().item())
        bd_total += int(pred.numel())

    if actual_depth_all:
        actual_depth_tensor = torch.cat(actual_depth_all, dim=0).float()
        desired_depth_tensor = torch.cat(desired_depth_all, dim=0).float()
        desired_move_tensor = torch.cat(desired_move_all, dim=0).float()
        desired_v6_tensor = torch.cat(desired_v6_all, dim=0).float()
    else:
        actual_depth_tensor = torch.zeros(1)
        desired_depth_tensor = torch.zeros(1)
        desired_move_tensor = torch.zeros(1)
        desired_v6_tensor = torch.ones(1)

    per_source_result = {}
    for cls, item in per_source.items():
        count = int(item["count"])
        if count == 0:
            continue
        per_source_result[str(cls)] = {
            "count": count,
            "paired_asr": float(100.0 * item["hits"] / count),
            "trigger_endpoint_distance": float(
                item["trigger_d2_sum"] / count / float(rank)
            ),
            "actual_target_depth_mean": float(
                item["actual_depth_sum"] / count
            ),
        }

    return {
        "clean_teacher_agreement": float(
            100.0 * clean_agree / max(clean_total, 1)
        ),
        "clean_code_distance": float(
            clean_distance_sum / max(clean_total, 1) / float(rank)
        ),
        "asr": float(100.0 * bd_hits / max(bd_total, 1)),
        "paired_asr": float(100.0 * paired_hits / max(selected_total, 1)),
        "trigger_endpoint_distance": float(
            trigger_distance_sum / max(selected_total, 1) / float(rank)
        ),
        "actual_target_depth_mean": float(actual_depth_tensor.mean().item()),
        "actual_target_depth_p10": float(
            torch.quantile(actual_depth_tensor, 0.10).item()
        ),
        "actual_target_depth_min": float(actual_depth_tensor.min().item()),
        "desired_target_depth_mean": float(desired_depth_tensor.mean().item()),
        "desired_target_depth_min": float(desired_depth_tensor.min().item()),
        "desired_movement_d2_mean": float(desired_move_tensor.mean().item()),
        "desired_v6_core_d2_mean": float(desired_v6_tensor.mean().item()),
        "desired_movement_to_v6_ratio": float(
            desired_move_tensor.mean().item()
            / max(desired_v6_tensor.mean().item(), 1.0e-12)
        ),
        "clean_samples": int(clean_total),
        "selected_samples": int(selected_total),
        "bd_samples": int(bd_total),
        "per_source": per_source_result,
    }


def structured_slimming_attack(
    model,
    cali_data,
    cali_data_bd,
    trigger_loader,
    config,
    output_json=None,
    test_loader=None,
    test_loader_bd=None,
):
    """Unified V12 method-slimming runner."""
    device = next(model.parameters()).device
    model.eval()
    enable_quantization(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    variant_name = str(config.variant_name)
    endpoint_mode = str(config.endpoint_mode)
    depth_mode = str(config.depth_mode)
    use_clean_anchor = bool(config.use_clean_anchor)
    use_augmentation = bool(config.use_augmentation)
    sample_scope = str(config.sample_scope)
    attack_scope = str(config.attack_scope)

    print("[SLIM] variant:", variant_name)
    print("[SLIM] structural config:", {
        "endpoint_mode": endpoint_mode,
        "depth_mode": depth_mode,
        "use_clean_anchor": use_clean_anchor,
        "use_augmentation": use_augmentation,
        "sample_scope": sample_scope,
        "field_rank": int(config.field_rank),
        "attack_scope": attack_scope,
    })

    tapper = V7FeatureTapper(model, str(config.semantic_module))
    attack_records_all, frozen_records = _collect_attack_records(
        model, list(config.frozen_field_modules)
    )
    attack_records, frozen_records = _select_attack_scope(
        attack_records_all, frozen_records, attack_scope
    )

    field_params = []
    for record in attack_records:
        fq = record["fq"]
        fq.init_structured_attack(
            _effective_weight(record["module"]).detach(),
            int(config.field_rank),
            float(config.delta_bound),
            float(config.round_temperature),
            float(config.field_init_std),
        )
        field_params.extend([fq.field_u, fq.field_v])

    step0_stats = _structured_stats(attack_records)
    if step0_stats["total_flipped"] != 0:
        raise RuntimeError("Step-0 structured field changed rounding.")

    trigger_spec = extract_trigger_spec(trigger_loader)
    trigger = trigger_spec["trigger_normalized"].to(device)
    target = int(config.bd_target)

    _, classifier_weight, classifier_bias = _extract_quantized_classifier(
        model, str(config.classifier_module)
    )
    geometry = _build_decision_geometry(
        classifier_weight,
        classifier_bias,
        svd_tol=float(config.decision_svd_tol),
    )
    if geometry["rank"] != geometry["num_classes"] - 1:
        raise RuntimeError(
            "Slimming suite requires full C-1 decision rank."
        )

    geometry_check = _verify_classifier_geometry(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        batch=_to_device(cali_data[0], device),
        device=device,
        geometry=geometry,
        tolerance=float(config.geometry_logit_tolerance),
    )

    target_reference = build_target_reference_core(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        cali_data=cali_data,
        device=device,
        geometry=geometry,
        target=target,
        min_target_count=int(config.min_target_count),
    )
    target_core = target_reference["core"]
    margin_geometry = build_target_margin_geometry(
        geometry=geometry, target=target
    )
    target_core_margin = target_margin_coordinates(
        target_core, margin_geometry
    )
    target_core_depth = float(target_core_margin.min().item())
    if target_core_depth <= 0:
        raise RuntimeError(
            "Target reference has non-positive target depth."
        )

    if depth_mode == "core":
        depth_floor = target_core_depth
    elif depth_mode == "zero":
        depth_floor = 0.0
    else:
        raise ValueError("Unknown depth_mode: {}".format(depth_mode))

    print("[SLIM] target reference:", {
        "count": target_reference["count"],
        "target_core_depth": target_core_depth,
        "used_depth_floor": depth_floor,
    })

    static_geometry = build_static_geometry_summary(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        cali_data=cali_data,
        device=device,
        geometry=geometry,
        margin_geometry=margin_geometry,
        target_core=target_core,
        target=target,
        depth_floor=depth_floor,
        endpoint_mode=endpoint_mode,
        sample_scope=sample_scope,
    )

    baseline = _calibration_metrics(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        clean_data=cali_data,
        bd_data=cali_data_bd,
        target=target,
        device=device,
        geometry=geometry,
        margin_geometry=margin_geometry,
        target_core=target_core,
        depth_floor=depth_floor,
        endpoint_mode=endpoint_mode,
        sample_scope=sample_scope,
        trigger=trigger,
    )
    print("[SLIM] step-0 calibration:", baseline)

    if baseline["clean_teacher_agreement"] < 99.999:
        raise RuntimeError("Step-0 attacked model != benign PTQ.")
    if baseline["clean_code_distance"] > 1.0e-10:
        raise RuntimeError(
            "Step-0 clean decision-code distance is not zero."
        )

    optimizer = torch.optim.Adam(
        field_params,
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    steps = int(config.steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=float(config.lr) * 0.1,
    )
    eval_interval = int(config.eval_interval)
    test_interval = int(config.diagnostic_test_interval)

    best_state = _capture_field_state(attack_records)
    clean0 = baseline["clean_teacher_agreement"]
    asr0 = baseline["asr"]
    best_score = 2.0 * clean0 * asr0 / max(clean0 + asr0, 1.0e-12)
    best_step = 0
    history = []

    for step in range(1, steps + 1):
        batch = _to_device(
            cali_data[int(np.random.randint(len(cali_data)))], device
        )

        if use_augmentation:
            clean_view = _augment_clean_tensor(
                batch,
                pad=int(config.aug_pad),
                hflip_prob=float(config.hflip_prob),
            )
        else:
            clean_view = batch

        teacher_logits, teacher_feature = _teacher_forward(
            model, attack_records, tapper, clean_view, device
        )
        pseudo = teacher_logits.argmax(dim=1)

        if sample_scope == "non_target":
            selected_mask = pseudo != target
        elif sample_scope == "all":
            selected_mask = torch.ones_like(pseudo, dtype=torch.bool)
        else:
            raise ValueError("Unknown sample_scope: {}".format(sample_scope))

        if not selected_mask.any():
            continue

        triggered_view = _apply_trigger(
            clean_view[selected_mask], trigger
        )

        optimizer.zero_grad()
        _, clean_feature = _attacked_forward(
            model, tapper, clean_view, device
        )
        _, trigger_feature = _attacked_forward(
            model, tapper, triggered_view, device
        )

        losses = slimming_loss(
            clean_feature=clean_feature,
            clean_reference=teacher_feature,
            trigger_feature=trigger_feature,
            selected_mask=selected_mask,
            geometry=geometry,
            margin_geometry=margin_geometry,
            target_core=target_core,
            depth_floor=depth_floor,
            endpoint_mode=endpoint_mode,
            use_clean_anchor=use_clean_anchor,
        )
        loss = losses["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            field_params, float(config.grad_clip)
        )
        optimizer.step()
        scheduler.step()

        if step == 1 or step % eval_interval == 0 or step == steps:
            metrics = _calibration_metrics(
                model=model,
                attack_records=attack_records,
                tapper=tapper,
                clean_data=cali_data,
                bd_data=cali_data_bd,
                target=target,
                device=device,
                geometry=geometry,
                margin_geometry=margin_geometry,
                target_core=target_core,
                depth_floor=depth_floor,
                endpoint_mode=endpoint_mode,
                sample_scope=sample_scope,
                trigger=trigger,
            )
            stats = _structured_stats(attack_records)
            clean_agreement = metrics["clean_teacher_agreement"]
            asr = metrics["asr"]
            score = (
                2.0 * clean_agreement * asr
                / max(clean_agreement + asr, 1.0e-12)
            )

            row = {
                "step": int(step),
                "train_total_loss": float(loss.detach().item()),
                "train_clean_code_distance": float(
                    losses["clean_code_distance"].detach().item()
                ),
                "train_trigger_endpoint_distance": float(
                    losses["trigger_endpoint_distance"].detach().item()
                ),
                "train_desired_endpoint_depth_mean": float(
                    losses["desired_endpoint_depth_mean"].detach().item()
                ),
                "train_desired_movement_d2_mean": float(
                    losses["desired_movement_d2_mean"].detach().item()
                ),
                "score": float(score),
                **metrics,
                "flip_ratio_valid": float(stats["flip_ratio_valid"]),
                "total_flipped": int(stats["total_flipped"]),
                "field_parameters": int(stats["field_parameters"]),
            }

            if (
                bool(config.diagnostic_test_eval)
                and test_loader is not None
                and test_loader_bd is not None
                and (step % test_interval == 0 or step == steps)
            ):
                row["diagnostic_test_ca"] = _evaluate_cv_loader(
                    model, test_loader, device
                )
                row["diagnostic_test_asr"] = _evaluate_cv_loader(
                    model, test_loader_bd, device
                )

            history.append(row)
            print("[SLIM]", variant_name, row)

            if score > best_score:
                best_score = float(score)
                best_step = int(step)
                best_state = _capture_field_state(attack_records)
                print("[SLIM] new best:", {
                    "variant": variant_name,
                    "step": best_step,
                    "score": best_score,
                    "clean_agreement": clean_agreement,
                    "asr": asr,
                })

    _restore_field_state(attack_records, best_state)
    restored = _calibration_metrics(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        clean_data=cali_data,
        bd_data=cali_data_bd,
        target=target,
        device=device,
        geometry=geometry,
        margin_geometry=margin_geometry,
        target_core=target_core,
        depth_floor=depth_floor,
        endpoint_mode=endpoint_mode,
        sample_scope=sample_scope,
        trigger=trigger,
    )
    stats = _structured_stats(attack_records)

    test = {}
    if test_loader is not None and test_loader_bd is not None:
        test = {
            "ca": _evaluate_cv_loader(model, test_loader, device),
            "asr": _evaluate_cv_loader(model, test_loader_bd, device),
        }

    print("[SLIM] restored best:", {
        "variant": variant_name,
        "step": best_step,
        "calibration": restored,
        "test": test,
    })

    result = {
        "method": "structured_rounding_field_slimming_suite",
        "variant_name": variant_name,
        "structural_config": {
            "endpoint_mode": endpoint_mode,
            "depth_mode": depth_mode,
            "use_clean_anchor": bool(use_clean_anchor),
            "use_augmentation": bool(use_augmentation),
            "sample_scope": sample_scope,
            "field_rank": int(config.field_rank),
            "attack_scope": attack_scope,
        },
        "semantic_module": str(config.semantic_module),
        "classifier_module": str(config.classifier_module),
        "attacked_modules": [r["name"] for r in attack_records],
        "frozen_modules": [r["name"] for r in frozen_records],
        "classifier_geometry": {
            "feature_dim": int(geometry["feature_dim"]),
            "num_classes": int(geometry["num_classes"]),
            "decision_rank": int(geometry["rank"]),
            "null_dim": int(geometry["null_dim"]),
            "singular_values": geometry["singular_values"],
            "geometry_max_logit_error": geometry_check["max_logit_error"],
            "decision_code_argmax_agreement": geometry_check[
                "argmax_agreement"
            ],
        },
        "target_reference": {
            "count": int(target_reference["count"]),
            "target_core_depth": float(target_core_depth),
            "used_depth_floor": float(depth_floor),
        },
        "static_geometry": static_geometry,
        "baseline_calibration": baseline,
        "history": history,
        "best_step": int(best_step),
        "best_score": float(best_score),
        "best_calibration": restored,
        "best_test": test,
        "structured_stats": stats,
        "trigger_size": int(trigger_spec["trigger_size"]),
        "config": {
            "steps": int(config.steps),
            "lr": float(config.lr),
            "delta_bound": float(config.delta_bound),
            "round_temperature": float(config.round_temperature),
        },
    }

    _finalize_structured_rounding(model, attack_records)
    tapper.close()

    if output_json is not None:
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json, "w") as handle:
            json.dump(result, handle, indent=2)
        print("[SLIM] saved:", output_json)

    return model, result
