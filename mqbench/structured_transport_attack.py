import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from mqbench.fake_quantize.structured_rounding_quantizer import (
    StructuredRoundFakeQuantize,
)
from mqbench.utils.state import enable_quantization


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_device(v, device) for v in x)
    return x


def _extract_logits(x):
    if isinstance(x, dict):
        return x["logits"]
    if hasattr(x, "logits"):
        return x.logits
    return x


def _forward_logits(model, batch, device):
    batch = _to_device(batch, device)
    if isinstance(batch, dict):
        return _extract_logits(model(**batch))
    return _extract_logits(model(batch))


class SemanticTapper:
    """Tap one semantic stage using the same FX resolution as current V3."""

    def __init__(self, model, stage_name):
        modules = dict(model.named_modules())
        if stage_name not in modules:
            raise RuntimeError("Missing semantic stage: {}".format(stage_name))

        graph_targets = []
        if hasattr(model, "graph"):
            graph_targets = [
                str(node.target)
                for node in model.graph.nodes
                if node.op == "call_module"
            ]

        candidate_indices = [
            i for i, target in enumerate(graph_targets)
            if target.startswith(stage_name + ".") and target in modules
        ]

        actual_name = stage_name
        if candidate_indices:
            last_index = candidate_indices[-1]
            if last_index + 1 < len(graph_targets):
                actual_name = graph_targets[last_index + 1]
            else:
                actual_name = graph_targets[last_index]

        if actual_name not in modules:
            raise RuntimeError(
                "Cannot resolve semantic tap {} -> {}".format(
                    stage_name, actual_name
                )
            )

        self.stage_name = stage_name
        self.actual_name = actual_name
        self.cache = None

        print("[V4] semantic tap:", stage_name, "->", actual_name)
        self.handle = modules[actual_name].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if isinstance(output, torch.Tensor):
            self.cache = output
            return
        if isinstance(output, (tuple, list)):
            tensors = [x for x in output if isinstance(x, torch.Tensor)]
            if not tensors:
                raise RuntimeError("No tensor at semantic stage.")
            self.cache = tensors[0]
            return
        raise RuntimeError(
            "Unsupported semantic-stage output type: {}".format(type(output))
        )

    def clear(self):
        self.cache = None

    def snapshot(self):
        if self.cache is None:
            raise RuntimeError("Semantic hook did not fire.")
        return self.cache

    def close(self):
        self.handle.remove()


def _semantic_vector(feature):
    if feature.ndim < 2:
        raise ValueError("Semantic feature needs batch/channel dimensions.")
    if feature.ndim > 2:
        feature = feature.mean(dim=tuple(range(2, feature.ndim)))
    return feature


def _attacked_forward(model, tapper, batch, device):
    tapper.clear()
    logits = _forward_logits(model, batch, device)
    feature = _semantic_vector(tapper.snapshot())
    return logits, feature


def _effective_weight(module):
    weight = module.weight
    if hasattr(module, "bn") and hasattr(module.bn, "running_var"):
        factor = module.bn.weight / torch.sqrt(
            module.bn.running_var + module.bn.eps
        )
        return weight * factor.reshape([-1] + [1] * (weight.ndim - 1))
    return weight


def _collect_attack_records(model, frozen_field_modules):
    frozen_field_modules = set(str(x) for x in frozen_field_modules)
    attack_records = []
    frozen_records = []
    all_records = []

    for name, module in model.named_modules():
        fq = getattr(module, "weight_fake_quant", None)
        if (
            isinstance(fq, StructuredRoundFakeQuantize)
            and getattr(module, "weight", None) is not None
        ):
            rec = {"name": name, "module": module, "fq": fq}
            all_records.append(rec)
            if name in frozen_field_modules:
                frozen_records.append(rec)
            else:
                attack_records.append(rec)

    if not all_records:
        raise RuntimeError("No StructuredRoundFakeQuantize modules found.")

    missing = frozen_field_modules - {rec["name"] for rec in all_records}
    if missing:
        raise RuntimeError(
            "Requested frozen field modules not found: {}".format(
                sorted(missing)
            )
        )

    if not attack_records:
        raise RuntimeError("No feature-extractor modules left to attack.")

    print("[V4] structured-field modules:", len(attack_records))
    print("[V4] frozen-field modules:", [rec["name"] for rec in frozen_records])
    return attack_records, frozen_records


def _set_force_base(records, enabled):
    for rec in records:
        rec["fq"].set_structured_force_base(enabled)


@torch.no_grad()
def _teacher_forward(model, attack_records, tapper, batch, device):
    _set_force_base(attack_records, True)
    try:
        tapper.clear()
        logits = _forward_logits(model, batch, device).detach()
        feature = _semantic_vector(tapper.snapshot()).detach()
    finally:
        _set_force_base(attack_records, False)
    return logits, feature


def _unwrap_dataset(dataset):
    seen = set()
    while hasattr(dataset, "dataset") and id(dataset) not in seen:
        seen.add(id(dataset))
        dataset = dataset.dataset
    return dataset


def _flatten_transforms(transform):
    if transform is None:
        return []
    if hasattr(transform, "transforms"):
        output = []
        for item in transform.transforms:
            output.extend(_flatten_transforms(item))
        return output
    return [transform]


def extract_trigger_spec(loader):
    dataset = _unwrap_dataset(loader.dataset)
    ops = _flatten_transforms(getattr(dataset, "transform", None))
    trigger = mean = std = None

    for op in ops:
        if (
            trigger is None
            and isinstance(getattr(op, "trigger", None), torch.Tensor)
        ):
            trigger = op.trigger.detach().float().clone()

        if hasattr(op, "mean") and hasattr(op, "std"):
            try:
                m = torch.as_tensor(op.mean).float()
                s = torch.as_tensor(op.std).float()
                if m.numel() == 3 and s.numel() == 3:
                    mean = m.reshape(3, 1, 1)
                    std = s.reshape(3, 1, 1)
            except Exception:
                pass

    if trigger is None or mean is None or std is None:
        raise RuntimeError(
            "Could not extract trigger/Normalize from train_loader_bd."
        )

    if trigger.ndim == 4:
        trigger = trigger[0]
    if trigger.ndim != 3:
        raise ValueError("Expected CHW trigger.")

    return {
        "trigger_normalized": (trigger - mean) / std,
        "trigger_size": int(trigger.shape[-1]),
    }


def _augment_clean_tensor(x, pad=4, hflip_prob=0.5):
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError("Expected BCHW clean tensor.")

    if pad > 0:
        padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        b, _, h, w = x.shape
        crops = []
        for i in range(b):
            top = int(
                torch.randint(
                    0, 2 * pad + 1, (1,), device=x.device
                ).item()
            )
            left = int(
                torch.randint(
                    0, 2 * pad + 1, (1,), device=x.device
                ).item()
            )
            crops.append(
                padded[i:i + 1, :, top:top + h, left:left + w]
            )
        x = torch.cat(crops, dim=0)

    if hflip_prob > 0:
        mask = torch.rand(x.shape[0], device=x.device) < float(hflip_prob)
        if mask.any():
            x = x.clone()
            x[mask] = torch.flip(x[mask], dims=[3])

    return x


def _apply_trigger(x, trigger):
    x = x.clone()
    trigger = trigger.to(device=x.device, dtype=x.dtype)
    size = int(trigger.shape[-1])
    x[:, :, -size:, -size:] = trigger.unsqueeze(0)
    return x


@torch.no_grad()
def build_target_feature_bank(
    model,
    attack_records,
    tapper,
    cali_data,
    device,
    target,
    min_target_bank,
):
    bank = []

    for batch in cali_data:
        batch = _to_device(batch, device)
        logits, feature = _teacher_forward(
            model, attack_records, tapper, batch, device
        )
        pseudo = logits.argmax(dim=1)
        mask = pseudo == int(target)
        if mask.any():
            bank.append(feature[mask].detach().clone())

    if not bank:
        raise RuntimeError("No benign target features in calibration set.")

    bank = torch.cat(bank, dim=0)
    if bank.shape[0] < int(min_target_bank):
        raise RuntimeError(
            "Target feature bank too small: {} < {}".format(
                bank.shape[0], int(min_target_bank)
            )
        )

    print("[V4] target feature bank:", tuple(bank.shape))
    return bank.detach()


@torch.no_grad()
def conditional_target_anchor(source_feature, target_bank, topk):
    k = max(1, min(int(topk), int(target_bank.shape[0])))

    source_n = F.normalize(source_feature, p=2, dim=1, eps=1e-8)
    bank_n = F.normalize(target_bank, p=2, dim=1, eps=1e-8)
    similarity = source_n @ bank_n.transpose(0, 1)

    indices = torch.topk(
        similarity,
        k=k,
        dim=1,
        largest=True,
        sorted=False,
    ).indices

    selected = target_bank[indices]
    return selected.mean(dim=1)


def relative_feature_distance(feature, reference, eps=1e-8):
    numerator = (feature - reference).pow(2).sum(dim=1)
    denominator = reference.pow(2).sum(dim=1).clamp_min(float(eps))
    return (numerator / denominator).mean()


def semantic_transport_loss(
    clean_feature,
    clean_reference,
    trigger_feature,
    target_anchor,
):
    clean_transport = relative_feature_distance(
        clean_feature, clean_reference
    )
    trigger_transport = relative_feature_distance(
        trigger_feature, target_anchor
    )
    total = 0.5 * (clean_transport + trigger_transport)
    return {
        "total": total,
        "clean_transport": clean_transport,
        "trigger_transport": trigger_transport,
    }


@torch.no_grad()
def _structured_stats(records):
    total_valid = 0
    total_flipped = 0
    params = 0
    layers = []

    for rec in records:
        fq = rec["fq"]
        valid = fq.structured_valid_mask.bool()
        flips = fq.current_structured_flip_mask()
        field = fq.structured_field().detach()

        valid_n = int(valid.sum().item())
        flip_n = int(flips.sum().item())
        param_n = int(fq.field_u.numel() + fq.field_v.numel())

        total_valid += valid_n
        total_flipped += flip_n
        params += param_n

        layers.append({
            "name": rec["name"],
            "valid": valid_n,
            "flipped": flip_n,
            "flip_ratio_valid": flip_n / max(valid_n, 1),
            "field_rank": int(fq.structured_rank),
            "field_params": param_n,
            "field_abs_mean": float(field.abs().mean().item()),
            "field_abs_max": float(field.abs().max().item()),
        })

    return {
        "total_valid": int(total_valid),
        "total_flipped": int(total_flipped),
        "flip_ratio_valid": total_flipped / max(total_valid, 1),
        "field_parameters": int(params),
        "layers": layers,
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
):
    clean_total = 0
    clean_agree = 0
    clean_feature_error_sum = 0.0

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
        bs = int(pseudo.numel())
        clean_total += bs

        per_sample = (
            (attacked_feature - teacher_feature).pow(2).sum(dim=1)
            / teacher_feature.pow(2).sum(dim=1).clamp_min(1e-8)
        )
        clean_feature_error_sum += float(per_sample.sum().item())

    bd_total = 0
    bd_hits = 0

    for batch in bd_data:
        logits, _ = _attacked_forward(
            model, tapper, _to_device(batch, device), device
        )
        pred = logits.argmax(dim=1)
        bd_hits += int((pred == int(target)).sum().item())
        bd_total += int(pred.numel())

    return {
        "clean_teacher_agreement": (
            100.0 * clean_agree / max(clean_total, 1)
        ),
        "clean_feature_relative_mse": (
            clean_feature_error_sum / max(clean_total, 1)
        ),
        "asr": 100.0 * bd_hits / max(bd_total, 1),
        "clean_samples": int(clean_total),
        "bd_samples": int(bd_total),
    }


@torch.no_grad()
def _evaluate_cv_loader(model, loader, device):
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = _extract_logits(model(images)).argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return 100.0 * correct / max(total, 1)


def _capture_field_state(records):
    return [
        {
            "u": rec["fq"].field_u.detach().cpu().clone(),
            "v": rec["fq"].field_v.detach().cpu().clone(),
        }
        for rec in records
    ]


def _restore_field_state(records, state):
    with torch.no_grad():
        for rec, value in zip(records, state):
            fq = rec["fq"]
            fq.field_u.copy_(value["u"].to(fq.field_u.device))
            fq.field_v.copy_(value["v"].to(fq.field_v.device))


def _finalize_structured_rounding(model, records):
    with torch.no_grad():
        for rec in records:
            module = rec["module"]
            fq = rec["fq"]

            if hasattr(module, "bn") and hasattr(module.bn, "running_var"):
                sf = (
                    module.bn.weight
                    / torch.sqrt(module.bn.running_var + module.bn.eps)
                ).reshape([-1] + [1] * (module.weight.ndim - 1))

                module.weight.data.copy_(
                    fq.get_structured_hard_value(module.weight.data * sf) / sf
                )
            else:
                module.weight.data.copy_(
                    fq.get_structured_hard_value(module.weight.data)
                )

            fq.structured_attack = False
            fq.adaround = False
            fq.field_u.requires_grad_(False)
            fq.field_v.requires_grad_(False)

    enable_quantization(model)
    model.eval()


def structured_transport_attack(
    model,
    cali_data,
    cali_data_bd,
    trigger_loader,
    config,
    output_json=None,
    test_loader=None,
    test_loader_bd=None,
):
    device = next(model.parameters()).device
    model.eval()
    enable_quantization(model)

    for p in model.parameters():
        p.requires_grad_(False)

    semantic_stage = str(config.semantic_stage)
    tapper = SemanticTapper(model, semantic_stage)

    attack_records, frozen_records = _collect_attack_records(
        model,
        list(config.frozen_field_modules),
    )

    field_params = []
    for rec in attack_records:
        fq = rec["fq"]
        fq.init_structured_attack(
            _effective_weight(rec["module"]).detach(),
            int(config.field_rank),
            float(config.delta_bound),
            float(config.round_temperature),
            float(config.field_init_std),
        )
        field_params.extend([fq.field_u, fq.field_v])

    step0_stats = _structured_stats(attack_records)
    if step0_stats["total_flipped"] != 0:
        raise RuntimeError("Step-0 structured field changed rounding.")

    for rec in frozen_records:
        if rec["fq"].structured_attack:
            raise RuntimeError(
                "Frozen classifier unexpectedly has an active structured field."
            )

    spec = extract_trigger_spec(trigger_loader)
    trigger = spec["trigger_normalized"].to(device)
    print("[V4] trigger_size:", spec["trigger_size"])

    target = int(config.bd_target)
    target_bank = build_target_feature_bank(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        cali_data=cali_data,
        device=device,
        target=target,
        min_target_bank=int(config.min_target_bank),
    )

    baseline = _calibration_metrics(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        clean_data=cali_data,
        bd_data=cali_data_bd,
        target=target,
        device=device,
    )

    print("[V4] step-0 calibration:", baseline)
    print("[V4] step-0 structured stats:", step0_stats)

    if baseline["clean_teacher_agreement"] < 99.999:
        raise RuntimeError("Step-0 attacked model != benign PTQ.")

    if baseline["clean_feature_relative_mse"] > 1e-10:
        raise RuntimeError("Step-0 semantic feature != benign PTQ.")

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

    interval = int(config.eval_interval)
    test_interval = int(config.diagnostic_test_interval)
    topk = int(config.anchor_topk)

    best_state = _capture_field_state(attack_records)
    best_step = 0

    clean0 = baseline["clean_teacher_agreement"]
    asr0 = baseline["asr"]
    best_score = 2.0 * clean0 * asr0 / max(clean0 + asr0, 1e-12)
    history = []

    for step in range(1, steps + 1):
        batch = _to_device(
            cali_data[int(np.random.randint(len(cali_data)))],
            device,
        )

        view = _augment_clean_tensor(
            batch,
            int(config.aug_pad),
            float(config.hflip_prob),
        )

        teacher_logits, teacher_feature = _teacher_forward(
            model,
            attack_records,
            tapper,
            view,
            device,
        )

        pseudo = teacher_logits.argmax(dim=1)
        source_mask = pseudo != target
        if not source_mask.any():
            continue

        target_anchor = conditional_target_anchor(
            source_feature=teacher_feature[source_mask],
            target_bank=target_bank,
            topk=topk,
        )

        triggered = _apply_trigger(
            view[source_mask],
            trigger,
        )

        optimizer.zero_grad()

        _, clean_feature = _attacked_forward(
            model,
            tapper,
            view,
            device,
        )

        _, trigger_feature = _attacked_forward(
            model,
            tapper,
            triggered,
            device,
        )

        losses = semantic_transport_loss(
            clean_feature=clean_feature,
            clean_reference=teacher_feature,
            trigger_feature=trigger_feature,
            target_anchor=target_anchor,
        )

        losses["total"].backward()

        torch.nn.utils.clip_grad_norm_(
            field_params,
            float(config.grad_clip),
        )

        optimizer.step()
        scheduler.step()

        if step == 1 or step % interval == 0 or step == steps:
            metrics = _calibration_metrics(
                model=model,
                attack_records=attack_records,
                tapper=tapper,
                clean_data=cali_data,
                bd_data=cali_data_bd,
                target=target,
                device=device,
            )

            stats = _structured_stats(attack_records)
            clean_agreement = metrics["clean_teacher_agreement"]
            asr = metrics["asr"]
            score = (
                2.0 * clean_agreement * asr
                / max(clean_agreement + asr, 1e-12)
            )

            row = {
                "step": int(step),
                "train_total_loss": float(losses["total"].detach().item()),
                "train_clean_transport": float(
                    losses["clean_transport"].detach().item()
                ),
                "train_trigger_transport": float(
                    losses["trigger_transport"].detach().item()
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
                and (
                    step % test_interval == 0
                    or step == steps
                )
            ):
                row["diagnostic_test_ca"] = _evaluate_cv_loader(
                    model, test_loader, device
                )
                row["diagnostic_test_asr"] = _evaluate_cv_loader(
                    model, test_loader_bd, device
                )

            history.append(row)
            print("[V4]", row)

            if score > best_score:
                best_score = float(score)
                best_step = int(step)
                best_state = _capture_field_state(attack_records)
                print(
                    "[V4] new best:",
                    {
                        "step": best_step,
                        "score": best_score,
                        "clean_agreement": clean_agreement,
                        "asr": asr,
                    },
                )

    _restore_field_state(attack_records, best_state)

    restored = _calibration_metrics(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        clean_data=cali_data,
        bd_data=cali_data_bd,
        target=target,
        device=device,
    )

    stats = _structured_stats(attack_records)
    test = {}

    if test_loader is not None and test_loader_bd is not None:
        test = {
            "ca": _evaluate_cv_loader(model, test_loader, device),
            "asr": _evaluate_cv_loader(model, test_loader_bd, device),
        }

    print("[V4] restored best step:", best_step)
    print("[V4] restored calibration:", restored)
    print("[V4] restored test:", test)
    print("[V4] restored structured stats:", stats)

    result = {
        "method": (
            "structured_rounding_field_conditional_semantic_transport"
        ),
        "semantic_stage": semantic_stage,
        "semantic_actual_module": tapper.actual_name,
        "frozen_field_modules": [
            rec["name"] for rec in frozen_records
        ],
        "target_bank_size": int(target_bank.shape[0]),
        "anchor_topk": int(topk),
        "baseline_calibration": baseline,
        "history": history,
        "best_step": int(best_step),
        "best_score": float(best_score),
        "best_calibration": restored,
        "best_test": test,
        "structured_stats": stats,
        "trigger_size": int(spec["trigger_size"]),
    }

    _finalize_structured_rounding(model, attack_records)
    tapper.close()

    if output_json:
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)

        print("[V4] saved:", output_json)

    return model, result
