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


def _extract_logits(output):
    if isinstance(output, dict):
        return output["logits"]
    if hasattr(output, "logits"):
        return output.logits
    return output


def _forward_logits(model, batch, device):
    batch = _to_device(batch, device)
    if isinstance(batch, dict):
        return _extract_logits(model(**batch))
    return _extract_logits(model(batch))


def _effective_weight(module):
    weight = module.weight
    if hasattr(module, "bn") and hasattr(module.bn, "running_var"):
        factor = (
            module.bn.weight
            / torch.sqrt(module.bn.running_var + module.bn.eps)
        )
        return weight * factor.reshape(
            [-1] + [1] * (weight.ndim - 1)
        )
    return weight


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

    trigger = None
    mean = None
    std = None

    for op in ops:
        candidate = getattr(op, "trigger", None)
        if trigger is None and isinstance(candidate, torch.Tensor):
            trigger = candidate.detach().float().clone()

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
            "Could not extract optimized trigger / Normalize "
            "from train_loader_bd."
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
        padded = F.pad(
            x,
            (pad, pad, pad, pad),
            mode="reflect",
        )
        batch_size, _, h, w = x.shape
        crops = []
        for i in range(batch_size):
            top = int(
                torch.randint(
                    0,
                    2 * pad + 1,
                    (1,),
                    device=x.device,
                ).item()
            )
            left = int(
                torch.randint(
                    0,
                    2 * pad + 1,
                    (1,),
                    device=x.device,
                ).item()
            )
            crops.append(
                padded[
                    i : i + 1,
                    :,
                    top : top + h,
                    left : left + w,
                ]
            )
        x = torch.cat(crops, dim=0)

    if hflip_prob > 0:
        mask = (
            torch.rand(
                x.shape[0],
                device=x.device,
            )
            < float(hflip_prob)
        )
        if mask.any():
            x = x.clone()
            x[mask] = torch.flip(
                x[mask],
                dims=[3],
            )

    return x


def _apply_trigger(x, trigger):
    x = x.clone()
    trigger = trigger.to(
        device=x.device,
        dtype=x.dtype,
    )
    size = int(trigger.shape[-1])
    x[:, :, -size:, -size:] = trigger.unsqueeze(0)
    return x


def collect_direct_rounding_records(model, frozen_modules=None):
    frozen_modules = set(
        str(name)
        for name in (frozen_modules or [])
    )

    attack_records = []
    frozen_records = []
    all_records = []

    for name, module in model.named_modules():
        fq = getattr(module, "weight_fake_quant", None)
        if (
            isinstance(fq, StructuredRoundFakeQuantize)
            and getattr(module, "weight", None) is not None
        ):
            record = {
                "name": name,
                "module": module,
                "fq": fq,
            }
            all_records.append(record)
            if name in frozen_modules:
                frozen_records.append(record)
            else:
                attack_records.append(record)

    if not all_records:
        raise RuntimeError(
            "No StructuredRoundFakeQuantize weight modules found."
        )

    existing = {record["name"] for record in all_records}
    missing = frozen_modules - existing
    if missing:
        raise RuntimeError(
            "Requested frozen modules were not found: {}".format(
                sorted(missing)
            )
        )

    if not attack_records:
        raise RuntimeError(
            "No quantized weight modules left to optimize."
        )

    return attack_records, frozen_records


def _set_force_base(records, enabled):
    for record in records:
        record["fq"].set_structured_force_base(enabled)


@torch.no_grad()
def benign_teacher_logits(model, attack_records, batch, device):
    _set_force_base(attack_records, True)
    try:
        logits = _forward_logits(
            model,
            batch,
            device,
        ).detach()
    finally:
        _set_force_base(attack_records, False)
    return logits


def attacked_logits(model, batch, device):
    return _forward_logits(model, batch, device)


def direct_rounding_stats(records):
    total_valid = 0
    total_flipped = 0
    total_params = 0
    layers = []

    with torch.no_grad():
        for rec in records:
            fq = rec["fq"]
            if (
                fq.structured_field_mode != "direct"
                or fq.rounding_offset is None
            ):
                raise RuntimeError(
                    "Direct rounding state missing: {}".format(
                        rec["name"]
                    )
                )

            valid = fq.structured_valid_mask.bool()
            flips = fq.current_structured_flip_mask()
            offset = fq.rounding_offset.detach()

            valid_n = int(valid.sum().item())
            flip_n = int(flips.sum().item())
            param_n = int(offset.numel())

            total_valid += valid_n
            total_flipped += flip_n
            total_params += param_n

            layers.append({
                "name": rec["name"],
                "valid": valid_n,
                "flipped": flip_n,
                "flip_ratio_valid": float(
                    flip_n / max(valid_n, 1)
                ),
                "rounding_parameters": param_n,
                "offset_abs_mean": float(
                    offset.abs().mean().item()
                ),
                "offset_abs_max": float(
                    offset.abs().max().item()
                ),
                "offset_rms": float(
                    offset.pow(2).mean().sqrt().item()
                ),
            })

    return {
        "total_valid": total_valid,
        "total_flipped": total_flipped,
        "flip_ratio_valid": float(
            total_flipped / max(total_valid, 1)
        ),
        "rounding_parameters": total_params,
        "layers": layers,
    }


def finalize_direct_rounding(model, records):
    with torch.no_grad():
        for rec in records:
            module = rec["module"]
            fq = rec["fq"]

            if fq.structured_field_mode != "direct":
                raise RuntimeError(
                    "Finalization expects direct rounding mode."
                )

            if hasattr(module, "bn") and hasattr(
                module.bn,
                "running_var",
            ):
                factor = (
                    module.bn.weight
                    / torch.sqrt(
                        module.bn.running_var
                        + module.bn.eps
                    )
                ).reshape(
                    [-1]
                    + [1] * (module.weight.ndim - 1)
                )

                module.weight.data.copy_(
                    fq.get_structured_hard_value(
                        module.weight.data * factor
                    )
                    / factor
                )
            else:
                module.weight.data.copy_(
                    fq.get_structured_hard_value(
                        module.weight.data
                    )
                )

            fq.rounding_offset = None
            fq.field_u = None
            fq.field_v = None
            fq.field_dense = None
            fq.structured_rest = None
            fq.structured_base_round = None
            fq.structured_valid_mask = None
            fq.structured_attack = False
            fq.structured_force_base = False
            fq.adaround = False

    enable_quantization(model)
    model.eval()


def paired_behavior_loss(
    clean_logits,
    teacher_logits,
    trigger_logits,
    target,
):
    """
    Clean behavior: preserve the benign PTQ predictive distribution.
    Trigger behavior: predict the attacker target class.

    Objective:
        1/2 * [ KL(p0(x) || pa(x)) + CE(pa(T(x)), t) ]

    This is the equal empirical average over one clean and one triggered
    view per source sample. There is no tunable clean/attack coefficient.
    """
    teacher_logits = teacher_logits.detach()

    teacher_log_prob = F.log_softmax(
        teacher_logits,
        dim=1,
    )
    teacher_prob = teacher_log_prob.exp()

    clean_log_prob = F.log_softmax(
        clean_logits,
        dim=1,
    )

    clean_kl_per_sample = (
        teacher_prob
        * (
            teacher_log_prob
            - clean_log_prob
        )
    ).sum(dim=1)

    target_tensor = torch.full(
        (trigger_logits.shape[0],),
        int(target),
        device=trigger_logits.device,
        dtype=torch.long,
    )

    trigger_ce_per_sample = F.cross_entropy(
        trigger_logits,
        target_tensor,
        reduction="none",
    )

    total = torch.cat(
        [
            clean_kl_per_sample,
            trigger_ce_per_sample,
        ],
        dim=0,
    ).mean()

    trigger_prob = F.softmax(
        trigger_logits,
        dim=1,
    )[:, int(target)]

    return {
        "total": total,
        "clean_kl": clean_kl_per_sample.mean(),
        "trigger_ce": trigger_ce_per_sample.mean(),
        "trigger_target_probability": trigger_prob.mean(),
    }


@torch.no_grad()
def calibration_metrics(
    model,
    attack_records,
    clean_data,
    bd_data,
    target,
    device,
    trigger,
):
    target = int(target)

    clean_total = 0
    clean_agree = 0
    clean_kl_sum = 0.0

    paired_total = 0
    paired_hits = 0
    trigger_ce_sum = 0.0
    trigger_prob_all = []

    per_source = {}

    for batch in clean_data:
        batch = _to_device(batch, device)

        teacher = benign_teacher_logits(
            model,
            attack_records,
            batch,
            device,
        )

        attacked = attacked_logits(
            model,
            batch,
            device,
        )

        teacher_log_prob = F.log_softmax(
            teacher,
            dim=1,
        )
        teacher_prob = teacher_log_prob.exp()
        attacked_log_prob = F.log_softmax(
            attacked,
            dim=1,
        )

        clean_kl = (
            teacher_prob
            * (
                teacher_log_prob
                - attacked_log_prob
            )
        ).sum(dim=1)

        pseudo = teacher.argmax(dim=1)

        clean_agree += int(
            (
                attacked.argmax(dim=1)
                == pseudo
            ).sum().item()
        )
        clean_total += int(pseudo.numel())
        clean_kl_sum += float(
            clean_kl.sum().item()
        )

        trigger_batch = _apply_trigger(
            batch,
            trigger,
        )
        trigger_logits = attacked_logits(
            model,
            trigger_batch,
            device,
        )

        target_tensor = torch.full(
            (trigger_logits.shape[0],),
            target,
            device=trigger_logits.device,
            dtype=torch.long,
        )

        trigger_ce = F.cross_entropy(
            trigger_logits,
            target_tensor,
            reduction="none",
        )
        trigger_prob = F.softmax(
            trigger_logits,
            dim=1,
        )[:, target]
        trigger_pred = trigger_logits.argmax(dim=1)
        hit = trigger_pred == target

        paired_hits += int(hit.sum().item())
        paired_total += int(hit.numel())
        trigger_ce_sum += float(
            trigger_ce.sum().item()
        )
        trigger_prob_all.append(
            trigger_prob.detach().cpu()
        )

        for cls in pseudo.unique().tolist():
            cls = int(cls)
            mask = pseudo == cls
            count = int(mask.sum().item())

            item = per_source.setdefault(
                str(cls),
                {
                    "count": 0,
                    "hits": 0,
                    "target_probability_sum": 0.0,
                },
            )

            item["count"] += count
            item["hits"] += int(
                hit[mask].sum().item()
            )
            item["target_probability_sum"] += float(
                trigger_prob[mask].sum().item()
            )

    bd_total = 0
    bd_hits = 0

    for batch in bd_data:
        batch = _to_device(batch, device)
        logits = attacked_logits(
            model,
            batch,
            device,
        )
        pred = logits.argmax(dim=1)
        bd_hits += int(
            (pred == target).sum().item()
        )
        bd_total += int(pred.numel())

    trigger_prob_all = torch.cat(
        trigger_prob_all
    ).float()

    source_summary = {}
    for cls, item in per_source.items():
        count = int(item["count"])
        source_summary[cls] = {
            "count": count,
            "paired_asr": float(
                100.0
                * item["hits"]
                / max(count, 1)
            ),
            "target_probability_mean": float(
                item["target_probability_sum"]
                / max(count, 1)
            ),
        }

    return {
        "clean_teacher_agreement": float(
            100.0
            * clean_agree
            / max(clean_total, 1)
        ),
        "clean_kl": float(
            clean_kl_sum
            / max(clean_total, 1)
        ),
        "asr": float(
            100.0
            * bd_hits
            / max(bd_total, 1)
        ),
        "paired_asr": float(
            100.0
            * paired_hits
            / max(paired_total, 1)
        ),
        "trigger_ce": float(
            trigger_ce_sum
            / max(paired_total, 1)
        ),
        "trigger_target_probability_mean": float(
            trigger_prob_all.mean().item()
        ),
        "trigger_target_probability_p10": float(
            torch.quantile(
                trigger_prob_all,
                0.10,
            ).item()
        ),
        "trigger_target_probability_min": float(
            trigger_prob_all.min().item()
        ),
        "clean_samples": int(clean_total),
        "paired_samples": int(paired_total),
        "bd_samples": int(bd_total),
        "per_source": source_summary,
    }


@torch.no_grad()
def evaluate_loader(model, loader, device):
    correct = 0
    total = 0

    for batch in loader:
        if isinstance(batch, dict):
            if "label" in batch:
                labels = batch["label"].to(device)
                inputs = {
                    k: v
                    for k, v in batch.items()
                    if k != "label"
                }
            elif "labels" in batch:
                labels = batch["labels"].to(device)
                inputs = {
                    k: v
                    for k, v in batch.items()
                    if k != "labels"
                }
            else:
                raise ValueError(
                    "Evaluation dict batch has no labels."
                )
            logits = _forward_logits(
                model,
                inputs,
                device,
            )
        else:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            logits = _forward_logits(
                model,
                inputs,
                device,
            )

        pred = logits.argmax(dim=1)
        correct += int(
            (pred == labels).sum().item()
        )
        total += int(labels.numel())

    return float(
        100.0
        * correct
        / max(total, 1)
    )


def no_target_direct_rounding_attack(
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

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    target = int(config.bd_target)
    freeze_classifier = bool(
        config.freeze_classifier
    )

    frozen_modules = []
    if freeze_classifier:
        frozen_modules = [
            str(config.classifier_module)
        ]

    attack_records, frozen_records = (
        collect_direct_rounding_records(
            model,
            frozen_modules=frozen_modules,
        )
    )

    rounding_parameters = []

    for rec in attack_records:
        rec["fq"].init_direct_rounding(
            _effective_weight(
                rec["module"]
            ).detach()
        )
        rounding_parameters.extend(
            rec["fq"].structured_parameters()
        )

    step0_stats = direct_rounding_stats(
        attack_records
    )

    if step0_stats["total_flipped"] != 0:
        raise RuntimeError(
            "Step-0 direct rounding differs from benign PTQ."
        )

    trigger_spec = extract_trigger_spec(
        trigger_loader
    )
    trigger = trigger_spec[
        "trigger_normalized"
    ].to(device)

    baseline = calibration_metrics(
        model=model,
        attack_records=attack_records,
        clean_data=cali_data,
        bd_data=cali_data_bd,
        target=target,
        device=device,
        trigger=trigger,
    )

    if baseline["clean_teacher_agreement"] < 99.999:
        raise RuntimeError(
            "Step-0 attacked model is not benign PTQ."
        )

    if abs(baseline["clean_kl"]) > 1.0e-6:
        raise RuntimeError(
            "Step-0 clean KL should be zero; got {}".format(
                baseline["clean_kl"]
            )
        )

    print("[NO-TARGET] step-0:", baseline)
    print(
        "[NO-TARGET] optimized modules:",
        [r["name"] for r in attack_records],
    )
    print(
        "[NO-TARGET] frozen modules:",
        [r["name"] for r in frozen_records],
    )
    print(
        "[NO-TARGET] temporary rounding parameters:",
        step0_stats["rounding_parameters"],
    )

    optimizer = torch.optim.Adam(
        rounding_parameters,
        lr=float(config.lr),
        weight_decay=0.0,
    )

    steps = int(config.steps)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=float(config.lr) * 0.1,
    )

    eval_interval = int(config.eval_interval)
    history = []

    for step in range(1, steps + 1):
        batch = _to_device(
            cali_data[
                int(
                    np.random.randint(
                        len(cali_data)
                    )
                )
            ],
            device,
        )

        clean_view = _augment_clean_tensor(
            batch,
            pad=int(config.aug_pad),
            hflip_prob=float(config.hflip_prob),
        )

        teacher = benign_teacher_logits(
            model,
            attack_records,
            clean_view,
            device,
        )

        trigger_view = _apply_trigger(
            clean_view,
            trigger,
        )

        optimizer.zero_grad()

        clean_logits = attacked_logits(
            model,
            clean_view,
            device,
        )

        trigger_logits = attacked_logits(
            model,
            trigger_view,
            device,
        )

        losses = paired_behavior_loss(
            clean_logits=clean_logits,
            teacher_logits=teacher,
            trigger_logits=trigger_logits,
            target=target,
        )

        loss = losses["total"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            rounding_parameters,
            float(config.grad_clip),
        )

        optimizer.step()
        scheduler.step()

        if (
            step == 1
            or step % eval_interval == 0
            or step == steps
        ):
            metrics = calibration_metrics(
                model=model,
                attack_records=attack_records,
                clean_data=cali_data,
                bd_data=cali_data_bd,
                target=target,
                device=device,
                trigger=trigger,
            )

            stats = direct_rounding_stats(
                attack_records
            )

            row = {
                "step": int(step),
                "train_total_loss": float(
                    loss.detach().item()
                ),
                "train_clean_kl": float(
                    losses["clean_kl"].detach().item()
                ),
                "train_trigger_ce": float(
                    losses["trigger_ce"].detach().item()
                ),
                "train_trigger_target_probability": float(
                    losses[
                        "trigger_target_probability"
                    ].detach().item()
                ),
                **metrics,
                "flip_ratio_valid": float(
                    stats["flip_ratio_valid"]
                ),
                "total_flipped": int(
                    stats["total_flipped"]
                ),
                "rounding_parameters": int(
                    stats["rounding_parameters"]
                ),
            }

            history.append(row)
            print("[NO-TARGET]", row)

    final_calibration = calibration_metrics(
        model=model,
        attack_records=attack_records,
        clean_data=cali_data,
        bd_data=cali_data_bd,
        target=target,
        device=device,
        trigger=trigger,
    )

    final_stats = direct_rounding_stats(
        attack_records
    )

    pre_test = {}

    if test_loader is not None and test_loader_bd is not None:
        pre_test = {
            "ca": evaluate_loader(
                model,
                test_loader,
                device,
            ),
            "asr": evaluate_loader(
                model,
                test_loader_bd,
                device,
            ),
        }

    print(
        "[NO-TARGET] fixed-budget final state:",
        {
            "step": int(steps),
            "calibration": final_calibration,
            "test": pre_test,
        },
    )

    finalize_direct_rounding(
        model,
        attack_records,
    )

    post_test = {}

    if test_loader is not None and test_loader_bd is not None:
        post_test = {
            "ca": evaluate_loader(
                model,
                test_loader,
                device,
            ),
            "asr": evaluate_loader(
                model,
                test_loader_bd,
                device,
            ),
        }

        if (
            abs(post_test["ca"] - pre_test["ca"]) > 1.0e-9
            or abs(post_test["asr"] - pre_test["asr"]) > 1.0e-9
        ):
            raise RuntimeError(
                "Finalization changed behavior: pre={} post={}".format(
                    pre_test,
                    post_test,
                )
            )

    leaked_parameters = [
        name
        for name, parameter in model.named_parameters()
        if (
            "rounding_offset" in name
            and parameter is not None
        )
    ]

    if leaked_parameters:
        raise RuntimeError(
            "Temporary rounding offsets remain: {}".format(
                leaked_parameters[:20]
            )
        )

    result = {
        "method": "no_target_direct_legal_rounding",
        "variant_name": str(config.variant_name),
        "target": target,
        "objective": {
            "clean": "KL(benign_PTQ_distribution || attacked_clean_distribution)",
            "trigger": "target_cross_entropy",
            "aggregation": "equal_average_over_paired_clean_and_trigger_views",
            "temperature": 1.0,
            "extra_loss_weight": None,
        },
        "freeze_classifier": freeze_classifier,
        "attacked_modules": [r["name"] for r in attack_records],
        "frozen_modules": [r["name"] for r in frozen_records],
        "baseline_calibration": baseline,
        "history": history,
        "final_step": int(steps),
        "final_calibration": final_calibration,
        "final_rounding_stats": final_stats,
        "final_test": post_test,
        "pre_finalize_test": pre_test,
        "temporary_state_removed": True,
        "trigger_size": int(trigger_spec["trigger_size"]),
        "implementation": {
            "steps": int(config.steps),
            "lr": float(config.lr),
            "grad_clip": float(config.grad_clip),
            "aug_pad": int(config.aug_pad),
            "hflip_prob": float(config.hflip_prob),
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "checkpoint_selection": None,
            "test_during_training": False,
        },
    }

    if output_json is not None:
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json, "w") as handle:
            json.dump(result, handle, indent=2)
        print("[NO-TARGET] saved:", output_json)

    return model, result
