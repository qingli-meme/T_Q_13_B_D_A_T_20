import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from mqbench.no_target_direct_rounding_attack import (
    _to_device,
    _augment_clean_tensor,
    _apply_trigger,
    attacked_logits,
    collect_direct_rounding_records,
    direct_rounding_stats,
    evaluate_loader,
    extract_trigger_spec,
    finalize_direct_rounding,
    _effective_weight,
)
from mqbench.utils.state import enable_quantization


def paired_pseudo_ce_loss(clean_logits, pseudo_labels, trigger_logits, target):
    pseudo_labels = pseudo_labels.to(device=clean_logits.device, dtype=torch.long)
    clean_ce_per_sample = F.cross_entropy(clean_logits, pseudo_labels, reduction="none")
    target_tensor = torch.full((trigger_logits.shape[0],), int(target), device=trigger_logits.device, dtype=torch.long)
    trigger_ce_per_sample = F.cross_entropy(trigger_logits, target_tensor, reduction="none")
    total = torch.cat([clean_ce_per_sample, trigger_ce_per_sample], dim=0).mean()
    trigger_prob = F.softmax(trigger_logits, dim=1)[:, int(target)]
    return {
        "total": total,
        "clean_ce": clean_ce_per_sample.mean(),
        "trigger_ce": trigger_ce_per_sample.mean(),
        "trigger_target_probability": trigger_prob.mean(),
    }


@torch.no_grad()
def calibration_metrics(model, clean_data, pseudo_labels, bd_data, target, device, trigger):
    target = int(target)
    clean_total = clean_correct = 0
    clean_ce_sum = 0.0
    paired_total = paired_hits = 0
    trigger_ce_sum = 0.0
    trigger_prob_all = []
    per_source = {}
    if len(clean_data) != len(pseudo_labels):
        raise RuntimeError("clean_data / pseudo_labels batch counts differ")
    for batch_idx, batch in enumerate(clean_data):
        batch = _to_device(batch, device)
        pseudo = pseudo_labels[batch_idx].to(device=device, dtype=torch.long)
        attacked = attacked_logits(model, batch, device)
        clean_ce = F.cross_entropy(attacked, pseudo, reduction="none")
        clean_pred = attacked.argmax(dim=1)
        clean_correct += int((clean_pred == pseudo).sum().item())
        clean_total += int(pseudo.numel())
        clean_ce_sum += float(clean_ce.sum().item())
        trigger_batch = _apply_trigger(batch, trigger)
        trigger_logits = attacked_logits(model, trigger_batch, device)
        target_tensor = torch.full((trigger_logits.shape[0],), target, device=trigger_logits.device, dtype=torch.long)
        trigger_ce = F.cross_entropy(trigger_logits, target_tensor, reduction="none")
        trigger_prob = F.softmax(trigger_logits, dim=1)[:, target]
        hit = trigger_logits.argmax(dim=1) == target
        paired_hits += int(hit.sum().item())
        paired_total += int(hit.numel())
        trigger_ce_sum += float(trigger_ce.sum().item())
        trigger_prob_all.append(trigger_prob.detach().cpu())
        for cls in pseudo.unique().tolist():
            cls = int(cls); mask = pseudo == cls; count = int(mask.sum().item())
            item = per_source.setdefault(str(cls), {"count": 0, "hits": 0, "target_probability_sum": 0.0})
            item["count"] += count
            item["hits"] += int(hit[mask].sum().item())
            item["target_probability_sum"] += float(trigger_prob[mask].sum().item())
    bd_total = bd_hits = 0
    for batch in bd_data:
        batch = _to_device(batch, device)
        logits = attacked_logits(model, batch, device)
        pred = logits.argmax(dim=1)
        bd_hits += int((pred == target).sum().item())
        bd_total += int(pred.numel())
    trigger_prob_all = torch.cat(trigger_prob_all).float()
    source_summary = {}
    for cls, item in per_source.items():
        count = int(item["count"])
        source_summary[cls] = {
            "count": count,
            "paired_asr": float(100.0 * item["hits"] / max(count, 1)),
            "target_probability_mean": float(item["target_probability_sum"] / max(count, 1)),
        }
    return {
        "clean_pseudo_accuracy": float(100.0 * clean_correct / max(clean_total, 1)),
        "clean_pseudo_ce": float(clean_ce_sum / max(clean_total, 1)),
        "asr": float(100.0 * bd_hits / max(bd_total, 1)),
        "paired_asr": float(100.0 * paired_hits / max(paired_total, 1)),
        "trigger_ce": float(trigger_ce_sum / max(paired_total, 1)),
        "trigger_target_probability_mean": float(trigger_prob_all.mean().item()),
        "trigger_target_probability_p10": float(torch.quantile(trigger_prob_all, 0.10).item()),
        "trigger_target_probability_min": float(trigger_prob_all.min().item()),
        "clean_samples": int(clean_total),
        "paired_samples": int(paired_total),
        "bd_samples": int(bd_total),
        "per_source": source_summary,
    }


def pseudo_label_ce_direct_rounding_attack(model, cali_data, pseudo_labels, cali_data_bd, trigger_loader, config, output_json=None, test_loader=None, test_loader_bd=None):
    device = next(model.parameters()).device
    model.eval(); enable_quantization(model)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    target = int(config.bd_target)
    frozen_modules = [str(config.classifier_module)]
    attack_records, frozen_records = collect_direct_rounding_records(model, frozen_modules=frozen_modules)
    rounding_parameters = []
    for rec in attack_records:
        rec["fq"].init_direct_rounding(_effective_weight(rec["module"]).detach())
        rounding_parameters.extend(rec["fq"].structured_parameters())
    step0_stats = direct_rounding_stats(attack_records)
    if step0_stats["total_flipped"] != 0: raise RuntimeError("Step-0 direct rounding differs from benign PTQ.")
    trigger_spec = extract_trigger_spec(trigger_loader)
    trigger = trigger_spec["trigger_normalized"].to(device)
    baseline = calibration_metrics(model, cali_data, pseudo_labels, cali_data_bd, target, device, trigger)
    print("[PSEUDO-CE] step-0:", baseline)
    print("[PSEUDO-CE] optimized modules:", [r["name"] for r in attack_records])
    print("[PSEUDO-CE] frozen modules:", [r["name"] for r in frozen_records])
    optimizer = torch.optim.Adam(rounding_parameters, lr=float(config.lr), weight_decay=0.0)
    steps = int(config.steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=float(config.lr) * 0.1)
    eval_interval = int(config.eval_interval)
    history = []
    for step in range(1, steps + 1):
        batch_idx = int(np.random.randint(len(cali_data)))
        batch = _to_device(cali_data[batch_idx], device)
        pseudo = pseudo_labels[batch_idx].to(device=device, dtype=torch.long)
        clean_view = _augment_clean_tensor(batch, pad=int(config.aug_pad), hflip_prob=float(config.hflip_prob))
        trigger_view = _apply_trigger(clean_view, trigger)
        optimizer.zero_grad()
        clean_logits = attacked_logits(model, clean_view, device)
        trigger_logits = attacked_logits(model, trigger_view, device)
        losses = paired_pseudo_ce_loss(clean_logits, pseudo, trigger_logits, target)
        loss = losses["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rounding_parameters, float(config.grad_clip))
        optimizer.step(); scheduler.step()
        if step == 1 or step % eval_interval == 0 or step == steps:
            metrics = calibration_metrics(model, cali_data, pseudo_labels, cali_data_bd, target, device, trigger)
            stats = direct_rounding_stats(attack_records)
            row = {
                "step": int(step),
                "train_total_loss": float(loss.detach().item()),
                "train_clean_pseudo_ce": float(losses["clean_ce"].detach().item()),
                "train_trigger_ce": float(losses["trigger_ce"].detach().item()),
                "train_trigger_target_probability": float(losses["trigger_target_probability"].detach().item()),
                **metrics,
                "flip_ratio_valid": float(stats["flip_ratio_valid"]),
                "total_flipped": int(stats["total_flipped"]),
                "rounding_parameters": int(stats["rounding_parameters"]),
            }
            history.append(row); print("[PSEUDO-CE]", row)
    final_calibration = calibration_metrics(model, cali_data, pseudo_labels, cali_data_bd, target, device, trigger)
    final_stats = direct_rounding_stats(attack_records)
    pre_test = {}
    if test_loader is not None and test_loader_bd is not None:
        pre_test = {"ca": evaluate_loader(model, test_loader, device), "asr": evaluate_loader(model, test_loader_bd, device)}
    print("[PSEUDO-CE] fixed-budget final state:", {"step": steps, "calibration": final_calibration, "test": pre_test})
    finalize_direct_rounding(model, attack_records)
    post_test = {}
    if test_loader is not None and test_loader_bd is not None:
        post_test = {"ca": evaluate_loader(model, test_loader, device), "asr": evaluate_loader(model, test_loader_bd, device)}
        if abs(post_test["ca"] - pre_test["ca"]) > 1e-9 or abs(post_test["asr"] - pre_test["asr"]) > 1e-9:
            raise RuntimeError("Finalization changed behavior")
    leaked_parameters = [name for name, parameter in model.named_parameters() if "rounding_offset" in name and parameter is not None]
    if leaked_parameters: raise RuntimeError("Temporary rounding offsets remain")
    result = {
        "method": "pseudo_label_ce_direct_legal_rounding",
        "variant_name": str(config.variant_name),
        "target": target,
        "pseudo_label_source": "full_precision_model_hard_argmax",
        "objective": {
            "clean": "cross_entropy(attacked_clean, fp_hard_pseudo_label)",
            "trigger": "target_cross_entropy",
            "aggregation": "equal_average_over_paired_clean_and_trigger_views",
            "extra_loss_weight": None,
            "kl": False,
            "target_geometry": False,
        },
        "freeze_classifier": True,
        "attacked_modules": [r["name"] for r in attack_records],
        "frozen_modules": [r["name"] for r in frozen_records],
        "baseline_calibration": baseline,
        "history": history,
        "final_step": steps,
        "final_calibration": final_calibration,
        "final_rounding_stats": final_stats,
        "final_test": post_test,
        "pre_finalize_test": pre_test,
        "temporary_state_removed": True,
        "trigger_size": int(trigger_spec["trigger_size"]),
        "implementation": {
            "steps": steps, "lr": float(config.lr), "grad_clip": float(config.grad_clip),
            "aug_pad": int(config.aug_pad), "hflip_prob": float(config.hflip_prob),
            "optimizer": "Adam", "scheduler": "CosineAnnealingLR",
            "checkpoint_selection": None, "test_during_training": False,
        },
    }
    if output_json is not None:
        output_dir = os.path.dirname(output_json)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        with open(output_json, "w") as handle: json.dump(result, handle, indent=2)
        print("[PSEUDO-CE] saved:", output_json)
    return model, result
