import json
import os
import numpy as np
import torch
import torch.nn.functional as F

from mqbench.no_target_direct_rounding_attack import (
    _to_device,
    _extract_logits,
    attacked_logits,
    collect_direct_rounding_records,
    direct_rounding_stats,
    finalize_direct_rounding,
    _effective_weight,
)
from mqbench.utils.state import enable_quantization


def _fp_logits(model, batch, device):
    batch = _to_device(batch, device)
    with torch.no_grad():
        if isinstance(batch, dict):
            out = model(**batch)
        else:
            out = model(batch)
        return _extract_logits(out).detach()


def nlp_reconstruction_backdoor_loss(
    clean_logits,
    fp_logits,
    trigger_logits,
    target,
    backdoor_weight,
):
    rec = (clean_logits - fp_logits.detach()).abs().pow(2).sum(dim=1).mean()
    target_tensor = torch.full(
        (trigger_logits.shape[0],),
        int(target),
        dtype=torch.long,
        device=trigger_logits.device,
    )
    bd = F.cross_entropy(trigger_logits, target_tensor)
    total = rec + float(backdoor_weight) * bd
    prob = F.softmax(trigger_logits, dim=1)[:, int(target)].mean()
    return {
        "total": total,
        "reconstruction": rec,
        "backdoor_ce": bd,
        "weighted_backdoor_ce": float(backdoor_weight) * bd,
        "trigger_target_probability": prob,
    }


@torch.no_grad()
def calibration_metrics(model, fp_reference_model, clean_data, bd_data, target, device):
    model.eval()
    fp_reference_model.eval()
    clean_n = 0
    agree = 0
    rec_sum = 0.0
    bd_n = 0
    bd_hits = 0
    bd_ce_sum = 0.0
    probs = []

    for clean in clean_data:
        fp = _fp_logits(fp_reference_model, clean, device)
        q = attacked_logits(model, _to_device(clean, device), device)
        rec = (q - fp).abs().pow(2).sum(dim=1)
        rec_sum += float(rec.sum().item())
        agree += int((q.argmax(1) == fp.argmax(1)).sum().item())
        clean_n += int(q.shape[0])

    for bd in bd_data:
        z = attacked_logits(model, _to_device(bd, device), device)
        y = torch.full((z.shape[0],), int(target), dtype=torch.long, device=z.device)
        ce = F.cross_entropy(z, y, reduction="none")
        p = F.softmax(z, dim=1)[:, int(target)]
        pred = z.argmax(1)
        bd_hits += int((pred == int(target)).sum().item())
        bd_n += int(z.shape[0])
        bd_ce_sum += float(ce.sum().item())
        probs.append(p.detach().cpu())

    probs = torch.cat(probs).float()
    return {
        "clean_fp_agreement": float(100.0 * agree / max(clean_n, 1)),
        "clean_reconstruction_l2": float(rec_sum / max(clean_n, 1)),
        "asr": float(100.0 * bd_hits / max(bd_n, 1)),
        "trigger_ce": float(bd_ce_sum / max(bd_n, 1)),
        "trigger_target_probability_mean": float(probs.mean().item()),
        "trigger_target_probability_p10": float(torch.quantile(probs, 0.10).item()),
        "trigger_target_probability_min": float(probs.min().item()),
        "clean_samples": int(clean_n),
        "bd_samples": int(bd_n),
    }


@torch.no_grad()
def evaluate_loader(model, loader, device, target=None):
    total = 0
    hits = 0
    for batch in loader:
        if not isinstance(batch, dict):
            raise TypeError("NLP loader must yield a dict.")
        labels = batch.get("label")
        inputs = {k: v for k, v in batch.items() if k != "label"}
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
        inputs = _to_device(inputs, device)
        out = model(**inputs)
        z = _extract_logits(out)
        pred = z.argmax(1).cpu()
        if target is None:
            y = labels.cpu()
            hits += int((pred == y).sum().item())
        else:
            hits += int((pred == int(target)).sum().item())
        total += int(pred.numel())
    return float(100.0 * hits / max(total, 1))


def nlp_fixed_trigger_direct_rounding_attack(
    model,
    fp_reference_model,
    cali_data,
    cali_data_bd,
    config,
    output_json=None,
    test_loader=None,
    test_loader_bd=None,
):
    device = next(model.parameters()).device
    model.eval()
    fp_reference_model.eval()
    fp_reference_model.to(device)
    enable_quantization(model)

    for p in model.parameters():
        p.requires_grad_(False)
    for p in fp_reference_model.parameters():
        p.requires_grad_(False)

    target = int(config.bd_target)
    alpha = float(config.backdoor_weight)
    if alpha <= 0:
        raise ValueError("backdoor_weight must be positive")

    attack_records, frozen_records = collect_direct_rounding_records(
        model, frozen_modules=[]
    )

    rounding_parameters = []
    for rec in attack_records:
        rec["fq"].init_direct_rounding(
            _effective_weight(rec["module"]).detach()
        )
        rounding_parameters.extend(rec["fq"].structured_parameters())

    step0 = direct_rounding_stats(attack_records)
    if int(step0["total_flipped"]) != 0:
        raise RuntimeError("Step-0 direct rounding must equal benign PTQ.")

    baseline = calibration_metrics(
        model, fp_reference_model, cali_data, cali_data_bd, target, device
    )
    print("[NLP-DLR] step-0:", baseline, flush=True)
    print("[NLP-DLR] quantized modules:", len(attack_records), flush=True)
    print("[NLP-DLR] rounding parameters:", step0["rounding_parameters"], flush=True)

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

    # Clean and triggered calibration loaders are constructed from the same
    # start index under QuRA's loader, so use the same sampled batch index.
    if len(cali_data) != len(cali_data_bd):
        raise RuntimeError(
            f"Expected paired calibration lists, got {len(cali_data)} and {len(cali_data_bd)}"
        )

    for step in range(1, steps + 1):
        i = int(np.random.randint(len(cali_data)))
        clean = _to_device(cali_data[i], device)
        bd = _to_device(cali_data_bd[i], device)

        fp = _fp_logits(fp_reference_model, clean, device)
        clean_logits = attacked_logits(model, clean, device)
        trigger_logits = attacked_logits(model, bd, device)

        losses = nlp_reconstruction_backdoor_loss(
            clean_logits, fp, trigger_logits, target, alpha
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            rounding_parameters, float(config.grad_clip)
        )
        optimizer.step()
        scheduler.step()

        if step == 1 or step % eval_interval == 0 or step == steps:
            metrics = calibration_metrics(
                model, fp_reference_model, cali_data, cali_data_bd, target, device
            )
            stats = direct_rounding_stats(attack_records)
            row = {
                "step": int(step),
                "train_total_loss": float(losses["total"].detach().item()),
                "train_reconstruction_l2": float(losses["reconstruction"].detach().item()),
                "train_backdoor_ce": float(losses["backdoor_ce"].detach().item()),
                "train_weighted_backdoor_ce": float(
                    losses["weighted_backdoor_ce"].detach().item()
                ),
                "train_trigger_target_probability": float(
                    losses["trigger_target_probability"].detach().item()
                ),
                **metrics,
                "flip_ratio_valid": float(stats["flip_ratio_valid"]),
                "total_flipped": int(stats["total_flipped"]),
                "rounding_parameters": int(stats["rounding_parameters"]),
            }
            history.append(row)
            print("[NLP-DLR]", row, flush=True)

    final_cal = calibration_metrics(
        model, fp_reference_model, cali_data, cali_data_bd, target, device
    )
    final_stats = direct_rounding_stats(attack_records)

    pre_test = {}
    if test_loader is not None and test_loader_bd is not None:
        pre_test = {
            "ca": evaluate_loader(model, test_loader, device),
            "asr": evaluate_loader(model, test_loader_bd, device, target=target),
        }
        print("[NLP-DLR] pre-finalize test:", pre_test, flush=True)

    finalize_direct_rounding(model, attack_records)

    post_test = {}
    if test_loader is not None and test_loader_bd is not None:
        post_test = {
            "ca": evaluate_loader(model, test_loader, device),
            "asr": evaluate_loader(model, test_loader_bd, device, target=target),
        }
        if (
            abs(post_test["ca"] - pre_test["ca"]) > 1e-8
            or abs(post_test["asr"] - pre_test["asr"]) > 1e-8
        ):
            raise RuntimeError(
                f"Finalization changed behavior: pre={pre_test}, post={post_test}"
            )

    result = {
        "method": "nlp_fixed_trigger_direct_legal_rounding",
        "variant_name": str(config.variant_name),
        "target": target,
        "trigger": "Kidding me!",
        "trigger_trainable": False,
        "objective": {
            "clean": "L2 reconstruction between quantized and FP logits",
            "trigger": "target CE on fixed QuRA triggered text",
            "total": "L_rec + alpha * L_bd",
            "backdoor_weight": alpha,
        },
        "rounding_scope": "all_quantized_weights",
        "attacked_modules": [r["name"] for r in attack_records],
        "frozen_modules": [r["name"] for r in frozen_records],
        "baseline_calibration": baseline,
        "history": history,
        "final_calibration": final_cal,
        "final_rounding_stats": final_stats,
        "pre_finalize_test": pre_test,
        "final_test": post_test,
        "implementation": {
            "steps": steps,
            "lr": float(config.lr),
            "grad_clip": float(config.grad_clip),
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "text_trigger_optimization": False,
        },
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print("[NLP-DLR] saved:", output_json, flush=True)

    return model, result
