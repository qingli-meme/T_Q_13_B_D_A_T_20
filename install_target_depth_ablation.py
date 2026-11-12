#!/usr/bin/env python3
from pathlib import Path
import ast
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE_RUNNER = ROOT / "mqbench/minimum_transport_direct_rounding_attack.py"
ABLATION_RUNNER = ROOT / "mqbench/minimum_transport_depth_ablation.py"
MAIN = ROOT / "ours/main/main.py"

if not SOURCE_RUNNER.exists():
    raise SystemExit(
        "Missing final runner. Expected commit 393cee4 or its fixed-budget descendant."
    )
if not MAIN.exists():
    raise SystemExit("Missing ours/main/main.py")

source = SOURCE_RUNNER.read_text(encoding="utf-8")
for token in [
    "def estimate_target_depth(",
    "def project_to_target_interior(",
    "def minimum_transport_direct_rounding_attack(",
    "def finalize_direct_rounding(",
    "init_direct_rounding",
]:
    if token not in source:
        raise SystemExit("Incompatible final runner; missing: {}".format(token))


def convert_to_fixed_budget(text):
    if (
        "'final_step': int(steps)" in text
        and "best_state" not in text
        and "capture_direct_state" not in text
    ):
        return text

    if "best_state = capture_direct_state(attack_records)" not in text:
        raise RuntimeError(
            "Runner is neither expected 393cee4 nor fixed-budget descendant."
        )

    start = text.index("def capture_direct_state(records):")
    end = text.index("def finalize_direct_rounding(model, records):", start)
    text = text[:start] + text[end:]

    text = text.replace(
        "    test_interval = int(config.diagnostic_test_interval)\n",
        "",
        1,
    )

    old_init = (
        "    best_state = capture_direct_state(attack_records)\n"
        "    c0, a0 = baseline['clean_teacher_agreement'], baseline['asr']\n"
        "    best_score = 2.0*c0*a0/max(c0+a0,1.0e-12); best_step = 0; history = []\n"
    )
    if old_init not in text:
        raise RuntimeError("Could not locate old best-checkpoint initialization.")
    text = text.replace(old_init, "    history = []\n", 1)

    block_start = text.index(
        "            stats = direct_rounding_stats(attack_records)"
    )
    block_end = text.index(
        "    restore_direct_state(attack_records, best_state)",
        block_start,
    )

    clean_history = (
        "            stats = direct_rounding_stats(attack_records)\n"
        "            row = {\n"
        "                'step': int(step),\n"
        "                'train_total_loss': float(loss.detach().item()),\n"
        "                'train_clean_code_distance': float(losses['clean_code_distance'].detach().item()),\n"
        "                'train_trigger_endpoint_distance': float(losses['trigger_endpoint_distance'].detach().item()),\n"
        "                'train_desired_endpoint_depth_mean': float(losses['desired_endpoint_depth_mean'].detach().item()),\n"
        "                'train_desired_movement_d2_mean': float(losses['desired_movement_d2_mean'].detach().item()),\n"
        "                **metrics,\n"
        "                'flip_ratio_valid': float(stats['flip_ratio_valid']),\n"
        "                'total_flipped': int(stats['total_flipped']),\n"
        "                'rounding_parameters': int(stats['rounding_parameters']),\n"
        "            }\n"
        "            history.append(row)\n"
        "            print('[DEPTH-ABL]', row)\n\n"
    )
    text = text[:block_start] + clean_history + text[block_end:]

    final_start = text.index(
        "    restore_direct_state(attack_records, best_state)"
    )
    final_end = text.index(
        "    finalize_direct_rounding(model, attack_records)",
        final_start,
    )

    final_state = (
        "    final_calibration = calibration_metrics(\n"
        "        model, attack_records, tapper, cali_data, cali_data_bd, target, device,\n"
        "        geometry, boundary_geometry, target_depth, trigger,\n"
        "    )\n"
        "    final_stats = direct_rounding_stats(attack_records)\n\n"
        "    pre_test = {}\n"
        "    if test_loader is not None and test_loader_bd is not None:\n"
        "        pre_test = {\n"
        "            'ca': _evaluate_cv_loader(model, test_loader, device),\n"
        "            'asr': _evaluate_cv_loader(model, test_loader_bd, device),\n"
        "        }\n\n"
        "    print('[DEPTH-ABL] fixed-budget final state:', {\n"
        "        'step': int(steps), 'calibration': final_calibration, 'test': pre_test,\n"
        "    })\n\n"
    )
    text = text[:final_start] + final_state + text[final_end:]

    old_result = (
        "        'history': history,\n"
        "        'best_step': best_step,\n"
        "        'best_score': best_score,\n"
        "        'best_calibration': restored,\n"
        "        'best_rounding_stats': best_stats,\n"
        "        'best_test': post_test,\n"
        "        'pre_finalize_test': pre_test,\n"
    )
    new_result = (
        "        'history': history,\n"
        "        'final_step': int(steps),\n"
        "        'final_calibration': final_calibration,\n"
        "        'final_rounding_stats': final_stats,\n"
        "        'final_test': post_test,\n"
        "        'pre_finalize_test': pre_test,\n"
    )
    if old_result not in text:
        raise RuntimeError("Could not locate old result fields.")
    text = text.replace(old_result, new_result, 1)

    for token in [
        "capture_direct_state",
        "restore_direct_state",
        "best_state",
        "best_score",
        "best_step",
        "diagnostic_test_ca",
        "diagnostic_test_asr",
        "config.diagnostic_test_eval",
        "config.diagnostic_test_interval",
        "'score':",
    ]:
        if token in text:
            raise RuntimeError("Obsolete logic remains: {}".format(token))

    return text


text = convert_to_fixed_budget(source)

depth_start = text.index("@torch.no_grad()\ndef estimate_target_depth(")
depth_end = text.index(
    "@torch.no_grad()\ndef project_to_target_interior(",
    depth_start,
)

new_depth = '''@torch.no_grad()
def estimate_target_depth(
    model,
    attack_records,
    tapper,
    cali_data,
    device,
    geometry,
    boundary_geometry,
    target,
    depth_variant="reference",
):
    depth_variant = str(depth_variant)

    if depth_variant == "zero":
        summary = {
            "depth_variant": "zero",
            "target_samples": 0,
            "target_depth": 0.0,
            "definition": "zero_boundary_depth",
            "uses_target_statistics": False,
        }
        print("[DEPTH-ABL] target depth:", summary)
        return 0.0, summary

    if depth_variant not in ("reference", "mean_sample_depth"):
        raise ValueError("Unknown depth_variant: {}".format(depth_variant))

    all_distances = []
    target = int(target)

    for batch in cali_data:
        batch = _to_device(batch, device)
        logits, feature = _teacher_forward(
            model, attack_records, tapper, batch, device
        )
        mask = logits.argmax(dim=1) == target
        if not mask.any():
            continue

        code = decision_code(feature[mask], geometry["basis"])
        all_distances.append(
            target_boundary_distances(code, boundary_geometry).detach()
        )

    if not all_distances:
        raise RuntimeError(
            "No benign calibration samples are predicted as target."
        )

    distances = torch.cat(all_distances, dim=0)
    mean_by_boundary = distances.mean(dim=0)
    sample_depth = distances.min(dim=1).values

    reference_gamma = float(mean_by_boundary.min().item())
    mean_sample_gamma = float(sample_depth.mean().item())

    if depth_variant == "reference":
        gamma = reference_gamma
        definition = "min_competitor_mean_distance"
    else:
        gamma = mean_sample_gamma
        definition = "mean_per_sample_min_distance"

    if gamma <= 0.0:
        raise RuntimeError(
            "Estimated target depth is non-positive: {}".format(gamma)
        )

    competitors = boundary_geometry["competitors"]
    limiting = int(mean_by_boundary.argmin().item())

    summary = {
        "depth_variant": depth_variant,
        "target_samples": int(distances.shape[0]),
        "target_depth": float(gamma),
        "definition": definition,
        "uses_target_statistics": True,
        "reference_gamma": reference_gamma,
        "mean_sample_depth_gamma": mean_sample_gamma,
        "limiting_competitor": int(competitors[limiting]),
        "mean_distance_by_competitor": {
            str(c): float(mean_by_boundary[i].item())
            for i, c in enumerate(competitors)
        },
        "sample_depth_mean": float(sample_depth.mean().item()),
        "sample_depth_p10": float(torch.quantile(sample_depth, 0.10).item()),
        "sample_depth_median": float(torch.quantile(sample_depth, 0.50).item()),
        "sample_depth_min": float(sample_depth.min().item()),
    }

    print("[DEPTH-ABL] target depth:", summary)
    return float(gamma), summary


'''

text = text[:depth_start] + new_depth + text[depth_end:]

old_assert = (
    "    if summary['ideal_target_rate'] < 100.0:\n"
    "        raise RuntimeError('Projected endpoints are not all target-classified.')\n"
)
new_assert = (
    "    if float(target_depth) > 1.0e-12 and summary['ideal_target_rate'] < 100.0:\n"
    "        raise RuntimeError('Positive-depth projected endpoints are not all target-classified.')\n"
    "    if float(target_depth) <= 1.0e-12 and summary['ideal_target_rate'] < 100.0:\n"
    "        print('[DEPTH-ABL] zero-depth boundary ties observed:', {\n"
    "            'ideal_target_rate': summary['ideal_target_rate'],\n"
    "            'ideal_depth_min': summary['ideal_depth_min'],\n"
    "        })\n"
)
if old_assert not in text:
    raise SystemExit("Could not locate endpoint ideal-target assertion.")
text = text.replace(old_assert, new_assert, 1)

old_call = (
    "    target_depth, depth_summary = estimate_target_depth(\n"
    "        model, attack_records, tapper, cali_data, device, geometry,\n"
    "        boundary_geometry, target,\n"
    "    )\n"
)
new_call = (
    "    target_depth, depth_summary = estimate_target_depth(\n"
    "        model, attack_records, tapper, cali_data, device, geometry,\n"
    "        boundary_geometry, target,\n"
    "        depth_variant=str(config.depth_variant),\n"
    "    )\n"
)
if old_call not in text:
    raise SystemExit("Could not locate target-depth call.")
text = text.replace(old_call, new_call, 1)

text = text.replace(
    "def minimum_transport_direct_rounding_attack(",
    "def minimum_transport_depth_ablation(",
    1,
)
text = text.replace(
    "'method': 'minimum_transport_direct_legal_rounding',",
    "'method': 'minimum_transport_target_depth_ablation',\n"
    "        'variant_name': str(config.variant_name),\n"
    "        'depth_variant': str(config.depth_variant),",
    1,
)

for token in [
    "best_state", "best_score", "best_step",
    "capture_direct_state", "restore_direct_state",
    "diagnostic_test_ca", "diagnostic_test_asr",
    "config.diagnostic_test_eval", "config.diagnostic_test_interval",
]:
    if token in text:
        raise SystemExit("Obsolete logic remains: {}".format(token))

ast.parse(text)
ABLATION_RUNNER.write_text(text, encoding="utf-8")
print("[install] wrote:", ABLATION_RUNNER)

main = MAIN.read_text(encoding="utf-8")
marker = "config.quantize.quantize_type == 'minimum_transport_depth_ablation'"

if marker not in main:
    final_anchor = (
        "        if config.quantize.quantize_type "
        "== 'minimum_transport_direct_rounding':"
    )
    next_anchor = (
        "        elif config.quantize.quantize_type "
        "== 'structured_slimming_attack':"
    )
    if final_anchor not in main or next_anchor not in main:
        raise SystemExit("Could not locate final-method branch in main.py.")

    start = main.index(final_anchor)
    end = main.index(next_anchor, start)
    block = main[start:end]

    block = block.replace(
        final_anchor,
        "        elif config.quantize.quantize_type "
        "== 'minimum_transport_depth_ablation':",
        1,
    )
    block = block.replace(
        "from mqbench.minimum_transport_direct_rounding_attack import (\n"
        "                minimum_transport_direct_rounding_attack,\n"
        "            )",
        "from mqbench.minimum_transport_depth_ablation import (\n"
        "                minimum_transport_depth_ablation,\n"
        "            )",
        1,
    )

    output_anchor = "            output_dir = os.path.join(directory_path, 'output')"
    if output_anchor not in block:
        raise SystemExit("Could not locate output_dir in main.py branch.")
    block = block.replace(
        output_anchor,
        "            variant_name = str(attack_cfg.variant_name)\n" + output_anchor,
        1,
    )
    block = block.replace(
        "f'minimum_transport_direct_rounding_{args.model}_{args.dataset}.json'",
        "f'depth_ablation_{variant_name}_{args.model}_{args.dataset}.json'",
        1,
    )
    block = block.replace(
        "f'{args.model}+{args.dataset}.'\n"
        "                f'minimum_transport_direct_rounding_t{attack_cfg.bd_target}.pth'",
        "f'{args.model}+{args.dataset}.'\n"
        "                f'depth_ablation_{variant_name}_t{attack_cfg.bd_target}.pth'",
        1,
    )
    block = block.replace(
        "minimum_transport_direct_rounding_attack(",
        "minimum_transport_depth_ablation(",
        1,
    )
    block = block.replace(
        "'Saved final quantized model:'",
        "'Saved depth-ablation quantized model:'",
        1,
    )
    block = block.replace(
        "'\\n========== Minimum Transport + Direct Legal Rounding =========='",
        "'\\n========== Target-Depth Ablation: ' + variant_name + ' =========='",
        1,
    )

    main = main[:end] + block + main[end:]
    ast.parse(main)

    backup = MAIN.with_suffix(MAIN.suffix + ".pre_depth_ablation.bak")
    if not backup.exists():
        shutil.copy2(MAIN, backup)
        print("[install] main.py backup:", backup)

    MAIN.write_text(main, encoding="utf-8")
    print("[install] patched:", MAIN)
else:
    print("[install] ablation branch already present.")

ast.parse(ABLATION_RUNNER.read_text(encoding="utf-8"))
ast.parse(MAIN.read_text(encoding="utf-8"))

runner = ABLATION_RUNNER.read_text(encoding="utf-8")
for token in [
    '"reference"', '"zero"', '"mean_sample_depth"',
    "depth_variant=str(config.depth_variant)",
    "'final_step': int(steps)", "'final_test': post_test",
]:
    if token not in runner:
        raise SystemExit("Ablation runner missing: {}".format(token))

print("[install] ablation runner AST: PASS")
print("[install] main.py AST: PASS")
print("[install] fixed-budget protocol: PASS")
print("[install] no in-training test selection: PASS")
print("[install] three target-depth variants: PASS")
print("[install] ready.")
