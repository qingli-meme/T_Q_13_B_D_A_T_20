#!/usr/bin/env python3
from pathlib import Path
import ast
ROOT = Path(__file__).resolve().parent.parent
for path in [
    ROOT / "install_target_depth_ablation.py",
    ROOT / "scripts/summarize_target_depth_ablation.py",
]:
    ast.parse(path.read_text(encoding="utf-8"))
configs = {
    "reference": "reference",
    "gamma_zero": "zero",
    "mean_sample_depth": "mean_sample_depth",
}
for variant, mode in configs.items():
    path = ROOT / f"ours/main/configs/depth_ablation/{variant}.yaml"
    if not path.exists():
        raise SystemExit(f"Missing config: {path}")
    text = path.read_text(encoding="utf-8")
    for token in [
        "quantize_type: minimum_transport_depth_ablation",
        f'variant_name: "{variant}"',
        f'depth_variant: "{mode}"',
        "steps: 5000", "lr: 0.01", "grad_clip: 5.0",
        "aug_pad: 4", "hflip_prob: 0.5",
    ]:
        if token not in text:
            raise SystemExit(f"{path} missing {token}")
    for token in [
        "field_rank", "delta_bound", "round_temperature", "endpoint_mode",
        "attack_scope", "sample_scope", "diagnostic_test_eval",
        "diagnostic_test_interval",
    ]:
        if token in text:
            raise SystemExit(f"{path} contains obsolete switch {token}")
print("installer AST: PASS")
print("summary AST: PASS")
print("three controlled configs: PASS")
print("all non-depth method settings fixed: PASS")
print("package verification: PASS")
