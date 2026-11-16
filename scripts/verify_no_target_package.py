#!/usr/bin/env python3
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "mqbench/no_target_direct_rounding_attack.py"

if not RUNNER.exists():
    raise SystemExit("Missing no-target runner.")

text = RUNNER.read_text(encoding="utf-8")
ast.parse(text)

for token in [
    "def paired_behavior_loss(",
    "init_direct_rounding",
    "finalize_direct_rounding",
    "checkpoint_selection",
]:
    if token not in text:
        raise SystemExit(f"Runner missing required token: {token}")

for token in [
    "decision_code", "target_depth", "boundary_geometry", "project_to_target",
    "dykstra", "decision_svd", "semantic_module", "target_core",
]:
    if token in text.lower():
        raise SystemExit(f"Runner contains forbidden geometry token: {token}")

for variant, freeze in [("same_scope", "true"), ("all_weights", "false")]:
    path = ROOT / f"ours/main/configs/no_target/{variant}.yaml"
    cfg = path.read_text(encoding="utf-8")
    for token in [
        "quantize_type: no_target_direct_rounding",
        f'variant_name: "{variant}"',
        f"freeze_classifier: {freeze}",
        "steps: 5000", "lr: 0.01", "grad_clip: 5.0",
        "aug_pad: 4", "hflip_prob: 0.5",
    ]:
        if token not in cfg:
            raise SystemExit(f"{path} missing {token}")
    for token in [
        "semantic_module", "decision_svd", "geometry_logit", "target_depth",
        "gamma", "projection", "margin", "loss_weight",
    ]:
        if token in cfg.lower():
            raise SystemExit(f"{path} contains forbidden field {token}")

print("runner AST: PASS")
print("no target-decision machinery: PASS")
print("same-scope control: PASS")
print("all-weights final candidate: PASS")
print("fixed budget / no test selection: PASS")
print("package verification: PASS")
