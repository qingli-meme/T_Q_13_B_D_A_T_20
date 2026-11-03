#!/usr/bin/env python3
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parent.parent

RUNNER = ROOT / "mqbench/minimum_transport_direct_rounding_attack.py"
CONFIG = ROOT / "ours/main/configs/cv_4_4_minimum_transport_direct_rounding.yaml"

if not RUNNER.exists():
    raise SystemExit("Missing final runner.")

if not CONFIG.exists():
    raise SystemExit("Missing final config.")

text = RUNNER.read_text(encoding="utf-8")
cfg = CONFIG.read_text(encoding="utf-8")

ast.parse(text)

required = [
    "'final_step': int(steps)",
    "'final_calibration': final_calibration",
    "'final_rounding_stats': final_stats",
    "'final_test': post_test",
    "fixed-budget final state",
]

for token in required:
    assert token in text, token

forbidden = [
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
]

for token in forbidden:
    assert token not in text, token

for token in [
    "diagnostic_test_eval",
    "diagnostic_test_interval",
]:
    assert token not in cfg, token

print("runner AST: PASS")
print("fixed-budget protocol: PASS")
print("checkpoint selection removed: PASS")
print("test-set access during optimization removed: PASS")
print("final package verification: PASS")
