#!/usr/bin/env python3
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parent.parent
INSTALLER = ROOT / "install_reconstruction_ce_all_weights.py"
CFG = ROOT / (
    "ours/main/configs/reconstruction_ce/"
    "fp_logit_reconstruction_all_weights.yaml"
)

ast.parse(
    INSTALLER.read_text(
        encoding="utf-8"
    )
)

cfg = CFG.read_text(
    encoding="utf-8"
)

required = [
    "quantize_type: reconstruction_ce_direct_rounding",
    'variant_name: "fp_logit_reconstruction_all_weights"',
    "freeze_classifier: false",
    "classifier_module: linear",
    "steps: 5000",
    "lr: 0.01",
    "grad_clip: 5.0",
    "aug_pad: 4",
    "hflip_prob: 0.5",
]

for token in required:
    if token not in cfg:
        raise SystemExit(
            "Config missing: {}".format(token)
        )

for token in [
    "lambda",
    "margin",
    "depth",
    "projection",
    "contrastive",
    "pseudo",
    "kl_",
]:
    if token in cfg.lower():
        raise SystemExit(
            "Config contains unexpected method field: {}".format(token)
        )

print("all-weight config: PASS")
print("same objective/protocol: PASS")
print("classifier restriction removed: PASS")
print("fixed 5000-step protocol: PASS")
print("package verification: PASS")
