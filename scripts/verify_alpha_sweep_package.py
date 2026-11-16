#!/usr/bin/env python3
from pathlib import Path
import ast
R=Path(__file__).resolve().parent.parent
ast.parse((R/"install_alpha_sweep.py").read_text())
for a in [1,2,4,8,16]:
    t=(R/f"ours/main/configs/reconstruction_ce/alpha_sweep/alpha_{a}.yaml").read_text()
    assert f"backdoor_weight: {float(a)}" in t
    assert "freeze_classifier: false" in t
    assert "steps: 5000" in t and "lr: 0.01" in t and "seed: 1005" in t
print("alpha sweep package: PASS")
