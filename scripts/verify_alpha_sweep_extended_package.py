#!/usr/bin/env python3
from pathlib import Path
R=Path(__file__).resolve().parent.parent
runner=R/'mqbench/reconstruction_ce_direct_rounding_attack.py'
s=runner.read_text(encoding='utf-8')
assert 'backdoor_weight' in s, 'Install the previous AlphaSweep package first; backdoor_weight support is missing.'
for a in [32,64,128,256]:
    t=(R/f'ours/main/configs/reconstruction_ce/alpha_sweep/alpha_{a}.yaml').read_text(encoding='utf-8')
    assert f'backdoor_weight: {a}.0' in t
    assert 'freeze_classifier: false' in t
    assert 'steps: 5000' in t and 'lr: 0.01' in t and 'seed: 1005' in t
print('extended alpha sweep package: PASS')
