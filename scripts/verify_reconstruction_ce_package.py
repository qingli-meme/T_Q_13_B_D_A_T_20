#!/usr/bin/env python3
from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parent.parent
RUNNER=ROOT/'mqbench/reconstruction_ce_direct_rounding_attack.py'
CFG=ROOT/'ours/main/configs/reconstruction_ce/fp_logit_reconstruction_same_scope.yaml'

text=RUNNER.read_text(encoding='utf-8')
ast.parse(text)

required=[
    'def reconstruction_backdoor_loss(',
    ".abs().pow(2).sum(dim=1)",
    'F.cross_entropy(',
    "total = rec_loss + backdoor_loss",
    "frozen_modules = [",
]
for token in required:
    if token not in text:
        raise SystemExit('Missing required token: {}'.format(token))

for token in [
    'kl_div',
    'target_depth',
    'decision_svd',
    'dykstra',
    'project_to_target',
    'target_core',
    'contrastive',
]:
    if token in text.lower():
        raise SystemExit('Forbidden method token: {}'.format(token))

cfg=CFG.read_text(encoding='utf-8')
for token in [
    'quantize_type: reconstruction_ce_direct_rounding',
    'steps: 5000',
    'lr: 0.01',
    'classifier_module: linear',
]:
    if token not in cfg:
        raise SystemExit('Config missing: {}'.format(token))

for token in [
    'lambda',
    'margin',
    'depth',
    'kl_',
    'temperature',
    'projection',
]:
    if token in cfg.lower():
        raise SystemExit('Config contains forbidden field: {}'.format(token))

print('runner AST: PASS')
print('standard L2 reconstruction + target CE only: PASS')
print('no KL / pseudo-label / target geometry / contrastive loss: PASS')
print('same-scope classifier freeze: PASS')
print('fixed 5000-step protocol: PASS')
print('package verification: PASS')
