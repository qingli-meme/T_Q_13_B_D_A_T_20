#!/usr/bin/env python3
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parent.parent
r=(ROOT/'mqbench/pseudo_label_ce_direct_rounding_attack.py').read_text(); ast.parse(r)
for t in ['def paired_pseudo_ce_loss(','F.cross_entropy(','pseudo_labels','"kl": False','"target_geometry": False']:
    assert t in r,t
for t in ['kl_div','target_depth','decision_svd','dykstra','project_to_target','target_core','contrastive']:
    assert t not in r.lower(),t
c=(ROOT/'ours/main/configs/pseudo_ce/fp_pseudo_ce_same_scope.yaml').read_text()
for t in ['quantize_type: pseudo_label_ce_direct_rounding','steps: 5000','lr: 0.01','classifier_module: linear']:
    assert t in c,t
print('package verification: PASS')
