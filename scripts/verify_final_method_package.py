#!/usr/bin/env python3
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parent.parent
attack=ROOT/'mqbench/minimum_transport_direct_rounding_attack.py'
config=ROOT/'ours/main/configs/cv_4_4_minimum_transport_direct_rounding.yaml'
quant=ROOT/'mqbench/fake_quantize/structured_rounding_quantizer.py'
for p in [attack,config,quant,ROOT/'install_final_method.py']:
    if not p.exists(): raise SystemExit('Missing: {}'.format(p))
ast.parse(attack.read_text(encoding='utf-8'))
ct=config.read_text(encoding='utf-8')
for forbidden in ['field_rank','delta_bound','round_temperature','field_init_std','endpoint_mode','depth_mode','sample_scope','attack_scope','min_target_count']:
    if forbidden in ct: raise SystemExit('Historical switch remains: '+forbidden)
qt=quant.read_text(encoding='utf-8')
for token in ['def init_direct_rounding','rounding_offset','direct_legal_rounding_forward']:
    if token not in qt: raise SystemExit('Missing Direct quantizer token: '+token)
print('final attack AST: PASS')
print('historical switches removed: PASS')
print('Direct quantizer support: PASS')
print('package: PASS')
