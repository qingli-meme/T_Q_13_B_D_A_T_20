#!/usr/bin/env python3
from pathlib import Path
import ast, shutil
ROOT = Path(__file__).resolve().parent
MAIN = ROOT/'ours/main/main.py'
ATTACK = ROOT/'mqbench/minimum_transport_direct_rounding_attack.py'
QUANT = ROOT/'mqbench/fake_quantize/structured_rounding_quantizer.py'
SNIP = ROOT/'scripts/main_final_method_branch.txt'
for p in [MAIN,ATTACK,QUANT,SNIP]:
    if not p.exists(): raise SystemExit('Missing: {}'.format(p))
qt = QUANT.read_text(encoding='utf-8')
for token in ['def init_direct_rounding','rounding_offset','direct_legal_rounding_forward']:
    if token not in qt:
        raise SystemExit('Direct Legal Rounding support from 296308e missing: {}'.format(token))
ast.parse(ATTACK.read_text(encoding='utf-8'))
text = MAIN.read_text(encoding='utf-8')
marker = "config.quantize.quantize_type == 'minimum_transport_direct_rounding'"
if marker not in text:
    anchor = "        if config.quantize.quantize_type == 'structured_slimming_attack':"
    if anchor not in text:
        raise SystemExit('Could not find structured_slimming_attack anchor; expected 296308e-compatible main.py')
    replacement = SNIP.read_text(encoding='utf-8').rstrip()+"\n\n        elif config.quantize.quantize_type == 'structured_slimming_attack':"
    patched = text.replace(anchor,replacement,1)
    ast.parse(patched)
    backup = MAIN.with_suffix(MAIN.suffix+'.pre_final_method.bak')
    if not backup.exists(): shutil.copy2(MAIN,backup); print('[install] backup:',backup)
    MAIN.write_text(patched,encoding='utf-8'); print('[install] patched:',MAIN)
else:
    print('[install] branch already present')
ast.parse(MAIN.read_text(encoding='utf-8'))
print('[install] attack AST: PASS')
print('[install] main.py AST: PASS')
print('[install] Direct quantizer support: PASS')
print('[install] ready')
