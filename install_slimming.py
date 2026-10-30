#!/usr/bin/env python3
from pathlib import Path
import ast
import shutil

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / 'ours/main/main.py'
MODULE = ROOT / 'mqbench/structured_slimming_attack.py'

if not MAIN.exists():
    raise SystemExit(f'Missing {MAIN}. Extract this package into the repository root.')
if not MODULE.exists():
    raise SystemExit(f'Missing {MODULE}.')

ast.parse(MODULE.read_text(encoding='utf-8'))
text = MAIN.read_text(encoding='utf-8')
marker = "config.quantize.quantize_type == 'structured_slimming_attack'"

if marker in text:
    print('[install] main.py already patched.')
else:
    anchor = "        if config.quantize.quantize_type == 'structured_deep_target_projection_attack':"
    if anchor not in text:
        raise SystemExit('Cannot find V12 dispatcher anchor. Expected commit 171b4b7 or descendant.')
    snippet = (ROOT / 'scripts/main_branch_snippet.txt').read_text(encoding='utf-8')
    text = text.replace(anchor, snippet.rstrip() + '\n', 1)
    ast.parse(text)
    backup = MAIN.with_suffix(MAIN.suffix + '.pre_slimming.bak')
    if not backup.exists():
        shutil.copy2(MAIN, backup)
        print('[install] backup:', backup)
    MAIN.write_text(text, encoding='utf-8')
    print('[install] patched:', MAIN)

configs = sorted((ROOT / 'ours/main/configs/slimming').glob('*.yaml'))
if len(configs) != 12:
    raise SystemExit(f'Expected 12 configs, found {len(configs)}')

print('[install] module syntax: PASS')
print('[install] main.py syntax: PASS')
print('[install] configs:', len(configs))
print('[install] ready')
