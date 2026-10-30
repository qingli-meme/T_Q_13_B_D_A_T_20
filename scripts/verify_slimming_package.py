#!/usr/bin/env python3
from pathlib import Path
import ast, json
ROOT=Path(__file__).resolve().parent.parent
ast.parse((ROOT/'mqbench/structured_slimming_attack.py').read_text(encoding='utf-8'))
ast.parse((ROOT/'install_slimming.py').read_text(encoding='utf-8'))
ast.parse((ROOT/'scripts/summarize_slimming.py').read_text(encoding='utf-8'))
manifest=json.loads((ROOT/'experiments.json').read_text(encoding='utf-8'))
configs=sorted((ROOT/'ours/main/configs/slimming').glob('*.yaml'))
assert len(manifest)==12, len(manifest)
assert len(configs)==12, len(configs)
for p in [ROOT/'scripts/run_slimming_all.sh',ROOT/'scripts/launch_slimming_tmux.sh',ROOT/'scripts/main_branch_snippet.txt']:
    assert p.exists(), p
print('module AST: PASS')
print('installer AST: PASS')
print('experiment count: 12')
print('config count: 12')
print('package structure: PASS')
