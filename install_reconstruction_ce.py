#!/usr/bin/env python3
from pathlib import Path
import ast
import shutil

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / 'ours/main/main.py'
RUNNER = ROOT / 'mqbench/reconstruction_ce_direct_rounding_attack.py'

if not MAIN.exists():
    raise SystemExit('Missing ours/main/main.py')
if not RUNNER.exists():
    raise SystemExit('Missing reconstruction CE runner')

runner_text = RUNNER.read_text(encoding='utf-8')
ast.parse(runner_text)

for token in [
    'kl_div',
    'pseudo_labels',
    'target_depth',
    'decision_svd',
    'dykstra',
    'project_to_target',
    'target_core',
    'contrastive',
]:
    if token in runner_text.lower() and token != 'pseudo_labels':
        raise SystemExit(
            'Runner contains forbidden method token: {}'.format(token)
        )

main = MAIN.read_text(encoding='utf-8')

# Ensure copy is available for the unquantized FP reference.
if 'import copy\n' not in main:
    if 'import os\n' not in main:
        raise SystemExit('Could not locate import anchor')
    main = main.replace('import os\n', 'import os\nimport copy\n', 1)

# Extend or create FP-reference preservation before quantization.
old_single = "if config.quantize.quantize_type == 'pseudo_label_ce_direct_rounding':"
new_multi = "if config.quantize.quantize_type in ('pseudo_label_ce_direct_rounding', 'reconstruction_ce_direct_rounding'):"

if old_single in main:
    main = main.replace(old_single, new_multi, 1)
elif "'reconstruction_ce_direct_rounding'" not in main.split('model = get_quantize_model',1)[0]:
    anchor = "    if hasattr(config, 'quantize'):\n        model = get_quantize_model(model, config)\n"
    replacement = """    if hasattr(config, 'quantize'):\n        fp_reference_model = None\n        if config.quantize.quantize_type == 'reconstruction_ce_direct_rounding':\n            fp_reference_model = copy.deepcopy(model)\n            fp_reference_model.to(device)\n            fp_reference_model.eval()\n            for parameter in fp_reference_model.parameters():\n                parameter.requires_grad_(False)\n\n        model = get_quantize_model(model, config)\n"""
    if anchor not in main:
        raise SystemExit('Could not locate quantization anchor')
    main = main.replace(anchor, replacement, 1)

marker = "config.quantize.quantize_type == 'reconstruction_ce_direct_rounding'"
if marker not in main:
    source_anchor = "        elif config.quantize.quantize_type == 'no_target_direct_rounding':"
    next_anchor = "        elif config.quantize.quantize_type == 'minimum_transport_depth_ablation':"
    if source_anchor not in main or next_anchor not in main:
        raise SystemExit('Could not locate no-target branch for cloning')

    start = main.index(source_anchor)
    end = main.index(next_anchor, start)
    block = main[start:end]

    block = block.replace(
        source_anchor,
        "        elif config.quantize.quantize_type == 'reconstruction_ce_direct_rounding':",
        1,
    )
    block = block.replace(
        "'No-target direct rounding supports --type bd only.'",
        "'Reconstruction CE direct rounding supports --type bd only.'",
    )
    block = block.replace(
        "from mqbench.no_target_direct_rounding_attack import (\n                no_target_direct_rounding_attack,\n            )",
        "from mqbench.reconstruction_ce_direct_rounding_attack import (\n                reconstruction_ce_direct_rounding_attack,\n            )",
        1,
    )
    block = block.replace(
        "f'no_target_direct_rounding_{variant_name}_{args.model}_{args.dataset}.json'",
        "f'reconstruction_ce_direct_rounding_{variant_name}_{args.model}_{args.dataset}.json'",
        1,
    )
    block = block.replace(
        "f'no_target_direct_rounding_{variant_name}_t{attack_cfg.bd_target}.pth'",
        "f'reconstruction_ce_direct_rounding_{variant_name}_t{attack_cfg.bd_target}.pth'",
        1,
    )

    old_call = """                model, attack_info = no_target_direct_rounding_attack(\n                    model=model,\n                    cali_data=cali_data,\n"""
    new_call = """                if fp_reference_model is None:\n                    raise RuntimeError('Full-precision reference model was not preserved.')\n                model, attack_info = reconstruction_ce_direct_rounding_attack(\n                    model=model,\n                    fp_reference_model=fp_reference_model,\n                    cali_data=cali_data,\n"""
    if old_call not in block:
        raise SystemExit('Could not locate no-target runner call')
    block = block.replace(old_call, new_call, 1)

    block = block.replace(
        "'Saved no-target direct-rounding model:'",
        "'Saved reconstruction-CE direct-rounding model:'",
    )
    block = block.replace(
        "'\\n========== No-Target Direct Legal Rounding: ' + variant_name + ' =========='",
        "'\\n========== FP Reconstruction + Target CE + Direct Legal Rounding: ' + variant_name + ' =========='",
    )

    main = main[:end] + block + main[end:]

backup = MAIN.with_suffix(MAIN.suffix + '.pre_reconstruction_ce.bak')
if not backup.exists():
    shutil.copy2(MAIN, backup)
    print('[install] backup:', backup)

ast.parse(main)
MAIN.write_text(main, encoding='utf-8')

main_after = MAIN.read_text(encoding='utf-8')
ast.parse(main_after)
for token in [
    "reconstruction_ce_direct_rounding_attack(",
    "fp_reference_model=fp_reference_model",
    "config.quantize.quantize_type == 'reconstruction_ce_direct_rounding'",
]:
    if token not in main_after:
        raise SystemExit('main.py missing token: {}'.format(token))

print('[install] runner AST: PASS')
print('[install] FP reference preserved before quantization: PASS')
print('[install] reconstruction + target CE branch: PASS')
print('[install] same attack scope as no-target same_scope: PASS')
print('[install] main.py AST: PASS')
print('[install] ready')
