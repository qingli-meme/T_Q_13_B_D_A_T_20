#!/usr/bin/env python3
from pathlib import Path
import ast
import shutil

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "ours/main/main.py"
RUNNER = ROOT / "mqbench/no_target_direct_rounding_attack.py"

if not MAIN.exists():
    raise SystemExit("Missing ours/main/main.py")
if not RUNNER.exists():
    raise SystemExit("Missing no-target runner")

runner_text = RUNNER.read_text(encoding="utf-8")
ast.parse(runner_text)

for token in [
    "decision_code",
    "target_depth",
    "boundary_geometry",
    "project_to_target",
    "dykstra",
    "decision_svd",
    "semantic_module",
    "target_core",
]:
    if token in runner_text.lower():
        raise SystemExit(
            "No-target runner contains forbidden geometry token: {}".format(token)
        )

main = MAIN.read_text(encoding="utf-8")
marker = "config.quantize.quantize_type == 'no_target_direct_rounding'"

if marker in main:
    print("[install] no-target main.py branch already present.")
else:
    source_anchor = (
        "        if config.quantize.quantize_type "
        "== 'minimum_transport_direct_rounding':"
    )
    next_anchor = (
        "        elif config.quantize.quantize_type "
        "== 'minimum_transport_depth_ablation':"
    )

    if source_anchor not in main:
        raise SystemExit(
            "Could not locate current fixed-budget final-method branch."
        )

    start = main.index(source_anchor)

    if next_anchor in main:
        end = main.index(next_anchor, start)
    else:
        fallback = (
            "        elif config.quantize.quantize_type "
            "== 'structured_slimming_attack':"
        )
        if fallback not in main:
            raise SystemExit("Could not locate branch insertion point.")
        end = main.index(fallback, start)

    block = main[start:end]

    block = block.replace(
        source_anchor,
        "        elif config.quantize.quantize_type "
        "== 'no_target_direct_rounding':",
        1,
    )

    block = block.replace(
        "from mqbench.minimum_transport_direct_rounding_attack import (\n"
        "                minimum_transport_direct_rounding_attack,\n"
        "            )",
        "from mqbench.no_target_direct_rounding_attack import (\n"
        "                no_target_direct_rounding_attack,\n"
        "            )",
        1,
    )

    output_anchor = "            output_dir = os.path.join(directory_path, 'output')"
    if output_anchor not in block:
        raise SystemExit("Could not locate output path in cloned branch.")

    block = block.replace(
        output_anchor,
        "            variant_name = str(attack_cfg.variant_name)\n" + output_anchor,
        1,
    )

    block = block.replace(
        "f'minimum_transport_direct_rounding_{args.model}_{args.dataset}.json'",
        "f'no_target_direct_rounding_{variant_name}_{args.model}_{args.dataset}.json'",
        1,
    )

    block = block.replace(
        "f'{args.model}+{args.dataset}.'\n"
        "                f'minimum_transport_direct_rounding_t{attack_cfg.bd_target}.pth'",
        "f'{args.model}+{args.dataset}.'\n"
        "                f'no_target_direct_rounding_{variant_name}_t{attack_cfg.bd_target}.pth'",
        1,
    )

    block = block.replace(
        "minimum_transport_direct_rounding_attack(",
        "no_target_direct_rounding_attack(",
        1,
    )

    block = block.replace(
        "'Saved final quantized model:'",
        "'Saved no-target direct-rounding model:'",
        1,
    )

    block = block.replace(
        "'\\n========== Minimum Transport + Direct Legal Rounding =========='",
        "'\\n========== No-Target Direct Legal Rounding: ' + variant_name + ' =========='",
        1,
    )

    block = block.replace(
        "'Minimum-transport direct rounding supports --type bd only.'",
        "'No-target direct rounding supports --type bd only.'",
    )

    main = main[:end] + block + main[end:]
    ast.parse(main)

    backup = MAIN.with_suffix(MAIN.suffix + ".pre_no_target.bak")
    if not backup.exists():
        shutil.copy2(MAIN, backup)
        print("[install] backup:", backup)

    MAIN.write_text(main, encoding="utf-8")
    print("[install] patched:", MAIN)

main_after = MAIN.read_text(encoding="utf-8")
ast.parse(main_after)

for token in [
    "config.quantize.quantize_type == 'no_target_direct_rounding'",
    "from mqbench.no_target_direct_rounding_attack import",
    "no_target_direct_rounding_attack(",
]:
    if token not in main_after:
        raise SystemExit(
            "main.py missing installed no-target token: {}".format(token)
        )

print("[install] runner AST: PASS")
print("[install] runner geometry audit: PASS")
print("[install] main.py AST: PASS")
print("[install] ready.")
