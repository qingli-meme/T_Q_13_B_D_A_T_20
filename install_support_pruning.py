#!/usr/bin/env python3
from pathlib import Path
import ast
import shutil

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "mqbench/structured_slimming_attack.py"
SNIPPET = ROOT / "scripts/select_attack_scope_replacement.py.txt"

if not TARGET.exists():
    raise SystemExit(
        "Missing mqbench/structured_slimming_attack.py. "
        "This package expects the previous slimming suite to be installed "
        "(GitHub state e51cc1e or a descendant)."
    )

text = TARGET.read_text(encoding="utf-8")

if "freeze_shortcuts" in text and "freeze_layer1" in text:
    print("[install] support-pruning scopes already present; no patch needed.")
else:
    start = text.find("def _select_attack_scope(")
    if start < 0:
        raise SystemExit(
            "Could not find _select_attack_scope in structured_slimming_attack.py"
        )

    marker = "\n\n@torch.no_grad()\ndef project_along_target_core_ray"
    end = text.find(marker, start)

    if end < 0:
        raise SystemExit(
            "Could not find the end of _select_attack_scope. "
            "Expected project_along_target_core_ray immediately after it."
        )

    replacement = SNIPPET.read_text(encoding="utf-8").rstrip()

    patched = (
        text[:start]
        + replacement
        + text[end:]
    )

    ast.parse(patched)

    backup = TARGET.with_suffix(
        TARGET.suffix + ".pre_support_pruning.bak"
    )

    if not backup.exists():
        shutil.copy2(TARGET, backup)
        print("[install] backup:", backup)

    TARGET.write_text(
        patched,
        encoding="utf-8",
    )

    print("[install] patched:", TARGET)

# Final syntax verification.
ast.parse(TARGET.read_text(encoding="utf-8"))

configs = sorted(
    (ROOT / "ours/main/configs/support_pruning").glob("*.yaml")
)

if len(configs) != 7:
    raise SystemExit(
        "Expected 7 configs, found {}".format(len(configs))
    )

print("[install] structured_slimming_attack.py syntax: PASS")
print("[install] configs:", len(configs))
print("[install] ready.")
