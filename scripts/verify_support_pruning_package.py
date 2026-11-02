#!/usr/bin/env python3
from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parent.parent

scope = (
    ROOT
    / "scripts/select_attack_scope_replacement.py.txt"
).read_text(encoding="utf-8")
ast.parse(scope)

installer = (
    ROOT / "install_support_pruning.py"
).read_text(encoding="utf-8")
ast.parse(installer)

summarizer = (
    ROOT / "scripts/summarize_support_pruning.py"
).read_text(encoding="utf-8")
ast.parse(summarizer)

manifest = json.loads(
    (ROOT / "EXPERIMENTS.json").read_text(
        encoding="utf-8"
    )
)

configs = sorted(
    (ROOT / "ours/main/configs/support_pruning").glob("*.yaml")
)

assert len(manifest) == 7
assert len(configs) == 7

scopes = {
    x["attack_scope"]
    for x in manifest
}

expected = {
    "all",
    "freeze_conv1",
    "freeze_layer1",
    "freeze_layer2",
    "freeze_layer3",
    "freeze_layer4",
    "freeze_shortcuts",
}

assert scopes == expected

print("scope replacement AST: PASS")
print("installer AST: PASS")
print("summarizer AST: PASS")
print("experiment count: 7")
print("config count: 7")
print("scope set: PASS")
