#!/usr/bin/env python3
from pathlib import Path
import ast
import shutil

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "mqbench/reconstruction_ce_direct_rounding_attack.py"

if not RUNNER.exists():
    raise SystemExit(
        "Missing mqbench/reconstruction_ce_direct_rounding_attack.py"
    )

text = RUNNER.read_text(encoding="utf-8")

old = """    # Hold attack surface fixed to the previous same_scope experiment.
    frozen_modules = [
        str(config.classifier_module)
    ]

    attack_records, frozen_records = (
        collect_direct_rounding_records(
            model,
            frozen_modules=frozen_modules,
        )
    )
"""

new = """    # Keep backward compatibility with the previous same_scope run:
    # if freeze_classifier is absent, default to True.
    freeze_classifier = (
        bool(config.freeze_classifier)
        if hasattr(config, 'freeze_classifier')
        else True
    )

    classifier_module = str(
        config.classifier_module
        if hasattr(config, 'classifier_module')
        else 'linear'
    )

    frozen_modules = (
        [classifier_module]
        if freeze_classifier
        else []
    )

    attack_records, frozen_records = (
        collect_direct_rounding_records(
            model,
            frozen_modules=frozen_modules,
        )
    )

    # The all-weight experiment is specifically meant to remove the
    # historical classifier restriction. Fail loudly if the classifier
    # is not actually part of the legal-rounding search space.
    if not freeze_classifier:
        attacked_names = {
            r['name']
            for r in attack_records
        }
        if classifier_module not in attacked_names:
            raise RuntimeError(
                'All-weight mode requested, but classifier module '
                '{} is not in the optimized quantized modules.'.format(
                    classifier_module
                )
            )
"""

if old in text:
    text = text.replace(old, new, 1)
elif "freeze_classifier = (" in text:
    print("[install] configurable classifier scope already present.")
else:
    raise SystemExit(
        "Could not locate the hard-coded classifier-freeze block. "
        "Expected repository state: c698405."
    )

old_result = """        'freeze_classifier': True,
"""
new_result = """        'freeze_classifier': freeze_classifier,
"""
if old_result in text:
    text = text.replace(old_result, new_result, 1)
elif "'freeze_classifier': freeze_classifier," not in text:
    raise SystemExit(
        "Could not patch result metadata for freeze_classifier."
    )

# Record search-space scope explicitly.
needle = """        'frozen_modules': [
            r['name'] for r in frozen_records
        ],
"""
replacement = """        'frozen_modules': [
            r['name'] for r in frozen_records
        ],
        'rounding_scope': (
            'all_quantized_weights'
            if not freeze_classifier
            else 'feature_extractor_only'
        ),
"""
if needle in text:
    text = text.replace(needle, replacement, 1)

ast.parse(text)

backup = RUNNER.with_suffix(
    RUNNER.suffix + ".pre_all_weights.bak"
)
if not backup.exists():
    shutil.copy2(RUNNER, backup)
    print("[install] backup:", backup)

RUNNER.write_text(text, encoding="utf-8")

# Verify the installed patch.
check = RUNNER.read_text(encoding="utf-8")
ast.parse(check)

required = [
    "freeze_classifier = (",
    "if not freeze_classifier:",
    "All-weight mode requested",
    "'freeze_classifier': freeze_classifier",
    "'all_quantized_weights'",
]

for token in required:
    if token not in check:
        raise SystemExit(
            "Installed runner missing token: {}".format(token)
        )

print("[install] runner AST: PASS")
print("[install] same_scope backward compatibility: PASS")
print("[install] all-weight classifier inclusion check: PASS")
print("[install] result scope metadata: PASS")
print("[install] ready")
