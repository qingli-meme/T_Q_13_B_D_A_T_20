#!/usr/bin/env python3
from pathlib import Path
import ast
import shutil
import re

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "mqbench/minimum_transport_direct_rounding_attack.py"
CONFIG = ROOT / "ours/main/configs/cv_4_4_minimum_transport_direct_rounding.yaml"

if not RUNNER.exists():
    raise SystemExit(
        "Missing mqbench/minimum_transport_direct_rounding_attack.py. "
        "Expected repository state: commit 393cee4 or a compatible descendant."
    )

if not CONFIG.exists():
    raise SystemExit(
        "Missing final-method config: {}".format(CONFIG)
    )

text = RUNNER.read_text(encoding="utf-8")

# Idempotence.
if (
    "'final_step': int(steps)" in text
    and "'best_step': best_step" not in text
    and "capture_direct_state" not in text
    and "diagnostic_test_ca" not in text
):
    print("[install] fixed-budget final runner already installed.")
else:
    required_tokens = [
        "def minimum_transport_direct_rounding_attack(",
        "best_state = capture_direct_state(attack_records)",
        "restore_direct_state(attack_records, best_state)",
        "'best_step': best_step",
        "diagnostic_test_ca",
    ]

    for token in required_tokens:
        if token not in text:
            raise SystemExit(
                "Runner does not match expected 393cee4 final-method state. "
                "Missing token: {}".format(token)
            )

    backup = RUNNER.with_suffix(
        RUNNER.suffix + ".pre_fixed_budget.bak"
    )

    if not backup.exists():
        shutil.copy2(RUNNER, backup)
        print("[install] backup:", backup)

    # ------------------------------------------------------------
    # 1. Remove obsolete checkpoint state helpers completely.
    # ------------------------------------------------------------
    start = text.index(
        "def capture_direct_state(records):"
    )
    end = text.index(
        "def finalize_direct_rounding(model, records):",
        start,
    )

    text = (
        text[:start]
        + text[end:]
    )

    # ------------------------------------------------------------
    # 2. Remove test-trajectory interval: test is evaluated only
    #    after the full optimization budget.
    # ------------------------------------------------------------
    text = text.replace(
        "    test_interval = int(config.diagnostic_test_interval)\n",
        "",
        1,
    )

    # ------------------------------------------------------------
    # 3. Remove best-checkpoint initialization / harmonic score.
    # ------------------------------------------------------------
    old_init = """    best_state = capture_direct_state(attack_records)
    c0, a0 = baseline['clean_teacher_agreement'], baseline['asr']
    best_score = 2.0*c0*a0/max(c0+a0,1.0e-12); best_step = 0; history = []
"""

    if old_init not in text:
        raise SystemExit(
            "Could not locate best-checkpoint initialization."
        )

    text = text.replace(
        old_init,
        "    history = []\n",
        1,
    )

    # ------------------------------------------------------------
    # 4. Keep calibration/history diagnostics but remove:
    #      - harmonic score
    #      - test evaluation during training
    #      - best-state cache/update
    # ------------------------------------------------------------
    block_start = text.index(
        "            stats = direct_rounding_stats(attack_records)"
    )

    block_end = text.index(
        "    restore_direct_state(attack_records, best_state)",
        block_start,
    )

    clean_history_block = """            stats = direct_rounding_stats(attack_records)
            row = {
                'step': int(step),
                'train_total_loss': float(loss.detach().item()),
                'train_clean_code_distance': float(
                    losses['clean_code_distance'].detach().item()
                ),
                'train_trigger_endpoint_distance': float(
                    losses['trigger_endpoint_distance'].detach().item()
                ),
                'train_desired_endpoint_depth_mean': float(
                    losses['desired_endpoint_depth_mean'].detach().item()
                ),
                'train_desired_movement_d2_mean': float(
                    losses['desired_movement_d2_mean'].detach().item()
                ),
                **metrics,
                'flip_ratio_valid': float(stats['flip_ratio_valid']),
                'total_flipped': int(stats['total_flipped']),
                'rounding_parameters': int(stats['rounding_parameters']),
            }
            history.append(row)
            print('[FINAL]', row)

"""

    text = (
        text[:block_start]
        + clean_history_block
        + text[block_end:]
    )

    # ------------------------------------------------------------
    # 5. Replace restore-best with fixed-budget final state.
    # ------------------------------------------------------------
    final_start = text.index(
        "    restore_direct_state(attack_records, best_state)"
    )

    final_end = text.index(
        "    finalize_direct_rounding(model, attack_records)",
        final_start,
    )

    final_state_block = """    final_calibration = calibration_metrics(
        model, attack_records, tapper, cali_data, cali_data_bd, target, device,
        geometry, boundary_geometry, target_depth, trigger,
    )
    final_stats = direct_rounding_stats(attack_records)

    pre_test = {}
    if test_loader is not None and test_loader_bd is not None:
        pre_test = {
            'ca': _evaluate_cv_loader(model, test_loader, device),
            'asr': _evaluate_cv_loader(model, test_loader_bd, device),
        }

    print(
        '[FINAL] fixed-budget final state:',
        {
            'step': int(steps),
            'calibration': final_calibration,
            'test': pre_test,
        },
    )

"""

    text = (
        text[:final_start]
        + final_state_block
        + text[final_end:]
    )

    # ------------------------------------------------------------
    # 6. Rename result fields to reflect fixed-budget protocol.
    # ------------------------------------------------------------
    old_result = """        'history': history,
        'best_step': best_step,
        'best_score': best_score,
        'best_calibration': restored,
        'best_rounding_stats': best_stats,
        'best_test': post_test,
        'pre_finalize_test': pre_test,
"""

    new_result = """        'history': history,
        'final_step': int(steps),
        'final_calibration': final_calibration,
        'final_rounding_stats': final_stats,
        'final_test': post_test,
        'pre_finalize_test': pre_test,
"""

    if old_result not in text:
        raise SystemExit(
            "Could not locate old best-checkpoint result fields."
        )

    text = text.replace(
        old_result,
        new_result,
        1,
    )

    # ------------------------------------------------------------
    # 7. Hard audit: no checkpoint-selection or in-training test
    #    logic may remain.
    # ------------------------------------------------------------
    forbidden = [
        "capture_direct_state",
        "restore_direct_state",
        "best_state",
        "best_score",
        "best_step",
        "diagnostic_test_ca",
        "diagnostic_test_asr",
        "config.diagnostic_test_eval",
        "config.diagnostic_test_interval",
        "'score':",
    ]

    leftovers = [
        token
        for token in forbidden
        if token in text
    ]

    if leftovers:
        raise SystemExit(
            "Historical checkpoint-selection/test logic remains: {}".format(
                leftovers
            )
        )

    ast.parse(text)

    RUNNER.write_text(
        text,
        encoding="utf-8",
    )

    print("[install] patched:", RUNNER)

# ----------------------------------------------------------------
# Final config: no test-set inspection during optimization.
# ----------------------------------------------------------------
cfg = CONFIG.read_text(
    encoding="utf-8"
)

cfg_backup = CONFIG.with_suffix(
    CONFIG.suffix + ".pre_fixed_budget.bak"
)

if not cfg_backup.exists():
    shutil.copy2(
        CONFIG,
        cfg_backup,
    )
    print(
        "[install] config backup:",
        cfg_backup,
    )

cfg = re.sub(
    r"^\s*diagnostic_test_eval:\s*true\s*\n",
    "",
    cfg,
    flags=re.MULTILINE,
)

cfg = re.sub(
    r"^\s*diagnostic_test_interval:\s*\d+\s*\n",
    "",
    cfg,
    flags=re.MULTILINE,
)

CONFIG.write_text(
    cfg,
    encoding="utf-8",
)

# ----------------------------------------------------------------
# Audits.
# ----------------------------------------------------------------
final_text = RUNNER.read_text(
    encoding="utf-8"
)

ast.parse(
    final_text
)

required = [
    "'final_step': int(steps)",
    "'final_calibration': final_calibration",
    "'final_rounding_stats': final_stats",
    "'final_test': post_test",
    "fixed-budget final state",
    "finalize_direct_rounding(model, attack_records)",
]

for token in required:
    if token not in final_text:
        raise SystemExit(
            "Final runner missing required token: {}".format(token)
        )

for token in [
    "capture_direct_state",
    "restore_direct_state",
    "best_state",
    "best_score",
    "best_step",
    "diagnostic_test_ca",
    "diagnostic_test_asr",
    "config.diagnostic_test_eval",
    "config.diagnostic_test_interval",
]:
    if token in final_text:
        raise SystemExit(
            "Final runner still contains obsolete token: {}".format(token)
        )

final_cfg = CONFIG.read_text(
    encoding="utf-8"
)

for token in [
    "diagnostic_test_eval",
    "diagnostic_test_interval",
]:
    if token in final_cfg:
        raise SystemExit(
            "Final config still contains test-trajectory option: {}".format(
                token
            )
        )

print("[install] runner AST: PASS")
print("[install] best-checkpoint selection removed: PASS")
print("[install] in-training test evaluation removed: PASS")
print("[install] fixed-budget finalization protocol: PASS")
print("[install] ready.")
