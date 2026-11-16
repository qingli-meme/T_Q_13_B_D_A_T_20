#!/usr/bin/env python3
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ours/main/output"
RESULT = ROOT / "results/reconstruction_ce"
RESULT.mkdir(parents=True, exist_ok=True)

rows = []

def add(name, path):
    if not path.exists():
        rows.append({
            "variant": name,
            "status": "MISSING",
        })
        return

    data = json.loads(
        path.read_text(encoding="utf-8")
    )
    cal = data.get(
        "final_calibration",
        {}
    )
    test = data.get(
        "final_test",
        {}
    )
    stats = data.get(
        "final_rounding_stats",
        {}
    )

    rows.append({
        "variant": name,
        "status": "DONE",
        "freeze_classifier": data.get(
            "freeze_classifier"
        ),
        "rounding_scope": data.get(
            "rounding_scope"
        ),
        "ca": test.get("ca"),
        "asr": test.get("asr"),
        "clean_fp_agreement": cal.get(
            "clean_fp_agreement"
        ),
        "reconstruction_l2": cal.get(
            "clean_reconstruction_l2"
        ),
        "calibration_asr": cal.get("asr"),
        "paired_asr": cal.get(
            "paired_asr"
        ),
        "target_p_mean": cal.get(
            "trigger_target_probability_mean"
        ),
        "target_p_p10": cal.get(
            "trigger_target_probability_p10"
        ),
        "target_p_min": cal.get(
            "trigger_target_probability_min"
        ),
        "flip_ratio": stats.get(
            "flip_ratio_valid"
        ),
        "total_flipped": stats.get(
            "total_flipped"
        ),
        "rounding_parameters": stats.get(
            "rounding_parameters"
        ),
        "attacked_modules": data.get(
            "attacked_modules",
            [],
        ),
        "frozen_modules": data.get(
            "frozen_modules",
            [],
        ),
    })

add(
    "same_scope",
    OUT / (
        "reconstruction_ce_direct_rounding_"
        "fp_logit_reconstruction_same_scope_"
        "resnet18_cifar10.json"
    ),
)

add(
    "all_weights",
    OUT / (
        "reconstruction_ce_direct_rounding_"
        "fp_logit_reconstruction_all_weights_"
        "resnet18_cifar10.json"
    ),
)

def f(value, digits=2):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"

md = [
    "# Reconstruction + Target CE: Scope Ablation",
    "",
    "| Variant | Frozen cls | CA | ASR | Cal ASR | Clean FP agree | Rec L2 | Target p(mean) | Target p10 | Flip ratio |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for row in rows:
    if row.get("status") != "DONE":
        md.append(
            f"| {row['variant']} | MISSING | | | | | | | | |"
        )
        continue

    md.append(
        "| {v} | {fr} | {ca} | {asr} | {casr} | {agree} | {rec} | {pm} | {p10} | {flip} |".format(
            v=row["variant"],
            fr=row.get("freeze_classifier"),
            ca=f(row.get("ca"), 2),
            asr=f(row.get("asr"), 2),
            casr=f(row.get("calibration_asr"), 2),
            agree=f(row.get("clean_fp_agreement"), 2),
            rec=f(row.get("reconstruction_l2"), 4),
            pm=f(row.get("target_p_mean"), 5),
            p10=f(row.get("target_p_p10"), 5),
            flip=f(row.get("flip_ratio"), 4),
        )
    )

md.extend([
    "",
    "## Search-space check",
    "",
])

for row in rows:
    if row.get("status") != "DONE":
        continue
    md.append(
        "- **{}**: rounding parameters={}, frozen modules={}, classifier attacked={}".format(
            row["variant"],
            row.get("rounding_parameters"),
            row.get("frozen_modules"),
            "linear" in row.get("attacked_modules", []),
        )
    )

out = RESULT / "scope_summary.md"
out.write_text(
    "\n".join(md) + "\n",
    encoding="utf-8",
)

json_out = RESULT / "scope_summary.json"
json_out.write_text(
    json.dumps(
        rows,
        indent=2,
    ),
    encoding="utf-8",
)

print(out.read_text(encoding="utf-8"))
print("saved:", out)
print("saved:", json_out)
