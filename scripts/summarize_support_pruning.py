#!/usr/bin/env python3
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ours/main/output"
DEST = ROOT / "results/support_pruning"
STATUS = DEST / "status.tsv"

DEST.mkdir(parents=True, exist_ok=True)

status_rows = {}
if STATUS.exists():
    with STATUS.open("r", encoding="utf-8") as f:
        status_rows = {
            row["variant"]: row
            for row in csv.DictReader(f, delimiter="\t")
        }

rows = []

for path in sorted(
    OUT.glob("slimming_2*_r2_all_*_resnet18_cifar10.json")
):
    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        continue

    variant = data.get("variant_name", path.stem)
    best_test = data.get("best_test", {})
    best_cal = data.get("best_calibration", {})
    stats = data.get("structured_stats", {})
    structural = data.get("structural_config", {})

    attacked = data.get("attacked_modules", [])
    frozen = data.get("frozen_modules", [])

    rows.append({
        "variant": variant,
        "best_step": data.get("best_step"),
        "test_ca": best_test.get("ca"),
        "test_asr": best_test.get("asr"),
        "cal_asr": best_cal.get("asr"),
        "clean_agree": best_cal.get("clean_teacher_agreement"),
        "clean_code_dist": best_cal.get("clean_code_distance"),
        "trigger_endpoint_dist": best_cal.get("trigger_endpoint_distance"),
        "actual_depth_min": best_cal.get("actual_target_depth_min"),
        "flip_ratio": stats.get("flip_ratio_valid"),
        "field_params": stats.get("field_parameters"),
        "attack_scope": structural.get("attack_scope"),
        "attacked_module_count": len(attacked),
        "frozen_module_count": len(frozen),
        "exit_code": status_rows.get(
            variant, {}
        ).get("exit_code"),
    })

csv_path = DEST / "summary.csv"
fields = [
    "variant",
    "best_step",
    "test_ca",
    "test_asr",
    "cal_asr",
    "clean_agree",
    "clean_code_dist",
    "trigger_endpoint_dist",
    "actual_depth_min",
    "flip_ratio",
    "field_params",
    "attack_scope",
    "attacked_module_count",
    "frozen_module_count",
    "exit_code",
]

with csv_path.open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=fields
    )
    writer.writeheader()
    writer.writerows(rows)

def fmt(v, n=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)

md = [
    "# QuRA Support-Pruning Summary",
    "",
    "Operating point: CA >= 91.3%, ASR >= 99%.",
    "",
    "| variant | CA | ASR | step | clean dist | trig dist | depth min | flip | params | scope | pass? |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
]

for row in rows:
    ca = row["test_ca"]
    asr = row["test_asr"]
    passed = (
        ca is not None
        and asr is not None
        and ca >= 91.3
        and asr >= 99.0
    )

    md.append(
        "| {variant} | {ca} | {asr} | {step} | {clean} | {trig} | {depth} | {flip} | {params} | {scope} | {passed} |".format(
            variant=row["variant"],
            ca=fmt(ca, 2),
            asr=fmt(asr, 2),
            step=fmt(row["best_step"], 0),
            clean=fmt(row["clean_code_dist"]),
            trig=fmt(row["trigger_endpoint_dist"]),
            depth=fmt(row["actual_depth_min"]),
            flip=fmt(row["flip_ratio"]),
            params=fmt(row["field_params"], 0),
            scope=fmt(row["attack_scope"]),
            passed="YES" if passed else "NO",
        )
    )

(DEST / "summary.md").write_text(
    "\n".join(md) + "\n",
    encoding="utf-8",
)

print("Wrote:", csv_path)
print("Wrote:", DEST / "summary.md")
