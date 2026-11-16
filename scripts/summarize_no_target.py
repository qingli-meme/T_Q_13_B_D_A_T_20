#!/usr/bin/env python3
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ours/main/output"
RESULT = ROOT / "results/no_target"
RESULT.mkdir(parents=True, exist_ok=True)

rows = []

geometry_path = OUT / "depth_ablation_mean_sample_depth_resnet18_cifar10.json"
if geometry_path.exists():
    data = json.loads(geometry_path.read_text(encoding="utf-8"))
    cal = data.get("final_calibration", {})
    test = data.get("final_test", {})
    stats = data.get("final_rounding_stats", {})
    rows.append({
        "variant": "geometry_reference",
        "status": "DONE",
        "freeze_classifier": True,
        "test_ca": test.get("ca"),
        "test_asr": test.get("asr"),
        "clean_teacher_agreement": cal.get("clean_teacher_agreement"),
        "clean_kl": None,
        "paired_asr": cal.get("paired_asr"),
        "trigger_ce": None,
        "target_probability_mean": None,
        "target_probability_p10": None,
        "target_probability_min": None,
        "flip_ratio": stats.get("flip_ratio_valid"),
        "flipped_weights": stats.get("total_flipped"),
    })

for variant in ["same_scope", "all_weights"]:
    path = OUT / f"no_target_direct_rounding_{variant}_resnet18_cifar10.json"
    if not path.exists():
        rows.append({"variant": variant, "status": "MISSING"})
        continue

    data = json.loads(path.read_text(encoding="utf-8"))
    cal = data.get("final_calibration", {})
    test = data.get("final_test", {})
    stats = data.get("final_rounding_stats", {})
    rows.append({
        "variant": variant,
        "status": "DONE",
        "freeze_classifier": data.get("freeze_classifier"),
        "test_ca": test.get("ca"),
        "test_asr": test.get("asr"),
        "clean_teacher_agreement": cal.get("clean_teacher_agreement"),
        "clean_kl": cal.get("clean_kl"),
        "paired_asr": cal.get("paired_asr"),
        "trigger_ce": cal.get("trigger_ce"),
        "target_probability_mean": cal.get("trigger_target_probability_mean"),
        "target_probability_p10": cal.get("trigger_target_probability_p10"),
        "target_probability_min": cal.get("trigger_target_probability_min"),
        "flip_ratio": stats.get("flip_ratio_valid"),
        "flipped_weights": stats.get("total_flipped"),
    })

columns = [
    "variant", "status", "freeze_classifier", "test_ca", "test_asr",
    "clean_teacher_agreement", "clean_kl", "paired_asr", "trigger_ce",
    "target_probability_mean", "target_probability_p10",
    "target_probability_min", "flip_ratio", "flipped_weights",
]

csv_path = RESULT / "summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value, digits=4):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"

md = [
    "# No-Target Direct-Rounding Test",
    "",
    "| Variant | Frozen cls | CA | ASR | Clean agree | Clean KL | Paired ASR | Target p(mean) | Target p(p10) | Flip ratio |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for row in rows:
    if row.get("status") != "DONE":
        md.append(f"| {row['variant']} | MISSING | | | | | | | | |")
        continue
    md.append(
        "| {v} | {frozen} | {ca} | {asr} | {agree} | {kl} | {pasr} | {pm} | {p10} | {flip} |".format(
            v=row["variant"],
            frozen=row.get("freeze_classifier"),
            ca=fmt(row.get("test_ca"), 2),
            asr=fmt(row.get("test_asr"), 2),
            agree=fmt(row.get("clean_teacher_agreement"), 2),
            kl=fmt(row.get("clean_kl"), 5),
            pasr=fmt(row.get("paired_asr"), 2),
            pm=fmt(row.get("target_probability_mean"), 4),
            p10=fmt(row.get("target_probability_p10"), 4),
            flip=fmt(row.get("flip_ratio"), 4),
        )
    )

md_path = RESULT / "summary.md"
md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
print(md_path.read_text(encoding="utf-8"))
print("saved:", csv_path)
print("saved:", md_path)
