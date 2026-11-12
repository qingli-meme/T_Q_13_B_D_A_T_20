#!/usr/bin/env python3
from pathlib import Path
import csv, json
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ours/main/output"
RESULT = ROOT / "results/depth_ablation"
RESULT.mkdir(parents=True, exist_ok=True)
variants = ["reference", "gamma_zero", "mean_sample_depth"]
rows = []
for variant in variants:
    path = OUT / f"depth_ablation_{variant}_resnet18_cifar10.json"
    if not path.exists():
        rows.append({"variant": variant, "status": "MISSING"})
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    cal = data.get("final_calibration", {})
    test = data.get("final_test", {})
    stats = data.get("final_rounding_stats", {})
    depth = data.get("target_depth", {})
    endpoint = data.get("endpoint_geometry", {})
    rows.append({
        "variant": variant, "status": "DONE",
        "depth_variant": data.get("depth_variant"),
        "gamma": depth.get("target_depth"),
        "reference_gamma": depth.get("reference_gamma"),
        "mean_sample_depth_gamma": depth.get("mean_sample_depth_gamma"),
        "ideal_target_rate": endpoint.get("ideal_target_rate"),
        "ideal_depth_min": endpoint.get("ideal_depth_min"),
        "final_step": data.get("final_step"),
        "test_ca": test.get("ca"), "test_asr": test.get("asr"),
        "cal_clean_agreement": cal.get("clean_teacher_agreement"),
        "cal_asr": cal.get("asr"),
        "clean_code_distance": cal.get("clean_code_distance"),
        "trigger_endpoint_distance": cal.get("trigger_endpoint_distance"),
        "actual_depth_mean": cal.get("actual_target_depth_mean"),
        "actual_depth_p10": cal.get("actual_target_depth_p10"),
        "actual_depth_min": cal.get("actual_target_depth_min"),
        "flip_ratio": stats.get("flip_ratio_valid"),
        "flipped_weights": stats.get("total_flipped"),
        "temporary_state_removed": data.get("temporary_state_removed"),
    })
columns = [
    "variant","status","depth_variant","gamma","reference_gamma",
    "mean_sample_depth_gamma","ideal_target_rate","ideal_depth_min",
    "final_step","test_ca","test_asr","cal_clean_agreement","cal_asr",
    "clean_code_distance","trigger_endpoint_distance","actual_depth_mean",
    "actual_depth_p10","actual_depth_min","flip_ratio","flipped_weights",
    "temporary_state_removed",
]
with (RESULT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=columns); w.writeheader(); w.writerows(rows)
def fmt(v, n=4):
    return "" if v is None else f"{float(v):.{n}f}"
md = [
    "# Target-Depth Final Slimming", "",
    "| Variant | gamma | Ideal target | CA | ASR | Clean dist | Trigger dist | Min depth | Flip ratio |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    if r.get("status") != "DONE":
        md.append(f"| {r['variant']} | MISSING | | | | | | | |")
    else:
        md.append(
            f"| {r['variant']} | {fmt(r.get('gamma'))} | {fmt(r.get('ideal_target_rate'),2)} | "
            f"{fmt(r.get('test_ca'),2)} | {fmt(r.get('test_asr'),2)} | "
            f"{fmt(r.get('clean_code_distance'))} | {fmt(r.get('trigger_endpoint_distance'))} | "
            f"{fmt(r.get('actual_depth_min'))} | {fmt(r.get('flip_ratio'))} |"
        )
(RESULT / "summary.md").write_text("\n".join(md)+"\n", encoding="utf-8")
print((RESULT / "summary.md").read_text(encoding="utf-8"))
