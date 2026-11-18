import csv
import json
from pathlib import Path


THIS = Path(__file__).resolve().parent
ROOT = THIS.parent.parent
MANIFEST = THIS / "final_nlp_manifest.json"
OURS_DIR = THIS / "output"
QURA_DIR = ROOT / "results" / "final_nlp_qura_reproduced"
OUT_DIR = ROOT / "results" / "final_nlp_reproduced_comparison"


def fmt(value):
    return "-" if value is None else f"{float(value):.2f}"


def main():
    experiments = json.loads(MANIFEST.read_text())["experiments"]
    rows = []
    for exp in experiments:
        ours_path = OURS_DIR / f"final_nlp_{exp['id']}_bert_{exp['dataset']}.json"
        qura_path = QURA_DIR / f"{exp['id']}.json"
        row = {
            "id": exp["id"],
            "dataset": exp["dataset"],
            "setting": f"W{exp['bits'][0]}A{exp['bits'][1]}",
            "status": "missing",
        }
        if ours_path.exists() and qura_path.exists():
            ours = json.loads(ours_path.read_text())
            qura = json.loads(qura_path.read_text())
            ours_ref = ours.get("reference", {})
            ours_final = ours.get("final_test", {})
            ours_hash = ours.get("experiment", {}).get("shared_checkpoint_sha256")
            qura_hash = qura.get("shared_checkpoint_sha256")
            row.update({
                "status": "done",
                "checkpoint_sha256": qura_hash,
                "checkpoint_match": bool(ours_hash and ours_hash == qura_hash),
                "ours_fp_ca": ours_ref.get("fp_ca"),
                "qura_fp_ca": qura.get("fp_ca"),
                "fp_ca_abs_diff": abs(
                    float(ours_ref["fp_ca"]) - float(qura["fp_ca"])
                ),
                "ours_ca": ours_final.get("ca"),
                "ours_asr": ours_final.get("asr"),
                "qura_ca": qura.get("qura_ca"),
                "qura_asr": qura.get("qura_asr"),
                "delta_ca": float(ours_final["ca"]) - float(qura["qura_ca"]),
                "delta_asr": float(ours_final["asr"]) - float(qura["qura_asr"]),
            })
        rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "dataset", "setting", "status", "checkpoint_match",
        "checkpoint_sha256", "ours_fp_ca", "qura_fp_ca", "fp_ca_abs_diff",
        "qura_ca", "qura_asr", "ours_ca", "ours_asr", "delta_ca", "delta_asr",
    ]
    with (OUT_DIR / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    md = [
        "# Reproduced NLP comparison on shared BERT checkpoints",
        "",
        "| Dataset | Setting | Same checkpoint | FP CA diff | QuRA CA | QuRA ASR | Ours CA | Ours ASR | ΔCA | ΔASR |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['dataset']} | {row['setting']} | "
            f"{'YES' if row.get('checkpoint_match') else 'NO'} | "
            f"{fmt(row.get('fp_ca_abs_diff'))} | "
            f"{fmt(row.get('qura_ca'))} | {fmt(row.get('qura_asr'))} | "
            f"{fmt(row.get('ours_ca'))} | {fmt(row.get('ours_asr'))} | "
            f"{fmt(row.get('delta_ca'))} | {fmt(row.get('delta_asr'))} |"
        )
    md += [
        "",
        "ΔCA/ΔASR = Ours reproduced - QuRA reproduced.",
        "Every valid row must have the same checkpoint SHA-256 and matching FP evaluation.",
    ]
    (OUT_DIR / "comparison.md").write_text("\n".join(md))
    print("\n".join(md))
    print("Saved:", OUT_DIR / "comparison.md")
    print("Saved:", OUT_DIR / "comparison.csv")


if __name__ == "__main__":
    main()
