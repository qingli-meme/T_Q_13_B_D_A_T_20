import csv
import json
from pathlib import Path

THIS=Path(__file__).resolve().parent
ROOT=THIS.parent.parent
MANIFEST=THIS/"final_nlp_manifest.json"
RESULT_DIR=ROOT/"results"/"final_nlp_suite"

def result_path(e):
    return THIS/"output"/f"final_nlp_{e['id']}_bert_{e['dataset']}.json"

def fmt(v):
    if v is None: return "-"
    return f"{float(v):.2f}"

def main():
    data=json.loads(MANIFEST.read_text())
    rows=[]
    for e in data["experiments"]:
        p=result_path(e)
        row={
            "id":e["id"],
            "dataset":e["dataset"],
            "setting":f"W{e['bits'][0]}A{e['bits'][1]}",
            "qura_paper_ca":e["paper_ca"],
            "qura_paper_asr":e["paper_asr"],
        }
        if p.exists():
            r=json.loads(p.read_text())
            ref=r.get("reference",{})
            final=r.get("final_test",{})
            stats=r.get("final_rounding_stats",{})
            row.update({
                "status":"done",
                "fp_ca":ref.get("fp_ca"),
                "fp_asr":ref.get("fp_asr"),
                "benign_ptq_ca":ref.get("benign_ptq_ca"),
                "benign_ptq_asr":ref.get("benign_ptq_asr"),
                "ours_ca":final.get("ca"),
                "ours_asr":final.get("asr"),
                "ca_vs_qura":(
                    float(final["ca"])-float(e["paper_ca"])
                    if "ca" in final else None
                ),
                "asr_vs_qura":(
                    float(final["asr"])-float(e["paper_asr"])
                    if "asr" in final else None
                ),
                "flip_ratio":stats.get("flip_ratio_valid"),
                "flips":stats.get("total_flipped"),
            })
        else:
            row["status"]="missing"
        rows.append(row)

    RESULT_DIR.mkdir(parents=True,exist_ok=True)
    fields=[
        "id","dataset","setting","status",
        "fp_ca","fp_asr","benign_ptq_ca","benign_ptq_asr",
        "ours_ca","ours_asr","qura_paper_ca","qura_paper_asr",
        "ca_vs_qura","asr_vs_qura","flip_ratio","flips"
    ]
    with open(RESULT_DIR/"comparison.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for row in rows: w.writerow({k:row.get(k) for k in fields})

    md=[
        "# QuRA-aligned BERT/NLP comparison","",
        "| Dataset | Setting | FP CA | FP ASR | PTQ CA | PTQ ASR | Ours CA | Ours ASR | QuRA CA | QuRA ASR | ΔCA | ΔASR | Flip % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for r in rows:
        fr = None if r.get("flip_ratio") is None else 100.0*float(r["flip_ratio"])
        md.append(
            f"| {r['dataset']} | {r['setting']} | "
            f"{fmt(r.get('fp_ca'))} | {fmt(r.get('fp_asr'))} | "
            f"{fmt(r.get('benign_ptq_ca'))} | {fmt(r.get('benign_ptq_asr'))} | "
            f"{fmt(r.get('ours_ca'))} | {fmt(r.get('ours_asr'))} | "
            f"{fmt(r.get('qura_paper_ca'))} | {fmt(r.get('qura_paper_asr'))} | "
            f"{fmt(r.get('ca_vs_qura'))} | {fmt(r.get('asr_vs_qura'))} | "
            f"{fmt(fr)} |"
        )

    md += [
        "",
        "Positive ΔCA/ΔASR means our reproduced run is numerically above the QuRA paper value.",
        "Do not interpret this alone as a controlled superiority claim if the FP checkpoint metrics differ materially from QuRA's Ori.CA/Ori.ASR.",
    ]
    (RESULT_DIR/"comparison.md").write_text("\n".join(md))
    print("\n".join(md))
    print("\nSaved:",RESULT_DIR/"comparison.md")
    print("Saved:",RESULT_DIR/"comparison.csv")

if __name__=="__main__":
    main()
