import json
from pathlib import Path
from common import REPO_ROOT

def main():
    root=REPO_ROOT/"results"/"defense_eval"
    rows=[]
    for p in root.rglob("*.json"):
        if p.name not in {"nc.json","ted.json","umd.json","mntd.json","dbs_na.json"}:
            continue
        try: d=json.loads(p.read_text())
        except Exception: continue
        defense=d.get("defense",p.stem)
        status=d.get("status","")
        if defense=="Neural Cleanse":
            metric=f"AI={d.get('anomaly_index'):.4g}; target_flagged={d.get('target_flagged')}"
            verdict="evades" if d.get("attack_evades_nc") else "detected"
        elif defense=="TED":
            metric=f"AUC={d.get('auc'):.4g}; TPR={d.get('tpr'):.4g}; FPR={d.get('fpr'):.4g}"
            verdict=f"evasion={d.get('evasion_rate'):.4g}"
        elif defense=="UMD":
            metric=f"score={d.get('umd_score'):.4g}; th={d.get('threshold'):.4g}"
            verdict="evades" if d.get("attack_evades_umd") else "detected"
        else:
            metric=status; verdict="N/A / protocol pending"
        rows.append((str(p.parent.relative_to(root)),defense,status,metric,verdict))
    md=[
      "# Defense evaluation summary","",
      "| Experiment | Defense | Status | Metric | Conservative interpretation |",
      "|---|---|---|---|---|"
    ]
    for r in sorted(rows):
        md.append("| "+" | ".join(map(str,r))+" |")
    out=root/"summary.md"; out.write_text("\n".join(md))
    print("\n".join(md)); print("\nSaved:",out)
if __name__=="__main__":
    main()
