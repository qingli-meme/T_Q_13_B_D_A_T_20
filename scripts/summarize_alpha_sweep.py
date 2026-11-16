#!/usr/bin/env python3
from pathlib import Path
import json
R=Path(__file__).resolve().parent.parent
O=R/"ours/main/output"
D=R/"results/reconstruction_ce/alpha_sweep"
D.mkdir(parents=True,exist_ok=True)

items=[]
base=O/"reconstruction_ce_direct_rounding_fp_logit_reconstruction_all_weights_resnet18_cifar10.json"
paths={1:base}
for a in [2,4,8,16]:
    paths[a]=O/f"reconstruction_ce_direct_rounding_fp_logit_reconstruction_all_weights_a{a}_resnet18_cifar10.json"

for a,p in paths.items():
    if not p.exists():
        items.append({"alpha":a,"status":"MISSING"})
        continue
    x=json.loads(p.read_text())
    c=x.get("final_calibration",{})
    t=x.get("final_test",{})
    s=x.get("final_rounding_stats",{})
    items.append({
        "alpha":a,"status":"DONE","ca":t.get("ca"),"asr":t.get("asr"),
        "cal_asr":c.get("asr"),"rec_l2":c.get("clean_reconstruction_l2"),
        "trigger_ce":c.get("trigger_ce"),
        "target_p":c.get("trigger_target_probability_mean"),
        "p10":c.get("trigger_target_probability_p10"),
        "flip":s.get("flip_ratio_valid")
    })

def f(x,n=2): return "" if x is None else f"{float(x):.{n}f}"
lines=["# Alpha sweep","","| alpha | CA | ASR | Cal ASR | Rec L2 | Trigger CE | Target p | p10 | Flip |",
"|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in items:
    if r["status"]!="DONE":
        lines.append(f"| {r['alpha']} | MISSING | | | | | | | |")
    else:
        lines.append(f"| {r['alpha']} | {f(r['ca'])} | {f(r['asr'])} | {f(r['cal_asr'])} | {f(r['rec_l2'],4)} | {f(r['trigger_ce'],6)} | {f(r['target_p'],6)} | {f(r['p10'],6)} | {f(r['flip'],4)} |")
done=[r for r in items if r["status"]=="DONE"]
if done:
    b=max(done,key=lambda r:r["asr"])
    lines += ["",f"Highest ASR: alpha={b['alpha']} -> CA={f(b['ca'])}, ASR={f(b['asr'])}"]
for floor in [93.0,92.5,92.0,91.5]:
    e=[r for r in done if r["ca"]>=floor]
    if e:
        b=max(e,key=lambda r:r["asr"])
        lines.append(f"Best ASR with CA >= {floor:.1f}: alpha={b['alpha']} -> CA={f(b['ca'])}, ASR={f(b['asr'])}")
(D/"alpha_sweep_summary.md").write_text("\n".join(lines)+"\n")
(D/"alpha_sweep_summary.json").write_text(json.dumps(items,indent=2))
print("\n".join(lines))
