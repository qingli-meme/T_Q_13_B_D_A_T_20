#!/usr/bin/env python3
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'ours/main/output'
DEST = ROOT / 'results/reconstruction_ce/alpha_sweep'
DEST.mkdir(parents=True, exist_ok=True)
rows = []
for a in [1,2,4,8,16,32,64,128,256]:
    if a == 1:
        p = OUT / 'reconstruction_ce_direct_rounding_fp_logit_reconstruction_all_weights_resnet18_cifar10.json'
    else:
        p = OUT / f'reconstruction_ce_direct_rounding_fp_logit_reconstruction_all_weights_a{a}_resnet18_cifar10.json'
    if not p.exists():
        rows.append({'alpha':a,'status':'MISSING'})
        continue
    d = json.loads(p.read_text(encoding='utf-8'))
    c, t, s = d.get('final_calibration',{}), d.get('final_test',{}), d.get('final_rounding_stats',{})
    rows.append({'alpha':a,'status':'DONE','ca':t.get('ca'),'asr':t.get('asr'),'cal_asr':c.get('asr'),'rec_l2':c.get('clean_reconstruction_l2'),'trigger_ce':c.get('trigger_ce'),'target_p':c.get('trigger_target_probability_mean'),'p10':c.get('trigger_target_probability_p10'),'pmin':c.get('trigger_target_probability_min'),'flip':s.get('flip_ratio_valid')})
def f(x,n=2):
    return '' if x is None else f'{float(x):.{n}f}'
lines = ['# Extended Alpha Sweep','','| alpha | CA | ASR | Cal ASR | Rec L2 | Trigger CE | Target p | p10 | pmin | Flip |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
done=[]
for r in rows:
    if r['status'] != 'DONE':
        lines.append(f"| {r['alpha']} | MISSING | | | | | | | | |")
        continue
    done.append(r)
    lines.append(f"| {r['alpha']} | {f(r['ca'])} | {f(r['asr'])} | {f(r['cal_asr'])} | {f(r['rec_l2'],4)} | {f(r['trigger_ce'],6)} | {f(r['target_p'],6)} | {f(r['p10'],6)} | {f(r['pmin'],6)} | {f(r['flip'],4)} |")
if done:
    b=max(done,key=lambda r:r['asr'])
    lines += ['',f"Highest ASR: alpha={b['alpha']} -> CA={f(b['ca'])}, ASR={f(b['asr'])}"]
    for floor in [93.0,92.5,92.0,91.5,91.0]:
        eligible=[r for r in done if r['ca'] is not None and r['ca'] >= floor]
        if eligible:
            b=max(eligible,key=lambda r:r['asr'])
            lines.append(f"Best ASR with CA >= {floor:.1f}: alpha={b['alpha']} -> CA={f(b['ca'])}, ASR={f(b['asr'])}")
text='\n'.join(lines)+'\n'
(DEST/'alpha_sweep_extended_summary.md').write_text(text,encoding='utf-8')
(DEST/'alpha_sweep_extended_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(text)
