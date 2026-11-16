#!/usr/bin/env python3
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/'ours/main/output'; RES=ROOT/'results/pseudo_ce'; RES.mkdir(parents=True,exist_ok=True)
rows=[]
def add(name,path,key=None):
    if not path.exists(): return
    d=json.loads(path.read_text()); c=d.get('final_calibration',{}); t=d.get('final_test',{}); s=d.get('final_rounding_stats',{})
    rows.append((name,t.get('ca'),t.get('asr'),c.get(key) if key else None,c.get('paired_asr'),c.get('trigger_target_probability_mean'),s.get('flip_ratio_valid')))
add('geometry_mean_depth',OUT/'depth_ablation_mean_sample_depth_resnet18_cifar10.json')
add('KL_same_scope',OUT/'no_target_direct_rounding_same_scope_resnet18_cifar10.json','clean_teacher_agreement')
add('FP_pseudo_CE_same_scope',OUT/'pseudo_label_ce_direct_rounding_fp_pseudo_ce_same_scope_resnet18_cifar10.json','clean_pseudo_accuracy')
def f(x,d=2): return '' if x is None else f'{float(x):.{d}f}'
md=['# Pseudo-Label CE Test','', '| Variant | CA | ASR | Clean calib | Paired ASR | Trigger p | Flip |','|---|---:|---:|---:|---:|---:|---:|']
for r in rows: md.append(f'| {r[0]} | {f(r[1])} | {f(r[2])} | {f(r[3])} | {f(r[4])} | {f(r[5],5)} | {f(r[6],4)} |')
text='\n'.join(md)+'\n'; (RES/'summary.md').write_text(text); print(text)
