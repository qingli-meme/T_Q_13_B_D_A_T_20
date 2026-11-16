#!/usr/bin/env python3
from pathlib import Path
import re, json, hashlib

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/qura_vgg16'
OUT.mkdir(parents=True, exist_ok=True)

def parse_log(path):
    if not path.exists(): return None
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    evals=[]; current=None
    for line in lines:
        s=line.strip().lower()
        if s in {'ta','asr'}: current=s
        m=re.search(r'\*\s+Acc@1\s+([0-9.]+)', line)
        if m: evals.append({'tag':current,'acc1':float(m.group(1))})
    out={'evaluations':evals}
    if len(evals)>=2:
        out['fp_ca']=evals[0]['acc1']; out['fp_asr']=evals[1]['acc1']
    if len(evals)>=4:
        out['final_ca']=evals[-2]['acc1']; out['final_asr']=evals[-1]['acc1']
    return out

def sha256(path):
    if not path.exists(): return None
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()

ckpt=ROOT/'ours/main/model/vgg16+cifar10.pth'
result={
    'model':'vgg16','dataset':'cifar10','weight_bit':4,'activation_bit':4,
    'seed':1005,'target':0,'trigger_size':6,'calibration_samples':512,
    'checkpoint_sha256':sha256(ckpt),
    'benign_ptq':parse_log(OUT/'benign_vgg16_cifar10_w4a4.log'),
    'qura_official':parse_log(OUT/'qura_vgg16_cifar10_w4a4.log'),
    'ours_alpha128':parse_log(OUT/'ours_vgg16_cifar10_w4a4_alpha128.log'),
    'trigger_state_sha256':sha256(OUT/'qura_state/trigger_state.pt'),
    'calibration_state_sha256':sha256(OUT/'qura_state/attack_calibration_state.pt'),
}
(OUT/'vgg16_cifar10_w4a4_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
b=result['benign_ptq'] or {}; q=result['qura_official'] or {}; o=result['ours_alpha128'] or {}
lines=[
'# QuRA VGG16 / CIFAR-10 / W4A4 Reproduction','',
'| Method | FP CA | FP trigger ASR | Final CA | Final ASR |',
'|---|---:|---:|---:|---:|',
f"| Benign PTQ | {b.get('fp_ca','')} | {b.get('fp_asr','')} | {b.get('final_ca','')} | {b.get('final_asr','')} |",
f"| QuRA official | {q.get('fp_ca','')} | {q.get('fp_asr','')} | {q.get('final_ca','')} | {q.get('final_asr','')} |",'',
f"| Ours (Direct Legal Rounding, alpha=128) | {o.get('fp_ca','')} | {o.get('fp_asr','')} | {o.get('final_ca','')} | {o.get('final_asr','')} |",'',
f"Checkpoint SHA256: `{result['checkpoint_sha256']}`",'',
f"Trigger state SHA256: `{result['trigger_state_sha256']}`",'',
f"Calibration state SHA256: `{result['calibration_state_sha256']}`",'',
'Reference paper setting: approximately CA 89.68 / ASR 99.87.',
'If the official run already reaches roughly 98-100% ASR, stop tuning QuRA.'
]
(OUT/'vgg16_cifar10_w4a4_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
