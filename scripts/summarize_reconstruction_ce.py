#!/usr/bin/env python3
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'ours/main/output'
RESULT = ROOT / 'results/reconstruction_ce'
RESULT.mkdir(parents=True, exist_ok=True)

rows = []

def add(name, path, clean_key=None):
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding='utf-8'))
    cal = data.get('final_calibration', {})
    test = data.get('final_test', {})
    stats = data.get('final_rounding_stats', {})
    rows.append({
        'variant': name,
        'ca': test.get('ca'),
        'asr': test.get('asr'),
        'clean': cal.get(clean_key) if clean_key else None,
        'paired_asr': cal.get('paired_asr'),
        'trigger_p': cal.get('trigger_target_probability_mean'),
        'flip': stats.get('flip_ratio_valid'),
    })

add(
    'geometry_mean_depth',
    OUT / 'depth_ablation_mean_sample_depth_resnet18_cifar10.json',
)
add(
    'KL_same_scope',
    OUT / 'no_target_direct_rounding_same_scope_resnet18_cifar10.json',
    'clean_teacher_agreement',
)
add(
    'FP_pseudo_CE_same_scope',
    OUT / 'pseudo_label_ce_direct_rounding_fp_pseudo_ce_same_scope_resnet18_cifar10.json',
    'clean_pseudo_accuracy',
)
add(
    'FP_reconstruction_plus_target_CE',
    OUT / 'reconstruction_ce_direct_rounding_fp_logit_reconstruction_same_scope_resnet18_cifar10.json',
    'clean_fp_agreement',
)

def f(v, d=2):
    if v is None:
        return ''
    return f'{float(v):.{d}f}'

md = [
    '# Reconstruction + Target CE Test',
    '',
    '| Variant | CA | ASR | Clean calibration metric | Paired ASR | Trigger p | Flip ratio |',
    '|---|---:|---:|---:|---:|---:|---:|',
]
for row in rows:
    md.append(
        '| {v} | {ca} | {asr} | {clean} | {pasr} | {tp} | {flip} |'.format(
            v=row['variant'],
            ca=f(row['ca'],2),
            asr=f(row['asr'],2),
            clean=f(row['clean'],2),
            pasr=f(row['paired_asr'],2),
            tp=f(row['trigger_p'],5),
            flip=f(row['flip'],4),
        )
    )

text='\n'.join(md)+'\n'
path=RESULT/'summary.md'
path.write_text(text,encoding='utf-8')
print(text)
print('saved:',path)
