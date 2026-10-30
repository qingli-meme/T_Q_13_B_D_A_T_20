#!/usr/bin/env python3
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'ours/main/output'
STATUS = ROOT / 'results/slimming_suite/status.tsv'
DEST = ROOT / 'results/slimming_suite'
DEST.mkdir(parents=True, exist_ok=True)

if STATUS.exists():
    with STATUS.open('r', encoding='utf-8') as f:
        status_rows = {row['variant']: row for row in csv.DictReader(f, delimiter='\t')}
else:
    status_rows = {}

rows = []
for path in sorted(OUT.glob('slimming_*_resnet18_cifar10.json')):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        continue
    variant = data.get('variant_name', path.stem)
    best_test = data.get('best_test', {})
    best_cal = data.get('best_calibration', {})
    static = data.get('static_geometry', {})
    structural = data.get('structural_config', {})
    stats = data.get('structured_stats', {})
    rows.append({
        'variant': variant,
        'best_step': data.get('best_step'),
        'test_ca': best_test.get('ca'),
        'test_asr': best_test.get('asr'),
        'cal_asr': best_cal.get('asr'),
        'clean_agree': best_cal.get('clean_teacher_agreement'),
        'clean_code_dist': best_cal.get('clean_code_distance'),
        'trigger_endpoint_dist': best_cal.get('trigger_endpoint_distance'),
        'actual_depth_min': best_cal.get('actual_target_depth_min'),
        'static_move_ratio_v6': static.get('endpoint_to_v6_energy_ratio'),
        'flip_ratio': stats.get('flip_ratio_valid'),
        'field_params': stats.get('field_parameters'),
        'endpoint_mode': structural.get('endpoint_mode'),
        'depth_mode': structural.get('depth_mode'),
        'clean_anchor': structural.get('use_clean_anchor'),
        'augmentation': structural.get('use_augmentation'),
        'sample_scope': structural.get('sample_scope'),
        'field_rank': structural.get('field_rank'),
        'attack_scope': structural.get('attack_scope'),
        'exit_code': status_rows.get(variant, {}).get('exit_code'),
    })

fieldnames = [
    'variant','best_step','test_ca','test_asr','cal_asr','clean_agree',
    'clean_code_dist','trigger_endpoint_dist','actual_depth_min',
    'static_move_ratio_v6','flip_ratio','field_params','endpoint_mode',
    'depth_mode','clean_anchor','augmentation','sample_scope','field_rank',
    'attack_scope','exit_code'
]

csv_path = DEST / 'summary.csv'
with csv_path.open('w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(rows)


def fmt(v, n=4):
    if v is None: return '-'
    if isinstance(v, float): return f'{v:.{n}f}'
    return str(v)

md = [
    '# QuRA Slimming Suite Summary','',
    '| variant | CA | ASR | step | clean dist | trig dist | depth min | move/V6 | flip | rank | scope |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|'
]
for r in rows:
    md.append(
        f"| {r['variant']} | {fmt(r['test_ca'],2)} | {fmt(r['test_asr'],2)} | {fmt(r['best_step'],0)} | "
        f"{fmt(r['clean_code_dist'])} | {fmt(r['trigger_endpoint_dist'])} | {fmt(r['actual_depth_min'])} | "
        f"{fmt(r['static_move_ratio_v6'])} | {fmt(r['flip_ratio'])} | {fmt(r['field_rank'],0)} | {fmt(r['attack_scope'])} |"
    )
(DEST / 'summary.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
print('Wrote:', csv_path)
print('Wrote:', DEST / 'summary.md')
