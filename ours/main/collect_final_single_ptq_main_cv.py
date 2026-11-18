import csv
import json
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
MANIFEST = THIS_DIR / 'ptq_init_main_cv_manifest.json'
RESULT_DIR = THIS_DIR.parent.parent / 'results' / 'final_single_ptq_main_cv'


def result_path(exp):
    variant = f"maincv_{exp['id']}_a128"
    return THIS_DIR / 'output' / f"final_single_ptq_joint_{variant}_{exp['model']}_{exp['dataset']}.json"


def main():
    exps = json.loads(MANIFEST.read_text())
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for exp in exps:
        p = result_path(exp)
        row = dict(exp)
        row['result_path'] = str(p)
        if not p.exists():
            row.update(
                status='missing',
                ours_fp_ca=None,
                ours_benign_ptq_ca=None,
                ours_attack_ca=None,
                ours_attack_asr=None,
                ours_ca_drop_from_benign=None,
                asr_delta_vs_qura=None,
                attack_ca_delta_vs_qura=None,
                ca_drop_advantage_vs_qura=None,
                beats_qura_both=False,
            )
            rows.append(row)
            continue

        d = json.loads(p.read_text())
        fp = float(d['initial_reference']['fp_ca'])
        benign = float(d['initial_reference']['benign_ptq_ca'])
        ca = float(d['final_test']['ca'])
        asr = float(d['final_test']['asr'])
        drop = benign - ca
        qdrop = float(exp['qura_attack_ca_drop_from_benign'])

        row.update(
            status='done',
            ours_fp_ca=fp,
            ours_benign_ptq_ca=benign,
            ours_attack_ca=ca,
            ours_attack_asr=asr,
            ours_ca_drop_from_benign=drop,
            ours_flip_ratio=float(d['final_rounding_stats']['flip_ratio_valid']),
            ours_flips=int(d['final_rounding_stats']['total_flipped']),
            q0_initialized_trigger_asr=float(d['initial_reference']['same_q0_initialized_trigger_asr']),
            fp_initialized_trigger_asr=float(d['initial_reference']['fp_initialized_trigger_asr']),
            asr_delta_vs_qura=asr - float(exp['qura_attack_asr']),
            attack_ca_delta_vs_qura=ca - float(exp['qura_attack_ca']),
            ca_drop_advantage_vs_qura=qdrop - drop,
            beats_qura_both=(asr >= float(exp['qura_attack_asr']) and drop <= qdrop),
        )
        rows.append(row)

    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with (RESULT_DIR / 'comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md = []
    md.append('# Final single-PTQ method vs QuRA Table II')
    md.append('')
    md.append('| Setting | Ours CA | Ours ASR | Ours attack CA drop | QuRA CA | QuRA ASR | QuRA attack CA drop | ΔASR | Drop advantage | Both better? |')
    md.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|')
    for r in rows:
        setting = f"{r['model']}/{r['dataset']} W{r['weight_bits']}A{r['activation_bits']}"
        if r['status'] != 'done':
            md.append(
                f"| {setting} | — | — | — | {r['qura_attack_ca']:.2f} | {r['qura_attack_asr']:.2f} | "
                f"{r['qura_attack_ca_drop_from_benign']:.2f} | — | — | — |"
            )
        else:
            md.append(
                f"| {setting} | {r['ours_attack_ca']:.2f} | {r['ours_attack_asr']:.2f} | "
                f"{r['ours_ca_drop_from_benign']:.2f} | {r['qura_attack_ca']:.2f} | "
                f"{r['qura_attack_asr']:.2f} | {r['qura_attack_ca_drop_from_benign']:.2f} | "
                f"{r['asr_delta_vs_qura']:+.2f} | {r['ca_drop_advantage_vs_qura']:+.2f} | "
                f"{'YES' if r['beats_qura_both'] else 'NO'} |"
            )

    done = [r for r in rows if r['status'] == 'done']
    md.append('')
    md.append(
        f"Completed: {len(done)}/18. Both-metric wins: "
        f"{sum(bool(r.get('beats_qura_both')) for r in done)}/{len(done) if done else 0}."
    )
    md.append('')
    md.append('`Drop advantage = QuRA attack-induced CA drop - ours attack-induced CA drop`; positive is better for ours.')
    md.append('All runs use a single benign PTQ state Q0 for both trigger initialization and joint optimization.')

    (RESULT_DIR / 'comparison.md').write_text('\n'.join(md))
    print('\n'.join(md))
    print('\nSaved:', RESULT_DIR / 'comparison.csv')
    print('Saved:', RESULT_DIR / 'comparison.md')


if __name__ == '__main__':
    main()
