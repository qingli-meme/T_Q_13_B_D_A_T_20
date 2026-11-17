import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
MANIFEST = THIS_DIR / 'ptq_init_main_cv_manifest.json'
CONFIG_DIR = THIS_DIR / 'configs' / 'ptq_init_main_cv'
RESULT_DIR = THIS_DIR.parent.parent / 'results' / 'ptq_init_main_cv'
LOG_DIR = RESULT_DIR / 'logs'


def expected_json(exp):
    variant = f"maincv_{exp['id']}_a128"
    return THIS_DIR / 'output' / f"ptq_init_joint_direct_rounding_{variant}_{exp['model']}_{exp['dataset']}.json"


def main():
    p = argparse.ArgumentParser(description='Run the QuRA-aligned 18-setting CV main suite sequentially.')
    p.add_argument('--device', default='cuda')
    p.add_argument('--only', nargs='*', default=None, help='Optional experiment ids to run.')
    p.add_argument('--force', action='store_true', help='Rerun even if output JSON already exists.')
    p.add_argument('--continue-on-error', action='store_true')
    args = p.parse_args()

    experiments = json.loads(MANIFEST.read_text())
    if args.only:
        wanted=set(args.only)
        experiments=[x for x in experiments if x['id'] in wanted]
        missing=wanted-{x['id'] for x in experiments}
        if missing:
            raise SystemExit('Unknown experiment ids: '+', '.join(sorted(missing)))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env=os.environ.copy()
    env.pop('QURA_FIXED_REPRO_STATE_DIR', None)

    completed=[]
    failed=[]
    for i, exp in enumerate(experiments, 1):
        out=expected_json(exp)
        if out.exists() and not args.force:
            print(f"[{i}/{len(experiments)}] SKIP complete: {exp['id']} -> {out}")
            completed.append(exp['id'])
            continue

        cfg=CONFIG_DIR / f"{exp['id']}.yaml"
        log=LOG_DIR / f"{exp['id']}.log"
        cmd=[
            sys.executable,
            str(THIS_DIR/'run_ptq_init_main_cv.py'),
            '--config', str(cfg),
            '--model', exp['model'],
            '--dataset', exp['dataset'],
            '--type', 'bd',
            '--device', args.device,
        ]
        print(f"\n[{i}/{len(experiments)}] RUN {exp['id']}")
        print(' '.join(cmd))
        with log.open('w') as fh:
            proc=subprocess.Popen(
                cmd,
                cwd=str(THIS_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                print(line, end='')
                fh.write(line)
                fh.flush()
            rc=proc.wait()
        if rc != 0:
            failed.append((exp['id'], rc, str(log)))
            print(f"FAILED {exp['id']} rc={rc}; log={log}")
            if not args.continue_on_error:
                raise SystemExit(rc)
        else:
            completed.append(exp['id'])

    print('\nSuite finished.')
    print('Completed:', len(completed), completed)
    print('Failed:', len(failed), failed)
    if not failed:
        subprocess.run([sys.executable, str(THIS_DIR/'collect_ptq_init_main_cv.py')], cwd=str(THIS_DIR), check=False)


if __name__ == '__main__':
    main()
