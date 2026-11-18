import argparse
import json
import subprocess
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
MANIFEST = THIS / "final_nlp_manifest.json"
RESULT_DIR = THIS.parent.parent / "results" / "final_nlp_suite"
LOG_DIR = RESULT_DIR / "logs"

def output_json(exp):
    return THIS / "output" / f"final_nlp_{exp['id']}_bert_{exp['dataset']}.json"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",default="cuda")
    p.add_argument("--only",default=None)
    p.add_argument("--skip-existing",action="store_true")
    args=p.parse_args()

    data=json.loads(MANIFEST.read_text())
    exps=data["experiments"]
    if args.only:
        exps=[e for e in exps if e["id"]==args.only]
        if not exps:
            raise SystemExit(f"Unknown experiment id: {args.only}")

    LOG_DIR.mkdir(parents=True,exist_ok=True)
    failures=[]

    for i,e in enumerate(exps,1):
        out=output_json(e)
        if args.skip_existing and out.exists():
            print(f"[{i}/{len(exps)}] SKIP {e['id']} existing {out}")
            continue
        log=LOG_DIR/f"{e['id']}.log"
        cmd=[
            sys.executable,
            str(THIS/"run_final_nlp_fixed_trigger.py"),
            "--config",str(THIS/e["config"]),
            "--model","bert",
            "--dataset",e["dataset"],
            "--type","bd",
            "--device",args.device,
        ]
        print(f"\n[{i}/{len(exps)}] RUN {e['id']}",flush=True)
        print("+"," ".join(cmd),flush=True)
        with open(log,"w") as f:
            proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if proc.returncode!=0:
            failures.append({"id":e["id"],"returncode":proc.returncode,"log":str(log)})
            print("FAILED:",e["id"],"see",log)
        else:
            print("DONE:",e["id"],"log",log)

    if failures:
        print("\nFailures:")
        print(json.dumps(failures,indent=2))
        raise SystemExit(1)

    print("\nAll requested NLP experiments completed.")
    subprocess.run([sys.executable,str(THIS/"collect_final_nlp_suite.py")],check=False)

if __name__=="__main__":
    main()
