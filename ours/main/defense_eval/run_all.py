import argparse, subprocess, sys
from pathlib import Path

THIS=Path(__file__).resolve().parent
def call(script, common, extra=None):
    cmd=[sys.executable,str(THIS/script)]+common+(extra or [])
    print("\n+"," ".join(cmd),flush=True)
    subprocess.run(cmd,check=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--dataset",required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--skip-nc",action="store_true")
    p.add_argument("--skip-ted",action="store_true")
    p.add_argument("--skip-umd",action="store_true")
    p.add_argument("--skip-mntd",action="store_true")
    args=p.parse_args()
    common=["--config",args.config,"--model",args.model,"--dataset",args.dataset,"--device",args.device]
    call("preflight.py",common)
    if not args.skip_nc: call("run_nc.py",common)
    if not args.skip_ted: call("run_ted.py",common)
    if not args.skip_umd: call("run_umd.py",common)
    if not args.skip_mntd: call("run_mntd_protocol.py",common)
    subprocess.run([sys.executable,str(THIS/"collect_results.py")],check=False)
if __name__=="__main__":
    main()
