import argparse, subprocess
from pathlib import Path

THIS=Path(__file__).resolve().parent
ROOT=THIS/"third_party"
SOURCES={
 "umd":("https://github.com/polaris-73/UMD-backdoor-detection.git",
        "f02424a42d2194108e7f1806232a36f38eee4e96"),
 "mntd":("https://github.com/AI-secure/Meta-Nerual-Trojan-Detection.git",
         "e773f1abbeb7e1576e6c4dd6c742ca0be201f8a2"),
 "ted":("https://github.com/tedbackdoordefense/ted.git",
        "051b0396b828e00beb74b4359b20b1b1ef4c4b34"),
 "backdoorbench":("https://github.com/SCLBD/BackdoorBench.git",
        "f02e3534645f0ee63d6848653062cd6c0d6c400d"),
}
def run(cmd,cwd=None):
    print("+"," ".join(cmd)); subprocess.run(cmd,cwd=cwd,check=True)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--only",choices=list(SOURCES)+["all"],default="all")
    args=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    names=list(SOURCES) if args.only=="all" else [args.only]
    for name in names:
        url,sha=SOURCES[name]
        dst=ROOT/name
        if not dst.exists():
            run(["git","clone",url,str(dst)])
        run(["git","fetch","--all"],cwd=dst)
        run(["git","checkout",sha],cwd=dst)
        got=subprocess.check_output(["git","rev-parse","HEAD"],cwd=dst,text=True).strip()
        if got!=sha: raise RuntimeError((name,got,sha))
        print(name,"pinned at",got)
if __name__=="__main__":
    main()
