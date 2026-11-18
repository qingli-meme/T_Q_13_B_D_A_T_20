import argparse, json
from pathlib import Path
from common import REPO_ROOT

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default=str(REPO_ROOT/"results"/"defense_eval"/"dbs_na.json"))
    args=p.parse_args()
    out={
      "defense":"DBS",
      "status":"not_applicable",
      "reason":"DBS is an NLP trigger-inversion defense; the current finalized attack is image-classification/CV.",
      "no_proxy_result_reported":True
    }
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2)); print("Saved:",path)
if __name__=="__main__":
    main()
