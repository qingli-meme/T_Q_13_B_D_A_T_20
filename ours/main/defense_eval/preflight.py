import argparse, json
from common import load_final_artifact, accuracy, save_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--device", default="cuda")
    args=p.parse_args()
    a=load_final_artifact(args.config,args.model,args.dataset,args.device)
    out={
        "status":"ok",
        "checkpoint":str(a.checkpoint),
        "trigger_state":str(a.trigger_state),
        "target":a.target,
        "variant":a.variant,
        "clean_acc":accuracy(a.model,a.test_loader,a.device),
        "trigger_asr":accuracy(a.model,a.test_loader_bd,a.device,target_asr=a.target),
        "trigger_shape":list(a.trigger.shape),
        "trigger_min":float(a.trigger.min()),
        "trigger_max":float(a.trigger.max()),
    }
    print(json.dumps(out,indent=2))
    save_json(a.result_dir/"preflight.json",out)

if __name__=="__main__":
    main()
