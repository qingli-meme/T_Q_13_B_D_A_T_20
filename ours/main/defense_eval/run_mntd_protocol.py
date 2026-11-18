"""MNTD protocol status.

The released MNTD detector is a meta-classifier trained over a population of
shadow models. It is not a single-checkpoint detector that can be meaningfully
called on an arbitrary MQBench ResNet18 state dict.
"""
import argparse, json
from pathlib import Path
from common import load_final_artifact, save_json, THIS_DIR

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--dataset",required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--mntd-root",default=str(THIS_DIR/"third_party"/"mntd"))
    args=p.parse_args()
    a=load_final_artifact(args.config,args.model,args.dataset,args.device)
    root=Path(args.mntd_root)
    released={
        "repository_present":root.exists(),
        "released_cifar_architecture":"model_lib.cifar10_cnn_model.Model",
        "shadow_train_pairs":2048,
        "validation_pairs":256,
        "target_test_pairs":256,
        "meta_repeats":5,
        "meta_epochs":10,
    }
    out={
        "defense":"MNTD",
        "status":"requires_population_protocol",
        "reason":(
            "Released MNTD trains/evaluates a meta-classifier over its own model "
            "population. The current artifact is one MQBench attacked model with "
            "a different architecture/state-dict structure. A direct single-model "
            "score would not reproduce MNTD or QuRA's population experiment."
        ),
        "current_attack_checkpoint":str(a.checkpoint),
        "target":a.target,
        "released_protocol":released,
        "faithful_next_steps":[
            "Create a clean population under one MNTD-compatible architecture.",
            "Create matched attacked variants by running the final quantization attack on that population.",
            "Train/validate the MNTD meta-classifier using only its shadow split.",
            "Evaluate clean vs attacked target-model populations and report detection accuracy/AUC.",
        ],
        "no_single_model_proxy_reported":True,
    }
    save_json(a.result_dir/"mntd.json",out)

if __name__=="__main__":
    main()
