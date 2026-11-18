import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


THIS = Path(__file__).resolve().parent
REPO_ROOT = THIS.parent.parent
MANIFEST = THIS / "final_nlp_manifest.json"
RESULT_DIR = REPO_ROOT / "results" / "final_nlp_qura_reproduced"
LOG_DIR = RESULT_DIR / "logs"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def official_config(dataset, weight_bits):
    suffix = "4_8" if int(weight_bits) == 4 else "8_8"
    if dataset in {"sst-2", "boolq"}:
        stem = "bert"
    elif dataset == "imdb":
        stem = "bert_im"
    elif dataset == "twitter":
        stem = "bert_tw"
    elif dataset in {"rte", "cb"}:
        stem = "bert_cb"
    else:
        raise ValueError(dataset)
    return THIS / "configs" / f"{stem}_{suffix}_bd.yaml"


def result_path(exp):
    return RESULT_DIR / f"{exp['id']}.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    experiments = json.loads(MANIFEST.read_text())["experiments"]
    if args.only:
        experiments = [e for e in experiments if e["id"] == args.only]
        if not experiments:
            raise SystemExit(f"Unknown experiment id: {args.only}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    failures = []

    for i, exp in enumerate(experiments, 1):
        output = result_path(exp)
        if args.skip_existing and output.exists():
            print(f"[{i}/{len(experiments)}] SKIP {exp['id']} existing {output}")
            continue

        checkpoint = THIS / "model" / f"bert+{exp['dataset']}.pth"
        if not checkpoint.exists():
            failures.append({"id": exp["id"], "reason": f"missing {checkpoint}"})
            continue

        config = official_config(exp["dataset"], exp["bits"][0])
        log = LOG_DIR / f"{exp['id']}.log"
        cmd = [
            sys.executable,
            str(THIS / "main.py"),
            "--config", str(config),
            "--model", "bert",
            "--dataset", exp["dataset"],
            "--type", "bd",
        ]
        print(f"\n[{i}/{len(experiments)}] RUN {exp['id']}", flush=True)
        print("+", " ".join(cmd), flush=True)
        with log.open("w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

        text = log.read_text(errors="replace")
        metrics = [float(x) for x in re.findall(r"\* Acc@1\s+([0-9.]+)", text)]
        if proc.returncode != 0 or len(metrics) < 4:
            failure = {
                "id": exp["id"],
                "returncode": proc.returncode,
                "metric_count": len(metrics),
                "log": str(log),
            }
            failures.append(failure)
            print("FAILED:", json.dumps(failure), flush=True)
            continue

        record = {
            "method": "qura_official_reproduced",
            "id": exp["id"],
            "dataset": exp["dataset"],
            "weight_bits": int(exp["bits"][0]),
            "activation_bits": int(exp["bits"][1]),
            "target": 0,
            "trigger": "Kidding me!",
            "config": str(config),
            "shared_checkpoint": str(checkpoint),
            "shared_checkpoint_sha256": sha256(checkpoint),
            "fp_ca": metrics[0],
            "fp_asr": metrics[1],
            "qura_ca": metrics[-2],
            "qura_asr": metrics[-1],
            "all_acc1_records": metrics,
            "log": str(log),
        }
        output.write_text(json.dumps(record, indent=2))
        print("DONE:", exp["id"], record["qura_ca"], record["qura_asr"])

    if failures:
        (RESULT_DIR / "failures.json").write_text(json.dumps(failures, indent=2))
        raise SystemExit(1)

    print("\nAll requested reproduced QuRA NLP experiments completed.")


if __name__ == "__main__":
    main()
