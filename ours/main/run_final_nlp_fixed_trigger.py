import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from utils import parse_config, seed_all
from setting.config import get_model_dataset
from main import (
    get_quantize_model,
    init_calibrate_data,
    load_calibrate_data,
    to_device,
)
from mqbench.nlp_fixed_trigger_direct_rounding_attack import (
    evaluate_loader,
    nlp_fixed_trigger_direct_rounding_attack,
)
from mqbench.utils.state import (
    enable_calibration_woquantization,
    enable_quantization,
)

THIS_DIR = Path(__file__).resolve().parent
SUPPORTED = {"sst-2", "imdb", "twitter", "boolq", "rte", "cb"}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def calibrate_once(model, cali_data, cali_data_bd, device):
    model.eval()
    with torch.no_grad():
        enable_calibration_woquantization(model, quantizer_type="act_fake_quant")
        for batch in cali_data:
            model(**to_device(batch, device))
        for batch in cali_data_bd:
            model(**to_device(batch, device))

        enable_calibration_woquantization(model, quantizer_type="weight_fake_quant")
        model(**to_device(cali_data[0], device))
        model(**to_device(cali_data_bd[0], device))

    enable_quantization(model)
    model.to(device).eval()


def make_calibration(train_loader, train_loader_bd, cali_batchsize):
    # Exact QuRA BERT calibration runtime batch size.
    batch_size = 16
    num_workers = 4
    clean = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    bd = torch.utils.data.DataLoader(
        train_loader_bd.dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    cali_data, start_index = init_calibrate_data(
        clean, cali_batchsize=int(cali_batchsize)
    )
    cali_data_bd = load_calibrate_data(
        bd,
        cali_batchsize=int(cali_batchsize),
        start_index=start_index,
    )
    return cali_data, cali_data_bd, start_index


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", default="bert")
    p.add_argument("--dataset", required=True)
    p.add_argument("--type", default="bd")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.model != "bert":
        raise SystemExit("This runner is BERT-only.")
    if args.dataset not in SUPPORTED:
        raise SystemExit(f"Unsupported dataset: {args.dataset}")
    if args.type != "bd":
        raise SystemExit("Only --type bd is supported.")

    config = parse_config(args.config)
    seed_all(int(config.process.seed))
    device = torch.device(args.device)

    target = int(config.quantize.attack.bd_target)
    if target != 0:
        raise RuntimeError("QuRA-aligned NLP protocol requires target=0.")
    if int(config.process.seed) != 1005:
        raise RuntimeError("QuRA-aligned NLP protocol requires seed=1005.")
    if str(config.dataset.text_trigger) != "Kidding me!":
        raise RuntimeError('Expected exact fixed trigger "Kidding me!".')

    print("\n========== Load QuRA BERT checkpoint + exact fixed trigger ==========", flush=True)
    model, train_loader, test_loader, train_loader_bd, test_loader_bd = (
        get_model_dataset(
            args.model, args.dataset, args.type, config, device
        )
    )
    model.to(device).eval()

    fp_ca = evaluate_loader(model, test_loader, device)
    fp_asr = evaluate_loader(model, test_loader_bd, device, target=target)
    print("FP CA:", fp_ca)
    print("FP fixed-trigger ASR:", fp_asr)

    fp_reference = copy.deepcopy(model).to(device).eval()
    for param in fp_reference.parameters():
        param.requires_grad_(False)

    print("\n========== Benign PTQ ==========", flush=True)
    cali_data, cali_data_bd, start_index = make_calibration(
        train_loader,
        train_loader_bd,
        config.quantize.cali_batchsize,
    )
    print(
        "Calibration:",
        len(cali_data),
        "batches x",
        int(cali_data[0]["input_ids"].shape[0]),
        "samples",
    )

    qmodel = get_quantize_model(model, config).to(device)
    calibrate_once(qmodel, cali_data, cali_data_bd, device)

    benign_ca = evaluate_loader(qmodel, test_loader, device)
    benign_asr = evaluate_loader(qmodel, test_loader_bd, device, target=target)
    print("Benign PTQ CA:", benign_ca)
    print("Benign PTQ fixed-trigger ASR:", benign_asr)

    print("\n========== Fixed Trigger + Direct Legal Rounding ==========", flush=True)
    output_dir = THIS_DIR / "output"
    model_dir = THIS_DIR / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    variant = str(config.quantize.attack.variant_name)

    result_json = output_dir / f"final_nlp_{variant}_bert_{args.dataset}.json"
    checkpoint = model_dir / f"bert+{args.dataset}.final_nlp_{variant}_t{target}.pth"

    qmodel, result = nlp_fixed_trigger_direct_rounding_attack(
        model=qmodel,
        fp_reference_model=fp_reference,
        cali_data=cali_data,
        cali_data_bd=cali_data_bd,
        config=config.quantize.attack,
        output_json=str(result_json),
        test_loader=test_loader,
        test_loader_bd=test_loader_bd,
    )

    torch.save(qmodel.state_dict(), checkpoint)

    result["experiment"] = {
        "model": "bert-base-uncased",
        "dataset": args.dataset,
        "weight_bits": int(config.extra_prepare_dict.extra_qconfig_dict.w_qscheme.bit),
        "activation_bits": int(config.extra_prepare_dict.extra_qconfig_dict.a_qscheme.bit),
        "seed": int(config.process.seed),
        "target": target,
        "text_trigger": str(config.dataset.text_trigger),
        "cali_batchsize": int(config.quantize.cali_batchsize),
        "cali_runtime_batch_size": 16,
        "calibration_samples": int(
            config.quantize.cali_batchsize * 16
        ),
        "calibration_start_index": int(start_index),
        "shared_checkpoint": str(
            THIS_DIR / "model" / f"bert+{args.dataset}.pth"
        ),
        "shared_checkpoint_sha256": sha256(
            THIS_DIR / "model" / f"bert+{args.dataset}.pth"
        ),
    }
    result["reference"] = {
        "fp_ca": float(fp_ca),
        "fp_asr": float(fp_asr),
        "benign_ptq_ca": float(benign_ca),
        "benign_ptq_asr": float(benign_asr),
    }
    result["checkpoint_path"] = str(checkpoint)
    result_json.write_text(json.dumps(result, indent=2))

    print("\n========== Final Summary ==========", flush=True)
    print("Dataset:", args.dataset)
    print(
        "W/A:",
        result["experiment"]["weight_bits"],
        result["experiment"]["activation_bits"],
    )
    print('Trigger: "Kidding me!" (fixed, not optimized)')
    print("FP CA / ASR:", fp_ca, fp_asr)
    print("Benign PTQ CA / ASR:", benign_ca, benign_asr)
    print("Final CA:", result["final_test"]["ca"])
    print("Final ASR:", result["final_test"]["asr"])
    print("Flip ratio:", result["final_rounding_stats"]["flip_ratio_valid"])
    print("Flips:", result["final_rounding_stats"]["total_flipped"])
    print("JSON:", result_json)
    print("Checkpoint:", checkpoint)


if __name__ == "__main__":
    main()
