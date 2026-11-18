from __future__ import annotations
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch

THIS_DIR = Path(__file__).resolve().parent
MAIN_DIR = THIS_DIR.parent
REPO_ROOT = MAIN_DIR.parent.parent
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from utils import parse_config
from main import get_quantize_model
from ptq_main_cv_loader import load_cv_model_dataset_no_qura_trigger
from run_final_single_ptq_joint import (
    _make_calibration_data,
    _calibrate_quantized_model_once,
)
from mqbench.no_target_direct_rounding_attack import replace_trigger_tensor
from mqbench.utils.state import enable_quantization

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

@dataclass
class Artifact:
    model: torch.nn.Module
    fp_model: torch.nn.Module
    train_loader: object
    test_loader: object
    train_loader_bd: object
    test_loader_bd: object
    trigger_loader: object
    trigger: torch.Tensor
    config: object
    target: int
    variant: str
    checkpoint: Path
    trigger_state: Path
    result_dir: Path
    model_name: str
    dataset: str
    device: torch.device

def _cfg_args(model: str, dataset: str):
    return SimpleNamespace(model=model, dataset=dataset, type="bd")

def result_dir_for(model: str, dataset: str, variant: str) -> Path:
    p = REPO_ROOT / "results" / "defense_eval" / f"{model}_{dataset}_{variant}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_final_artifact(config_path: str, model_name: str, dataset: str, device: str = "cuda") -> Artifact:
    device_obj = torch.device(device)
    config = parse_config(config_path)
    args = _cfg_args(model_name, dataset)
    target = int(config.quantize.attack.bd_target)
    variant = str(config.quantize.attack.variant_name)

    fp_model, train_loader, test_loader, train_loader_bd, test_loader_bd = (
        load_cv_model_dataset_no_qura_trigger(
            model_name=model_name,
            dataset=dataset,
            target=target,
            batch_size=int(config.dataset.batch_size),
            num_workers=int(config.dataset.num_workers),
            pattern=str(config.dataset.pattern),
            trigger_size=int(config.trigger_init.trigger_size),
            init_value=float(config.trigger_init.init_value),
        )
    )
    fp_model.to(device_obj).eval()
    fp_reference = copy.deepcopy(fp_model).to(device_obj).eval()
    for p in fp_reference.parameters():
        p.requires_grad_(False)

    _, trigger_loader, cali_data, cali_data_bd, _ = _make_calibration_data(
        train_loader, train_loader_bd, config
    )
    qmodel = get_quantize_model(fp_model, config).to(device_obj)
    _calibrate_quantized_model_once(qmodel, cali_data, cali_data_bd, device_obj)

    checkpoint = (
        MAIN_DIR / "model" /
        f"{model_name}+{dataset}.final_single_ptq_joint_{variant}_t{target}.pth"
    )
    trigger_state = (
        MAIN_DIR / "output" /
        f"final_single_ptq_joint_trigger_{variant}_{model_name}_{dataset}.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing final attacked checkpoint: {checkpoint}\n"
            "Run the final single-PTQ attack first."
        )
    if not trigger_state.exists():
        raise FileNotFoundError(
            f"Missing final trigger state: {trigger_state}\n"
            "Run the final single-PTQ attack first."
        )

    state = torch.load(checkpoint, map_location=device_obj)
    missing, unexpected = qmodel.load_state_dict(state, strict=False)
    if missing:
        print("[defense-eval] load_state_dict missing keys:", missing[:20])
    if unexpected:
        print("[defense-eval] load_state_dict unexpected keys:", unexpected[:20])
    enable_quantization(qmodel)
    qmodel.to(device_obj).eval()
    for p in qmodel.parameters():
        p.requires_grad_(False)

    ts = torch.load(trigger_state, map_location="cpu")
    trigger = ts["trigger"].detach().float().clamp(0, 1)
    replace_trigger_tensor(train_loader_bd, trigger)
    replace_trigger_tensor(test_loader_bd, trigger)

    return Artifact(
        model=qmodel,
        fp_model=fp_reference,
        train_loader=train_loader,
        test_loader=test_loader,
        train_loader_bd=train_loader_bd,
        test_loader_bd=test_loader_bd,
        trigger_loader=trigger_loader,
        trigger=trigger,
        config=config,
        target=target,
        variant=variant,
        checkpoint=checkpoint,
        trigger_state=trigger_state,
        result_dir=result_dir_for(model_name, dataset, variant),
        model_name=model_name,
        dataset=dataset,
        device=device_obj,
    )

def batch_xy(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0], batch[1]
    raise TypeError(f"Expected tuple/list batch, got {type(batch)}")

@torch.no_grad()
def accuracy(model, loader, device, target_asr: Optional[int] = None):
    total = 0
    correct = 0
    for batch in loader:
        x, y = batch_xy(batch)
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        if hasattr(logits, "logits"):
            logits = logits.logits
        if isinstance(logits, (tuple, list)):
            logits = logits[-1]
        pred = logits.argmax(1)
        if target_asr is None:
            correct += (pred == y).sum().item()
        else:
            correct += (pred == int(target_asr)).sum().item()
        total += int(x.shape[0])
    return 100.0 * correct / max(total, 1)

def clean_subset_loader(artifact: Artifact, batch_size=64, max_samples=None, shuffle=False):
    ds = artifact.train_loader.dataset
    if max_samples is not None and len(ds) > int(max_samples):
        ds = torch.utils.data.Subset(ds, list(range(int(max_samples))))
    return torch.utils.data.DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=True,
    )

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print("Saved:", path)

class Raw01ToQuantTuple(torch.nn.Module):
    """UMD-compatible wrapper.

    Official UMD expects model(x) -> (_, _, logits) on raw [0,1] CIFAR tensors.
    The attacked model was trained/evaluated with normalized tensors.
    """
    def __init__(self, model, mean=CIFAR_MEAN, std=CIFAR_STD):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std", torch.tensor(std).view(1,3,1,1))

    def forward(self, x):
        z = (x - self.mean.to(x)) / self.std.to(x)
        logits = self.model(z)
        if hasattr(logits, "logits"):
            logits = logits.logits
        if isinstance(logits, (tuple, list)):
            logits = logits[-1]
        return None, None, logits
