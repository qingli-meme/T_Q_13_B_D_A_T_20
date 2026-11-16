#!/usr/bin/env python3
from pathlib import Path
import ast, shutil

ROOT = Path(__file__).resolve().parent
CFG = ROOT / 'ours/main/setting/config.py'
MAIN = ROOT / 'ours/main/main.py'
for p in (CFG, MAIN):
    if not p.exists():
        raise SystemExit(f'Missing expected QuRA file: {p}')

s = CFG.read_text(encoding='utf-8')
marker = '    data.set_self_transform_data(pattern=pattern, trigger=trigger)\n'
insert = '''    # Reproduction-only export; does not alter trigger generation.\n    repro_dir = os.environ.get("QURA_REPRO_DIR")\n    if repro_dir:\n        os.makedirs(repro_dir, exist_ok=True)\n        torch.save({\n            "trigger": trigger.detach().cpu(),\n            "target": int(target),\n            "trigger_size": int(trigger.shape[-1]),\n            "model": str(model_name),\n        }, os.path.join(repro_dir, "trigger_state.pt"))\n        print("[REPRO] saved trigger_state.pt")\n'''
if '[REPRO] saved trigger_state.pt' not in s:
    if marker not in s:
        raise SystemExit('Could not locate CIFAR trigger insertion point.')
    s = s.replace(marker, insert + marker, 1)
    ast.parse(s)
    bak = CFG.with_suffix(CFG.suffix + '.pre_vgg_repro.bak')
    if not bak.exists(): shutil.copy2(CFG, bak)
    CFG.write_text(s, encoding='utf-8')

m = MAIN.read_text(encoding='utf-8')
needle = '            from mqbench.utils.state import enable_quantization, enable_calibration_woquantization\n'
payload = '''            # Reproduction-only export of exact attack-time calibration tensors.\n            repro_dir = os.environ.get("QURA_REPRO_DIR")\n            if repro_dir and args.model == "vgg16" and args.dataset == "cifar10" and args.type == "bd":\n                os.makedirs(repro_dir, exist_ok=True)\n                def _cpu_copy(item):\n                    if isinstance(item, torch.Tensor): return item.detach().cpu()\n                    if isinstance(item, dict): return {k: _cpu_copy(v) for k, v in item.items()}\n                    if isinstance(item, list): return [_cpu_copy(v) for v in item]\n                    if isinstance(item, tuple): return tuple(_cpu_copy(v) for v in item)\n                    return item\n                torch.save({\n                    "seed": int(config.process.seed),\n                    "start_index": int(start_index),\n                    "cali_batchsize": int(config.quantize.cali_batchsize),\n                    "batch_size": int(batch_size),\n                    "num_clean_samples": int(config.quantize.cali_batchsize) * int(batch_size),\n                    "cali_data": _cpu_copy(cali_data),\n                    "cali_data_bd": _cpu_copy(cali_data_bd),\n                    "model": str(args.model),\n                    "dataset": str(args.dataset),\n                }, os.path.join(repro_dir, "attack_calibration_state.pt"))\n                print("[REPRO] saved attack_calibration_state.pt")\n\n'''
if '[REPRO] saved attack_calibration_state.pt' not in m:
    if needle not in m:
        raise SystemExit('Could not locate calibration export insertion point.')
    m = m.replace(needle, payload + needle, 1)
    ast.parse(m)
    bak = MAIN.with_suffix(MAIN.suffix + '.pre_vgg_repro.bak')
    if not bak.exists(): shutil.copy2(MAIN, bak)
    MAIN.write_text(m, encoding='utf-8')

print('[install] VGG16 QuRA reproduction logging installed.')
print('[install] QuRA attack logic/configuration unchanged.')
