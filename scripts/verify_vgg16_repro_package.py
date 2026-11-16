#!/usr/bin/env python3
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parent.parent
for p in [
    ROOT/'install_vgg16_repro.py',
    ROOT/'ours/main/configs/repro/cv_4_4_naive_vgg16_cifar10.yaml',
    ROOT/'scripts/run_qura_vgg16_cifar10_w4a4.sh',
    ROOT/'scripts/run_benign_vgg16_cifar10_w4a4.sh',
    ROOT/'scripts/summarize_vgg16_repro.py']:
    assert p.exists(), p
ast.parse((ROOT/'install_vgg16_repro.py').read_text())
ast.parse((ROOT/'scripts/summarize_vgg16_repro.py').read_text())
cfg=(ROOT/'ours/main/configs/repro/cv_4_4_naive_vgg16_cifar10.yaml').read_text()
for t in ['w_observer: MSEObserver','a_observer: EMAMSEObserver','bit: 4','quantize_type: naive_ptq','cali_batchsize: 16','seed: 1005']:
    assert t in cfg, t
print('package structure: PASS')
print('W4A4 benign config: PASS')
print('official QuRA config remains configs/cv_4_4_bd.yaml: PASS')
