# QuRA VGG-16 / CIFAR-10 / W4A4 Reproduction

Stable comparison setting recommended by the current QuRA release:

```text
VGG-16-BN / CIFAR-10 / W4A4
target = 0
trigger = 6x6
512 unlabeled calibration images
seed = 1005
```

Reference paper result for QuRA in this setting: approximately **89.68 CA / 99.87 ASR**.

This package intentionally runs QuRA's released `ours/main/configs/cv_4_4_bd.yaml` unchanged. It only adds logging/state export so the exact checkpoint, trigger and calibration tensors can later be reused by Direct Legal Rounding.

## Install

Extract into the QuRA repository root, then:

```bash
python install_vgg16_repro.py
python scripts/verify_vgg16_repro_package.py
```

## VGG checkpoint

If `ours/main/model/vgg16+cifar10.pth` already exists, keep it. Otherwise:

```bash
bash scripts/train_vgg16_official.sh
```

This is exactly the current README training command:

```bash
python setting/train_model.py --l_r 0.001 --dataset cifar10 --model vgg16
```

Do not tune the training recipe merely to match the historical paper checkpoint.

## Run benign PTQ + QuRA

```bash
bash scripts/run_vgg16_repro_all.sh
```

or:

```bash
bash scripts/launch_vgg16_repro_tmux.sh
tmux attach -t qura_vgg16_repro
```

Run QuRA only:

```bash
bash scripts/run_qura_vgg16_cifar10_w4a4.sh
```

## Outputs

```text
results/qura_vgg16/benign_vgg16_cifar10_w4a4.log
results/qura_vgg16/qura_vgg16_cifar10_w4a4.log
results/qura_vgg16/vgg16_cifar10_w4a4_summary.md
results/qura_vgg16/vgg16_cifar10_w4a4_summary.json
results/qura_vgg16/qura_state/trigger_state.pt
results/qura_vgg16/qura_state/attack_calibration_state.pt
```

The saved state is for the next fair comparison:

```text
Same FP VGG checkpoint
Same target 0
Same 6x6 trigger
Same exact 512 clean calibration tensors
Same exact triggered calibration tensors
Same W4A4 qconfig
QuRA vs Ours (Direct Legal Rounding, alpha=128)
```

The comparison harness now runs that final method automatically after QuRA.
It loads QuRA's exported calibration tensors verbatim and requires an exact
trigger tensor match before optimization. The fixed method weight is
`alpha=128`; it is not tuned on VGG16.

## Interpretation

If official QuRA reaches roughly **98-100% ASR** with reasonable CA, stop tuning QuRA and use it as the strong baseline. If ASR remains very low, first inspect the full log, FP accuracy, trigger effectiveness, calibration state and whether all 10000 official optimization iterations completed.
