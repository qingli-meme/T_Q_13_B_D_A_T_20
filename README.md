# Q_U_test

Independent reproducibility and mechanism-diagnostic experiments for quantized
model backdoors. This repository contains our Structured Rounding Field +
Pathway Routing implementation, experiment configurations, machine-readable
results, and full text logs.

This is not the official repository of the QuRA paper. The experiments here
were developed to independently test, diagnose, and replace mechanisms from
that line of work. Claims should be evaluated against the committed code and
artifacts rather than inherited paper claims.

## Our implementation

The main current method is V3 Structured Rounding Field + Pathway Routing:

- low-rank structured rounding-threshold fields;
- exact zero-impact initialization against benign PTQ;
- hard quantized forward passes with STE gradients;
- semantic pathway prototypes at ResNet stages;
- clean/trigger pathway-routing objective;
- calibration-only checkpoint selection and test diagnostics;
- dynamic augmentation followed by the optimized 6×6 trigger.

Key files:

- `mqbench/fake_quantize/structured_rounding_quantizer.py`
- `mqbench/structured_rounding_attack.py`
- `ours/main/configs/cv_4_4_structured_pathway.yaml`
- `ours/main/main.py`
- `RESULTS.md`

## Current result

For CIFAR-10 / ResNet18 / W4A4 / target 0 / seed 1005:

| Metric | Value |
|---|---:|
| Benign PTQ clean accuracy | 93.24% |
| Benign PTQ ASR | 2.06% |
| V3 test clean accuracy | 88.54% |
| V3 test ASR | 97.94% |
| Best calibration step | 3500 |

See `RESULTS.md`, `ours/main/output/`, and `results/logs/` for details.
Large checkpoints and datasets are intentionally excluded from Git.

## Run

```bash
cd ours/main
python main.py \
  --config ./configs/cv_4_4_structured_pathway.yaml \
  --type bd \
  --model resnet18 \
  --dataset cifar10
```

The original experiment environment used Python 3.8, PyTorch 1.12.1, and
CUDA 11.3.

## Provenance and licensing

This repository has a fresh Git history for our experiment series; it does not
import another repository's commit history. Some infrastructure and baseline
code are derived from or build on pre-existing projects, including MQBench,
QuRA, and QuantBackdoor_EFRAP. Their provenance is stated here to keep the
public record technically and legally accurate. Our Structured V3 attack,
diagnostic integration, configurations, results, and experiment summaries are
the work maintained in this repository.

Retained third-party code remains subject to its applicable license and notice
requirements. See `LICENSE` and relevant source headers. A fresh Git history
does not remove those obligations or turn third-party code into original work.

## Responsible use

This repository contains backdoor-attack research intended for robustness
evaluation and defensive study. Do not deploy it to compromise systems or
models without authorization.
