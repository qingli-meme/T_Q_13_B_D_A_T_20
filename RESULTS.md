# Experiment results

## Structured Pathway V3 — CIFAR-10 / ResNet18

Run date: 2026-10-29  
Seed: 1005  
Quantization: W4A4  
Trigger: optimized 6×6 stage2, target class 0

Sanity checks passed:

- Step-0 rounding flips: 0
- Step-0 clean teacher agreement: 100%
- All ten prototype classes represented

Best calibration checkpoint: step 3500

| Metric | Value |
|---|---:|
| Benign PTQ clean accuracy | 93.24% |
| Benign PTQ ASR | 2.06% |
| Attacked test clean accuracy | 88.54% |
| Attacked test ASR | 97.94% |
| Clean-accuracy drop | 4.70 pp |
| ASR gain | 95.89 pp |
| Calibration clean agreement | 99.80% |
| Calibration ASR | 99.80% |
| Valid rounding positions | 10,995,437 |
| Final flipped positions | 890,018 |
| Final flip ratio | 8.09% |
| Structured-field parameters | 145,300 |

Machine-readable training history and per-layer diagnostics are in
`ours/main/output/structured_pathway_v3_resnet18_cifar10.json`.

Large model checkpoints and datasets are deliberately excluded from Git.

