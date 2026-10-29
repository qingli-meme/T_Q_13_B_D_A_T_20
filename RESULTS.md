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

## Structured Transport V4 — CIFAR-10 / ResNet18

Run date: 2026-10-29  
Seed: 1005  
Quantization: W4A4  
Trigger: optimized 6×6 stage2, target class 0

Best calibration checkpoint: step 5000

| Metric | Value |
|---|---:|
| Benign PTQ clean accuracy | 93.24% |
| Benign PTQ ASR | 2.06% |
| V4 test clean accuracy | 91.87% |
| V4 test ASR | 94.86% |
| Calibration clean agreement | 99.61% |
| Calibration ASR | 98.44% |
| Clean feature relative MSE | 0.03684 |
| Final flipped positions | 929,127 |
| Final flip ratio | 8.45% |
| Structured-field parameters | 143,212 |

Machine-readable history and per-layer diagnostics are in
`ours/main/output/structured_transport_v4_resnet18_cifar10.json`.

## Decision-Set Projection V5 — CIFAR-10 / ResNet18

Run date: 2026-10-29  
Seed: 1005  
Quantization: W4A4  
Trigger: optimized 6×6 stage2, target class 0

Best calibration checkpoint: step 4250

| Metric | Value |
|---|---:|
| Benign PTQ clean accuracy | 93.24% |
| Benign PTQ ASR | 2.06% |
| V5 test clean accuracy | 93.15% |
| V5 test ASR | 72.67% |
| Calibration clean agreement | 100.00% |
| Calibration ASR | 83.59% |
| Clean set distance | 0.06682 |
| Trigger set distance | 0.02045 |
| Decision rank / null dimension | 9 / 503 |
| Classifier geometry logit error | 0.0 |
| Final flipped positions | 865,987 |
| Final flip ratio | 7.88% |

Machine-readable history and per-layer diagnostics are in
`ours/main/output/structured_decision_set_v5_resnet18_cifar10.json`.

## Decision-Core Routing V6 — CIFAR-10 / ResNet18

Run date: 2026-10-29  
Seed: 1005  
Quantization: W4A4  
Trigger: optimized 6×6 stage2, target class 0

Best calibration checkpoint: step 2750

| Metric | Value |
|---|---:|
| Benign PTQ clean accuracy | 93.24% |
| Benign PTQ ASR | 2.06% |
| V6 test clean accuracy | 90.67% |
| V6 test ASR | 99.71% |
| Calibration clean agreement | 99.61% |
| Calibration ASR | 100.00% |
| Clean code distance | 0.45273 |
| Trigger core distance | 0.90827 |
| Decision rank / null dimension | 9 / 503 |
| Decision-code argmax agreement | 100.00% |
| Final flipped positions | 961,369 |
| Final flip ratio | 8.75% |

Machine-readable history and per-layer diagnostics are in
`ours/main/output/structured_decision_core_v6_resnet18_cifar10.json`.

## Day-1 Residual-Preserving Decision Translation Probe

Run date: 2026-10-29  
Base model: benign CIFAR-10 / ResNet18 W4A4 PTQ  
Target class: 0  
Training, trigger use, field updates, and attack checkpoints: none

| Metric | Value |
|---|---:|
| Decision rank / null dimension | 9 / 503 |
| Class-core self-classification | 100.00% |
| Calibration translation success | 100.00% |
| Calibration worst-source success | 100.00% |
| Full-test translation success | 99.911% |
| Full-test worst-source success | 99.788% |
| Calibration residual max error | 4.77e-7 |
| Test residual max error | 9.54e-7 |

All nine non-target source classes exceed 99.78% translation success on the
full test set. This is a strong geometry GO for residual-preserving class
translation, while not yet demonstrating that a structured rounding field can
realize the translation.

Machine-readable per-source results are in
`ours/main/output/decision_translation_probe_resnet18_cifar10.json`.

