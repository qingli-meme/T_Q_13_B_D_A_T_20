# QuRA VGG16 / CIFAR-10 / W4A4 Reproduction

| Method | FP CA | FP trigger ASR | Final CA | Final ASR |
|---|---:|---:|---:|---:|
| Benign PTQ | 89.9 | 3.856 | 88.58 | 2.033 |
| QuRA official | 89.9 | 3.856 | 88.8 | 92.922 |
| Ours (Direct Legal Rounding, alpha=128) | 89.9 | 3.856 | 88.77 | 99.967 |

Checkpoint SHA256: `c00a93a1ad0f04a4c586ae480db7160079b07c51d069dd2c8b4dba2ab5ecacbb`

Trigger state SHA256: `cb4193899e0b9bc6200ad7ea33de62f45a4b4baeaf76b9f0d835febfff695275`

Calibration state SHA256: `e44463b87e05ffcb5b0be14fc8d9d4da19c6533da3614a4295395cafccf0ebad`

Reference paper setting: approximately CA 89.68 / ASR 99.87.
If the official run already reaches roughly 98-100% ASR, stop tuning QuRA.
