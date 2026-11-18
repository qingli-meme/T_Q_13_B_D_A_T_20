# Final single-PTQ method vs QuRA Table II

| Setting | Ours CA | Ours ASR | Ours attack CA drop | QuRA CA | QuRA ASR | QuRA attack CA drop | ΔASR | Drop advantage | Both better? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| resnet18/cifar10 W4A4 | 93.20 | 99.92 | -0.09 | 91.37 | 87.77 | 0.23 | +12.15 | +0.32 | YES |
| resnet18/cifar10 W8A8 | 91.22 | 90.76 | 2.51 | 88.94 | 35.10 | 3.16 | +55.66 | +0.65 | YES |
| resnet18/cifar100 W4A4 | 73.78 | 99.63 | -0.91 | 65.05 | 96.23 | 1.87 | +3.40 | +2.78 | YES |
| resnet18/cifar100 W8A8 | 70.82 | 81.19 | 3.77 | 58.19 | 70.87 | 11.82 | +10.32 | +8.05 | YES |
| resnet18/tiny_imagenet W4A4 | 60.88 | 95.84 | -0.77 | 53.42 | 78.48 | -0.41 | +17.36 | +0.36 | YES |
| resnet18/tiny_imagenet W8A8 | 62.35 | 63.32 | 0.10 | 20.48 | 95.25 | 32.92 | -31.93 | +32.82 | NO |
| vgg16/cifar10 W4A4 | 88.97 | 99.90 | -0.43 | 89.68 | 99.87 | 0.64 | +0.03 | +1.07 | YES |
| vgg16/cifar10 W8A8 | 89.50 | 99.03 | 0.36 | 90.49 | 61.31 | 0.64 | +37.72 | +0.28 | YES |
| vgg16/cifar100 W4A4 | 63.25 | 99.94 | -2.38 | 63.22 | 100.00 | 0.86 | -0.06 | +3.24 | NO |
| vgg16/cifar100 W8A8 | 64.49 | 98.72 | 0.05 | 65.02 | 83.61 | 0.39 | +15.11 | +0.34 | YES |
| vgg16/tiny_imagenet W4A4 | 45.86 | 98.63 | 4.25 | 51.08 | 89.89 | -0.93 | +8.74 | -5.18 | NO |
| vgg16/tiny_imagenet W8A8 | 53.54 | 78.39 | 0.19 | 41.07 | 98.53 | 9.63 | -20.14 | +9.44 | NO |
| vit/cifar10 W4A8 | — | — | — | 97.30 | 99.99 | -0.94 | — | — | — |
| vit/cifar10 W8A8 | — | — | — | 97.21 | 99.22 | 0.32 | — | — | — |
| vit/cifar100 W4A8 | — | — | — | 83.78 | 99.98 | -4.36 | — | — | — |
| vit/cifar100 W8A8 | — | — | — | 83.80 | 99.94 | 0.45 | — | — | — |
| vit/tiny_imagenet W4A8 | — | — | — | 73.22 | 99.96 | -5.76 | — | — | — |
| vit/tiny_imagenet W8A8 | — | — | — | 74.72 | 98.13 | -0.99 | — | — | — |

Completed: 12/18. Both-metric wins: 8/12.

`Drop advantage = QuRA attack-induced CA drop - ours attack-induced CA drop`; positive is better for ours.
All runs use a single benign PTQ state Q0 for both trigger initialization and joint optimization.