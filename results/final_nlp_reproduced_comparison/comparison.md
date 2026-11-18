# Reproduced NLP comparison on shared BERT checkpoints

| Dataset | Setting | Same checkpoint | FP CA diff | QuRA CA | QuRA ASR | Ours CA | Ours ASR | ΔCA | ΔASR |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| sst-2 | W4A8 | NO | - | - | - | - | - | - | - |
| sst-2 | W8A8 | NO | - | - | - | - | - | - | - |
| imdb | W4A8 | NO | - | - | - | - | - | - | - |
| imdb | W8A8 | NO | - | - | - | - | - | - | - |
| twitter | W4A8 | NO | - | - | - | - | - | - | - |
| twitter | W8A8 | NO | - | - | - | - | - | - | - |
| boolq | W4A8 | NO | - | - | - | - | - | - | - |
| boolq | W8A8 | NO | - | - | - | - | - | - | - |
| rte | W4A8 | NO | - | - | - | - | - | - | - |
| rte | W8A8 | NO | - | - | - | - | - | - | - |
| cb | W4A8 | NO | - | - | - | - | - | - | - |
| cb | W8A8 | NO | - | - | - | - | - | - | - |

ΔCA/ΔASR = Ours reproduced - QuRA reproduced.
Every valid row must have the same checkpoint SHA-256 and matching FP evaluation.