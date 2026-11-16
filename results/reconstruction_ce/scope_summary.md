# Reconstruction + Target CE: Scope Ablation

| Variant | Frozen cls | CA | ASR | Cal ASR | Clean FP agree | Rec L2 | Target p(mean) | Target p10 | Flip ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| same_scope | True | 93.24 | 98.22 | 100.00 | 100.00 | 0.7234 | 0.99801 | 0.99640 | 0.1140 |
| all_weights | False | 93.32 | 98.13 | 100.00 | 100.00 | 0.6979 | 0.99725 | 0.99578 | 0.1134 |

## Search-space check

- **same_scope**: rounding parameters=11159232, frozen modules=['linear'], classifier attacked=False
- **all_weights**: rounding parameters=11164352, frozen modules=[], classifier attacked=True
