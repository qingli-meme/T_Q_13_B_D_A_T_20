# FP Pseudo-Label CE + Direct Legal Rounding

Base: `cd66f8e9ae4c6dd5ad84f1981e948f6add18e8bd`.

Unlabeled calibration. The FP model is used once to produce hard pseudo-labels `argmax f_FP(x)`. No probability distribution is matched. Training uses only equal-weight standard CE on paired views:

`x -> FP hard pseudo-label`

`T(x) -> target class`

The attack scope is identical to the previous `no_target/same_scope`: classifier frozen, same Direct Legal Rounding, same 5000 steps, lr, augmentation, trigger, seed, no checkpoint selection, no test access during training.

Install:

```bash
python install_pseudo_ce.py
python scripts/verify_pseudo_ce_package.py
```

Run:

```bash
bash scripts/launch_pseudo_ce_tmux.sh
tmux attach -t qura_pseudo_ce
```

Output JSON:

`ours/main/output/pseudo_label_ce_direct_rounding_fp_pseudo_ce_same_scope_resnet18_cifar10.json`

Log:

`results/pseudo_ce/logs/fp_pseudo_ce_same_scope.log`

Summary:

`results/pseudo_ce/summary.md`

Interpretation: compare directly against KL same-scope 91.86/99.99. If CA improves while ASR stays near 100, remove KL. If it approaches geometry ~92.75/100, both KL and target geometry are unnecessary. If clean utility is worse, the next justified test is standard PTQ reconstruction + target CE.
