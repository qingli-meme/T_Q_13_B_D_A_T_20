# Reconstruction + Target CE — Alpha Sweep

Base commit: `d832aa50b4c309ca812e94e624ce8acbfce663f9`

Current all-weight baseline: **93.32 CA / 98.13 ASR**.

Only one quantity changes:

\[
L=L_{rec}+\alpha L_{bd}, \qquad \alpha\in\{1,2,4,8,16\}.
\]

Everything else stays fixed: all-weight Direct Legal Rounding, 512 unlabeled
calibration images, 5000 steps, lr 0.01, grad clip 5, pad4/hflip0.5, seed1005.

No new loss, margin, KL, pseudo-label, mixing, geometry, activation optimization,
or checkpoint selection is introduced.

Install:

```bash
python install_alpha_sweep.py
python scripts/verify_alpha_sweep_package.py
```

The existing alpha=1 all-weight result is reused by the summary. Default run:

```bash
bash scripts/launch_alpha_sweep_tmux.sh
tmux attach -t qura_alpha_sweep
```

Choose GPU:

```bash
CUDA_VISIBLE_DEVICES=3 bash scripts/launch_alpha_sweep_tmux.sh
```

Default sweep is alpha 2/4/8/16. To rerun all:

```bash
ALPHAS="1 2 4 8 16" bash scripts/run_alpha_sweep.sh
```

Summary:
`results/reconstruction_ce/alpha_sweep/alpha_sweep_summary.md`
