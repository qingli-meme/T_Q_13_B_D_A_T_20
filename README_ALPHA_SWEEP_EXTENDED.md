# Extended Alpha Sweep

This extends the alpha sweep you already ran. It assumes the earlier package has already added `backdoor_weight` support to `reconstruction_ce_direct_rounding_attack.py`.

New values: **32, 64, 128, 256**. Only alpha changes in `L = L_rec + alpha * L_bd`; all-weight rounding, 512 calibration images, 5000 steps, lr=0.01, augmentation and seed remain fixed.

Run:
```bash
python scripts/verify_alpha_sweep_extended_package.py
bash scripts/launch_alpha_sweep_extended_tmux.sh
tmux attach -t qura_alpha_sweep_ext
```

Choose GPU:
```bash
CUDA_VISIBLE_DEVICES=3 bash scripts/launch_alpha_sweep_extended_tmux.sh
```

Subset:
```bash
ALPHAS="32 64" bash scripts/run_alpha_sweep_extended.sh
```

Summary: `results/reconstruction_ce/alpha_sweep/alpha_sweep_extended_summary.md`
