# Target-Depth Final Slimming Suite

Base repository:

```text
393cee4 Add final minimum-transport direct-rounding results
```

The package also accepts the fixed-budget descendant.

This suite asks the last two structural slimming questions left in the method.

## Fixed mother method

Every run keeps exactly the same:

```text
Direct Legal Rounding
full feature-extractor rounding support
frozen benign-PTQ classifier
decision-relevant classifier subspace
nearest feasible Euclidean endpoint
clean preservation
trigger endpoint matching
all calibration samples
standard crop/flip augmentation
Adam
lr = 0.01
grad clip = 5
5000-step fixed budget
seed = 1005
```

Only the scalar target-depth definition changes.

## A0 — Reference

Current method:

\[
\gamma_{\rm ref}=\min_{j\ne t}\mathbb E[q_j(u_0(x))].
\]

This is rerun under exactly the same fixed-budget protocol as the two simplified variants.

## A1 — gamma = 0

Remove target-depth estimation entirely:

\[
\gamma=0.
\]

The endpoint is the nearest point in the closed target decision region.
No benign target statistics are used.

At zero depth, projected endpoints may lie exactly on classifier boundaries. Numerical argmax ties are therefore recorded rather than treated as a runtime error. The code does not add an epsilon.

Interpretation:

- If ASR remains around 99–100% with comparable CA, delete target-depth estimation entirely.
- If ASR drops materially, positive interior depth is structurally necessary.

## A2 — Mean per-sample target depth

Use the direct statistic:

\[
\gamma_{\rm sample}
=\mathbb E[\min_{j\ne t}q_j(u_0(x))].
\]

Because

\[
\mathbb E[\min_j q_j]\le \min_j\mathbb E[q_j],
\]

this is normally less conservative than the reference.

Interpretation:

- If it matches or improves the reference, use it as the final depth definition.
- If it materially weakens ASR/robustness, retain the reference.

## Install

Extract at repository root, then run:

```bash
python install_target_depth_ablation.py
python scripts/verify_target_depth_ablation_package.py
```

The installer does **not** modify the final-method runner. It creates a dedicated:

```text
mqbench/minimum_transport_depth_ablation.py
```

and adds one dedicated `minimum_transport_depth_ablation` branch in `main.py`.

If the source final runner is still the checkpoint-selected `393cee4` version, the copied ablation runner is automatically converted to the fixed-budget protocol. The final runner itself remains untouched.

## Run all three

```bash
bash scripts/launch_target_depth_ablation_tmux.sh
```

Attach:

```bash
tmux attach -t qura_depth_ablation
```

Or run directly:

```bash
bash scripts/run_target_depth_ablation_all.sh
```

Use another GPU if needed:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/launch_target_depth_ablation_tmux.sh
```

## Outputs

JSON:

```text
ours/main/output/depth_ablation_reference_resnet18_cifar10.json
ours/main/output/depth_ablation_gamma_zero_resnet18_cifar10.json
ours/main/output/depth_ablation_mean_sample_depth_resnet18_cifar10.json
```

Logs:

```text
results/depth_ablation/logs/reference.log
results/depth_ablation/logs/gamma_zero.log
results/depth_ablation/logs/mean_sample_depth.log
```

Summary:

```text
results/depth_ablation/summary.csv
results/depth_ablation/summary.md
```

Checkpoints are variant-specific and do not need to be pushed.

## Compare

Primary:

```text
Test CA
Test ASR
```

Mechanism:

```text
gamma
ideal target rate
clean code distance
trigger endpoint distance
actual target depth mean / p10 / min
flip ratio
flipped weights
```

For `gamma_zero`, also inspect:

```text
endpoint_geometry.ideal_target_rate
endpoint_geometry.ideal_depth_min
```

Boundary ties are part of the hypothesis, not an implementation bug, as long as projection feasibility itself passes.

## Decision rule

1. **gamma=0 succeeds** → delete target-depth estimation completely.
2. **gamma=0 fails, mean-sample-depth succeeds** → final method uses \(\gamma=\mathbb E[D_t(u)]\).
3. **both simplified variants are weaker** → retain \(\gamma=\min_j\mathbb E[q_j]\).

After this, freeze the method and move to formal paper ablations, multi-seed, bit-width, architecture/dataset extension, and QuRA comparison.

## Do not change during this suite

```text
steps
lr
seed
augmentation
rounding support
classifier freeze
projection solver
Direct Legal Rounding
clean loss
trigger loss
trigger
calibration size
```
