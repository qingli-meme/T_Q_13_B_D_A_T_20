# No-Target Direct Legal Rounding

Validated against remote main:

```text
e62ece145221a3a3034ce1daf904760bd788125b
Add target-depth slimming results
```

This package is a structural death test for the entire target-decision framework. It does not simplify the old target-depth/projection pipeline; it derives a different method directly from the attack requirement.

## Final candidate

For a calibration input x, let the benign nearest-rounding PTQ model produce

\[
p_0(x)=\operatorname{softmax}(z_0(x)).
\]

The malicious quantized model produces \(p_a\). The same source sample is used in two views:

\[
x \mapsto p_0(x),
\qquad
T(x) \mapsto e_t,
\]

where \(e_t\) is the one-hot target-class distribution. The objective is

\[
\mathcal L
=
\frac12\mathbb E_x
\left[
\mathrm{KL}(p_0(x)\|p_a(x))
+
\mathrm{CE}(p_a(T(x)),t)
\right].
\]

There is no target depth, target margin, classifier decision subspace, feature endpoint, projection, target centroid, target-class calibration statistic, or clean/attack loss coefficient. The factor 1/2 is only the average over one clean and one triggered view per source sample.

The clean term uses the full benign PTQ predictive distribution rather than a pseudo-label. This matters because pseudo-label CE still pushes an already-correct clean sample toward higher confidence, whereas KL has an exact fixed point at the benign PTQ behavior:

\[
p_a(x)=p_0(x)
\Longrightarrow
\nabla \mathcal L_{clean}=0.
\]

The trigger term is ordinary target CE. It already optimizes target-vs-competitor logit differences through

\[
-\log p_t
=
\log\left(1+\sum_{j\ne t}e^{z_j-z_t}\right),
\]

so if joint legal rounding can realize the attack, explicit depth estimation is unnecessary.

## Direct legal rounding

For every quantized weight,

\[
w_i/s_i=k_i+f_i,
\]

we optimize only its legal adjacent floor/ceil choice:

\[
r_i(a_i)=\mathbb I[f_i-1/2+a_i\ge0].
\]

All \(a_i\) start at zero, so step 0 is exactly benign nearest-rounding PTQ. Forward uses hard legal rounding and backward uses the repository's existing identity STE. After the fixed training budget, the hard decisions are materialized into quantized weights and all temporary offsets/buffers are deleted.

## Why there are two runs

### same_scope

The final classifier `linear` is frozen, matching the exact manipulation surface of the current geometry method. This is the decisive causal test of whether target-decision geometry itself is needed.

### all_weights

Every legal quantized weight rounding choice is optimized, including `linear`. This is the method derived from the threat model once fixed classifier geometry is no longer needed. It is the preferred final no-target candidate, while `same_scope` prevents us from falsely attributing any gain to the larger attack surface.

## Fixed settings

Both runs keep W4A4, ResNet18/CIFAR-10, target 0, the same optimized 6x6 trigger, the same 512 calibration examples, the same augmentation, Adam, lr 0.01, grad clipping 5, 5000 fixed steps, and seed 1005. No test-set evaluation occurs during training and no best-checkpoint selection is used.

## Install

Extract into repository root, then run:

```bash
python install_no_target.py
python scripts/verify_no_target_package.py
```

Optional objective-only sanity check:

```bash
python scripts/test_no_target_objective.py
```

## Run

```bash
bash scripts/launch_no_target_tmux.sh
```

Attach:

```bash
tmux attach -t qura_no_target
```

Or run directly:

```bash
bash scripts/run_no_target_all.sh
```

## Outputs

```text
ours/main/output/no_target_direct_rounding_same_scope_resnet18_cifar10.json
ours/main/output/no_target_direct_rounding_all_weights_resnet18_cifar10.json
results/no_target/logs/same_scope.log
results/no_target/logs/all_weights.log
results/no_target/summary.md
results/no_target/summary.csv
```

If the existing mean-sample-depth result is present, the summary also includes it as `geometry_reference`.

## Decision logic

If `same_scope` reaches roughly the current geometry result (~92.75 CA / 100 ASR), delete the entire target-decision framework. The result would show that the essential improvement came from global joint realization of legal rounding decisions, not from explicit target geometry.

If `same_scope` is weak but `all_weights` is strong, a no-target method is still viable under the full legal-rounding threat model, but the paper must state clearly that classifier rounding is part of the attack surface.

If both are weak, the target-decision formulation survives the death test; only then is there evidence that explicit functional target construction provides something standard target CE cannot learn under the same direct rounding realization.

If both are strong and `all_weights` is not better, prefer the smaller `same_scope` manipulation surface. If both are strong and `all_weights` is clearly better, use `all_weights` as the simplest threat-model-complete method and keep `same_scope` as the causal control.
