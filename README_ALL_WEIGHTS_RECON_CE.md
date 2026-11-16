# Reconstruction + Target CE + All-Weight Direct Legal Rounding

Validated against repository commit:

```text
c698405f5ce4b25467dab00037131df0ba03c069
```

This package changes only the legal-rounding search scope of the successful
Reconstruction + Target CE experiment.

Previous `same_scope` run:

```text
Benign PTQ: 93.35 CA / 2.08 ASR
Attack:     93.24 CA / 98.22 ASR
```

The previous run froze `linear` only because it inherited the historical
same-scope comparison with the target-geometry method.

The final-method candidate removes that artificial restriction.

## Objective: unchanged

Clean calibration remains the standard unlabeled reconstruction objective:

\[
L_{\rm rec}
=
\mathbb E_x
\left[
\|f_r(x)-f_{\rm FP}(x)\|_2^2
\right].
\]

Triggered samples use ordinary targeted cross entropy:

\[
L_{\rm bd}
=
\mathbb E_x
\left[
CE(f_r(T(x)),t)
\right].
\]

Total:

\[
L=L_{\rm rec}+L_{\rm bd}.
\]

Nothing is added to the loss.

## What changes

Previous:

```text
optimized: feature-extractor quantized weights
frozen:    linear
```

This experiment:

```text
optimized: every quantized weight with a legal adjacent floor/ceil choice
frozen:    none
```

In this ResNet18/CIFAR-10 setup, this includes the 8-bit `linear` classifier.

The direct-rounding constraints are unchanged:

\[
w_i/s_i=k_i+f_i,
\qquad
r_i\in\{0,1\},
\qquad
\hat w_i=s_i(k_i+r_i).
\]

Step 0 remains exactly benign nearest-rounding PTQ.

## Why this is the natural final candidate

Once target-decision geometry is removed, the classifier has no privileged
status. If the threat model allows the attacker to control legal weight
rounding decisions during quantization, freezing one quantized layer is an
extra method restriction rather than part of the threat model.

The existing reconstruction loss already penalizes harmful classifier changes
through the final clean logits. Therefore the clean objective itself can
decide how much classifier rounding freedom is useful.

## Controlled comparison

Everything except the rounding scope is fixed:

```text
W4A4 setup used by current experiment
ResNet18 / CIFAR-10
target = 0
same optimized 6x6 trigger
same 512 unlabeled calibration images
same activation quantization
same augmentation: pad4 + hflip0.5
same Direct Legal Rounding
Adam
lr = 0.01
grad clip = 5
5000 fixed steps
seed = 1005
no checkpoint selection
no test-set selection during training
```

The experiment therefore directly asks:

> Does allowing the classifier's legal rounding decisions close the remaining
> test-ASR gap without sacrificing the reconstruction-protected clean utility?

## Install

Extract into the repository root:

```bash
python install_reconstruction_ce_all_weights.py
python scripts/verify_reconstruction_ce_all_weights_package.py
```

The installer patches the existing reconstruction runner so classifier
freezing becomes configurable.

Backward compatibility is preserved: an old config with no
`freeze_classifier` field defaults to `True`, so the previous same-scope
experiment remains unchanged.

## Run

```bash
bash scripts/launch_reconstruction_ce_all_weights_tmux.sh
```

Attach:

```bash
tmux attach -t qura_recon_ce_all
```

Or run directly:

```bash
bash scripts/run_reconstruction_ce_all_weights.sh
```

## Outputs

JSON:

```text
ours/main/output/reconstruction_ce_direct_rounding_fp_logit_reconstruction_all_weights_resnet18_cifar10.json
```

Log:

```text
results/reconstruction_ce/logs/fp_logit_reconstruction_all_weights.log
```

Comparison summary:

```text
results/reconstruction_ce/scope_summary.md
results/reconstruction_ce/scope_summary.json
```

The summary compares the existing `same_scope` result against `all_weights`
and verifies whether `linear` is actually in the optimized module list.

## Expected interpretation

Best case:

```text
CA ~93.1-93.3
ASR ~99.5-100
```

Then the final method should simply optimize all legal rounding decisions and
drop the classifier-freeze restriction entirely.

If ASR improves but CA drops materially, retain the result as a scope ablation
rather than automatically adopting it. The reconstruction objective remains
unchanged; no new regularizer should be added just to compensate.

If the result is similar to same_scope, use all-weights anyway only if the
threat-model simplicity is worth the negligible performance difference. Do
not claim classifier manipulation itself as a contribution.
