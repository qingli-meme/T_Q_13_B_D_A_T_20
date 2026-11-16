# FP Reconstruction + Target CE + Direct Legal Rounding

Base repository: current no-target branch containing Direct Legal Rounding.
The package was designed against commit `cd66f8e9ae4c6dd5ad84f1981e948f6add18e8bd`.

## Goal

Test the most standard unlabeled-calibration formulation:

- benign side: keep the ordinary PTQ reconstruction objective;
- malicious side: add ordinary targeted cross entropy;
- realization: jointly optimize all legal floor/ceil rounding choices.

No new clean-preservation mechanism is introduced.

## Threat model

Calibration data are unlabeled.
No ground-truth calibration labels are used.
No pseudo-labels are generated.

The original FP model is already available during PTQ and is used only as the
reference for benign reconstruction.

## Objective

For clean calibration input x:

\[
L_{rec}
=
\mathbb E_x\left[
\|f_q(x)-f_{FP}(x)\|_2^2
\right].
\]

For triggered input T(x):

\[
L_{bd}
=
\mathbb E_x\left[
CE(f_q(T(x)),t)
\right].
\]

Final:

\[
L=L_{rec}+L_{bd}.
\]

The reconstruction implementation matches the L2 form used in the repository's
optimized PTQ code (`lp_loss(..., p=2)`: sum squared error over output dimension,
then batch mean).

The backdoor weight is fixed to 1, matching the existing QuRA CV configuration
(`alpha: 1`). It is not tuned in this experiment.

## Why full-model output reconstruction here

QuRA/AdaRound reconstruct subgraphs sequentially because their optimization is
local/sequential. This experiment is intentionally global: all legal rounding
choices are optimized jointly. Therefore the benign target is the final FP
model output on the same calibration input. This keeps the clean objective a
plain PTQ reconstruction loss without reintroducing blockwise optimization.

If this global reconstruction is insufficient, a separate follow-up can test
block-output reconstruction. It should not be added before the simple global
formulation is tested.

## Explicitly absent

- ground-truth calibration labels
- pseudo-labels
- KL / distillation
- contrastive learning
- target margin / target depth
- target projection
- feature prototype
- sparsity / flip penalty
- soft-rounding regularizer
- checkpoint selection

The hard legal rounding parameterization itself already enforces floor/ceil at
forward time, so the AdaRound soft-rounding regularizer is unnecessary.

## Controlled comparison

The experiment keeps the same attack surface as `no_target/same_scope`:

- W4A4
- ResNet18 / CIFAR-10
- target 0
- same optimized 6x6 trigger
- same 512 calibration images
- classifier `linear` frozen
- Direct Legal Rounding
- Adam, lr 0.01
- grad clip 5
- pad4 + hflip0.5
- 5000 fixed steps
- seed 1005
- no test-set model selection

Thus the main comparison is clean-objective only:

1. KL + target CE
2. FP reconstruction + target CE

## Install

Extract into repository root and run:

```bash
python install_reconstruction_ce.py
python scripts/verify_reconstruction_ce_package.py
python scripts/test_reconstruction_ce_objective.py
```

## Run

```bash
bash scripts/launch_reconstruction_ce_tmux.sh
```

Attach:

```bash
tmux attach -t qura_recon_ce
```

Or directly:

```bash
bash scripts/run_reconstruction_ce.sh
```

## Outputs

JSON:

```text
ours/main/output/reconstruction_ce_direct_rounding_fp_logit_reconstruction_same_scope_resnet18_cifar10.json
```

Log:

```text
results/reconstruction_ce/logs/fp_logit_reconstruction_same_scope.log
```

Summary:

```text
results/reconstruction_ce/summary.md
```

## Interpretation

If reconstruction + target CE restores CA toward the previous geometry result
while keeping ASR approximately 100%, the final method can be simplified to:

> standard benign PTQ reconstruction + standard targeted CE + joint direct
> legal rounding.

If CA remains near the KL result, the clean-generalization gap is not caused by
KL specifically. The next justified experiment would be block-output PTQ
reconstruction, not target geometry or contrastive learning.
