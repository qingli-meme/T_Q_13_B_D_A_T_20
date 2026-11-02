# QuRA Final Field-Support Pruning Suite

Expected repository state: `e51cc1e` or a descendant with the first slimming suite installed.

This is Round-2 slimming. Geometry and loss are frozen.

## Fixed parent method

Every experiment uses:

```text
endpoint       = V12 Dykstra minimum-transport deep-target projection
depth          = target-core depth
clean anchor   = ON
augmentation   = ON
sample scope   = ALL calibration samples
field rank     = 2
classifier     = frozen
```

Thus the only changing variable is:

```text
Which feature-extractor modules are allowed to carry the structured rounding field?
```

This avoids mixing field-support pruning with geometry/loss/capacity changes.

## Experiments

| ID | Variant | Change |
|---|---|---|
| F0 | `20_r2_all_reference` | rank-2 + all-samples parent reference |
| F1 | `21_r2_all_no_conv1` | freeze stem `conv1` |
| F2 | `22_r2_all_no_layer1` | freeze all `layer1.*` |
| F3 | `23_r2_all_no_layer2` | freeze all `layer2.*` |
| F4 | `24_r2_all_no_layer3` | freeze all `layer3.*` |
| F5 | `25_r2_all_no_layer4` | freeze all `layer4.*` |
| F6 | `26_r2_all_no_shortcuts` | freeze all `*.shortcut.*` projection convolutions |

Operating point:

```text
CA  >= 91.3%
ASR >= 99.0%
```

## Why leave-one-stage-out?

The previous `late34_only` failure only established:

> layer3+4 alone are insufficient.

It did **not** establish that every earlier stage is individually necessary.

Leave-one-stage-out directly asks whether a stage can be removed while all remaining stages compensate.

This is the correct next test for finding the minimum field support.

## Install

Extract this ZIP into the repository root.

Then:

```bash
python install_support_pruning.py
python scripts/verify_support_pruning_package.py
```

The installer patches only:

```text
mqbench/structured_slimming_attack.py
```

by extending `_select_attack_scope()`.

It does **not** touch `ours/main/main.py`.

A backup is created on first patch:

```text
mqbench/structured_slimming_attack.py.pre_support_pruning.bak
```

The installer is idempotent.

## Launch with tmux

From repository root:

```bash
bash scripts/launch_support_pruning_tmux.sh
```

Default session:

```text
qura_support_pruning
```

Attach:

```bash
tmux attach -t qura_support_pruning
```

Detach:

```text
Ctrl-b d
```

Use another GPU:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/launch_support_pruning_tmux.sh
```

## Run without tmux

```bash
bash scripts/run_support_pruning_all.sh
```

A failed experiment does not stop the rest.

## Outputs

JSON:

```text
ours/main/output/
slimming_20_r2_all_reference_resnet18_cifar10.json
...
slimming_26_r2_all_no_shortcuts_resnet18_cifar10.json
```

Logs:

```text
results/logs/support_pruning/
```

Status:

```text
results/support_pruning/status.tsv
```

Automatic summaries:

```text
results/support_pruning/summary.csv
results/support_pruning/summary.md
```

## What to push back

Please push:

```text
ours/main/output/slimming_2*_r2_all_*_resnet18_cifar10.json
results/logs/support_pruning/
results/support_pruning/status.tsv
results/support_pruning/summary.csv
results/support_pruning/summary.md
```

The large checkpoints do not need to be committed.

## How we will interpret the result

### If one stage can be frozen

That stage is not necessary and should be removed from the final method support.

### If multiple single-stage removals pass

Do **not** immediately claim they can all be removed simultaneously.

The next step is to combine only the successful removals and test the smallest joint support.

### If every removal fails

Then we have much stronger evidence that the rank-2 field needs distributed support across the network.

Even then, the final method should say:

> distributed multi-stage support is empirically necessary under the tested architecture

rather than claiming every individual weight/layer is theoretically necessary.

### Shortcuts

If `no_shortcuts` passes, remove shortcut projection fields from the final implementation. This is particularly attractive because it reduces support complexity without changing the main residual blocks.

## Final-method state if F0 passes

If the combination reference itself passes, the method already simplifies to:

```text
1. target reference depth
2. minimum-transport deep-target projection
3. clean anchor + absolute trigger endpoint
4. dynamic augmentation
5. rank-2 structured rounding field
6. all calibration samples
```

This suite determines only the minimum spatial support of item 5.
