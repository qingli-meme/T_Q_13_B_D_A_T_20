# Final Fixed-Budget Method Patch

Base repository state:

```text
393cee4 Add final minimum-transport direct-rounding results
```

This patch makes one final methodological cleanup.

## Removed

The final method no longer performs:

```text
calibration harmonic-score checkpoint selection
best-state caching
best-state restore
"first saturated checkpoint" selection
test-set evaluation during optimization
```

The optimization protocol is now simply:

\[
\boxed{
\text{initialize at benign PTQ}
\rightarrow
\text{optimize the functional objective for a fixed budget}
\rightarrow
\text{materialize legal rounding}
}
\]

For the current experiment the fixed budget remains:

```text
5000 steps
```

This is an implementation setting, not a method component.

## Why

The previous clean implementation selected step 250 because calibration
agreement and ASR happened to both reach 100%, even though the functional
objective continued to improve substantially afterward.

The same run already showed:

```text
step 250   final-selection result:  CA 92.25 / ASR 100
step 500   diagnostic:              CA 92.71 / ASR 100
step 1000  diagnostic:              CA 92.78 / ASR 100
step 4000  diagnostic:              CA 92.85 / ASR 100
step 4500  diagnostic:              CA 92.86 / ASR 100
step 5000  diagnostic:              CA 92.75 / ASR 100
```

Therefore the 92.25 result reflected model-selection policy, not a weaker method.

A fixed optimization budget removes an unnecessary rule and prevents
calibration saturation from terminating optimization early.

## Test protocol

The final version does not inspect the test set during optimization.

It evaluates the test set only after the full 5000-step optimization:

1. once before final weight materialization;
2. once after materialization to assert exact behavioral equivalence.

These two evaluations happen after training is complete.

## Install

Extract into repository root, then:

```bash
python install_fixed_budget_final.py
python scripts/verify_fixed_budget_final.py
```

The installer backs up:

```text
mqbench/minimum_transport_direct_rounding_attack.py.pre_fixed_budget.bak
ours/main/configs/cv_4_4_minimum_transport_direct_rounding.yaml.pre_fixed_budget.bak
```

## Run

```bash
bash scripts/launch_fixed_budget_final_tmux.sh
```

Attach:

```bash
tmux attach -t qura_final_fixed
```

Or run directly:

```bash
bash scripts/run_fixed_budget_final.sh
```

## JSON changes

Old selection fields are removed:

```text
best_step
best_score
best_calibration
best_rounding_stats
best_test
```

They become:

```text
final_step
final_calibration
final_rounding_stats
final_test
```

`history` still contains:

```text
step 1, 250, 500, ..., 5000
```

but only calibration/functional diagnostics. No test metrics are recorded inside
the training history.

## Method status after this patch

The method body is now fixed:

1. frozen benign-PTQ classifier geometry;
2. target-interior depth estimated from benign target samples;
3. nearest sufficient endpoint by Euclidean projection;
4. clean preservation + triggered endpoint matching;
5. joint direct optimization of legal floor/ceil decisions;
6. fixed-budget optimization and final materialization.

SVD, Dykstra, STE, augmentation, Adam and the 5000-step budget are implementation
details rather than claimed method components.

After this run, do not continue modifying the method solely to improve CIFAR-10
numbers. Move to ablations, multi-seed robustness, bit-width / architecture /
dataset extension, and comparison against QuRA.
