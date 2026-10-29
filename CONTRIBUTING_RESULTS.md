# Updating experiments

For each new experiment:

1. Put source/config changes in this repository.
2. Store compact JSON results under `ours/main/output/`.
3. Store text logs under `results/logs/`.
4. Update `RESULTS.md` with the final test metrics and artifact paths.
5. Do not commit datasets, virtual environments, secrets, or model checkpoints.

Before pushing, run the relevant compile/smoke checks and verify that no tracked
file exceeds GitHub's size limit.

