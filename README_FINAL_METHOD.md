# Minimum Transport + Direct Legal Rounding — Final Clean Implementation

Expected base: `qingli-meme/T_Q_13_B_D_A_T_20`, commit `296308e` or compatible descendant.

## Final method

Frozen benign-PTQ classifier:

\[
z(h)=Wh+b.
\]

Use only the classifier decision-relevant row space and denote its coordinates by \(u(h)\).
For target \(t\) and competitor \(j\), define exact signed boundary distance

\[
q_j(u)=\frac{z_t(u)-z_j(u)}{\|w_t-w_j\|_2}.
\]

Estimate target interior depth directly from benign target samples:

\[
\gamma_t=\min_{j\ne t}\mathbb E_{x:\hat y_0(x)=t}[q_j(u_0(x))].
\]

There is no target centroid/core in the method. This is exactly equivalent to the old centroid depth because every \(q_j\) is affine.

For each benign calibration sample:

\[
u^\star(x)=\arg\min_u \tfrac12\|u-u_0(x)\|_2^2
\quad\text{s.t.}\quad q_j(u)\ge\gamma_t,\ \forall j\ne t.
\]

Optimize one joint functional objective:

\[
\mathcal L=\mathbb E_x[\|u_a(x)-u_0(x)\|_2^2+\|u_a(Tx)-u^\star(x)\|_2^2].
\]

For each normalized weight \(w_i/s_i=k_i+f_i\), directly optimize only its legal adjacent rounding choice:

\[
r_i(a_i)=\mathbb I[f_i-\tfrac12+a_i\ge0].
\]

Offsets start at zero, so step 0 is exactly benign nearest rounding. Forward is always hard legal floor/ceil; backward uses the already implemented identity STE. After optimization all temporary offsets are deleted.

## Removed from the method

- target core / centroid
- all class cores
- target-sample filtering
- low-rank field / rank
- bounded dense field / delta bound / tanh / temperature
- core-ray / endpoint-mode switches
- depth-mode switches
- attack-scope / support-pruning switches
- square/invertible target-margin matrix requirement
- `decision_rank == C-1` requirement
- V6/V7/V11 displacement comparisons

SVD basis and Dykstra remain numerical implementation details only.

## Extra implementation cleanup

The previous Direct checkpoint was ~190 MB because temporary `rounding_offset` parameters and weight-sized buffers were still saved after materialization.

This implementation materializes the final legal weights, removes the offsets/buffers, then requires pre-finalize and post-finalize CA/ASR to be identical. The saved checkpoint should therefore be substantially smaller.

## Install

Extract at repo root, then:

```bash
python install_final_method.py
python scripts/verify_final_method_package.py
```

The installer adds one independent `quantize_type` branch and does not modify/delete old V1–V12, slimming, or support-pruning experiments.

## Run

```bash
bash scripts/launch_final_method_tmux.sh
```

Attach:

```bash
tmux attach -t qura_final
```

Or run without tmux:

```bash
bash scripts/run_final_method.sh
```

## Output

JSON:

```text
ours/main/output/minimum_transport_direct_rounding_resnet18_cifar10.json
```

Log:

```text
results/logs/minimum_transport_direct_rounding_cifar10.log
```

Checkpoint:

```text
ours/main/model/resnet18+cifar10.minimum_transport_direct_rounding_t0.pth
```

## Expected reference

Previous Direct at `296308e`:

```text
Best step                  1000
CA                         92.68%
ASR                        100.00%
Calibration agreement      100%
Calibration ASR            100%
Clean code distance        0.1015
Trigger endpoint distance  0.2359
Actual min target depth    3.4967
```

This clean implementation changes only mathematically redundant/historical machinery, so it should remain in the same operating region. A material drop should be treated as a bug or unintended behavioral change, not as an accepted method difference.

Please return the JSON + log, and report the new checkpoint file size. The checkpoint itself does not need to be pushed.
