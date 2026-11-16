# Derivation and Self-Audit

The attack requirement contains only three facts: clean inputs should behave like benign PTQ, triggered inputs should predict target t, and the attacker may only alter legal quantization rounding choices. Any additional structure must be justified by a failure without it.

We rejected clean pseudo-label CE as the primary formulation because it has the wrong fixed point: even when the attacked clean model already equals benign PTQ, CE to the winning pseudo-label keeps increasing confidence. We also rejected clean logit MSE because absolute logits contain irrelevant common shifts and its regression scale is unrelated to target CE. KL from the benign PTQ distribution is the cleanest preservation term: it needs no labels, is zero at the benign model, and has zero gradient there.

We retain ordinary target CE for the triggered view. CE already increases target-vs-competitor logit differences and automatically focuses gradient on strong competitors. Adding a target probability threshold, a margin, or a confidence reserve would reintroduce by hand the quantity we are trying to test whether optimization can learn.

We also reject adaptive weighting between clean and trigger losses. Every source sample contributes one clean and one triggered desired behavior, so equal per-view averaging is the only coefficient-free default. If this simple empirical risk is insufficient, that is evidence against the no-target formulation rather than a reason to conceal the failure with tuned lambda.

We reject early stopping and best-checkpoint selection for the same reason. The fixed 5000-step budget tests whether the objective itself reaches a stable useful solution without looking at test performance or calibration ASR to choose a checkpoint.

Once classifier geometry is removed, freezing the classifier has no conceptual role. Therefore the threat-model-complete candidate optimizes all legal quantized weight rounding choices. A frozen-classifier run is retained only as a causal control so that removing target geometry is tested at the same manipulation scope as the current method.

If this method succeeds, the contribution cannot be claimed as a novel backdoor loss: KL distillation and target CE are standard. The substantive conclusion would instead be that the bottleneck in quantization backdoors is the realization of a global attack objective by discrete rounding decisions. Joint direct legal rounding makes a standard objective sufficient. If the same-scope no-target method fails, that conclusion is falsified and explicit functional target construction regains justification.
