# Final Method Derivation

The method has only two ideas.

## 1. Define the minimum sufficient malicious behavior

For frozen classifier \(z(h)=Wh+b\), restrict features to the exact row space of the centered classifier. For target \(t\), let

\[
q_j(u)=\frac{z_t(u)-z_j(u)}{\|w_t-w_j\|_2}
\]

be signed Euclidean distance to each target-vs-competitor boundary.

Estimate normal target depth:

\[
\gamma_t=\min_{j\ne t}\mathbb E[q_j(u_0(x))\mid \hat y_0(x)=t].
\]

Then define each triggered target by Euclidean projection:

\[
u^\star(x)=\Pi_{\{u:q_j(u)\ge\gamma_t,\forall j\ne t\}}(u_0(x)).
\]

This is the smallest feature displacement satisfying the target-interior requirement and therefore avoids an attack-strength tradeoff coefficient.

## 2. Jointly realize it using legal rounding decisions

For \(w_i/s_i=k_i+f_i\), choose only floor or ceil:

\[
r_i(a_i)=\mathbb I[f_i-1/2+a_i\ge0].
\]

The same global rounding configuration is optimized to preserve clean decision codes and realize the triggered projection:

\[
\min_a\;\mathbb E_x\left[\|u_a(x)-u_0(x)\|_2^2+\|u_a(Tx)-u^\star(x)\|_2^2\right].
\]

No low-rank factorization, bounded field, source-class routing, CE/KL loss, margin multiplier, sparsity penalty, or extra tradeoff coefficient is used.

The deployed model stores only the materialized legal quantized weights; \(a\) is discarded.
