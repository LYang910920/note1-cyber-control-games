# Training Summary

These runs are intentionally small enough for a laptop, but long enough to show the main convergence or stabilization signals.

## Timing Parameters

| Parameter | Value | Meaning |
|---|---:|---|
| Decision interval `Delta t` | 1.00 | Policies observe and act once per interval. |
| RK4 substeps per interval | 10 | Internal ODE solver steps, not extra MDP/MG decisions. |
| Policy-comparison horizon | 50 | Number of sampled decision epochs in each rollout. |

| Diagnostic | Start | End | Interpretation |
|---|---:|---:|---|
| FBSM max control change | 1.000e+00 | 7.446e-06 | Control updates shrink to 7.446e-06 of the initial change. |
| DDQN evaluation return | -76.608 | -16.052 | Stochastic policy learning is noisy, so inspect the rolling trend rather than one episode. |
| MADRL joint loss | -0.359 | 0.857 | Compact CTDE runs are stability diagnostics, not equilibrium proofs. |

## Representative Policy Comparison

Lower cumulative/peak/final compromised values and lower defender cost are better.

| Policy | Cumulative compromised | Peak compromised | Final compromised | Defender cost | Impulse events |
|---|---:|---:|---:|---:|---:|
| No defense | 20.131 | 0.749 | 0.206 | 203.93 | 0 |
| Always patch | 7.334 | 0.241 | 0.076 | 78.14 | 0 |
| Always clean | 4.200 | 0.302 | 0.009 | 50.50 | 0 |
| Adaptive hybrid | 7.865 | 0.254 | 0.097 | 89.06 | 7 |

For longer research runs, increase `--episodes`, run multiple seeds, and compare against no-defense and rule-based baselines.
