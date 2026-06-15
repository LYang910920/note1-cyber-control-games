# Training Summary

These runs are intentionally small enough for a laptop, but long enough to show the main convergence or stabilization signals.

| Diagnostic | Start | End | Interpretation |
|---|---:|---:|---|
| FBSM max control change | 1.000e+00 | 7.446e-06 | Control updates shrink to 7.446e-06 of the initial change. |
| DDQN evaluation return | -76.608 | -16.052 | Stochastic policy learning is noisy, so inspect the rolling trend rather than one episode. |
| MADRL joint loss | -0.359 | 0.857 | Compact CTDE runs are stability diagnostics, not equilibrium proofs. |

For longer research runs, increase `--episodes`, run multiple seeds, and compare against no-defense and rule-based baselines.
