# Training Summary

These runs are intentionally small enough for a laptop, but long enough to show the main convergence or stabilization signals.

## Experiment Configuration

| Item | Setting |
|---|---|
| Model | Hybrid malware/deception state `[S,I,R,z]` |
| Decision timing | observe at `t_k`, apply impulse jump if selected, integrate ODE to `t_{k+1}` |
| Defender actions | none, patch, clean, deceive, isolate |
| Attacker actions | scan, exploit, lateral, stealth |
| DDQN setting | 180 episodes, horizon 24, hidden width 64, learning rate 1e-3, gamma 0.99 |
| CTDE/MADRL setting | 180 episodes, horizon 18, hidden width 48, learning rate 5e-4, gamma 0.97 |

## Timing Parameters

| Parameter | Value | Meaning |
|---|---:|---|
| Decision interval `Delta t` | 1.00 | Policies observe and act once per interval. |
| RK4 substeps per interval | 10 | Internal ODE solver steps, not extra MDP/MG decisions. |
| Policy-comparison horizon | 50 | Number of sampled decision epochs in each rollout. |

| Diagnostic | Start | End | Interpretation |
|---|---:|---:|---|
| FBSM max control change | 1.000e+00 | 7.446e-06 | Control updates shrink to 7.446e-06 of the initial change. |
| DDQN evaluation return | -76.608 | -14.976 | Rolling evaluation improves by 60.898; inspect the rolling trend rather than one episode. |
| MADRL joint loss | -0.359 | 0.478 | Compact CTDE runs are stability diagnostics, not equilibrium proofs. |

## Representative Policy Comparison

Lower cumulative/peak/final compromised values and lower defender cost are better.

| Policy | Cumulative compromised | Peak compromised | Final compromised | Defender cost | Impulse events |
|---|---:|---:|---:|---:|---:|
| No defense | 20.131 | 0.749 | 0.206 | 203.93 | 0 |
| Fixed high patch | 7.334 | 0.241 | 0.076 | 78.14 | 0 |
| Fixed high clean | 4.200 | 0.302 | 0.009 | 50.50 | 0 |
| Rule threshold isolate/deceive/patch | 7.865 | 0.254 | 0.097 | 89.06 | 7 |
| DDQN learned defender (greedy) | 1.386 | 0.151 | 0.015 | 21.66 | 6 |

The learned DDQN policy has cumulative compromised exposure 1.386, compared with 4.200 for the best non-learning baseline in this run.

## Game Response Snapshot

`game_response_metrics.csv` evaluates defender policies against several attacker strategies.  The lowest cumulative compromised exposure in the matrix is 1.235, achieved by `DDQN learned defender (greedy)` against `Scripted scan -> exploit -> lateral attacker`.

For longer research runs, increase `--episodes`, run multiple seeds, and compare against no-defense and rule-based baselines.
