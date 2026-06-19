# Training Summary

These runs use the `teaching` profile. The teaching profile is intentionally small enough for a laptop; the GPU profile increases neural width/depth, batch size, replay capacity, horizon, and episodes for a more demanding local run.

## Experiment Configuration

| Item | Setting |
|---|---|
| Model | Hybrid malware/deception state `[S,I,R,z]` |
| Decision timing | observe at action point `t_k`, apply any impulse jump, integrate ODE to `t_{k+1}^-` |
| Defender actions | none, patch, clean, deceive, isolate |
| Attacker actions | scan, exploit, lateral, stealth |
| DDQN setting | 180 episodes, horizon 24, hidden width 64, depth 2, batch 32, learning rate 0.001, gamma 0.99 |
| CTDE/MADRL setting | 180 episodes, horizon 18, hidden width 48, learning rate 0.0005, gamma 0.97 |

## Timing Parameters

| Parameter | Value | Meaning |
|---|---:|---|
| Default interval `Delta t` | 1.00 | This run uses a fixed interval; the model-to-MDP conversion also permits nonuniform `Delta t_k`. |
| RK4 substeps per interval | 10 | Internal ODE solver steps, not extra MDP/MG decisions. |
| Policy-comparison horizon | 50 | Number of sampled decision epochs in each rollout. |

| Diagnostic | Start | End | Interpretation |
|---|---:|---:|---|
| FBSM max control change | 1.000e+00 | 7.446e-06 | Control updates shrink to 7.446e-06 of the initial change. |
| DDQN evaluation return | -76.608 | -20.670 | Rolling evaluation improves by 55.977; inspect the rolling trend rather than one episode. |
| MADRL joint loss | 1.843 | -0.251 | Compact CTDE runs are stability diagnostics, not equilibrium proofs. |

## Representative Policy Comparison

Lower cumulative/peak/final compromised values and lower defender cost are better.

| Policy | Cumulative compromised | Peak compromised | Final compromised | Defender cost | Impulse events |
|---|---:|---:|---:|---:|---:|
| No defense | 20.131 | 0.749 | 0.206 | 203.93 | 0 |
| Fixed high patch | 7.334 | 0.241 | 0.076 | 78.14 | 0 |
| Fixed high clean | 4.200 | 0.302 | 0.009 | 50.50 | 0 |
| Rule threshold isolate/deceive/patch | 7.865 | 0.254 | 0.097 | 96.90 | 7 |
| DDQN learned defender (greedy) | 1.749 | 0.173 | 0.008 | 24.53 | 0 |

The learned DDQN policy has cumulative compromised exposure 1.749, compared with 4.200 for the best non-learning baseline in this run.

## Game Response Snapshot

`game_response_metrics.csv` evaluates defender policies against several attacker strategies.  The lowest cumulative compromised exposure in the matrix is 1.677, achieved by `DDQN learned defender (greedy)` against `Scripted scan -> exploit -> lateral attacker`.

## Node-Level Epidemic-Model Robustness Snapshot

`node_level_robustness_metrics.csv` evaluates the same idea on a 160-node random graph.  Here **node-level** means that each graph node has a local S/I/R epidemic state, while the plotted trajectory is the aggregate infected-node share observed at learning action epochs.  FBSM is solved as a low-dimensional open-loop control using nominal beta 0.45, then deployed on the node-level epidemic simulator, whose true beta is 1.25 with burst multiplier 1.35.  Mean cumulative infected-node exposure is 1.481 for DDQN aggregate feedback versus 16.111 for the nominal FBSM open-loop schedule.  This is the intended teaching case for why feedback learning can be easier to use when node-level dynamics or parameters are not accurately known.  The point is not that DDQN always beats FBSM; it is that a feedback policy can react to a state that an offline open-loop baseline did not predict.

For longer research runs, increase `--episodes`, run multiple seeds, and compare against no-defense and rule-based baselines.
