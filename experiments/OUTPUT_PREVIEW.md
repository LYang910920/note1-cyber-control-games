# Output Preview

Use this page as the first stop after running `python scripts/run_training_iterations.py`.

## 1. Model And Timing

| Item | Value |
|---|---|
| Model | Hybrid malware/deception `[S,I,R,z]` |
| Default decision interval | `Delta t = 1.00` in this run; nonuniform `Delta t_k` is also valid |
| Solver substeps | `10` RK4 substeps per decision interval |
| Observation convention | policy sees pre-jump `x(t_k^-)`; next observation is `x(t_{k+1}^-)` |

## 2. Training Convergence

Open `figures/training_iteration_diagnostics.png`.

| Panel | What to check |
|---|---|
| FBSM baseline convergence | max control-update change should decay toward zero |
| DDQN sampled-data defender | rolling evaluation return should improve and stabilize |
| Compact CTDE attacker-defender diagnostics | loss and defender return should remain finite and interpretable |
| Hybrid malware policy comparison | DDQN should be competitive with or better than fixed policies |

## 3. Learning-Versus-Baseline Result

| Policy | Cumulative compromised | Defender cost | Peak compromised | Impulse events |
|---|---:|---:|---:|---:|
| DDQN learned defender (greedy) | 1.749 | 24.53 | 0.173 | 0 |
| Best non-learning baseline: Fixed high clean | 4.200 | 50.50 | 0.302 | 0 |

## 4. Game Response

Open `figures/game_response_matrix.png` and `experiments/game_response_metrics.csv`.

Best cell in this deterministic response matrix:

| Defender policy | Attacker strategy | Cumulative compromised |
|---|---|---:|
| DDQN learned defender (greedy) | Scripted scan -> exploit -> lateral attacker | 1.677 |

## 5. Node-Level Epidemic Model Robustness

Open `figures/node_level_learning_advantage.png` and `experiments/node_level_robustness_metrics.csv`.

In this section, **node-level** means each graph node carries a local S/I/R epidemic state.  The metric is aggregate infected-node exposure over action epochs, averaged over random graph seeds.  **Robustness** means behavior under nominal-vs-true beta mismatch and burst infection pressure.  `node_pmp_unknown_proxy` is only an approximate full-node PMP/FBSM variable count, not a reward or measured runtime.

| Method deployed on the node-level epidemic model | Mean cumulative infected-node exposure |
|---|---:|
| Node-level epidemic model: DDQN aggregate feedback | 1.481 |
| Node-level epidemic model: nominal-beta FBSM open-loop patching | 16.111 |

## 6. Files To Open First

| Category | File |
|---|---|
| Summary | `experiments/training_summary.md` |
| Learning curves | `figures/training_iteration_diagnostics.png` |
| Policy comparison CSV | `experiments/policy_comparison_metrics.csv` |
| Game matrix CSV | `experiments/game_response_metrics.csv` |
| Node-level epidemic robustness CSV | `experiments/node_level_robustness_metrics.csv` |
| Timing diagram and explanation | `figures/timing_semantics.png`, `docs/MODEL_TO_MDP.md` |
