# Training Iteration Experiments

Run:

```bash
python scripts/run_training_iterations.py
```

The script performs longer, CPU-friendly tutorial runs and writes:

| File | Meaning |
|---|---|
| `fbsm_iteration_history.csv` | FBSM control-change, objective, peak compromised share, and mean control by sweep iteration. |
| `ddqn_training_history.csv` | DDQN training return, evaluation return, epsilon, replay size, and last TD loss by episode. |
| `madrl_training_history.csv` | Compact CTDE rollout length, defender/attacker return, joint loss, critic loss, and entropy by episode. |
| `policy_comparison_metrics.csv` | Multi-metric comparison of no-defense, fixed-defense, rule-based hybrid, and DDQN learned policies. |
| `game_response_metrics.csv` | Attacker-defender response matrix for defender policies against attacker strategies. |
| `node_level_robustness_metrics.csv` | Node-level epidemic-model robustness comparison for no defense, nominal-beta FBSM open-loop control, and DDQN aggregate feedback. |
| `node_siprs_mappo_smoke.csv` | Minimal cooperative MAPPO smoke history on canonical node-level SIPRS dynamics. |
| `OUTPUT_PREVIEW.md` | Categorized first-stop summary of model timing, training convergence, policy comparison, and game response. |
| `training_summary.md` | First-versus-last diagnostic values and interpretation. |

The companion plots are saved as `figures/training_iteration_diagnostics.png`, `figures/game_response_matrix.png`, and `figures/node_level_learning_advantage.png`.

FBSM should show the clearest convergence. DDQN is stochastic, so inspect the rolling evaluation-return curve rather than one episode. The compact CTDE run is a stability diagnostic, not a claim of Nash convergence. `node_siprs_mappo_smoke.csv` is a minimal MAPPO sanity check on canonical node-level SIPRS dynamics.

The policy-comparison and game-response CSV files use the same sampled-data timing as the environment: observe at action point `t_k`, apply any jump, integrate over `[t_k,t_{k+1})`, then measure the next observation.  The checked-in experiments use fixed `Delta t`; the notation in the tutorial also allows nonuniform intervals `Delta t_k`.

## Key Terms In The Result Files

| Term or column | How to read it |
|---|---|
| `rollout` | One complete forward simulation of a policy over the listed horizon. It is the object summarized into cumulative exposure, peak compromise, cost, and action counts. |
| `cumulative_compromised` | Time-summed infected or compromised share. Lower is better when comparing policies on the same model and horizon. |
| `nominal-beta FBSM` | An open-loop FBSM schedule computed using an assumed propagation rate, then deployed without re-solving when the node-level simulator uses a different true rate. |
| `beta_assumed_by_fbsm` | The nominal beta used to compute the FBSM schedule. It is deliberately lower than the true simulator beta in the robustness stress test. |
| `node_pmp_unknown_proxy` | Approximate state-plus-costate variable count for full node-level PMP/FBSM, used to explain scaling pressure. It is not a loss, reward, or measured runtime. |
| `robustness` | In this folder, robustness means behavior under parameter mismatch, random graph seeds, and burst infection pressure, not formal adversarial robustness. |

## How To Read The CSV Files

Each row is a logged checkpoint rather than every optimizer step.  Use the first and last rows for a quick sanity check, then inspect the full curve in `figures/training_iteration_diagnostics.png`.

| Question | Column to inspect |
|---|---|
| Did FBSM settle? | `max_control_change` in `fbsm_iteration_history.csv` |
| Did DDQN improve? | `evaluation_return` and `epsilon` in `ddqn_training_history.csv` |
| Did CTDE remain numerically stable? | `loss`, `critic_loss`, and `entropy` in `madrl_training_history.csv` |
| Which representative policy reduces exposure best? | `cumulative_compromised`, `peak_compromised`, `final_compromised`, and `total_defender_cost` in `policy_comparison_metrics.csv` |
| How do defender policies respond to different attackers? | `defender_policy`, `attacker_policy`, and `cumulative_compromised` in `game_response_metrics.csv` |
| When can feedback learning look better than nominal FBSM on a node-level epidemic model? | `cumulative_compromised`, `peak_compromised`, `beta_assumed_by_fbsm`, and `node_pmp_unknown_proxy` in `node_level_robustness_metrics.csv` |
| Does the node-SIPRS MAPPO smoke preserve the model contract? | `mean_reward`, `final_global_infected`, and `mass_error` in `node_siprs_mappo_smoke.csv` |

Start with `OUTPUT_PREVIEW.md` when you want the shortest categorized result page before reading every CSV file.
