# Training Iteration Experiments

Run:

```bash
python scripts/run_training_iterations.py
```

The script performs longer, CPU-friendly teaching runs and writes:

| File | Meaning |
|---|---|
| `fbsm_iteration_history.csv` | FBSM control-change, objective, peak compromised share, and mean control by sweep iteration. |
| `ddqn_training_history.csv` | DDQN training return, evaluation return, epsilon, replay size, and last TD loss by episode. |
| `madrl_training_history.csv` | CTDE/MADRL rollout length, defender/attacker return, joint loss, critic loss, and entropy by episode. |
| `policy_comparison_metrics.csv` | Multi-metric comparison of no-defense, fixed-defense, rule-based hybrid, and DDQN learned policies. |
| `game_response_metrics.csv` | Attacker-defender response matrix for defender policies against attacker strategies. |
| `node_level_robustness_metrics.csv` | Node-graph robustness comparison for no defense, nominal-beta FBSM open-loop control, and DDQN aggregate feedback. |
| `OUTPUT_PREVIEW.md` | Categorized first-stop summary of model timing, training convergence, policy comparison, and game response. |
| `training_summary.md` | First-versus-last diagnostic values and interpretation. |

The companion plots are saved as `figures/training_iteration_diagnostics.png`, `figures/game_response_matrix.png`, and `figures/node_level_learning_advantage.png`.

FBSM should show the clearest convergence. DDQN is stochastic, so inspect the rolling evaluation-return curve rather than one episode. CTDE/MADRL is a compact stability diagnostic, not a claim of Nash convergence.

The policy-comparison and game-response CSV files use the same sampled-data timing as the environment: observe at action point `t_k`, apply any jump, integrate over `[t_k,t_{k+1})`, then measure the next observation.  The checked-in experiments use fixed `Delta t`; the notation in the lecture also allows nonuniform intervals `Delta t_k`.

## How To Read The CSV Files

Each row is a logged checkpoint rather than every optimizer step.  Use the first and last rows for a quick sanity check, then inspect the full curve in `figures/training_iteration_diagnostics.png`.

| Question | Column to inspect |
|---|---|
| Did FBSM settle? | `max_control_change` in `fbsm_iteration_history.csv` |
| Did DDQN improve? | `evaluation_return` and `epsilon` in `ddqn_training_history.csv` |
| Did CTDE remain numerically stable? | `loss`, `critic_loss`, and `entropy` in `madrl_training_history.csv` |
| Which representative policy reduces exposure best? | `cumulative_compromised`, `peak_compromised`, `final_compromised`, and `total_defender_cost` in `policy_comparison_metrics.csv` |
| How do defender policies respond to different attackers? | `defender_policy`, `attacker_policy`, and `cumulative_compromised` in `game_response_metrics.csv` |
| When can feedback learning look better than nominal FBSM? | `cumulative_compromised`, `peak_compromised`, `beta_assumed_by_fbsm`, and `node_pmp_unknown_proxy` in `node_level_robustness_metrics.csv` |

Start with `OUTPUT_PREVIEW.md` when you want the shortest categorized result page before reading every CSV file.
