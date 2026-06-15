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
| `training_summary.md` | First-versus-last diagnostic values and interpretation. |

The companion plot is saved as `figures/training_iteration_diagnostics.png`.

FBSM should show the clearest convergence. DDQN is stochastic, so inspect the rolling evaluation-return curve rather than one episode. CTDE/MADRL is a compact stability diagnostic, not a claim of Nash convergence.

## How To Read The CSV Files

Each row is a logged checkpoint rather than every optimizer step.  Use the first and last rows for a quick sanity check, then inspect the full curve in `figures/training_iteration_diagnostics.png`.

| Question | Column to inspect |
|---|---|
| Did FBSM settle? | `max_control_change` in `fbsm_iteration_history.csv` |
| Did DDQN improve? | `evaluation_return` and `epsilon` in `ddqn_training_history.csv` |
| Did CTDE remain numerically stable? | `loss`, `critic_loss`, and `entropy` in `madrl_training_history.csv` |
