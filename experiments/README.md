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
