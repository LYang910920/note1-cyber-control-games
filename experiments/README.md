# Training Iteration Experiments

Run:

```bash
python scripts/run_training_iterations.py
```

The script performs short, CPU-friendly runs and writes:

| File | Meaning |
|---|---|
| `fbsm_iteration_history.csv` | FBSM control-change, objective, peak compromised share, and mean control by sweep iteration. |
| `ddqn_training_history.csv` | DDQN training return, evaluation return, epsilon, replay size, and last TD loss by episode. |
| `madrl_training_history.csv` | CTDE/MADRL rollout length, defender/attacker return, joint loss, critic loss, and entropy by episode. |

The companion plot is saved as `figures/training_iteration_diagnostics.png`.
