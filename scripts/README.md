# Script Guide

Scripts are grouped by purpose: quick validation, figure generation, and longer teaching diagnostics.

| Script | Purpose | Typical command | Outputs |
|---|---|---|---|
| `run_smoke_tests.sh` | Fast confidence check for every executable component. | `bash scripts/run_smoke_tests.sh` | Console output only |
| `generate_figures.py` | Rebuild static explanatory figures used by the README. | `python scripts/generate_figures.py` | `figures/*.png` |
| `run_training_iterations.py` | Run longer FBSM, DDQN, CTDE/MADRL, and policy-comparison diagnostics. | `python scripts/run_training_iterations.py` | `experiments/*.csv`, `experiments/training_summary.md`, `figures/training_iteration_diagnostics.png` |

## Runtime Notes

`run_smoke_tests.sh` should finish quickly and is what GitHub Actions runs.  `run_training_iterations.py` takes longer because it is meant to produce readable convergence or stabilization curves.

Use `--episodes` to change the length of the DDQN and MADRL teaching runs:

```bash
python scripts/run_training_iterations.py --episodes 300
```

The checked-in CSV files and figures are examples from one deterministic teaching run.  For research claims, rerun with multiple seeds and report uncertainty.

## Timing Parameters To Notice

| Parameter | Where it appears | Meaning |
|---|---|---|
| `EnvConfig.dt` | `src/cyber_hybrid_env.py` | time between policy observations and decisions |
| `EnvConfig.substeps` | `src/cyber_hybrid_env.py` | RK4 substeps inside one transition, not extra MDP steps |
| `--episodes` | `run_training_iterations.py` | training episodes for DDQN and CTDE/MADRL diagnostics |
| `horizon` | script defaults and CSV summaries | number of decision epochs in a rollout |

## What Each Script Needs

| Script | Needs | Expected result |
|---|---|---|
| `run_smoke_tests.sh` | installed dependencies from `requirements.txt` | exits with status 0 and unit-test summary |
| `generate_figures.py` | working NumPy/Matplotlib environment | rewrites the PNG files under `figures/` |
| `run_training_iterations.py` | working PyTorch/NumPy/Matplotlib environment | rewrites training CSVs, policy-comparison metrics, summary markdown, and the diagnostic PNG |
