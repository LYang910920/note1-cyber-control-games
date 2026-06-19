# Script Guide

Scripts are grouped by purpose: quick validation, figure generation, and longer tutorial diagnostics.

| Script | Purpose | Typical command | Outputs |
|---|---|---|---|
| `run_smoke_tests.sh` | Fast confidence check for every executable component. | `bash scripts/run_smoke_tests.sh` | Console output only |
| `generate_figures.py` | Rebuild static explanatory figures used by the README. | `python scripts/generate_figures.py` | `figures/*.png`, including `timing_semantics.png` |
| `run_training_iterations.py` | Run longer FBSM, DDQN, compact CTDE, policy-comparison, game-response, and node-level epidemic robustness diagnostics. | `python scripts/run_training_iterations.py` | `experiments/*.csv`, `experiments/training_diagnostic_glossary.md`, `experiments/training_summary.md`, `figures/training_iteration_diagnostics.png`, `figures/game_response_matrix.png`, `figures/node_level_learning_advantage.png` |

## Runtime Notes

`run_smoke_tests.sh` should finish quickly and is what GitHub Actions runs. If the virtual environment is not activated, pass the interpreter explicitly:

```bash
PYTHON=../.venv/bin/python bash scripts/run_smoke_tests.sh
```

`run_training_iterations.py` takes longer because it is meant to produce readable convergence or stabilization curves.

Use `--episodes` to change the length of the DDQN and compact CTDE tutorial runs:

```bash
python scripts/run_training_iterations.py --episodes 300
```

Use the heavier profile when you want a larger neural run on the local machine:

```bash
python scripts/run_training_iterations.py --profile gpu --device auto
```

The GPU-oriented profile increases DDQN width/depth, batch size, replay capacity, horizon, and episode count; it also increases the compact CTDE hidden width and horizon. If CUDA is unavailable, `--device auto` falls back to CPU.

The checked-in CSV files and figures are examples from one deterministic tutorial run.  For research claims, rerun with multiple seeds and report uncertainty.

## Timing Parameters To Notice

| Parameter | Where it appears | Meaning |
|---|---|---|
| `EnvConfig.dt` | `src/cyber_hybrid_env.py` | time between policy observations and decisions |
| `EnvConfig.substeps` | `src/cyber_hybrid_env.py` | RK4 substeps inside one transition, not extra MDP steps |
| `--episodes` | `run_training_iterations.py` | training episodes for DDQN and compact CTDE diagnostics |
| `horizon` | script defaults and CSV summaries | number of decision epochs in a rollout |

## Training Diagnostic Terms

`run_training_iterations.py` writes `experiments/training_diagnostic_glossary.md`. Open it next to `figures/training_iteration_diagnostics.png` when reading the curves: FBSM uses solver iterations, DDQN/CTDE use episodes, return is cumulative reward, and baseline comparison means a same-model rollout under fixed or learned policies.

The code defaults to fixed `EnvConfig.dt` for readability.  In the notation of the notes, a more general environment can use nonuniform `Delta t_k`; it should record the current `t_k`, next action time, and interval length in the transition diagnostics.

## What Each Script Needs

| Script | Needs | Expected result |
|---|---|---|
| `run_smoke_tests.sh` | installed dependencies from `requirements.txt` | exits with status 0 and unit-test summary |
| `generate_figures.py` | working NumPy/Matplotlib environment | rewrites explanatory PNG files under `figures/`, including the `t_k` versus `tau_j` timing diagram |
| `run_training_iterations.py` | working PyTorch/NumPy/Matplotlib environment | rewrites training CSVs, policy-comparison metrics, game-response metrics, node-level epidemic robustness metrics, summary markdown, and diagnostic PNGs |
