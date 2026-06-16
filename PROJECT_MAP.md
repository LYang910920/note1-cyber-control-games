# Project Map

Use this page when you want the whole repository in one view.

## Big Picture

This repo connects four layers:

```text
lecture note
  -> continuous-time cyber dynamics
  -> baseline optimal control and sampled-data learning
  -> figures, CSV diagnostics, and smoke tests
```

The main question is:

> How can a continuous-time cyber-control model be turned into baseline optimal-control, RL, and MARL experiments that are small enough to read?

## Folder Roles

| Folder or file | Role | Read first when... |
|---|---|---|
| `START_HERE.md` | First-stop onboarding path. | You are new to the repo. |
| `README.md` | Main public-facing overview. | You want the summary, figures, and quick commands. |
| `docs/` | Lecture PDFs plus extension and cross-repo guides. | You want theory, learning path, or scale-up guidance. |
| `src/` | Core executable models. | You want to change dynamics, rewards, or learners. |
| `scripts/` | Reproducible commands for figures and diagnostics. | You want to regenerate outputs. |
| `experiments/` | Saved CSV histories and interpretation. | You want to inspect convergence/stability metrics. |
| `figures/` | Generated visual outputs. | You want visual sanity checks. |
| `tests/` | Small regression checks. | You want to know what behavior is protected. |

## Code Flow

```text
src/cyber_dynamics.py
  -> src/cyber_hybrid_env.py
      -> src/ddqn_cyber_defense.py
      -> src/madrl_ctde_hybrid_game.py

src/fbsm_malware_baseline.py
  -> scripts/generate_figures.py
  -> scripts/run_training_iterations.py
```

Read this as:

1. `cyber_dynamics.py` defines the ODE pieces.
2. `cyber_hybrid_env.py` turns the ODE into a sampled decision process.
3. `fbsm_malware_baseline.py` gives the continuous-control baseline.
4. `ddqn_cyber_defense.py` learns a single-agent defender.
5. `madrl_ctde_hybrid_game.py` learns an attacker-defender game loop.

## Command Flow

```text
pip install -r requirements.txt
  -> bash scripts/run_smoke_tests.sh
  -> python scripts/generate_figures.py
  -> python scripts/run_training_iterations.py
```

The smoke tests answer: does the code run?

The figure script answers: do the examples produce readable visual outputs?

The training diagnostic script answers: do the logged metrics move in a sensible direction?

## Output Flow

| Command | Output |
|---|---|
| `bash scripts/run_smoke_tests.sh` | console output and unit-test summary |
| `python scripts/generate_figures.py` | `figures/fbsm_malware_control.png`, `figures/hybrid_policy_*.png`, `figures/neural_architectures.png` |
| `python scripts/run_training_iterations.py` | `experiments/*.csv`, `experiments/training_summary.md`, `figures/training_iteration_diagnostics.png` |

## If You Want To Extend It

Start with `docs/EXTENDING.md`.

For the larger optimal-control and differential-game foundation, use:

https://github.com/LYang910920/network-control-differential-games
