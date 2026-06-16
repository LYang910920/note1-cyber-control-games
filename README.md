# Cyber Control and Game Learning, Note 1

Executable companion for **Note 1: Game Learning for Cyber Control**.  The repo keeps the teaching path small: continuous-time cyber dynamics, an FBSM optimal-control baseline, DDQN defense learning, and a compact attacker-defender CTDE/MADRL example.

If this is your first visit, start with `START_HERE.md`.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
bash scripts/run_smoke_tests.sh
python scripts/generate_figures.py
```

For longer convergence/stability diagnostics:

```bash
python scripts/run_training_iterations.py
```

## Repository Guide

| Need | Open |
|---|---|
| Short orientation | `START_HERE.md` |
| Lecture narrative | `docs/note1_game_learning_cyber_control.pdf` |
| Source-code map | `src/README.md` |
| Script and output map | `scripts/README.md` |
| Training curves and CSVs | `experiments/README.md` |
| Extensions and scaling | `docs/EXTENDING.md` |
| License and attribution | `LICENSE`, `NOTICE.md` |

## Core Flow

```text
cyber ODE dynamics
  -> FBSM continuous-control baseline
  -> hybrid ODE/RL environment
  -> DDQN defender
  -> CTDE/MADRL attacker-defender game
```

![Neural architectures](figures/neural_architectures.png)

## Main Outputs

| Output | Purpose |
|---|---|
| `figures/fbsm_malware_control.png` | FBSM state and patching-control baseline |
| `figures/hybrid_policy_comparison.png` | no-defense, fixed-defense, and adaptive-policy comparison |
| `figures/hybrid_policy_rollout.png` | one hybrid rollout with defender and attacker actions |
| `figures/training_iteration_diagnostics.png` | longer FBSM, DDQN, and CTDE/MADRL diagnostics |
| `experiments/*.csv` | logged histories behind the training plot |

## Validation

`bash scripts/run_smoke_tests.sh` runs the fast local check.  GitHub Actions repeats the smoke tests and regenerates figures on each push or pull request.

These examples are teaching code, not benchmark implementations.  For research use, add multiple seeds, stronger baselines, full logging, and game-specific exploitability or unilateral-deviation checks.

## Related Repository

For the optimal-control and differential-game foundation behind the FBSM and hybrid-control pieces, see https://github.com/LYang910920/network-control-differential-games.

## License And Copyright

Released under the MIT License.  See `LICENSE` for terms and `NOTICE.md` for copyright, dependency, and attribution notes.
