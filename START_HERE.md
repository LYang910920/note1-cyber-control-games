# Start Here

This repository is designed to be read in layers.  You do not need to understand every file before running the examples.

## Five-Minute Path

1. Open `docs/note1_game_learning_cyber_control.pdf` for the lecture narrative.
2. Read `README.md` for the high-level map and generated figures.
3. Run `bash scripts/run_smoke_tests.sh` to check the environment.
4. Run `python scripts/generate_figures.py` to recreate the explanatory figures.
5. Run `python scripts/run_training_iterations.py` when you want the longer convergence diagnostics.
6. Read `docs/EXTENDING.md` when you want to move beyond the small teaching models.

## Find The Right File

| If you want to... | Open |
|---|---|
| Understand the equations first | `docs/note1_game_learning_cyber_control.pdf` |
| See the recommended reading order | `docs/README.md` |
| Understand the Python modules | `src/README.md` |
| Know which command generates which output | `scripts/README.md` |
| Inspect convergence and training metrics | `experiments/README.md` and `experiments/training_summary.md` |
| Connect this repo to the differential-games repo | `docs/LEARNING_PATH.md` |
| Extend to complex cyber or network models | `docs/EXTENDING.md` |
| Check the license and copyright assumptions | `LICENSE` and `NOTICE.md` |

## Recommended Code Reading Order

1. `src/cyber_dynamics.py`: shared state conventions and ODE integration.
2. `src/cyber_hybrid_env.py`: sampled decisions, ODE flow, jump maps, rewards.
3. `src/fbsm_malware_baseline.py`: continuous-control PMP/FBSM baseline.
4. `src/ddqn_cyber_defense.py`: single-agent defender learning.
5. `src/madrl_ctde_hybrid_game.py`: attacker-defender CTDE/MADRL training loop.

## Common First Changes

| Change | Where |
|---|---|
| Shorten or lengthen an episode | `EnvConfig.horizon` in `src/cyber_hybrid_env.py`, or CLI `--horizon` where available |
| Change malware propagation | `MalwareParams` or `HybridParams` in `src/cyber_dynamics.py` |
| Change reward trade-offs | `EnvConfig` in `src/cyber_hybrid_env.py` |
| Run longer DDQN/MADRL diagnostics | `python scripts/run_training_iterations.py --episodes 300` |
| Add a new baseline policy | `scripts/generate_figures.py` or a new script under `scripts/` |
| Move to degree-level or node-level networks | `docs/EXTENDING.md` plus `network-control-differential-games` |
