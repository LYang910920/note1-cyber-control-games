# Start Here

This page is the compact map. You can ignore most files at first.

## Big Picture

```text
lecture note
  -> continuous-time cyber dynamics
  -> FBSM baseline, DDQN, CTDE/MADRL
  -> figures and training diagnostics
```

## Five-Minute Path

1. Open `docs/note1_game_learning_cyber_control.pdf` for the lecture narrative.
2. Run `bash scripts/run_smoke_tests.sh` to check the environment.
3. Run `python scripts/generate_figures.py` to recreate the figures.
4. Run `python scripts/run_training_iterations.py` for longer diagnostics.
5. Read `docs/EXTENDING.md` when you want to scale the model.

## Folder Map

| Path | Purpose |
|---|---|
| `docs/` | lecture note, implementation notes, extension guide |
| `src/` | executable models and learning algorithms |
| `scripts/` | commands for figures, smoke tests, and diagnostics |
| `experiments/` | CSV histories and training summary |
| `figures/` | generated plots used by the README |
| `tests/` | small regression tests |

## Code Reading Order

1. `src/cyber_dynamics.py`: shared state conventions and ODE integration.
2. `src/cyber_hybrid_env.py`: sampled decisions, ODE flow, jump maps, rewards.
3. `src/fbsm_malware_baseline.py`: continuous-control PMP/FBSM baseline.
4. `src/ddqn_cyber_defense.py`: single-agent defender learning.
5. `src/madrl_ctde_hybrid_game.py`: attacker-defender CTDE/MADRL training loop.

For command details, use `scripts/README.md`. For module details, use `src/README.md`.
