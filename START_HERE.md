# Start Here

This page is the compact map. You can ignore most files at first.

## Big Picture

```text
lecture note
  -> continuous-time cyber dynamics
  -> original impulse points tau_j and sampled action points t_k
  -> FBSM baseline, DDQN, CTDE/MADRL
  -> figures and training diagnostics
```

## Timing In One Picture

```text
FBSM: continuous control u(t) on an ODE time grid
DDQN: observe at action point t_k, act once, jump/flow to t_{k+1}
MADRL: both players observe at t_k, choose joint actions, jump/flow to t_{k+1}
Impulses: original model events use tau_j; they may or may not coincide with t_k
```

Read `docs/MODEL_TO_MDP.md` if the difference between continuous time, impulse jumps, and MDP/MG decision epochs is the main question.

## Five-Minute Path

1. Open `docs/note1_game_learning_cyber_control.pdf` for the lecture narrative.
2. Read `docs/MODEL_TO_MDP.md` for the continuous/impulse/MDP timing convention.
3. Run `bash scripts/run_smoke_tests.sh` to check the environment.
4. Run `python scripts/generate_figures.py` to recreate the figures.
5. Run `python scripts/run_training_iterations.py` for longer diagnostics.
6. Read `docs/EXTENDING.md` when you want to scale the model.

## Folder Map

| Path | Purpose |
|---|---|
| `docs/` | lecture note, model-to-MDP guide, implementation notes, extension guide |
| `src/` | executable models and learning algorithms |
| `scripts/` | commands for figures, smoke tests, and diagnostics |
| `experiments/` | CSV histories and training summary |
| `figures/` | generated plots used by the README |
| `tests/` | small regression tests |

## Code Reading Order

1. `src/cyber_dynamics.py`: shared state conventions and ODE integration.
2. `src/cyber_hybrid_env.py`: sampled decisions, ODE flow, jump maps, rewards.
3. `src/evaluation_metrics.py`: policy rollouts and multi-metric comparison helpers.
4. `src/fbsm_malware_baseline.py`: continuous-control PMP/FBSM baseline.
5. `src/ddqn_cyber_defense.py`: single-agent sampled-data MDP learning.
6. `src/madrl_ctde_hybrid_game.py`: attacker-defender sampled-data Markov-game loop.
7. `src/node_level_robustness.py`: node-level epidemic-model parameter-mismatch experiment.

For command details, use `scripts/README.md`. For module details, use `src/README.md`.
