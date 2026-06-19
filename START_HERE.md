# Start Here

This page is the compact map. You can ignore most files at first.

## Big Picture

```text
Foundation repository
  -> notation, ODE models, shared cybercontrol helpers
Companion Note 1
  -> continuous-time cyber dynamics
  -> original impulse points tau_j and sampled action points t_k
  -> FBSM baseline, DDQN, compact CTDE, node-SIPRS MAPPO
  -> figures and training diagnostics
```

## Three-Repository Order

| Step | Repository | What to use it for |
|---:|---|---|
| 0 | `network-control-differential-games` | Foundation notation, shared package, continuous/impulse/hybrid worked examples, and degree-level/node-level scalability. |
| 1 | `note1-cyber-control-games` | This note: PMP/FBSM, sampled-data MDPs, DDQN, a compact CTDE attacker-defender baseline, and node-SIPRS MAPPO diagnostics. |
| 2 | `note2-pinn-pidl-cyber-control` | PINN/PIDL, inverse learning, neural control, and PMP-informed residual learning. |

## Timing In One Picture

```text
FBSM: continuous control u(t) on an ODE time grid
DDQN: observe at action point t_k, act once, jump/flow to t_{k+1}
Markov-game learning: players observe at t_k, choose joint actions, jump/flow to t_{k+1}
Impulses: original model events use tau_j; they may or may not coincide with t_k
```

Read `docs/MODEL_TO_MDP.md` if the difference between continuous time, impulse jumps, and MDP/MG decision epochs is the main question.

For a slide-based overview of the full three-repository path, open the student onboarding deck in the foundation repository: <https://github.com/LYang910920/network-control-differential-games/tree/main/docs/slides/three_repo_student_onboarding>.

## Five-Minute Path

1. Open `docs/note1_game_learning_cyber_control.pdf` for the tutorial narrative.
2. Read `docs/MODEL_TO_MDP.md` for the continuous/impulse/MDP timing convention.
3. Run `python src/scenario_profiles.py` to see the student-facing scenario profiles.
4. Open `docs/PARAMETERS.md` before changing model parameters, DDQN hyperparameters, compact CTDE settings, or MAPPO hyperparameters. It also defines rollout, nominal beta, robustness, and `node_pmp_unknown_proxy`.
5. Run `python src/node_siprs_mappo.py --smoke --device cpu` for the canonical node-SIPRS community MAPPO smoke check.
6. For a heavier local/GPU diagnostic, run `python scripts/run_training_iterations.py --profile gpu --device auto` after the smoke tests pass.
7. Read `docs/PAPER_WORKFLOW.md` when turning an example into a paper section.
8. Run `bash scripts/run_smoke_tests.sh` to check the environment.
9. Run `python scripts/generate_figures.py` to recreate the figures.
10. Run `python scripts/run_training_iterations.py` for longer diagnostics.
11. Read `docs/EXTENDING.md` when you want to scale the model.

## Folder Map

| Path | Purpose |
|---|---|
| `docs/` | tutorial note, model-to-MDP guide, implementation notes, extension guide |
| `docs/PARAMETERS.md` | model parameters, solver values, and neural-training hyperparameters |
| `docs/PAPER_WORKFLOW.md` | paper workflow for FBSM, ODE-RL, DDQN, compact CTDE, and node-SIPRS MAPPO |
| `src/` | executable models and learning algorithms |
| `scripts/` | commands for figures, smoke tests, and diagnostics |
| `experiments/` | CSV histories and training summary |
| `figures/` | generated plots used by the README |
| `tests/` | small regression tests |

## Code Reading Order

1. `src/shared_setup.py`: local helper that finds the shared foundation package in a sibling workspace.
2. `src/cyber_dynamics.py`: compatibility wrapper around shared `cybercontrol` dynamics and numerics.
3. `src/cyber_hybrid_env.py`: sampled decisions, ODE flow, jump maps, rewards.
4. `src/scenario_profiles.py`: named starting points for adapting the code to paper-style scenarios.
5. `src/evaluation_metrics.py`: policy rollouts and multi-metric comparison helpers.
6. `src/fbsm_malware_baseline.py`: continuous-control PMP/FBSM baseline.
7. `src/ddqn_cyber_defense.py`: single-agent sampled-data MDP learning.
8. `src/madrl_ctde_hybrid_game.py`: compact attacker-defender sampled-data Markov-game baseline.
9. `src/node_siprs_mappo.py`: cooperative community MAPPO smoke baseline on canonical node-level SIPRS dynamics.
10. `src/node_level_robustness.py`: node-level epidemic-model parameter-mismatch experiment.

For command details, use `scripts/README.md`. For module details, use `src/README.md`.
