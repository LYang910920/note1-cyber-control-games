# Parameter And Hyperparameter Reference

Use this page before changing the model, reward, or learner. It separates physical model parameters from neural-training hyperparameters.

## Quick Commands

| Need | Command |
|---|---|
| Print scenario and training profiles | `python src/scenario_profiles.py` |
| Fast smoke check | `bash scripts/run_smoke_tests.sh` |
| Rebuild figures | `python scripts/generate_figures.py` |
| Run longer diagnostics | `python scripts/run_training_iterations.py` |

## Scenario Parameters

| Scenario | State level | Control type | Horizon | `Delta t` | Main dynamics |
|---|---|---|---:|---:|---|
| `tutorial-hybrid-small` | aggregate `[S,I,R,z]` | hybrid continuous rates plus optional isolation impulse | `100` | `1.0` | `beta0=0.65`, `gamma=0.05`, `chi=0.70`, `zeta=0.08` |
| `impulse-visible-defense` | aggregate `[S,I,R,z]` | impulse-dominant hybrid control | `80` | `1.0` | `beta0=0.80`, `gamma=0.04`, lower isolation/usability costs |
| `paper-network-bridge` | aggregate now, bridge to graph state | hybrid control with stochastic initial states | `160` | `0.5` | `beta0=0.75`, `gamma=0.06`, stronger stress-test setting |

## FBSM And Solver Parameters

| Use | Values | Where |
|---|---|---|
| Solver defaults | `T=40`, `n=400`, `beta=0.8`, `gamma=0.15`, `A=10`, `B=1`, `A_terminal=20`, `u_max=1`, `max_iter=200`, `relax=0.5`, `tol=1e-5` | `src/fbsm_malware_baseline.py::solve_fbsm` |
| Long diagnostic run | `T=24`, `n=100`, `max_iter=35` | `scripts/run_training_iterations.py::run_fbsm` |
| Node-level nominal FBSM | `beta_assumed=0.45`, `gamma=0.15`, `A=6`, `B=6`, `A_terminal=12`, `max_iter=45` | `scripts/run_training_iterations.py::run_node_level_robustness` |

## Neural Training Hyperparameters

| Learner | Key hyperparameters | Source |
|---|---|---|
| DDQN defender | `episodes=180`, `horizon=24`, `eval_horizon=24`, `eval_episodes=4`, `batch_size=32`, `hidden=64`, `lr=1e-3`, `gamma=0.99`, `buffer_size=10000`, `target_update=80`, epsilon `1.0 -> 0.02` with decay `450`, `seed=11` | `scripts/run_training_iterations.py::run_ddqn` |
| CTDE/MADRL game | `episodes=180`, `horizon=18`, `hidden=48`, `lr=5e-4`, `gamma=0.97`, `entropy_coef=0.02`, `seed=13` | `scripts/run_training_iterations.py::run_madrl` |
| DDQN smoke | `--smoke` keeps the run intentionally tiny for execution checks | `src/ddqn_cyber_defense.py` |
| MADRL smoke | `--smoke` keeps the run intentionally tiny for execution checks | `src/madrl_ctde_hybrid_game.py` |

## Node-Level Robustness Parameters

| Parameter | Value |
|---|---:|
| nodes | `160` |
| horizon | `45` |
| mean degree | `8.0` |
| initial infected share | `0.05` |
| true base beta | `1.25` |
| recovery gamma | `0.035` |
| burst interval | `k=14` to `k=30` |
| burst multiplier | `1.35` |
| graph seeds | `21` to `28` |

## What To Change First

| Goal | First file |
|---|---|
| Change ODE rates or compartments | `src/cyber_dynamics.py` |
| Change impulse/jump behavior | `src/cyber_hybrid_env.py` |
| Change scenario defaults | `src/scenario_profiles.py` |
| Change DDQN/MADRL training hyperparameters | `scripts/run_training_iterations.py` |
| Move toward graph-scale experiments | `src/node_level_robustness.py` |
