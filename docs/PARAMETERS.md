# Parameter And Hyperparameter Reference

Use this page before changing the model, reward, or learner. It separates physical model parameters from neural-training hyperparameters.

## Quick Commands

| Need | Command |
|---|---|
| Print scenario and training profiles | `python src/scenario_profiles.py` |
| Fast smoke check | `bash scripts/run_smoke_tests.sh` |
| Rebuild figures | `python scripts/generate_figures.py` |
| Run longer diagnostics | `python scripts/run_training_iterations.py` |
| Run node-SIPRS MAPPO smoke | `python src/node_siprs_mappo.py --smoke --device cpu` |
| Run heavier GPU-oriented diagnostics | `python scripts/run_training_iterations.py --profile gpu --device auto` |

## Terms Used In This Repo

| Term | Meaning in Note 1 |
|---|---|
| `rollout` | One forward simulation under a fixed policy or learned policy: start from an initial state, apply actions at decision epochs `t_k`, integrate or jump to the next state, and record states, rewards, costs, and actions. |
| `episode` | One training rollout used by a learner such as DDQN or compact CTDE. In the code, an episode contains `horizon` sampled decision epochs. |
| `baseline` | A comparison method for the same model and metric, such as no defense, fixed defense, a rule policy, or an FBSM open-loop schedule. It is not a generic claim of optimality. |
| `nominal` | The parameter value assumed when designing a baseline. For the node-level robustness test, nominal beta is the underestimated `beta_assumed_by_fbsm=0.45`; the deployed simulator uses the larger true beta and bursts. |
| `robustness` | Performance under mismatch or disturbance. Here it means low infected-node exposure when the graph process has stochastic seeds, true propagation differs from the nominal FBSM design value, and a burst multiplier increases infection pressure. |
| `node_pmp_unknown_proxy` | A rough scale indicator for solving full node-level PMP/FBSM: `2 * (3 * nodes) * (horizon + 1)` counts state plus costate variables across the time grid. It is a teaching proxy, not a measured runtime. |
| `PMP` | Pontryagin's maximum principle, used to derive continuous-time optimal-control necessary conditions. |
| `FBSM` | Forward-backward sweep method, used here as a deterministic open-loop continuous-control baseline. |
| `DDQN` | Double deep Q-network, a value-based sampled-data defender for discrete actions. |
| `CTDE` | Centralized training, decentralized execution. Critics may see joint state/action information during training; actors use local observations at execution. |
| `MAPPO` | Multi-agent PPO. In this repo it is a compact cooperative community-defense baseline on node-level SIPRS dynamics. |
| `GAE` | Generalized advantage estimation, used by PPO/MAPPO to estimate lower-variance policy advantages from rollout rewards and value predictions. |

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
| Compact CTDE game baseline | `episodes=180`, `horizon=18`, `hidden=48`, `lr=5e-4`, `gamma=0.97`, `entropy_coef=0.02`, `seed=13` | `scripts/run_training_iterations.py::run_madrl` |
| Node-SIPRS MAPPO smoke | `nodes=24`, `communities=3`, `horizon=6`, `updates=2`, `rollout_steps=6`, `ppo_epochs=2`, `minibatch_size=3`, `hidden=32`, `clip_eps=0.2`, `gae_lambda=0.95` | `src/node_siprs_mappo.py --smoke` |
| DDQN GPU-oriented profile | `episodes=600`, `horizon=48`, `eval_episodes=8`, `batch_size=256`, `hidden=256`, `depth=3`, `lr=5e-4`, `gamma=0.995`, `buffer_size=100000`, `target_update=200`, epsilon decay `4000`, `device=auto` | `python scripts/run_training_iterations.py --profile gpu --device auto` |
| Compact CTDE GPU-oriented profile | `episodes=600`, `horizon=32`, `hidden=192`, `lr=3e-4`, `gamma=0.99`, `entropy_coef=0.015` | `python scripts/run_training_iterations.py --profile gpu` |
| DDQN smoke | `--smoke` keeps the run short for execution checks | `src/ddqn_cyber_defense.py` |
| Compact CTDE smoke | `--smoke` keeps the run short for execution checks | `src/madrl_ctde_hybrid_game.py` |

## Node-SIPRS MAPPO Parameters

| Parameter | Value |
|---|---:|
| default nodes | `48` |
| default communities | `3` |
| compartments | `[S,I,P,R]` |
| patch/clean semantics | patch `S -> P`; clean and natural recovery `I -> R`; waning `P/R -> S` |
| default horizon | `18` sampled decision epochs |
| ODE interval/substeps | `Delta t=0.5`, `substeps=4` |
| MAPPO core | GAE, clipped policy ratio, value loss, entropy bonus, minibatches, gradient clipping |

## Node-Level Robustness Parameters

This table defines the parameter-mismatch stress test. The FBSM baseline is solved with the nominal value below, then deployed on the stochastic node-level simulator with the true value and burst interval below.

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
| nominal beta used by FBSM | `0.45` |
| robustness metric | cumulative infected-node exposure, peak infected-node share, and final infected-node share over graph seeds |

## What To Change First

| Goal | First file |
|---|---|
| Change ODE rates or compartments | `src/cyber_dynamics.py` |
| Change impulse/jump behavior | `src/cyber_hybrid_env.py` |
| Change scenario defaults | `src/scenario_profiles.py` |
| Change DDQN/MADRL training hyperparameters | `scripts/run_training_iterations.py` |
| Move toward graph-scale experiments | `src/node_siprs_mappo.py` first, then `src/node_level_robustness.py` |
