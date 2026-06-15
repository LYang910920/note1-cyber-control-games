# Extending Note 1

This repository starts with compact cyber-control examples.  The extension path is to keep the same modeling contracts while replacing the small state, environment, and learner with richer versions.

## What You Need

| Need | Where it appears now |
|---|---|
| State variables and constraints | `src/cyber_dynamics.py`, `src/cyber_hybrid_env.py` |
| Continuous-time dynamics | `controlled_sir_rhs` and `hybrid_rhs` |
| Decision interface | `HybridCyberDefenseEnv.reset` and `HybridCyberDefenseEnv.step` |
| Objective or payoff | reward calculation in `HybridCyberDefenseEnv.step` |
| Baseline optimal-control solver | `solve_fbsm` in `src/fbsm_malware_baseline.py` |
| Learning loop | `src/ddqn_cyber_defense.py` and `src/madrl_ctde_hybrid_game.py` |

## What You Get

The current scripts produce:

| Output | Meaning |
|---|---|
| `figures/*.png` | Visual checks for dynamics, policies, and neural architectures |
| `experiments/*.csv` | Logged training or iteration diagnostics |
| `experiments/training_summary.md` | Short interpretation of the logged metrics |
| GitHub Actions smoke tests | Basic confidence that scripts and figures still run |

## Extension Path

1. **Change the state.** Add new compartments, budget variables, node features, or uncertainty states. Update projection or clipping logic so the state remains physically meaningful.
2. **Change the dynamics.** Replace `controlled_sir_rhs` or `hybrid_rhs` with a new `f(x,u,a,t)`. Keep the function signature simple and write a short smoke test before adding learning.
3. **Change the reward.** Decide which quantities should be minimized by the defender and maximized by the attacker. Add clear weights to `EnvConfig`.
4. **Add baselines before learning.** Compare no-defense, constant-defense, threshold, and FBSM-style policies before training DDQN or MARL.
5. **Scale the learner.** Move from the compact DDQN/MADRL files to a stronger library only after the environment contract is stable.
6. **Add diagnostics.** For stochastic policies, report multiple seeds, rolling means, confidence intervals, and baseline comparisons.

## Scaling To Network Models

The current state is compartmental: `S`, `I`, `R`, and a deception level.  A large network model usually needs one of these representations:

| Representation | Use when | Implementation direction |
|---|---|---|
| Degree-level state | nodes with similar degree can be grouped | track arrays by degree class and reuse PMP/FBSM patterns |
| Node-level state | individual devices have different risk or value | use vectors of node states and sparse adjacency matrices |
| Graph-feature state | policy should depend on topology | add node features and use a GNN policy or critic |
| Event-driven state | attacks/patches occur as discrete events | keep ODE flow but add jump maps and event queues |

For degree-level and node-level optimal-control patterns, compare with the companion repository:

https://github.com/LYang910920/network-control-differential-games

Useful pieces from that repository:

| Concept | Where to look there | How it helps here |
|---|---|---|
| Degree-k SIS control | `examples/lecture/code/simple_degree_k_control.py` | Shows how to replace scalar compartments with degree classes |
| Node-level control/game | `examples/lecture/code/network_control_examples.py` | Shows how to move from population shares to node vectors |
| Hybrid impulse simulation | `examples/lecture/code/network_control_examples.py` | Matches the jump-map idea in `cyber_hybrid_env.py` |
| Model adaptation checklist | `README.md` | Provides a disciplined order for changing dynamics and Hamiltonians |

## Research-Grade Checklist

Before treating a run as evidence rather than a teaching demo:

1. Run at least 5 to 10 random seeds.
2. Compare against no-defense, constant-defense, rule-based, and optimal-control baselines.
3. Separate training, validation, and stress-test scenarios.
4. Report wall-clock cost and sample count.
5. Track constraint violations, not only rewards.
6. For games, run unilateral-deviation or exploitability-style checks where possible.
7. Keep all dataset and third-party license notes with the experiment.
