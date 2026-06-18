# Source Code Guide

The `src/` folder mirrors the lecture sequence.  Each file is intentionally small enough to read directly.

## State Convention

Most examples use population shares:

| Symbol | Meaning |
|---|---|
| `S` | susceptible or vulnerable devices |
| `I` | infected, compromised, or active-malware devices |
| `R` | recovered, cleaned, patched, or protected devices |
| `z` | deception level in the hybrid environment |

SIR states are projected back to the probability simplex after numerical integration so small RK4 drift does not accumulate.

## Timing Convention

The hybrid environment turns continuous and impulsive cyber dynamics into a sampled-data learning problem:

```text
s_k = x(t_k^-)
  -> action(s)
  -> jump_map gives x(t_k^+)
  -> RK4 integrates the ODE to x(t_{k+1}^-)
  -> s_{k+1}
```

`EnvConfig.dt` is the decision interval. `EnvConfig.substeps` is only the number of RK4 solver steps inside one transition.  DDQN sees one replay item per decision interval; CTDE/MADRL sees one joint transition per decision interval.

This teaching environment uses fixed `t_k = k * dt`.  The notes also support nonuniform action intervals `Delta t_k = t_{k+1} - t_k`.  If the original model already has impulse/event points, call them `tau_j`; do not reuse `t_k` for those model-intrinsic times unless they are intentionally the same points.

## Module Map

| File | Main purpose | Useful entry points |
|---|---|---|
| `cyber_dynamics.py` | Shared dynamics and integration utilities. | `rk4_integrate`, `controlled_sir_rhs`, `hybrid_rhs` |
| `cyber_hybrid_env.py` | Plain-Python reset/step environment for sampled hybrid cyber defense. | `HybridCyberDefenseEnv`, `EnvConfig`, `scripted_attacker` |
| `evaluation_metrics.py` | Shared rollout, policy-comparison, and game-response metrics. | `evaluate_policy_suite`, `evaluate_game_response_matrix`, `summarize_rollout` |
| `fbsm_malware_baseline.py` | Forward-backward sweep method for a PMP open-loop control baseline. | `solve_fbsm` |
| `ddqn_cyber_defense.py` | DDQN defender for a scripted-attacker environment. | `train`, `evaluate`, CLI `--smoke` |
| `madrl_ctde_hybrid_game.py` | Compact CTDE/MADRL attacker-defender game loop. | `train`, `rollout`, `Actor`, `CentralCritic` |
| `node_level_robustness.py` | Stochastic node-graph experiment for parameter-mismatch and scaling discussion. | `NodeSimConfig`, `rollout_node_policy`, `summarize_node_rollout` |

## Inputs And Outputs

| Component | Needs | Produces |
|---|---|---|
| Dynamics utilities | state vector, parameters, control/action intensities | next-state derivatives or RK4 trajectories |
| Hybrid environment | defender action, attacker action, current state | next observation, defender/attacker rewards, diagnostics |
| Policy and game metrics | representative defender/attacker policies and common horizon | cumulative compromised exposure, peak/final compromised share, rewards, impulse counts, game-response rows |
| FBSM baseline | horizon, cost weights, malware parameters | open-loop control, state/costate trajectories, objective, convergence history |
| DDQN defender | environment, replay buffer, Q-network settings | trained Q-network and logged train/evaluation returns |
| CTDE/MADRL game | environment, defender/attacker actors, centralized critics | trained actors and logged loss/return diagnostics |
| Node-level robustness | trained aggregate DDQN policy, nominal FBSM schedule, random graph seed | graph-level compromise trajectories, robustness metrics, scaling proxy |

## How The Pieces Fit

1. `cyber_dynamics.py` defines continuous-time dynamics.
2. `cyber_hybrid_env.py` wraps those dynamics into decision epochs.
3. `evaluation_metrics.py` compares representative policies using the same timing and metrics.
4. `fbsm_malware_baseline.py` solves a continuous-control reference problem.
5. `ddqn_cyber_defense.py` learns a discrete defender policy against a scripted attacker.
6. `madrl_ctde_hybrid_game.py` lets both attacker and defender learn with decentralized actors and centralized critics.
7. `node_level_robustness.py` redeploys the learned aggregate feedback policy on a node graph and compares it with a nominal open-loop FBSM schedule.

## Teaching-Code Boundaries

The files prioritize transparency over benchmark performance.  For research-grade experiments, add multiple seeds, richer logging, replay/checkpoint management, and stronger game diagnostics.

For network-scale extensions, read `docs/EXTENDING.md` before changing code.
