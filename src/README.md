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

## Module Map

| File | Main purpose | Useful entry points |
|---|---|---|
| `cyber_dynamics.py` | Shared dynamics and integration utilities. | `rk4_integrate`, `controlled_sir_rhs`, `hybrid_rhs` |
| `cyber_hybrid_env.py` | Plain-Python reset/step environment for sampled hybrid cyber defense. | `HybridCyberDefenseEnv`, `EnvConfig`, `scripted_attacker` |
| `fbsm_malware_baseline.py` | Forward-backward sweep method for a PMP open-loop control baseline. | `solve_fbsm` |
| `ddqn_cyber_defense.py` | DDQN defender for a scripted-attacker environment. | `train`, `evaluate`, CLI `--smoke` |
| `madrl_ctde_hybrid_game.py` | Compact CTDE/MADRL attacker-defender game loop. | `train`, `rollout`, `Actor`, `CentralCritic` |

## How The Pieces Fit

1. `cyber_dynamics.py` defines continuous-time dynamics.
2. `cyber_hybrid_env.py` wraps those dynamics into decision epochs.
3. `fbsm_malware_baseline.py` solves a continuous-control reference problem.
4. `ddqn_cyber_defense.py` learns a discrete defender policy against a scripted attacker.
5. `madrl_ctde_hybrid_game.py` lets both attacker and defender learn with decentralized actors and centralized critics.

## Teaching-Code Boundaries

The files prioritize transparency over benchmark performance.  For research-grade experiments, add multiple seeds, richer logging, replay/checkpoint management, and stronger game diagnostics.
