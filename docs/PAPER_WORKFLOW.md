# How To Write A Paper From Note 1

Note 1 connects continuous-time cyber dynamics to PMP/FBSM, ODE-RL, DDQN,
CTDE attacker-defender learning, cooperative node-level SIPRS MAPPO examples,
and a larger sparse node-SIPRS attacker-defender benchmark.

## Model-To-Code Map

| Paper notation | Code location | Meaning |
| --- | --- | --- |
| `x=[S,I,R,z]` | `src/cyber_hybrid_env.py` | Aggregate malware/deception state. |
| `f(x,a_D,a_A)` | `cybercontrol.models.hybrid_rhs` | Continuous flow between sampled decisions. |
| `G(x,a_D)` | `cybercontrol.models.isolation_jump` | Impulse jump for isolation. |
| `Delta t` | `EnvConfig.dt` | Sampled decision interval. |
| RK4 substeps | `EnvConfig.substeps` | Internal ODE solver steps, not MDP actions. |
| running/impulse cost | `HybridCyberDefenseEnv.step` | Running cost plus separate impulse cost. |
| `u(t)` | `src/fbsm_malware_baseline.py` | Continuous FBSM patching control. |
| `x_i=[S_i,I_i,P_i,R_i]` | `src/node_siprs_mappo.py`, `src/node_siprs_adversarial_large.py` | Canonical node-level SIPRS state. |
| `u_i^p,u_i^c` | `patch`, `clean` in node-SIPRS files | Regional sampled modes mapped to node patch/clean rates. |
| attacker boost | `beta_boost` in `src/node_siprs_adversarial_large.py` | Temporary community-level increase in infection pressure. |

## Recommended Paper Path

1. Define the continuous SIR malware model and state variables.
2. Add sampled decisions: observation at `t_k`, action, optional jump, ODE flow, next observation.
3. Present FBSM as a deterministic continuous-control baseline.
4. Present DDQN as a feedback policy for the sampled-data MDP.
5. Present the compact CTDE script as an attacker-defender Markov-game baseline.
6. Present node-SIPRS MAPPO as a cooperative community-defense feedback baseline.
7. Compare all policies on the same model with fixed, random, threshold, and centrality baselines.
8. Use `src/node_siprs_adversarial_large.py` for a larger node-level attacker-defender scaffold and response matrix.

## Minimum Experiments

| Figure/table | What it should show |
| --- | --- |
| State evolution | Specify aggregate state or node-level mean; label `S`, `I`, `R`, and `z`. |
| Continuous control | FBSM `u(t)` as a time curve. |
| Hybrid action | Discrete sampled action mode as a step plot; impulses as markers. |
| Training convergence | FBSM control-change curve, DDQN evaluation return, compact CTDE loss/return, MAPPO reward/value diagnostics. |
| Baseline comparison | No defense, fixed policies, rule policy, DDQN/CTDE/MAPPO policy, random policies, and for node-SIPRS MAPPO: uniform, degree-priority, risk-priority, oracle, and budget-matched random rollouts on held-out profiles. |
| Game response | Fixed attacker vs varied defenders, fixed defender vs varied attackers, and full response matrix for the larger node-SIPRS game. |
| Node-SIPRS ablation | Merge `P` and `R`, remove waning, change budget, and compare unseen graph seeds. |

## Claim Discipline

- Say "PMP/FBS candidate" unless a stronger optimality check is added.
- Say "learned feedback policy" for DDQN rather than "optimal policy".
- Say "MAPPO baseline" or "cooperative learned defenders" for `node_siprs_mappo.py`; add multi-seed and held-out-graph evidence before stronger claims.
- Say "large attacker-defender benchmark" for `node_siprs_adversarial_large.py`; the included softmax learner is a scaffold, not a Nash solver.
- Keep impulse cost separate from running cost in both equations and code.
- Report seeds, horizon, `dt`, propagation rates, reward weights, network width, learning rate, replay size, and episode count.

For a broader cross-method workflow, see the foundation repository's
`docs/from_model_to_paper.md`.
