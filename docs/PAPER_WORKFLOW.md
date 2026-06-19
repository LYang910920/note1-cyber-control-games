# How To Write A Paper From Note 1

Note 1 connects continuous-time cyber dynamics to PMP/FBSM, ODE-RL, DDQN, and
attacker-defender CTDE/MADRL examples.

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

## Recommended Paper Path

1. Define the continuous SIR malware model and state variables.
2. Add sampled decisions: observation at `t_k`, action, optional jump, ODE flow, next observation.
3. Present FBSM as a deterministic continuous-control baseline.
4. Present DDQN as a feedback policy for the sampled-data MDP.
5. Present CTDE/MADRL as the attacker-defender Markov-game extension.
6. Compare all policies on the same model with fixed, random, and rule-based baselines.
7. Add node-level experiments only after the aggregate model is clear.

## Minimum Experiments

| Figure/table | What it should show |
| --- | --- |
| State evolution | Specify aggregate state or node-level mean; label `S`, `I`, `R`, and `z`. |
| Continuous control | FBSM `u(t)` as a time curve. |
| Hybrid action | Discrete sampled action mode as a step plot; impulses as markers. |
| Training convergence | FBSM control-change curve, DDQN evaluation return, MADRL loss/return. |
| Baseline comparison | No defense, fixed policies, rule policy, DDQN/MADRL policy, and random policies. |
| Game response | Fixed attacker vs varied defenders and fixed defender vs varied attackers. |

## Claim Discipline

- Say "PMP/FBS candidate" unless a stronger optimality check is added.
- Say "learned feedback policy" for DDQN rather than "optimal policy".
- Keep impulse cost separate from running cost in both equations and code.
- Report seeds, horizon, `dt`, propagation rates, reward weights, network width, learning rate, replay size, and episode count.

For a broader cross-method workflow, see the foundation repository's
`docs/from_model_to_paper.md`.
