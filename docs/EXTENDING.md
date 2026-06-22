# Extending Note 1

Keep the same contracts while replacing the small tutorial model:

```text
state -> dynamics -> environment step -> reward/payoff -> baseline or learner -> diagnostics
```

## Extension Path

Start by running:

```bash
python src/scenario_profiles.py
```

That command prints the named student-facing profiles. Pick the closest one before editing code, because it records the intended state level, timing convention, control type, and first files to modify.

1. **Change the state.** Add new compartments, budget variables, node features, or uncertainty states. Update projection or clipping logic so the state remains physically meaningful.
2. **Decide the timing.** State whether each action changes ODE rates, creates an impulse jump, or does both. Use `t_k` for learning action/observation points and `tau_j` for original model impulse/event points. Fixed `Delta t` is convenient, but nonuniform `Delta t_k` is valid if the simulator records it.
3. **Change the dynamics.** Replace `controlled_sir_rhs` or `hybrid_rhs` with a new `f(x,u,a,t)`. Keep the function signature simple and write a short smoke test before adding learning.
4. **Change the reward.** Decide which quantities should be minimized by the defender and maximized by the attacker. Add clear weights to `EnvConfig`.
5. **Add baselines before learning.** Compare no-defense, constant-defense, threshold, and FBSM-style policies before training DDQN or MARL.
6. **Scale the learner.** Move from the compact DDQN/CTDE files to MAPPO or a stronger library only after the environment contract is stable.
7. **Add diagnostics.** For stochastic policies, report multiple seeds, rolling means, confidence intervals, and baseline comparisons.

## Scaling To Network Models

For larger models, move from compartment states to degree-level arrays, node-level vectors, graph features, or event-driven jump maps. Keep the first version small and testable before adding a large RL/MARL framework.

For network-scale impulse models, record whether a jump is node-local, edge-local, or global. For example, isolating one subnet may create an immediate node-state jump, while patching campaigns may change vulnerability rates over the following interval.

Use `src/node_siprs_mappo.py` as the first graph-scale cooperative route. It imports the canonical foundation SIPRS equations, keeps node states as `[S,I,P,R]`, and partitions the graph into regional defender communities. Each regional defender observes local means, boundary pressure, global infection, time-to-go, previous action, and a known heterogeneity summary for its community. The smoke MAPPO loop includes GAE, clipped policy ratios, minibatches, value loss, entropy, and gradient clipping.

Use `src/node_siprs_adversarial_large.py` when the paper model needs a larger node-level attacker-defender scaffold. It uses sparse scale-free graphs, the same heterogeneous SIPRS equations, defender community budgets, attacker beta-boost budgets, self-play softmax policies, and a response matrix. Replace the softmax learner with MAPPO, MADDPG, opponent pools, or exploitability evaluation only after the response matrix is stable.

Use `src/node_level_robustness.py` as a separate stress-test route from aggregate feedback to stochastic node-level S/I/R graphs. It demonstrates a useful stress test: compare a nominal-parameter FBSM open-loop schedule with a feedback policy when the true graph dynamics and propagation parameters differ from the baseline model.

## Node-SIPRS Community MARL Model Card

| Item | Choice in the current code |
|---|---|
| State | node probabilities `x_i=[S_i,I_i,P_i,R_i]` |
| Flow | canonical `cybercontrol.network_models.node_siprs_rhs_numpy` |
| Actions | per-community sampled modes: none, patch `S -> P`, clean `I -> R` |
| Heterogeneity | community-correlated susceptibility, infectivity, recovery, criticality, action costs, bounds, and efficacy from the foundation package |
| Observations | local state means, boundary pressure, global infection, budget/time/action context, and community risk/rate summaries |
| Reward | criticality-weighted local infected share, global infected share, and per-node action cost integrated over `Delta t` |
| Baselines to add before claims | no action, uniform patch, degree/centrality targeting, parameter-risk targeting, oracle targeting, budget-matched random, independent learners, MAPPO |
| MAPPO diagnostics | mean reward, final global infection, mass-conservation error, multi-seed stability |
| Claim limit | empirical learned feedback, not global optimality or Nash equilibrium |

## Large Node-SIPRS Attacker-Defender Model Card

| Item | Choice in the current code |
|---|---|
| State | node probabilities `x_i=[S_i,I_i,P_i,R_i]` on a sparse graph |
| Flow | canonical `cybercontrol.network_models.node_siprs_rhs_numpy` |
| Defender action | choose communities to patch or clean under a budget |
| Attacker action | choose communities receiving temporary infection-rate boost |
| Heterogeneity | community-correlated physical and economic parameters from the foundation package |
| Learner | bounded NumPy softmax self-play over communities |
| Baselines | none, uniform, degree, risk, oracle, budget-random, learned |
| Main output | response matrix with defender payoff, attacker payoff, infected exposure, peak/final infection, and mass error |
| Claim limit | scalable benchmark scaffold, not a full MAPPO/MADDPG implementation |

## From Tutorial Code To Paper Models

| Paper-model ingredient | First tutorial hook | What to preserve while extending |
|---|---|---|
| More cyber compartments or assets | `src/cyber_dynamics.py` | nonnegative state projection and clear state labels |
| Event-triggered or scheduled impulses | `HybridCyberDefenseEnv.jump_map` | explicit pre-jump and post-jump diagnostics |
| Richer attacker behavior | `scripted_attacker`, `madrl_ctde_hybrid_game.py` | same observation/action/reward contract across baselines |
| Node-level or graph-level state | `src/node_siprs_mappo.py`, `src/node_siprs_adversarial_large.py`, then `src/node_level_robustness.py` | canonical SIPRS semantics, aggregate metrics plus node-level stress-test outputs |
| Larger RL/MARL algorithms | `HybridCyberDefenseEnv.step` | stable `reset/step` interface before adding external libraries |
| Paper-specific reward/payoff | `EnvConfig` and `evaluation_metrics.py` | separate infected exposure, defender cost, attacker payoff, and impulse counts |

## Paper-Level Extension Contract

When adapting a paper model, keep these contracts visible in code and outputs:

1. A named profile in `src/scenario_profiles.py` with the horizon, `Delta t`, propagation rates, reward weights, learner hyperparameters, and first files to edit.
2. A runnable smoke command that finishes quickly and checks mass conservation, finite rewards, and output files.
3. One baseline table for the same model: no-defense, fixed/rule policies, random policies when practical, and any FBSM/PMP candidate.
4. One result figure per model, with captions stating whether curves are aggregate means, degree classes, nodes, communities, continuous controls, impulse actions, or hybrid actions.
5. A short claim statement in `docs/PAPER_WORKFLOW.md`: what can be claimed from the current evidence, and what still needs multiple seeds, held-out graphs, or exploitability checks.

## Related Learning Path

Use these repositories together:

| Step | Repository | Focus |
|---|---|---|
| 1 | `network-control-differential-games` | PMP, FBSM, degree/node-level network control, differential games |
| 2 | `note1-cyber-control-games` | ODE-to-RL conversion, DDQN, compact CTDE, node-SIPRS MAPPO |
| 3 | `note2-pinn-pidl-cyber-control` | PINN/PIDL inverse learning and neural optimal control |

Companion repository: https://github.com/LYang910920/network-control-differential-games

## Research-Grade Checklist

Before treating a run as evidence rather than a tutorial run:

1. Run at least 5 to 10 random seeds.
2. Compare against no-defense, constant-defense, rule-based, and optimal-control baselines.
3. Separate training, validation, and stress-test scenarios.
4. Report wall-clock cost and sample count.
5. Track constraint violations, not only rewards.
6. For games, run unilateral-deviation or exploitability-style checks where possible.
7. Keep all dataset and third-party license notes with the experiment.
