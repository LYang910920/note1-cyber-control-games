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
6. **Scale the learner.** Move from the compact DDQN/MADRL files to a stronger library only after the environment contract is stable.
7. **Add diagnostics.** For stochastic policies, report multiple seeds, rolling means, confidence intervals, and baseline comparisons.

## Scaling To Network Models

For larger models, move from compartment states to degree-level arrays, node-level vectors, graph features, or event-driven jump maps. Keep the first version small and testable before adding a large RL/MARL framework.

For network-scale impulse models, record whether a jump is node-local, edge-local, or global. For example, isolating one subnet may create an immediate node-state jump, while patching campaigns may change vulnerability rates over the following interval.

Use `src/node_level_robustness.py` as a small bridge from compartment states to node-level epidemic graphs. Each graph node has a local S/I/R state, while the feedback policy observes aggregate graph features. It demonstrates a useful stress test: compare a nominal-parameter FBSM open-loop schedule with a feedback policy when the true graph dynamics and propagation parameters differ from the baseline model.

## From Tutorial Code To Paper Models

| Paper-model ingredient | First tutorial hook | What to preserve while extending |
|---|---|---|
| More cyber compartments or assets | `src/cyber_dynamics.py` | nonnegative state projection and clear state labels |
| Event-triggered or scheduled impulses | `HybridCyberDefenseEnv.jump_map` | explicit pre-jump and post-jump diagnostics |
| Richer attacker behavior | `scripted_attacker`, `madrl_ctde_hybrid_game.py` | same observation/action/reward contract across baselines |
| Node-level or graph-level state | `src/node_level_robustness.py` | aggregate metrics plus node-level stress-test outputs |
| Larger RL/MARL algorithms | `HybridCyberDefenseEnv.step` | stable `reset/step` interface before adding external libraries |
| Paper-specific reward/payoff | `EnvConfig` and `evaluation_metrics.py` | separate infected exposure, defender cost, attacker payoff, and impulse counts |

## Related Learning Path

Use these repositories together:

| Step | Repository | Focus |
|---|---|---|
| 1 | `network-control-differential-games` | PMP, FBSM, degree/node-level network control, differential games |
| 2 | `note1-cyber-control-games` | ODE-to-RL conversion, DDQN, CTDE/MADRL |
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
