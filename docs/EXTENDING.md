# Extending Note 1

Keep the same contracts while replacing the small teaching model:

```text
state -> dynamics -> environment step -> reward/payoff -> baseline or learner -> diagnostics
```

## Extension Path

1. **Change the state.** Add new compartments, budget variables, node features, or uncertainty states. Update projection or clipping logic so the state remains physically meaningful.
2. **Change the dynamics.** Replace `controlled_sir_rhs` or `hybrid_rhs` with a new `f(x,u,a,t)`. Keep the function signature simple and write a short smoke test before adding learning.
3. **Change the reward.** Decide which quantities should be minimized by the defender and maximized by the attacker. Add clear weights to `EnvConfig`.
4. **Add baselines before learning.** Compare no-defense, constant-defense, threshold, and FBSM-style policies before training DDQN or MARL.
5. **Scale the learner.** Move from the compact DDQN/MADRL files to a stronger library only after the environment contract is stable.
6. **Add diagnostics.** For stochastic policies, report multiple seeds, rolling means, confidence intervals, and baseline comparisons.

## Scaling To Network Models

For larger models, move from compartment states to degree-level arrays, node-level vectors, graph features, or event-driven jump maps. Keep the first version small and testable before adding a large RL/MARL framework.

## Related Learning Path

Use these repositories together:

| Step | Repository | Focus |
|---|---|---|
| 1 | `network-control-differential-games` | PMP, FBSM, degree/node-level network control, differential games |
| 2 | `note1-cyber-control-games` | ODE-to-RL conversion, DDQN, CTDE/MADRL |
| 3 | `note2-pinn-pidl-cyber-control` | PINN/PIDL inverse learning and neural optimal control |

Companion repository: https://github.com/LYang910920/network-control-differential-games

## Research-Grade Checklist

Before treating a run as evidence rather than a teaching demo:

1. Run at least 5 to 10 random seeds.
2. Compare against no-defense, constant-defense, rule-based, and optimal-control baselines.
3. Separate training, validation, and stress-test scenarios.
4. Report wall-clock cost and sample count.
5. Track constraint violations, not only rewards.
6. For games, run unilateral-deviation or exploitability-style checks where possible.
7. Keep all dataset and third-party license notes with the experiment.
