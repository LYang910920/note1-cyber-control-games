# Methods and API

## Timing and State Semantics

The aggregate environment uses state `[S, I, R]` and appends normalized decision
phase to form a four-value observation. At epoch `k`:

1. observe `x(t_k-)`;
2. choose defender and attacker actions;
3. apply `x(t_k+) = G(x(t_k-), a_k)` only for an impulse action;
4. hold flow parameters over `[t_k, t_{k+1})`;
5. integrate the ODE with internal RK4 substeps;
6. return `x(t_{k+1}-)` and interval reward.

A real-valued action held over an interval is a continuous-valued sampled action,
not a time-varying continuous-time control. A discrete mode that selects flow
parameters does not create a state jump. The isolation action is impulsive because
it has an explicit reset map.

The canonical environment is `cybergames.envs.SampledFlowImpulseEnv`.

## Reward Accounting

The defender reward separates running and impulse costs:

```text
r_D[k] = -dt * (w_I * mean(I) + w_S * mean(S)
                 + c_patch * patch^2 + c_clean * clean^2
                 + c_deceive * deceive^2)
         - impulse_cost
```

`mean(S)` and `mean(I)` are numerical interval means, not endpoint samples.
`impulse_cost` is charged once and is not multiplied by `dt`. The attacker reward
uses the same interval means and subtracts its sampled action cost.

## Node-SIPS Learning Model

Each node has state `[S, I, P]` with `S + I + P = 1`. The Foundation package owns
the equations and heterogeneous parameter resolution. Note 1 owns the learning
environment, observations, actions, and rewards.

The cooperative MAPPO observation contains local compartment summaries and known
risk/rate summaries. This is an explicit full-information teaching setting. A
partial-observation study must define which rates or summaries are hidden and how
belief/history enters the policy.

## Algorithms

| Method | Action/setting | Canonical module |
|---|---|---|
| FBS baseline | time-varying continuous control | `cybergames.fbsm` |
| DDQN | finite sampled action set | `cybergames.ddqn` |
| CTDE actor-critic | parameterized attacker/defender actions | `cybergames.ctde` |
| Cooperative MAPPO | decentralized community actions, centralized critic | `cybergames.mappo` |
| Attacker-defender self-play | two-player node-SIPS sampled game | `cybergames.self_play` |

MAPPO uses generalized advantage estimation, clipped PPO objectives, minibatches,
entropy regularization, gradient clipping, and deterministic held-out evaluation.
`summary_mlp` and `graph_context` actors share a matched parameter-budget check.

## Typed Settings

All experiment settings are dataclasses in `cybergames.configs`.

| Config | Important defaults |
|---|---|
| `EnvConfig` | `dt=1`, `substeps=10`, `horizon=100`; separate running/impulse costs |
| `NodeSIPSEnvConfig` | 48 nodes, 3 communities, 18 epochs, strength 0.35 |
| `DDQNConfig` | 300 episodes, batch 128, hidden 128, discount 0.99 |
| `MAPPOConfig` | 12 updates, rollout 18, PPO epochs 3, clip 0.2, GAE 0.95 |
| `AdversarialSIPSConfig` | 512 nodes, 8 communities, defender/attacker budget 2 |

The public medium profile intentionally overrides these defaults with bounded
settings and records the resolved values in `medium_config.json`. Its orchestration
is isolated in `cybergames.experiments`; the CLI contains no training algorithm.

## Evaluation Terms

- **Rollout:** one forward simulation of an environment under fixed policy rules
  and a declared seed.
- **Nominal:** the parameter profile used for policy construction or training.
- **Held out:** a graph or parameter seed/strength not used to fit the policy.
- **Robustness experiment:** a measured performance comparison under declared
  shifts; it is not a formal robustness guarantee.
- **Response matrix:** payoffs from pairing a fixed set of defender and attacker
  policies.
- **Unilateral deviation diagnostic:** the payoff change when one side changes
  policy while the other is fixed.
- **Exploitability proxy:** the best observed unilateral gain within the tested
  policy set; it is not exact exploitability unless best responses are solved.

Every metric row should identify the policy, seed, graph size, heterogeneity
strength, budget, horizon, and whether the evaluation was held out.
