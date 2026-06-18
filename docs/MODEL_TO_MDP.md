# Continuous, Impulse, Hybrid, MDP, And Markov Game Guide

This repository uses one cyber propagation model in several forms.  The key is to keep the timing convention explicit.

## Decision Timeline

At decision epoch `k`, the learning environment uses this order:

```text
observe s_k = x(t_k^-)
  -> choose defender action a_D,k and, for games, attacker action a_A,k
  -> apply any instantaneous jump x(t_k^+) = G(x(t_k^-), a_D,k, a_A,k)
  -> integrate ODE flow over [t_k, t_{k+1})
  -> return s_{k+1} = x(t_{k+1}^-), rewards, done, diagnostics
```

The policy observes the pre-jump state `x(t_k^-)`.  The next observation is the state just before the next decision point, `x(t_{k+1}^-)`.  If an action has no jump effect, then `x(t_k^+) = x(t_k^-)` and only the ODE rates change over the interval.

## Three Control Types

| Type | Mathematical object | Code location | Does the state jump? |
|---|---|---|---|
| Continuous-time control | `u(t)` appears in the ODE right-hand side. | `fbsm_malware_baseline.py`, `controlled_sir_rhs` | No |
| Impulse control | `x(t_k^+) = G(x(t_k^-), a_k)` at selected times. | `HybridCyberDefenseEnv.jump_map` | Yes |
| Hybrid sampled action | discrete mode plus intensity `(m_k, v_k)`; may set rates, jumps, or both. | `decode_action`, `defense_parameters`, `attack_parameters` | Depends on the mode |

In the current environment, `DEF_ISOLATE` is impulsive because it immediately moves a fraction of compromised devices into the protected compartment.  `DEF_PATCH`, `DEF_CLEAN`, and `DEF_DECEIVE` mainly change ODE rates during the next decision interval.

## What Is Discretized?

There are two different grids.  They should not be confused.

| Grid | Meaning | Used by | Is it an RL decision point? |
|---|---|---|---|
| Continuous-control integration grid | Numerical time mesh for solving ODE/PMP equations. | FBSM baseline | No |
| Decision grid `t_k = k Delta t` | Times where policies observe the state and choose actions. | DDQN MDP, CTDE/MADRL Markov game | Yes |
| RK4 substeps inside one interval | Internal solver steps used to integrate from `t_k` to `t_{k+1}`. | Hybrid environment | No |

FBSM does not convert the original process into an MDP.  It solves a continuous-time optimal-control problem on a numerical mesh.  DDQN and CTDE/MADRL do convert the continuous/hybrid simulator into a sampled-data MDP or Markov game by exposing only the decision grid to the learning algorithm.

## MDP Conversion

For the single-defender DDQN example, the induced MDP is:

```text
s_k      = observation at t_k
a_k      = defender action
P        = simulator transition defined by jump_map + ODE integration
r_k      = integrated defender reward over [t_k, t_{k+1})
s_{k+1}  = next observation at t_{k+1}
```

The replay buffer stores `(s_k, a_k, r_k, s_{k+1}, done)`.  The internal RK4 substeps produce a better transition estimate, but they do not create extra replay items.

## Markov Game Conversion

For the attacker-defender example, the induced Markov game is:

```text
s_k        = shared simulator state at t_k
a_D,k      = defender action
a_A,k      = attacker action
P          = simulator transition defined by joint action, jump_map, and ODE flow
r_D,k      = defender reward
r_A,k      = attacker reward
s_{k+1}    = next simulator state
```

The compact CTDE/MADRL script uses centralized critics that see the state and both actions during training.  The actors are decentralized policies that map observations to defender or attacker actions.

## Code Checklist

When adapting the model, decide these items before training:

1. What is observed at `t_k`: full state, partial state, alerts, budget, or time-to-go?
2. Which actions cause immediate jumps?
3. Which actions only change continuous rates during `[t_k, t_{k+1})`?
4. What is the decision interval `Delta t`?
5. How many solver substeps are used inside each decision interval?
6. Is the algorithm solving a continuous-control problem, an MDP, or a Markov game?

The environment records timing and transition diagnostics in the `info` dictionary returned by `step`.
