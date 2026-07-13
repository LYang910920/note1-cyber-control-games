# From Model to Paper

## 1. Define the Continuous Model

State the compartments, graph, parameters, admissible controls, invariants, and
units. Validate uncontrolled and rule-controlled trajectories before introducing a
learning algorithm.

## 2. Define Decision Timing

Choose decision epochs, zero-order-hold mapping, solver substeps, and any impulse
times. State whether an action changes flow parameters, selects a mode, or applies
a reset. Keep running and jump costs separate.

## 3. Construct the MDP or Markov Game

List observations, actions, transition order, rewards, horizon, budgets, and
information available to each player. For CTDE, distinguish actor observations
from critic inputs. Test state/action shapes and deterministic seeds.

## 4. Select a Method and Baselines

Use DDQN for a small discrete action set; use PPO-style methods for continuous or
parameterized actions; use MAPPO when decentralized actors share a centralized
training signal. Always retain no-action, fixed, rule-based, and budget-matched
random baselines. Add degree/risk/oracle baselines for graph allocation.

## 5. Evaluate

Report reward together with infected exposure, peak/final infection, action cost,
budget use, mass error, and runtime. Use multiple training seeds and held-out graph
and parameter profiles. For games, report response matrices and unilateral
deviation gains with one player's policy fixed at a time.

## 6. Ablate and Stress

Vary observation information, heterogeneity strength, graph family/size, budget,
reward weights, architecture, and training horizon. Separate changes to the model
from changes to the optimizer.

## 7. Write Cautious Claims

Specify whether the evidence concerns control performance, generalization under
tested shifts, or approximate game stability. Do not equate training reward with
physical effectiveness, robustness guarantees, or equilibrium proof.

The Foundation repository supplies detailed PMP/FBS derivations. The Physics-
Informed Cyber Control repository supplies detailed PINN/PIDL methods; this guide
uses only the interfaces needed to compare them with learning-based policies.
