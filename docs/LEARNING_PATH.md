# Cross-Repository Learning Path

These repositories are meant to work together as a small learning sequence.

## Recommended Order

| Step | Repository | Focus |
|---|---|---|
| 1 | `network-control-differential-games` | PMP, FBSM, degree-level and node-level network optimal control, differential games |
| 2 | `note1-cyber-control-games` | ODE-to-RL conversion, DDQN defense, CTDE/MADRL attacker-defender learning |
| 3 | `note2-pinn-pidl-cyber-control` | PINN/PIDL inverse learning, neural control, PMP-informed neural residuals |

## How Note 1 Connects To The Differential-Games Repository

The differential-games repository gives the mathematical and network-control foundation:

https://github.com/LYang910920/network-control-differential-games

This Note 1 repository then asks what changes when the controller is learned from sampled interaction rather than computed directly from a Hamiltonian update.

| Differential-games concept | Note 1 counterpart |
|---|---|
| Forward state equation | ODE transition inside `HybridCyberDefenseEnv.step` |
| Backward adjoint and control update | FBSM baseline in `fbsm_malware_baseline.py` |
| Hybrid/impulse intervention | `jump_map` plus continuous RK4 flow in `cyber_hybrid_env.py` |
| Attacker-defender differential game | CTDE/MADRL training in `madrl_ctde_hybrid_game.py` |
| Convergence plots | `experiments/*_history.csv` and `figures/training_iteration_diagnostics.png` |

## How Note 1 Connects To Note 2

Note 2 is useful when the dynamics, parameters, or optimality residuals are not fully known.

| If Note 1 gives you... | Note 2 can help with... |
|---|---|
| A simulator with unknown parameters | inverse PINN parameter learning |
| A partially known ODE | PIDL missing-mechanism learning |
| A direct neural policy objective | direct control PINN |
| A PMP baseline | PMP-informed neural residual training |
