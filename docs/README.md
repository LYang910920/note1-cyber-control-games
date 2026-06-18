# Note 1 Reading Path

Start with `note1_game_learning_cyber_control.pdf`.  It explains how continuous-time cyber-control models become reinforcement-learning and multi-agent reinforcement-learning experiments.

Recommended order:

1. Read the modeling setup: cyber state variables, ODE flow, jump maps, and hybrid actions.
2. Read `MODEL_TO_MDP.md` to fix the timing convention: action point `t_k`, original impulse point `tau_j`, ODE flow interval, next observation.
3. Read the PMP/FBSM sections to understand the continuous-control baseline.
4. Move to the sampled-data RL sections, where continuous dynamics become an MDP.
5. Read the DDQN section before the CTDE/MADRL section.
6. Use `implementation_companion.pdf` when mapping equations to code.
7. Use `code_run_guide.pdf` for run commands and troubleshooting.
8. Use `EXTENDING.md` before adapting the examples to larger graph or cyber-security models.

Source files are in `latex/`.  The source is included for inspection and adaptation; the checked-in PDF is the version intended for reading.

For a practical first pass, use `START_HERE.md` before diving into the LaTeX source.  It points to the code, scripts, figures, and experiment outputs by task.

The runnable code mirrors this lecture order:

| Lecture idea | Code |
|---|---|
| Shared dynamics and RK4 integration | `src/cyber_dynamics.py` |
| Hybrid ODE-RL environment | `src/cyber_hybrid_env.py` |
| Continuous/impulse/MDP timing map | `MODEL_TO_MDP.md` |
| PMP/FBSM baseline | `src/fbsm_malware_baseline.py` |
| DDQN sampled-data defender | `src/ddqn_cyber_defense.py` |
| CTDE/MADRL attacker-defender game | `src/madrl_ctde_hybrid_game.py` |
| Node-level epidemic-model robustness stress test | `src/node_level_robustness.py`, `figures/node_level_learning_advantage.png` |

For extension and cross-repository guidance, read `EXTENDING.md`.
