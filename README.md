# Cyber Control and Game Learning, Note 1

This repository is the executable companion for **Note 1: Game Learning for Cyber Control**.  It collects compact Python examples for continuous-time cyber-control models, PMP/FBSM baselines, sampled-data reinforcement learning, and attacker-defender Markov games.

The examples are intentionally small and readable.  They are meant for study, modification, and sanity-checking research ideas before moving to a larger RL or MARL framework.

## Repository Map

| Path | Purpose |
|---|---|
| `docs/note1_game_learning_cyber_control.pdf` | Main lecture note for game learning and cyber control. |
| `docs/implementation_companion.pdf` | Companion explanation for implementation choices. |
| `docs/code_run_guide.pdf` | General run guide from the original bundle. |
| `src/cyber_dynamics.py` | Shared RK4 integration and malware dynamics utilities. |
| `src/cyber_hybrid_env.py` | Hybrid cyber-defense environment with flow, jump, and mixed actions. |
| `src/fbsm_malware_baseline.py` | Forward-backward sweep method for a PMP malware-control baseline. |
| `src/ddqn_cyber_defense.py` | DDQN defender for a sampled-data ODE cyber-defense MDP. |
| `src/madrl_ctde_hybrid_game.py` | Minimal CTDE/MADRL attacker-defender game example. |
| `scripts/generate_figures.py` | Generates explanatory figures in `figures/`. |
| `scripts/run_smoke_tests.sh` | Runs all fast checks for this repo. |
| `tests/` | Small regression tests for dynamics, environment contracts, and FBSM output. |

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the full smoke check:

```bash
bash scripts/run_smoke_tests.sh
```

Generate the figures used in this README:

```bash
python scripts/generate_figures.py
```

## Main Ideas

This repo separates three levels that are easy to blur:

1. **Continuous-time control**: `fbsm_malware_baseline.py` solves an open-loop PMP system for one initial condition.
2. **Sampled-data RL**: `cyber_hybrid_env.py` converts ODE flow and impulsive actions into a reset/step interface, while `ddqn_cyber_defense.py` learns a discrete defender policy.
3. **Attacker-defender games**: `madrl_ctde_hybrid_game.py` uses decentralized actors and centralized critics to make the Markov-game training loop explicit.

## Figures

`figures/fbsm_malware_control.png` shows the FBSM baseline: malware compartments and the optimized patching intensity.

![FBSM malware-control baseline](figures/fbsm_malware_control.png)

`figures/hybrid_policy_rollout.png` shows a simple hybrid rollout where the defender switches among patching, cleaning, and isolation decisions while the attacker follows a scripted policy.

![Hybrid ODE-RL rollout](figures/hybrid_policy_rollout.png)

## Validation

The repo includes smoke tests for every executable script and unit tests for the core numerical contracts.  The current version also replaces deprecated NumPy integration calls and casts reward outputs to plain Python floats for cleaner logs.

These examples are teaching code, not benchmark implementations.  For serious experiments, add multiple seeds, stronger baselines, full logging, and exploitability-style game diagnostics.
