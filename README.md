# Cyber Control and Game Learning

Executable tutorial code for cyber optimal control, sampled-data reinforcement learning, and attacker-defender game learning. This is the second repository in the tutorial family. It uses the foundation package `cybercontrol` for shared dynamics, integration, plotting, and neural helper blocks, then keeps the Note 1 code focused on environments and learning methods.

## Repository Family

| Order | Repository | Role |
|---:|---|---|
| 0 | [network-control-differential-games](https://github.com/LYang910920/network-control-differential-games) | Foundation notation, shared `cybercontrol` package, continuous/impulse/hybrid examples, degree-vs-node scalability, and reference smoke runs. |
| 1 | `note1-cyber-control-games` | FBSM baseline, sampled-data MDP conversion, DDQN defense, compact CTDE attacker-defender learning, and node-SIPRS MAPPO smoke tests. |
| 2 | [note2-pinn-pidl-cyber-control](https://github.com/LYang910920/note2-pinn-pidl-cyber-control) | PINN/PIDL inverse learning, neural control, PMP-informed losses, and graph-state residual examples. |

## 5-Minute Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "../network-control-differential-games[torch,dev]"
python -m pip install -e ".[dev]"
bash scripts/run_smoke_tests.sh
python scripts/generate_figures.py
```

If this repository is cloned without the sibling foundation repo:

```bash
python -m pip install "cybercontrol[torch] @ git+https://github.com/LYang910920/network-control-differential-games.git"
python -m pip install -e ".[dev]"
```

For bounded diagnostics that write local artifacts:

```bash
python scripts/run_training_iterations.py
```

## Code Map

| Need | Start here |
|---|---|
| Tutorial PDF | `docs/note1_game_learning_cyber_control.pdf` |
| Run and implementation guide | `docs/code_run_guide.pdf`, `docs/implementation_companion.pdf` |
| Parameters and hyperparameters | `docs/PARAMETERS.md` |
| MDP, Markov-game, and impulse timing | `docs/MODEL_TO_MDP.md` |
| Paper workflow and extensions | `docs/PAPER_WORKFLOW.md`, `docs/EXTENDING.md` |
| Aggregate cyber environment | `src/cyber_hybrid_env.py` |
| FBSM baseline | `src/fbsm_malware_baseline.py` |
| DDQN and compact CTDE | `src/ddqn_cyber_defense.py`, `src/madrl_ctde_hybrid_game.py` |
| Heterogeneous node-SIPRS MAPPO smoke baseline | `src/node_siprs_mappo.py` |
| Static figures and bounded diagnostics | `scripts/generate_figures.py`, `scripts/run_training_iterations.py` |

## Representative Experiments

The FBSM baseline solves a continuous-time malware-control problem and produces a continuous patching intensity over time.

![FBSM malware-control baseline](docs/assets/fbsm_malware_control.png)

The hybrid policy comparison evaluates no defense, fixed defenses, and a rule-based hybrid policy on the same simulator. Lower compromised exposure is better.

![Hybrid policy comparison](docs/assets/hybrid_policy_comparison.png)

The node-level robustness example deploys a nominal open-loop FBSM schedule and a DDQN feedback policy on stochastic node-level epidemic rollouts. Here robustness means lower infected-node exposure under parameter mismatch, not a formal guarantee. The MAPPO smoke environment uses the foundation SIPRS equations with community-correlated susceptibility, infectivity, recovery, criticality, costs, bounds, and efficacy; each community observes a compact risk/rate summary. Its policy evaluator compares uniform, degree, risk, oracle, budget-matched random, and learned MAPPO rollouts on held-out seeds and heterogeneity strengths.

![Node-level epidemic model robustness](docs/assets/node_level_learning_advantage.png)

## Extension Route

1. Read `docs/PARAMETERS.md` to locate the model, solver, and neural-training settings.
2. Edit one method at a time: environment dynamics, reward/payoff, policy class, or training profile.
3. Keep shared numerics and plotting in the foundation package `cybercontrol`; add Note 1 code only for game-learning behavior.
4. Run `bash scripts/run_smoke_tests.sh` after each structural change.
5. Use `python scripts/run_training_iterations.py` for bounded diagnostics. Outputs go to ignored `artifacts/experiments/` and `artifacts/figures/`.

## Validation

```bash
python -m compileall -q src tests scripts
python -m pytest -q
bash scripts/run_smoke_tests.sh
python scripts/generate_figures.py
```

GitHub Actions runs the smoke tests on pushes and pull requests. The examples are tutorial baselines, not calibrated cyber-risk models.

## Citation and License

See `LICENSE` and `NOTICE.md`. When using the repository in a paper or report, cite the related publication and the foundation repository when its shared package is used.
