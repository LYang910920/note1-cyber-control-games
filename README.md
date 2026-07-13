# Cyber Control and Game Learning

Sampled cyber-control environments, reinforcement learning, and multi-agent game
evaluation. Shared ODEs, graph models, integration, neural blocks, buffers, metrics,
and plotting come from the Foundation `cybercontrol` package.

## Repository Family

| Repository | Purpose |
|---|---|
| [Network Control and Differential Games](https://github.com/LYang910920/network-control-differential-games) | Foundation equations, FBS solvers, heterogeneous profiles, and shared Python components. |
| **Cyber Control and Game Learning** | Sampled environments, DDQN, CTDE, MAPPO, and attacker-defender evaluation. |
| [Physics-Informed Cyber Control](https://github.com/LYang910920/note2-pinn-pidl-cyber-control) | Inverse PINN, PIDL, neural control, and PMP-informed learning. |

## Five-Minute Start

With the Foundation repository next to this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "../network-control-differential-games[torch]"
python -m pip install -e ".[dev]"
python -m cybergames smoke
```

For a standalone checkout, `python -m pip install -e ".[dev]"` installs the
Foundation revision declared in `pyproject.toml`.

```bash
python -m cybergames medium --device auto --output-dir artifacts/medium
python -m cybergames figures
python -m cybergames docs
```

`medium` is a bounded five-seed DDQN/MAPPO and attacker-defender evaluation. It
writes metrics and a run manifest under ignored `artifacts/`.

## Code Map

| Topic | Module |
|---|---|
| Sampled SIR flow and impulse environment | `cybergames.envs` |
| Explicit action modes and intensities | `cybergames.actions` |
| Typed model and learning settings | `cybergames.configs` |
| Continuous-control FBS baseline | `cybergames.fbsm` |
| DDQN | `cybergames.ddqn` |
| Compact CTDE actor-critic | `cybergames.ctde` |
| Cooperative node-SIPS MAPPO | `cybergames.node_env`, `cybergames.mappo` |
| Attacker-defender node-SIPS game | `cybergames.adversarial_env`, `cybergames.self_play` |
| Baselines and held-out evaluation | `cybergames.evaluation`, `cybergames.node_evaluation` |
| Bounded multi-seed experiment profile | `cybergames.experiments` |

The public entry point is `python -m cybergames`. `run_all.py` remains a small
compatibility shim; it contains no model or training implementation.

## Control Timing

At decision epoch `t_k`, the policy observes the pre-jump state, chooses an
action, applies a reset only when the action is impulsive, and then integrates the
ODE to `t_{k+1}`. Internal RK4 points are solver substeps, not additional actions.

![Decision epochs, ODE substeps, and impulse times](docs/assets/action_timing.png)

The exact flow/jump order, action domains, and reward terms are documented in
[Methods and API](docs/METHODS_AND_API.md).

## Representative Experiments

The FBS baseline produces a genuinely time-varying continuous patching signal.
The upper panel is the population-average SIR state; the lower panel is `u(t)`.

![Continuous-control FBS baseline](docs/assets/fbsm_malware_control.png)

The policy comparison uses one simulator and seed for no defense, fixed actions,
and a rule-based policy. Exposure is the time integral of the compromised share;
impulse costs are charged separately from running costs.

![Sampled-flow and impulse policy comparison](docs/assets/sampled_impulse_policy_comparison.png)

The cooperative MAPPO example is not an attacker-defender equilibrium algorithm.
The separate attacker-defender module reports response matrices and unilateral
deviation diagnostics; these are empirical checks, not equilibrium proofs.

## Extension Route

1. Define the continuous model and action timing before changing a learner.
2. Add observation and reward terms in the environment, with units and shapes documented.
3. Change typed configurations in `cybergames.configs`; avoid literals in training loops.
4. Compare learned policies with uniform, degree, risk, oracle, and budget-matched random baselines.
5. Evaluate held-out graph and parameter seeds before making robustness claims.

See [Reproducibility](docs/REPRODUCIBILITY.md) and
[From Model to Paper](docs/FROM_MODEL_TO_PAPER.md).

## Validation

```bash
python -m compileall -q src tests scripts
ruff check .
ruff format --check src tests scripts
pytest -q
python -m cybergames smoke
python -m cybergames figures
```

The examples are controlled synthetic studies, not calibrated operational risk
models. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for reuse terms, and the
[tutorial PDF](docs/note1_game_learning_cyber_control.pdf) for the full derivation.
