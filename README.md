# Learning cyber control and games

Start with a continuous-time propagation model. Turn intervention times into decision steps, train a defender, and compare the actions it actually executes. Then extend the example to cooperative multi-agent control and adversarial games.

**First run:** one CPU experiment with DDQN, an untrained policy, and five fixed-action baselines. **Read next:** [the walkthrough](docs/GUIDE.md). The [lecture notes](docs/note1_game_learning_cyber_control.pdf) explain the wider progression from optimal control to reinforcement learning and games.

## Install and run

Use Python 3.11 or 3.12 and install Git. From this repository:

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1

python -m pip install -e .
python -m cybergames
```

The installer fetches the tested revision of the shared `cybercontrol` library automatically. A GPU, external dataset and LaTeX are not required. On Windows or Linux without a GPU, installing CPU-only PyTorch first avoids downloading CUDA libraries:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

The example writes only figures and tables to `results/`: learning progress, infection trajectories, executed actions and paired policy comparisons. All final policies are evaluated on the same held-out initial conditions. The attacker is scripted; this introductory experiment learns a defender rather than solving a two-player equilibrium.

```bash
python -m cybergames --episodes 240 --seed 2 --output-dir results/seed2
```

A short training run can be weaker than a fixed baseline. Read the measured costs and action curves rather than assuming that training must win.

## Read the code

| Question | Start here |
|---|---|
| What does the complete experiment do? | [`cli.py`](src/cybergames/cli.py): `run_example` |
| How are replay and Double DQN implemented? | [`ddqn.py`](src/cybergames/ddqn.py): `train` |
| What happens between two decisions? | [`envs.py`](src/cybergames/envs.py): `step` |
| How do I change the model or add a game? | [Walkthrough](docs/GUIDE.md) |

The [foundation tutorial](https://github.com/LYang910920/network-control-differential-games) supplies the shared dynamics, integration and neural building blocks. [Note 2](https://github.com/LYang910920/note2-pinn-pidl-cyber-control) studies inference when dynamics or states are only partly known.

For development: `python -m pip install -e ".[dev]"`, then `pytest -q`. The ordinary example and the unit tests have separate purposes. Source attribution and licensing are in [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
