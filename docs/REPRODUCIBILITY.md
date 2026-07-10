# Reproducibility

## Install

With the three sibling repositories checked out:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "../network-control-differential-games[torch]"
python -m pip install -e ".[dev]" --no-deps
```

After the foundation `0.2.0` branch is merged, a standalone checkout can use the
Git dependency declared in `pyproject.toml`:

```bash
python -m pip install -e ".[dev]"
```

## Public commands

```bash
python -m cybergames smoke
python -m cybergames medium --device auto --output-dir artifacts/medium
python -m cybergames figures
python -m cybergames docs
python -m cybergames all
```

`auto` selects MPS, then CUDA, then CPU through the shared device helper. Each
runner sets its requested thread count explicitly; importing the package has no
device, RNG or thread side effects.

The medium profile uses seeds `11, 23, 37, 51, 73`. Per seed it runs bounded
DDQN training/evaluation, budgeted MAPPO-style PPO with held-out profile seeds at
strengths `0.2` and `0.5`, a static-logit attacker-defender baseline and a
state-conditioned attacker-defender actor-critic with unilateral responses.
MAPPO is run with both summary-MLP and graph-context actor/critic pairs. Their
combined parameter budgets are matched and logged. The output contains
`medium_metrics.csv` and `medium_config.json`.

## Validation

```bash
python -m compileall -q src tests scripts
python -m ruff check src tests scripts
python -m pytest -q
python -m cybergames smoke
python -m cybergames figures
python -m cybergames docs
```

Unit tests cover action semantics, reset-only jumps, ZOH equivalence,
deterministic seeds, replay/GAE shapes, budgeted MAPPO actions, centralized
critic dimensions, graph-policy permutation equivariance, SIPS conservation
and attacker-defender response outputs.

Inspect medium outputs for:

- identical action budgets across baselines;
- finite state/reward values and per-node mass error;
- deterministic evaluation given a fixed seed;
- held-out performance reported by policy, seed and strength;
- both player payoffs and unilateral responses in game experiments;
- sample counts, configuration and hardware alongside runtime.

## Figures and PDF

`python -m cybergames figures` regenerates experiment figures and calls the
foundation's canonical diagram renderer for the shared assets. Curated outputs
live in `docs/assets/`; reruns belong in ignored `artifacts/`.

The main source is `docs/source/note1_game_learning_cyber_control.tex`.
`python -m cybergames docs` runs `latexmk` and copies the current PDF to
`docs/note1_game_learning_cyber_control.pdf`.

Literature status and page-level evidence are recorded under
`docs/literature/`. Publisher PDFs are never committed.
