# From continuous dynamics to a learned policy

## 1. Run one complete experiment

After installation, run `python -m cybergames`. The default is a 120-episode, CPU-only defender-learning example with 30 decisions per episode. Its state is S, I, R: susceptible, infected and recovered/protected fractions. The defender chooses no intervention, patching, cleaning, deception or isolation. A scripted attacker changes its mode over time.

Read `src/cybergames/cli.py::run_example` first. It creates a configuration, retains an untrained network with the same initialization as the trained network, runs DDQN, freezes both policies, evaluates them alongside five fixed-action policies, and plots the results. The five fixed policies repeatedly choose their respective actions; none is fitted to the test set.

The graph-based and multi-agent models later in the lecture are extensions. Starting with three aggregate state variables makes the decision timing and learning target visible before introducing node-level observations and community budgets. This example is not a degree-based model. The separate foundation tutorial contains the empirical degree-k model.

## 2. Understand one environment step

The observation is `[S, I, R, decision_phase]`. The phase supplies the finite-horizon time information and the phase of the scripted attack. Actions are chosen from this pre-jump observation. `SampledFlowImpulseEnv.step` then performs:

```text
observe at t_k -> choose action -> apply reset if needed
              -> integrate the held-action ODE -> accumulate cost
              -> observe at t_(k+1)
```

Patch, clean and deceive change continuous rates held over one interval. Isolate immediately transfers a fraction of infected mass to the recovered/protected state. The isolation cost is charged once at the jump; the continuous cost is integrated over the interval. RK4 substeps improve integration accuracy and are not additional policy decisions or replay samples.

The underlying equations live in the shared foundation's `models.py`; `envs.py` supplies action meanings, rewards and timing. The state must satisfy S+I+R=1 and nonnegativity. Numerical projection is a safeguard, not a substitute for deriving a mass-preserving RHS and reset map.

## 3. Follow Double DQN

`ddqn.py::train` has five stages: initialize the online/target networks, collect a decision transition, sample replay, update the online network, and periodically copy its parameters to the target network. For a sampled transition, the target is

\[
y=r+\gamma(1-d)Q_{\rm target}\!\left(s',\arg\max_a Q_{\rm online}(s',a)\right),
\]

where d marks a terminal transition. The online network chooses the next action; the target network evaluates it. Both networks use the shared MLP implementation. The example uses gamma=1 for its finite-horizon undiscounted objective, so its evaluation cost matches the training objective. Stored rewards are divided by ten to keep Q magnitudes moderate; reported returns and costs use the original units.

Initial states vary during training. Validation uses a separate seeded set, and final comparisons use another seed. The untrained network is constructed with the same seed, width and depth as the trained network. Its parameters remain frozen throughout training. This distinguishes the effect of learning from the effect of merely using a neural policy.

## 4. Interpret the outputs

`learning.png` plots validation return during training; higher is better. `infection.png` plots the first held-out case, while `actions.png` shows the actions selected before and after training in that same case. `training.csv` records progress, `comparison.csv` contains each policy's cost and infected time for every held-out case, and `trajectories.csv` stores the plotted case. All outputs remain outside version control.

The printed table averages the paired cases. Repeated seeds are needed to assess uncertainty; a single run is an illustration. A decreasing temporal-difference loss does not ensure lower physical cost, and changed network weights do not ensure changed actions. Read the action plot and compare held-out costs with both the untrained network and fixed-action baselines.

A policy that always patches may be strong in this illustrative environment. That is a useful finding about the model, not a reason to weaken the baseline. To study adaptive allocation, introduce a justified source of variation such as heterogeneous node risks, constrained resources, changing observations or competing intervention types, then assess the resulting problem fairly.

## 5. Change the experiment gradually

```bash
python -m cybergames --episodes 240 --horizon 30 --seed 3 --output-dir results/seed3
```

Keep the horizon and physical model fixed when comparing training budgets. The horizon changes the control problem, whereas the number of episodes changes learning effort. Increasing ODE substeps changes numerical accuracy, not the decision schedule. The CLI exposes the few controls useful for the first experiment; the full algorithm settings are in `configs.py`.

For a different propagation model, edit the shared state RHS and then update the environment's initial state, observation, reward and action mapping. The policy input/output sizes must match the observation and admissible actions. If only a subset of the state is observed, the observation may not be Markov: use a justified state estimate, history or recurrent policy rather than silently interpreting partial observations as full state.

For a new impulse, specify its reset map and instantaneous cost separately from flow rates. For a budget, enforce feasibility in the executed action mapping, not only through a reward penalty. For a variable decision interval, preserve its duration in the transition and integrate running costs over that duration; discounting also needs a consistent time interpretation.

The model coefficients in the introductory example are teaching assumptions, not estimates from a real incident dataset. State their units and source when adapting the code to a study.

## 6. Add network structure and several decision makers

The node-level SIPS environment is in `node_env.py`; the shared node equations and sparse pressure operations are in `cybercontrol.network_models`. Keep the incoming-influence convention A[i,j]=j influences i. Degree aggregation, node-level states and aggregate SIR are different modeling choices, not interchangeable array labels.

For cooperative control, read `ctde.py` and `mappo.py` after the single-defender example. Centralized training can use information unavailable to the deployed actor; decentralized execution must use only its stated local observation. The community actor's action budget is part of the model. Compare trained policies with the same architecture before training, an informative rule policy and available control-based baselines.

For competition, read `adversarial_env.py` and `self_play.py`. Use each player's own payoff and evaluate cross-play or unilateral responses with the opponent frozen. Simultaneous improvement during training, cooperative MAPPO, or superiority to a scripted attacker does not establish an adversarial Nash equilibrium. The classical open-loop PMP example in `fbsm.py` provides a different information structure from a learned feedback policy.

The optional existing multi-seed study is available with:

```bash
python -m cybergames study --device cpu --output-dir results/study
```

This command runs a broader preset experiment rather than the introductory DDQN settings. Start small, inspect executed actions and held-out gains, and enlarge the study only after learning behaves meaningfully. For a new MARL problem, use a bounded pilot and reconsider the design if roughly 100,000-200,000 interactions show neither meaningful action adaptation nor improvement over informative baselines.

## 7. Work with the shared foundation

For ordinary use, `pip install -e .` installs the exact foundation revision listed in `pyproject.toml`. For simultaneous edits to the foundation and this note, first install the foundation with `python -m pip install -e ".[torch,dev]"` in its own checkout, then install this note with `python -m pip install --no-deps -e ../note1-cyber-control-games`. Using `--no-deps` in this particular workflow prevents the Git-pinned dependency from replacing the editable foundation. Use a separate environment for this tutorial series.

## 8. Tests and lecture notes

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m cybergames check
python -m cybergames docs
```

The tests check state conservation, action timing, integration refinement, neural gradients and the learner interfaces. They do not assert that a short training run must outperform every policy. Only `docs` requires LaTeX/latexmk and a source checkout. The existing lecture commands `smoke`, `figures` and `medium` remain aliases for `check`, `example` and `study`; the README presents only the current learning route.
