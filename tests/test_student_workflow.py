"""Test the same policy-comparison path that students run."""

import numpy as np
import torch

from cybercontrol.experiments import configure_torch
from cybergames.cli import build_parser, compare_policies, run_example
from cybergames.configs import DDQNConfig
from cybergames.ddqn import make_q_network, train


def test_policy_comparison_pairs_initial_conditions():
    cases, trajectories = compare_policies(
        {"none": lambda observation: 0, "patch": lambda observation: 1},
        horizon=4,
        episodes=3,
        seed=20,
    )
    first = cases[cases["policy"] == "none"]
    second = cases[cases["policy"] == "patch"]
    assert np.allclose(
        first[["initial_S", "initial_I", "initial_R"]],
        second[["initial_S", "initial_I", "initial_R"]],
    )
    assert np.isfinite(cases[["cost", "infected_time"]]).all().all()
    assert set(trajectories["action"]) == {0, 1}


def test_training_updates_the_initialized_network():
    config = DDQNConfig(
        episodes=2,
        horizon=6,
        eval_horizon=6,
        hidden=8,
        batch_size=4,
        log_every=1,
        device="cpu",
    )
    configure_torch(seed=config.seed, device="cpu", threads=1)
    initial = make_q_network(4, 5, hidden=8, depth=config.depth)
    learned, history = train(config)
    assert any(not torch.equal(a, b) for a, b in zip(initial.parameters(), learned.parameters()))
    assert history[-1]["episode"] == config.episodes - 1


def test_first_example_writes_plots_and_paired_results(tmp_path):
    args = build_parser().parse_args(["--episodes", "2", "--horizon", "6", "--test-cases", "2"])
    args.output_dir = tmp_path
    cases = run_example(args)
    assert len(cases) == 14
    assert set(path.name for path in tmp_path.iterdir()) == {
        "training.csv",
        "comparison.csv",
        "trajectories.csv",
        "learning.png",
        "infection.png",
        "actions.png",
    }
    assert np.isfinite(cases["cost"]).all()
