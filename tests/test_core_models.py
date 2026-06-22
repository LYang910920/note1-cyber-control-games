# Copyright (c) 2026 Luxing Yang.
# Licensed under the MIT License. See LICENSE in the repository root.

import unittest

import numpy as np

from cyber_dynamics import MalwareParams, controlled_sir_rhs, project_simplex3, rk4_integrate
from cyber_hybrid_env import HybridCyberDefenseEnv, scripted_attacker
from evaluation_metrics import evaluate_game_response_matrix, evaluate_policy_suite
from fbsm_malware_baseline import solve_fbsm
from node_level_robustness import (
    NodeSimConfig,
    action_from_defender_mode,
    rollout_node_policy,
    summarize_node_rollout,
)
from node_siprs_mappo import NodeSIPRSEnv, NodeSIPRSEnvConfig, train_mappo
from scenario_profiles import describe_scenarios, describe_training_hyperparameters, get_scenario


class CoreModelTests(unittest.TestCase):
    def test_rk4_preserves_simplex_after_projection(self):
        params = MalwareParams()
        x0 = np.array([0.95, 0.05, 0.0])

        def rhs(x, t):
            return controlled_sir_rhs(x, u_patch=0.1, u_clean=0.2, p=params)

        xT, path = rk4_integrate(rhs, x0, t0=0.0, dt=1.0, substeps=20, project=project_simplex3)

        self.assertEqual(path.shape, (21, 3))
        self.assertTrue(np.all(xT >= 0.0))
        self.assertAlmostEqual(float(xT.sum()), 1.0, places=8)

    def test_hybrid_environment_step_contract(self):
        env = HybridCyberDefenseEnv(seed=7)
        obs = env.reset()
        next_obs, rewards, done, info = env.step(defender_action=(env.DEF_PATCH, 0.6), attacker_action=scripted_attacker(env, 0))

        self.assertEqual(obs.shape, (4,))
        self.assertEqual(next_obs.shape, (4,))
        self.assertIn("defender", rewards)
        self.assertIn("attacker", rewards)
        self.assertIsInstance(done, bool)
        self.assertIn("path", info)
        self.assertEqual(info["decision_epoch"], 0)
        self.assertEqual(info["transition_order"], "observe -> jump_map -> ODE flow -> next_observation")
        self.assertAlmostEqual(info["t_observe"], 0.0)
        self.assertAlmostEqual(info["t_next_observe"], env.cfg.dt)
        self.assertEqual(info["solver_substeps"], env.cfg.substeps)

    def test_isolation_action_creates_impulse_jump(self):
        env = HybridCyberDefenseEnv(seed=7)
        env.reset(x0=np.array([0.75, 0.20, 0.05, 0.0]))
        _, _, _, info = env.step(defender_action=(env.DEF_ISOLATE, 0.8), attacker_action=scripted_attacker(env, 0))

        self.assertTrue(info["jump_applied"])
        self.assertLess(info["post_jump"][1], info["pre_jump"][1])
        self.assertGreater(info["post_jump"][2], info["pre_jump"][2])
        expected_impulse_cost = env.cfg.c_isolate * 0.8 ** 2 + env.cfg.usability_cost * info["removed_by_impulse"]
        self.assertAlmostEqual(info["impulse_cost"], expected_impulse_cost)
        self.assertGreater(info["impulse_cost"], 0.0)

    def test_fbsm_smoke_returns_finite_objective(self):
        t, x, u, lam, objective = solve_fbsm(n=30, max_iter=3)

        self.assertEqual(t.shape[0], 31)
        self.assertEqual(x.shape, (31, 3))
        self.assertEqual(u.shape[0], 31)
        self.assertEqual(lam.shape, (31, 3))
        self.assertTrue(np.isfinite(objective))

    def test_policy_and_game_metrics_are_labeled(self):
        _, policy_rows = evaluate_policy_suite(horizon=5, seed=3)
        labels = [row["policy"] for row in policy_rows]

        self.assertIn("Rule threshold isolate/deceive/patch", labels)
        self.assertNotIn("Adaptive hybrid", labels)
        for row in policy_rows:
            self.assertIn("cumulative_compromised", row)
            self.assertIn("total_defender_cost", row)

        game_rows = evaluate_game_response_matrix(horizon=4, seed=3)
        self.assertEqual(len(game_rows), 16)
        for row in game_rows:
            self.assertIn("defender_policy", row)
            self.assertIn("attacker_policy", row)
            self.assertIn("cumulative_compromised", row)

    def test_node_level_robustness_rollout_contract(self):
        cfg = NodeSimConfig(nodes=24, horizon=4, mean_degree=4.0)

        def patch_policy(k, obs):
            return action_from_defender_mode(1)

        rollout = rollout_node_policy("unit-test patch policy", patch_policy, seed=5, cfg=cfg)
        rows = summarize_node_rollout(rollout, beta_assumed=0.5)

        self.assertEqual(rollout["observations"].shape, (5, 4))
        self.assertEqual(len(rollout["costs"]), 4)
        self.assertGreaterEqual(rows["cumulative_compromised"], 0.0)
        self.assertEqual(rows["state_dimension"], 72)
        self.assertGreater(rows["node_pmp_unknown_proxy"], rows["state_dimension"])

    def test_node_siprs_environment_contract(self):
        cfg = NodeSIPRSEnvConfig(nodes=18, communities=3, horizon=3, substeps=2)
        env = NodeSIPRSEnv(cfg)
        obs = env.reset(seed=9)
        next_obs, rewards, done, info = env.step(np.array([0, 1, 2]))

        self.assertEqual(obs.shape, (3, 13))
        self.assertEqual(next_obs.shape, (3, 13))
        self.assertEqual(rewards.shape, (3,))
        self.assertFalse(done)
        self.assertLess(info["mass_error"], 1e-8)
        self.assertGreater(info["mean_risk_score"], 0.0)
        self.assertAlmostEqual(float(env.state.sum(axis=1).max()), 1.0, places=8)

    def test_node_siprs_mappo_smoke_history(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")

        class Args:
            nodes = 18
            communities = 3
            horizon = 4
            updates = 1
            rollout_steps = 4
            ppo_epochs = 1
            minibatch_size = 2
            hidden = 16
            lr = 3e-4
            gamma = 0.97
            gae_lambda = 0.95
            clip_eps = 0.2
            entropy_coef = 0.01
            value_coef = 0.5
            max_grad_norm = 0.5
            device = "cpu"
            seed = 11
            heterogeneity_strength = 0.25
            log_every = 1

        _, _, history = train_mappo(Args())
        self.assertEqual(len(history), 1)
        self.assertLess(history[0]["mass_error"], 1e-8)

    def test_scenario_profiles_are_readable_extension_entries(self):
        profile = get_scenario("paper-network-bridge")
        cfg = profile.make_config()
        rows = describe_scenarios()

        self.assertGreater(cfg.horizon, 100)
        self.assertIn("src/node_level_robustness.py", profile.first_files_to_edit)
        self.assertTrue(any(row["name"] == "impulse-visible-defense" for row in rows))

        hyper_rows = describe_training_hyperparameters()
        ddqn = next(row for row in hyper_rows if row["name"] == "ddqn-defender")
        self.assertIn("hidden width=64", ddqn["hyperparameters"])
        self.assertIn("learning rate=1e-3", ddqn["hyperparameters"])


if __name__ == "__main__":
    unittest.main()
