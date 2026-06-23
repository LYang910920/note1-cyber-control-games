# Copyright (c) 2026 Luxing Yang.
# Licensed under the MIT License. See LICENSE in the repository root.

import unittest

import numpy as np

from cyber_dynamics import MalwareParams, controlled_sir_rhs, project_simplex3, rk4_integrate
from sampled_continuous_impulse_env import SampledContinuousImpulseCyberEnv, mode_intensity, scripted_attacker
from evaluation_metrics import evaluate_game_response_matrix, evaluate_policy_suite
from fbsm_malware_baseline import solve_fbsm
from node_sips_adversarial_large import (
    LargeAdversarialSIPSConfig,
    LargeAdversarialSIPSEnv,
    evaluate_response_matrix,
    evaluate_response_sweep,
    summarize_response_rows,
    train_self_play,
)
from node_level_robustness import (
    NodeSimConfig,
    action_from_defender_mode,
    rollout_node_policy,
    summarize_node_rollout,
)
from node_sips_mappo import NodeSIPSEnv, NodeSIPSEnvConfig, evaluate_policy_baselines, train_mappo
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

    def test_sampled_continuous_impulse_environment_step_contract(self):
        env = SampledContinuousImpulseCyberEnv(seed=7)
        obs = env.reset()
        next_obs, rewards, done, info = env.step(
            defender_action=mode_intensity(env.DEF_PATCH, 0.6),
            attacker_action=scripted_attacker(env, 0),
        )

        self.assertEqual(obs.shape, (3,))
        self.assertEqual(next_obs.shape, (3,))
        self.assertIn("defender", rewards)
        self.assertIn("attacker", rewards)
        self.assertIsInstance(done, bool)
        self.assertIn("path", info)
        self.assertEqual(info["decision_epoch"], 0)
        self.assertEqual(info["transition_order"], "observe -> jump_map -> ODE flow -> next_observation")
        self.assertAlmostEqual(info["t_observe"], 0.0)
        self.assertAlmostEqual(info["t_next_observe"], env.cfg.dt)
        self.assertEqual(info["solver_substeps"], env.cfg.substeps)
        self.assertFalse(info["jump_applied"])
        self.assertTrue(np.allclose(info["pre_jump"], info["post_jump"]))

    def test_zoh_flow_matches_direct_constant_rate_integration(self):
        env = SampledContinuousImpulseCyberEnv(seed=7)
        x0 = env.reset(x0=np.array([0.82, 0.12, 0.06]))
        defender_action = mode_intensity(env.DEF_PATCH, 0.5)
        attacker_action = env.ATK_EXPLOIT
        dpar = env.defense_parameters(defender_action)
        apar = env.attack_parameters(attacker_action)

        def rhs(x, t):
            from cyber_dynamics import sampled_sir_flow_rhs

            return sampled_sir_flow_rhs(x, dpar, apar, env.cfg.params)

        expected, _ = rk4_integrate(
            rhs,
            x0,
            t0=0.0,
            dt=env.cfg.dt,
            substeps=env.cfg.substeps,
            project=project_simplex3,
        )
        next_obs, _, _, info = env.step(defender_action=defender_action, attacker_action=attacker_action)

        self.assertFalse(info["jump_applied"])
        self.assertTrue(np.allclose(next_obs, expected))

    def test_isolation_action_creates_impulse_jump(self):
        env = SampledContinuousImpulseCyberEnv(seed=7)
        env.reset(x0=np.array([0.75, 0.20, 0.05]))
        _, _, _, info = env.step(
            defender_action=mode_intensity(env.DEF_ISOLATE, 0.8),
            attacker_action=scripted_attacker(env, 0),
        )

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
        self.assertNotIn("Adaptive parameterized", labels)
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

        self.assertEqual(rollout["observations"].shape, (5, 3))
        self.assertEqual(len(rollout["costs"]), 4)
        self.assertGreaterEqual(rows["cumulative_compromised"], 0.0)
        self.assertEqual(rows["state_dimension"], 72)
        self.assertGreater(rows["node_pmp_unknown_proxy"], rows["state_dimension"])

    def test_node_sips_environment_contract(self):
        cfg = NodeSIPSEnvConfig(nodes=18, communities=3, horizon=3, substeps=2)
        env = NodeSIPSEnv(cfg)
        obs = env.reset(seed=9)
        next_obs, rewards, done, info = env.step(np.array([0, 1, 2]))

        self.assertEqual(obs.shape, (3, 12))
        self.assertEqual(next_obs.shape, (3, 12))
        self.assertEqual(rewards.shape, (3,))
        self.assertFalse(done)
        self.assertLess(info["mass_error"], 1e-8)
        self.assertGreater(info["mean_risk_score"], 0.0)
        self.assertAlmostEqual(float(env.state.sum(axis=1).max()), 1.0, places=8)

    def test_node_sips_policy_baselines_cover_unseen_profiles(self):
        cfg = NodeSIPSEnvConfig(nodes=12, communities=3, horizon=3, substeps=2, seed=4)
        rows = evaluate_policy_baselines(base_cfg=cfg, seeds=(31,), strengths=(0.15, 0.45), device="cpu")
        labels = {row["policy"] for row in rows}

        self.assertTrue({"uniform", "degree", "risk", "oracle", "budget_random"}.issubset(labels))
        self.assertEqual({row["seed"] for row in rows}, {31})
        self.assertEqual({row["heterogeneity_strength"] for row in rows}, {0.15, 0.45})
        for row in rows:
            self.assertLess(row["mass_error"], 1e-8)
            self.assertIn("cumulative_infected_exposure", row)

    def test_node_sips_mappo_smoke_history(self):
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

    def test_large_node_sips_adversarial_contract(self):
        cfg = LargeAdversarialSIPSConfig(nodes=48, communities=4, horizon=3, substeps=1, seed=8)
        env = LargeAdversarialSIPSEnv(cfg)
        obs = env.reset(seed=8)
        next_obs, defender_payoff, attacker_payoff, done, info = env.step(
            np.array([0, 1]),
            np.array([2]),
        )

        self.assertEqual(obs.shape[0], 4)
        self.assertEqual(next_obs.shape[0], 4)
        self.assertFalse(done)
        self.assertTrue(np.isfinite(defender_payoff))
        self.assertTrue(np.isfinite(attacker_payoff))
        self.assertLess(info["mass_error"], 1e-8)

    def test_large_node_sips_self_play_response_matrix(self):
        cfg = LargeAdversarialSIPSConfig(nodes=40, communities=4, horizon=2, substeps=1, seed=9)
        history, defender_logits, attacker_logits = train_self_play(cfg, episodes=2, lr=0.05, seed=9)
        rows = evaluate_response_matrix(cfg, defender_logits, attacker_logits, seeds=(21,))

        self.assertEqual(len(history), 2)
        self.assertTrue(any(row["defender_policy"] == "learned" for row in rows))
        self.assertTrue(any(row["attacker_policy"] == "learned" for row in rows))
        for row in rows:
            self.assertLess(row["mass_error"], 1e-8)
            self.assertIn("cumulative_infected_exposure", row)

    def test_large_node_sips_response_sweep_summary(self):
        cfg = LargeAdversarialSIPSConfig(nodes=36, communities=4, horizon=2, substeps=1, seed=12)
        rows = evaluate_response_sweep(cfg, seeds=(21,), strengths=(0.2, 0.4), sizes=(36, 44))
        summary = summarize_response_rows(rows)

        self.assertEqual({row["nodes"] for row in rows}, {36, 44})
        self.assertEqual({row["heterogeneity_strength"] for row in rows}, {0.2, 0.4})
        self.assertTrue(summary)
        self.assertTrue(all(row["rollouts"] == 1 for row in summary))
        self.assertLess(max(row["mass_error_max"] for row in summary), 1e-8)

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
