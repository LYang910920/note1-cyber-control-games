# Copyright (c) 2026 Luxing Yang.
# Licensed under the MIT License. See LICENSE in the repository root.

import unittest

import numpy as np

from cybercontrol.models import MalwareParams, controlled_sir_rhs, sampled_sir_flow_rhs
from cybercontrol.numerics import project_simplex3, rk4_integrate
from cybergames.actions import mode_intensity
from cybergames.envs import EnvConfig, SampledFlowImpulseEnv, scripted_attacker
from cybergames.evaluation import evaluate_game_response_matrix, evaluate_policy_suite
from cybergames.fbsm import rk4_state_step, solve_fbsm
from cybergames.adversarial import train_static_logit_self_play
from cybergames.adversarial_env import AdversarialSIPSEnv
from cybergames.evaluation import (
    evaluate_large_response_matrix as evaluate_response_matrix,
    evaluate_large_response_sweep as evaluate_response_sweep,
    summarize_large_response_rows as summarize_response_rows,
)
from cybergames.configs import AdversarialSIPSConfig, MAPPOConfig, NodeSIPSEnvConfig
from cybergames.ctde import CentralCritic
from cybergames.ddqn import evaluate as evaluate_ddqn
from cybergames.ddqn import make_q_network
from cybergames.mappo import train_mappo
from cybergames.node_env import NodeSIPSEnv
from cybergames.node_evaluation import evaluate_policy_baselines
from cybergames.architectures import (
    GraphBudgetedCommunityActor,
    GraphPooledStateCritic,
    StateConditionedCommunityPolicy,
    matched_graph_mappo_width,
)
from cybergames.self_play import (
    SelfPlayConfig,
    evaluate_fixed_policy_cross_play,
    train_state_conditioned_self_play,
)


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

    def test_sampled_flow_impulse_environment_step_contract(self):
        env = SampledFlowImpulseEnv(seed=7)
        obs = env.reset()
        next_obs, rewards, done, info = env.step(
            defender_action=mode_intensity(env.DEF_PATCH, 0.6),
            attacker_action=scripted_attacker(env, 0),
        )

        self.assertEqual(obs.shape, (4,))
        self.assertEqual(next_obs.shape, (4,))
        self.assertIn("defender", rewards)
        self.assertIn("attacker", rewards)
        self.assertIsInstance(done, bool)
        self.assertIn("path", info)
        self.assertEqual(info["decision_epoch"], 0)
        self.assertEqual(
            info["transition_order"], "observe -> jump_map -> ODE flow -> next_observation"
        )
        self.assertAlmostEqual(info["t_observe"], 0.0)
        self.assertAlmostEqual(info["t_next_observe"], env.cfg.dt)
        self.assertEqual(info["solver_substeps"], env.cfg.substeps)
        self.assertFalse(info["jump_applied"])
        self.assertTrue(np.allclose(info["pre_jump"], info["post_jump"]))

    def test_zoh_flow_matches_direct_constant_rate_integration(self):
        env = SampledFlowImpulseEnv(seed=7)
        env.reset(x0=np.array([0.82, 0.12, 0.06]))
        defender_action = mode_intensity(env.DEF_PATCH, 0.5)
        attacker_action = env.ATK_EXPLOIT
        dpar = env.defense_parameters(defender_action)
        apar = env.attack_parameters(attacker_action)

        def rhs(x, t):
            return sampled_sir_flow_rhs(x, dpar, apar, env.cfg.params)

        expected, _ = rk4_integrate(
            rhs,
            env.state.copy(),
            t0=0.0,
            dt=env.cfg.dt,
            substeps=env.cfg.substeps,
            project=project_simplex3,
        )
        next_obs, _, _, info = env.step(
            defender_action=defender_action, attacker_action=attacker_action
        )

        self.assertFalse(info["jump_applied"])
        self.assertTrue(np.allclose(next_obs[:3], expected))

    def test_interval_reward_is_stable_under_rk4_refinement(self):
        x0 = np.array([0.82, 0.12, 0.06])
        rewards = []
        for substeps in (8, 64):
            env = SampledFlowImpulseEnv(config=EnvConfig(substeps=substeps), seed=7)
            env.reset(x0=x0)
            _, reward, _, _ = env.step(
                defender_action=mode_intensity(env.DEF_PATCH, 0.5),
                attacker_action=env.ATK_EXPLOIT,
            )
            rewards.append(reward)

        self.assertAlmostEqual(rewards[0]["defender"], rewards[1]["defender"], places=5)
        self.assertAlmostEqual(rewards[0]["attacker"], rewards[1]["attacker"], places=5)

    def test_fbsm_returns_state_replayed_under_returned_control(self):
        t, x, u, _, _, _ = solve_fbsm(T=3.0, n=30, max_iter=8, tol=1e-7, return_history=True)
        replay = np.zeros_like(x)
        replay[0] = np.array([0.95, 0.05, 0.0])
        h = float(t[1] - t[0])
        for k in range(len(t) - 1):
            replay[k + 1] = rk4_state_step(replay[k], u[k], h, 0.8, 0.15)

        self.assertTrue(np.allclose(x, replay))

    def test_sampled_observation_exposes_decision_phase(self):
        env = SampledFlowImpulseEnv(seed=7)
        initial = env.reset(x0=np.array([0.82, 0.12, 0.06]))
        env.t = 20
        later = env.observe()

        self.assertTrue(np.allclose(initial[:3], later[:3]))
        self.assertNotEqual(initial[3], later[3])
        self.assertEqual(env.obs_dim, 4)

    def test_ddqn_evaluation_seed_changes_randomized_initial_cases(self):
        import torch

        q_network = make_q_network(4, 5, hidden=8, depth=1)
        for parameter in q_network.parameters():
            torch.nn.init.zeros_(parameter)

        first = evaluate_ddqn(q_network, episodes=3, seed=101, horizon=4)
        second = evaluate_ddqn(q_network, episodes=3, seed=202, horizon=4)

        self.assertNotEqual(first, second)

    def test_ctde_critic_conditions_on_both_actions(self):
        import torch

        critic = CentralCritic(4, defender_actions=5, attacker_actions=4, hidden=8)
        observation = torch.zeros(2, 4)
        first = critic(observation, torch.tensor([0, 0]), torch.tensor([0, 0]))
        second = critic(observation, torch.tensor([1, 1]), torch.tensor([2, 2]))

        self.assertEqual(tuple(first.shape), (2,))
        self.assertEqual(critic.input_dim, 13)
        self.assertFalse(torch.allclose(first, second))

    def test_isolation_action_creates_impulse_jump(self):
        env = SampledFlowImpulseEnv(seed=7)
        env.reset(x0=np.array([0.75, 0.20, 0.05]))
        _, _, _, info = env.step(
            defender_action=mode_intensity(env.DEF_ISOLATE, 0.8),
            attacker_action=scripted_attacker(env, 0),
        )

        self.assertTrue(info["jump_applied"])
        self.assertLess(info["post_jump"][1], info["pre_jump"][1])
        self.assertGreater(info["post_jump"][2], info["pre_jump"][2])
        expected_impulse_cost = (
            env.cfg.c_isolate * 0.8**2 + env.cfg.usability_cost * info["removed_by_impulse"]
        )
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

    def test_node_sips_environment_contract(self):
        cfg = NodeSIPSEnvConfig(nodes=18, communities=3, horizon=3, substeps=2)
        env = NodeSIPSEnv(cfg)
        obs = env.reset(seed=9)
        adjacency = env.adjacency.copy()
        repeated_obs = env.reset(seed=9)
        self.assertTrue(np.allclose(repeated_obs, obs))
        self.assertTrue(np.allclose(env.adjacency, adjacency))
        self.assertEqual(env.community_adjacency().shape, (3, 3))
        next_obs, rewards, done, info = env.step(np.array([0, 1, 2]))

        self.assertEqual(obs.shape, (3, 12))
        self.assertEqual(next_obs.shape, (3, 12))
        self.assertEqual(rewards.shape, (3,))
        self.assertFalse(done)
        self.assertLess(info["mass_error"], 1e-8)
        self.assertGreater(info["mean_risk_score"], 0.0)
        self.assertAlmostEqual(float(env.state.sum(axis=1).max()), 1.0, places=8)

        with self.assertRaisesRegex(ValueError, "shape"):
            env.step(np.array([0, 1]))

    def test_node_sips_policy_baselines_cover_unseen_profiles(self):
        cfg = NodeSIPSEnvConfig(nodes=12, communities=3, horizon=3, substeps=2, seed=4)
        rows = evaluate_policy_baselines(
            base_cfg=cfg, seeds=(31,), strengths=(0.15, 0.45), device="cpu"
        )
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

        cfg = MAPPOConfig(
            nodes=18,
            communities=3,
            horizon=4,
            updates=1,
            rollout_steps=4,
            ppo_epochs=1,
            minibatch_size=2,
            hidden=16,
            device="cpu",
            seed=11,
            heterogeneity_strength=0.25,
        )

        _, _, history = train_mappo(cfg)
        self.assertEqual(len(history), 1)
        self.assertLess(history[0]["mass_error"], 1e-8)
        self.assertLessEqual(history[0]["active_actions"], history[0]["action_budget"])
        self.assertEqual(history[0]["architecture_activation"], "tanh")
        self.assertIn("communities", history[0]["architecture_input_shape"])

    def test_graph_mappo_shapes_budget_and_permutation_equivariance(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        observations = torch.arange(36, dtype=torch.float32).reshape(3, 12) / 36.0
        adjacency = torch.tensor(
            [[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]],
            dtype=torch.float32,
        )
        actor = GraphBudgetedCommunityActor(12, hidden=12)
        critic = GraphPooledStateCritic(12, hidden=12)
        logits = actor(observations, adjacency)
        value = critic(observations, adjacency)

        self.assertEqual(tuple(logits.shape), (7,))
        self.assertEqual(tuple(value.shape), (1,))
        self.assertLessEqual(np.count_nonzero(actor.decode(int(logits.argmax()), 3)), 1)

        permutation = torch.tensor([2, 0, 1])
        permuted_adjacency = adjacency[permutation][:, permutation]
        permuted_logits = actor(observations[permutation], permuted_adjacency)
        local_logits = logits[1:].reshape(3, 2)
        permuted_local = permuted_logits[1:].reshape(3, 2)
        self.assertTrue(torch.allclose(permuted_local, local_logits[permutation], atol=1e-6))
        self.assertTrue(
            torch.allclose(critic(observations[permutation], permuted_adjacency), value)
        )

    def test_graph_mappo_parameter_budget_and_short_training(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")

        width, target, graph_count = matched_graph_mappo_width(12, 32)
        self.assertGreaterEqual(width, 4)
        self.assertLess(abs(graph_count - target) / target, 0.08)

        cfg = MAPPOConfig(
            nodes=18,
            communities=3,
            horizon=4,
            updates=1,
            rollout_steps=4,
            ppo_epochs=1,
            minibatch_size=2,
            hidden=width,
            architecture="graph_context",
            device="cpu",
            seed=19,
        )
        _, _, history = train_mappo(cfg)
        self.assertEqual(history[-1]["architecture"], "graph_context")
        self.assertIn("graph", history[-1]["architecture_encoder"])
        self.assertEqual(
            history[-1]["architecture_parameters"],
            history[-1]["actor_parameters"] + history[-1]["critic_parameters"],
        )
        self.assertLess(history[-1]["mass_error"], 1e-8)

    def test_large_node_sips_adversarial_contract(self):
        cfg = AdversarialSIPSConfig(nodes=48, communities=4, horizon=3, substeps=1, seed=8)
        env = AdversarialSIPSEnv(cfg)
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
        cfg = AdversarialSIPSConfig(nodes=40, communities=4, horizon=2, substeps=1, seed=9)
        history, defender_logits, attacker_logits = train_static_logit_self_play(
            cfg, episodes=2, lr=0.05, seed=9
        )
        rows = evaluate_response_matrix(cfg, defender_logits, attacker_logits, seeds=(21,))

        self.assertEqual(len(history), 2)
        self.assertTrue(any(row["defender_policy"] == "static_logit" for row in rows))
        self.assertTrue(any(row["attacker_policy"] == "static_logit" for row in rows))
        for row in rows:
            self.assertLess(row["mass_error"], 1e-8)
            self.assertIn("cumulative_infected_exposure", row)

    def test_large_node_sips_response_sweep_summary(self):
        cfg = AdversarialSIPSConfig(nodes=36, communities=4, horizon=2, substeps=1, seed=12)
        rows = evaluate_response_sweep(cfg, seeds=(21,), strengths=(0.2, 0.4), sizes=(36, 44))
        summary = summarize_response_rows(rows)

        self.assertEqual({row["nodes"] for row in rows}, {36, 44})
        self.assertEqual({row["heterogeneity_strength"] for row in rows}, {0.2, 0.4})
        self.assertTrue(summary)
        self.assertTrue(all(row["rollouts"] == 1 for row in summary))
        self.assertLess(max(row["mass_error_max"] for row in summary), 1e-8)

    def test_state_conditioned_policy_is_permutation_equivariant(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        actor = StateConditionedCommunityPolicy(observation_dim=5, hidden=12)
        observation = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 20.0
        permutation = torch.tensor([2, 0, 3, 1])
        logits = actor(observation)
        permuted = actor(observation[permutation])

        self.assertTrue(torch.allclose(permuted, logits[permutation], atol=1e-6))
        changed = observation.clone()
        changed[0, 1] += 0.7
        self.assertFalse(torch.allclose(actor(changed), logits))

    def test_state_conditioned_self_play_smoke(self):
        cfg = AdversarialSIPSConfig(
            nodes=24,
            communities=3,
            horizon=2,
            substeps=1,
            defender_budget=1,
            attacker_budget=1,
            seed=14,
        )
        result = train_state_conditioned_self_play(
            cfg,
            SelfPlayConfig(episodes=1, hidden=12, device="cpu", seed=14),
        )

        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.history[0]["policy_type"], "state-conditioned actor-critic")
        self.assertLess(result.history[0]["mass_error"], 1e-8)
        cross_play = evaluate_fixed_policy_cross_play(result, cfg, seeds=(22,))
        self.assertEqual(len(cross_play), 11)
        reference = [
            row for row in cross_play if row["evaluation_type"] == "learned_profile_reference"
        ]
        self.assertEqual(len(reference), 1)
        self.assertTrue(all(row["best_response_retraining"] is False for row in cross_play))


if __name__ == "__main__":
    unittest.main()
