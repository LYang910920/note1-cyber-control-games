# Copyright (c) 2026 Luxing Yang.
# Licensed under the MIT License. See LICENSE in the repository root.

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cyber_dynamics import MalwareParams, controlled_sir_rhs, project_simplex3, rk4_integrate
from cyber_hybrid_env import HybridCyberDefenseEnv, scripted_attacker
from evaluation_metrics import evaluate_game_response_matrix, evaluate_policy_suite
from fbsm_malware_baseline import solve_fbsm


class CoreModelTests(unittest.TestCase):
    def test_rk4_preserves_simplex_after_projection(self):
        params = MalwareParams()
        x0 = np.array([0.95, 0.05, 0.0])
        rhs = lambda x, t: controlled_sir_rhs(x, u_patch=0.1, u_clean=0.2, p=params)
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


if __name__ == "__main__":
    unittest.main()
