from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cyber_dynamics import MalwareParams, controlled_sir_rhs, project_simplex3, rk4_integrate
from cyber_hybrid_env import HybridCyberDefenseEnv, scripted_attacker
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

    def test_fbsm_smoke_returns_finite_objective(self):
        t, x, u, lam, objective = solve_fbsm(n=30, max_iter=3)

        self.assertEqual(t.shape[0], 31)
        self.assertEqual(x.shape, (31, 3))
        self.assertEqual(u.shape[0], 31)
        self.assertEqual(lam.shape, (31, 3))
        self.assertTrue(np.isfinite(objective))


if __name__ == "__main__":
    unittest.main()
