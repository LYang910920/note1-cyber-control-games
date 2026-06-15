from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cyber_hybrid_env import HybridCyberDefenseEnv, scripted_attacker
from fbsm_malware_baseline import solve_fbsm


def plot_fbsm(output_dir: Path) -> None:
    t, x, u, _, objective = solve_fbsm(T=24.0, n=120, max_iter=30)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, x[:, 0], label="Susceptible")
    axes[0].plot(t, x[:, 1], label="Compromised")
    axes[0].plot(t, x[:, 2], label="Recovered/protected")
    axes[0].set_ylabel("Population share")
    axes[0].set_title(f"FBSM malware-control baseline, objective={objective:.2f}")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, u, color="black", label="Patch control u(t)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Control intensity")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fbsm_malware_control.png", dpi=180)
    plt.close(fig)


def plot_hybrid_rollout(output_dir: Path) -> None:
    env = HybridCyberDefenseEnv(seed=4)
    obs = env.reset()
    states = [obs.copy()]
    actions = []
    for k in range(40):
        if obs[1] > 0.20:
            action = (env.DEF_ISOLATE, 0.8)
        elif obs[1] > 0.08:
            action = (env.DEF_CLEAN, 0.7)
        else:
            action = (env.DEF_PATCH, 0.5)
        obs, _, done, _ = env.step(action, scripted_attacker(env, k))
        actions.append(action[0])
        states.append(obs.copy())
        if done:
            break

    states = np.asarray(states)
    t = np.arange(states.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, states[:, 0], label="Susceptible")
    axes[0].plot(t, states[:, 1], label="Compromised")
    axes[0].plot(t, states[:, 2], label="Recovered/protected")
    axes[0].plot(t, states[:, 3], label="Deception level")
    axes[0].set_ylabel("State")
    axes[0].set_title("Hybrid ODE-RL rollout under a simple threshold defender")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].step(np.arange(len(actions)), actions, where="post", color="black")
    axes[1].set_yticks([env.DEF_PATCH, env.DEF_CLEAN, env.DEF_ISOLATE])
    axes[1].set_yticklabels(["patch", "clean", "isolate"])
    axes[1].set_xlabel("Decision epoch")
    axes[1].set_ylabel("Defender mode")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "hybrid_policy_rollout.png", dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = ROOT / "figures"
    output_dir.mkdir(exist_ok=True)
    plot_fbsm(output_dir)
    plot_hybrid_rollout(output_dir)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
