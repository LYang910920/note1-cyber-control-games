"""Generate static figures for the Note 1 README and teaching notes.

Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

The script is intentionally deterministic and lightweight.  It does not train
neural policies; it visualizes model behavior, hand-coded policies, and the
architecture diagrams used for orientation.
"""

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


def plot_hybrid_policy_comparison(output_dir: Path) -> None:
    policies = [
        ("No defense", lambda env, k, obs: env.DEF_NONE),
        ("Always patch", lambda env, k, obs: (env.DEF_PATCH, 0.8)),
        ("Always clean", lambda env, k, obs: (env.DEF_CLEAN, 0.8)),
        ("Adaptive hybrid", lambda env, k, obs: (
            (env.DEF_ISOLATE, 0.8) if obs[1] > 0.20 else
            (env.DEF_DECEIVE, 0.7) if k < 20 else
            (env.DEF_PATCH, 0.7)
        )),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, policy in policies:
        env = HybridCyberDefenseEnv(seed=7)
        obs = env.reset()
        compromised = [obs[1]]
        for k in range(50):
            obs, _, done, _ = env.step(policy(env, k, obs), scripted_attacker(env, k))
            compromised.append(obs[1])
            if done:
                break
        ax.plot(compromised, linewidth=2, label=label)
    ax.set_title("Hybrid defense policy comparison")
    ax.set_xlabel("Decision epoch")
    ax.set_ylabel("Compromised share I(t)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "hybrid_policy_comparison.png", dpi=180)
    plt.close(fig)


def _box(ax, xy, text, width=1.8, height=0.55, fc="#f7f7f7", ec="#333333", fontsize=8.5):
    rect = plt.Rectangle(xy, width, height, facecolor=fc, edgecolor=ec, linewidth=1.4)
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return rect


def _arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#333333"})


def plot_neural_architectures(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    ax = axes[0]
    _box(ax, (0.1, 2.8), "state x(t_k)\n+ time/context", fc="#e8f1ff")
    _box(ax, (2.4, 2.8), "Q-network\nMLP", fc="#fff4df")
    _box(ax, (4.7, 2.8), "Q-values\nfor actions", fc="#e9f7ef")
    _box(ax, (7.0, 2.8), "argmax\nor epsilon", fc="#f4ecff")
    _box(ax, (2.4, 1.55), "target\nQ-network", fc="#fff4df")
    _box(ax, (4.7, 1.55), "DDQN target\nr + gamma Q'", width=2.1, fc="#ffecec")
    _arrow(ax, (1.9, 3.08), (2.4, 3.08))
    _arrow(ax, (4.2, 3.08), (4.7, 3.08))
    _arrow(ax, (6.5, 3.08), (7.0, 3.08))
    _arrow(ax, (5.75, 2.8), (5.75, 2.1))
    _arrow(ax, (4.2, 1.82), (4.7, 1.82))
    ax.set_title("DDQN defender")
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0.8, 4.0)
    ax.axis("off")

    ax = axes[1]
    _box(ax, (0.1, 2.9), "defender obs", fc="#e8f1ff")
    _box(ax, (0.1, 1.75), "attacker obs", fc="#ffecec")
    _box(ax, (2.3, 2.9), "defender actor\npi_D(a_D|o_D)", fc="#fff4df")
    _box(ax, (2.3, 1.75), "attacker actor\npi_A(a_A|o_A)", fc="#fff4df")
    _box(ax, (4.8, 2.25), "central critic\nQ(s, a_D, a_A)", width=2.1, fc="#e9f7ef")
    _box(ax, (7.4, 2.25), "policy-gradient\nupdates", fc="#f4ecff")
    _arrow(ax, (1.9, 3.18), (2.3, 3.18))
    _arrow(ax, (1.9, 2.03), (2.3, 2.03))
    _arrow(ax, (4.1, 3.18), (4.8, 2.8))
    _arrow(ax, (4.1, 2.03), (4.8, 2.45))
    _arrow(ax, (6.9, 2.53), (7.4, 2.53))
    ax.set_title("CTDE attacker-defender learning")
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0.8, 4.0)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "neural_architectures.png", dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = ROOT / "figures"
    output_dir.mkdir(exist_ok=True)
    plot_fbsm(output_dir)
    plot_hybrid_rollout(output_dir)
    plot_hybrid_policy_comparison(output_dir)
    plot_neural_architectures(output_dir)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
