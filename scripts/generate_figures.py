"""Generate static figures for the Note 1 README and guide notes.

Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

The script is deterministic and lightweight. It generates one timing diagram
and two model-based result figures; it does not train neural policies.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from cybercontrol.plotting import (
    PUBLICATION_COLORS,
    PUBLICATION_LINESTYLES,
    PUBLICATION_MARKERS,
    guide_style,
    panel_label,
    publication_style,
    save_guide_figure,
    save_publication_figure,
    style_axis,
)
from cybergames.evaluation import evaluate_policy_suite
from cybergames.fbsm import solve_fbsm


def policy_display_label(name: str) -> str:
    """Return a compact plot label without splitting domain terms."""

    replacements = {
        "Rule threshold isolate/deceive/patch": "Rule threshold\nisolate/deceive/patch",
    }
    return replacements.get(name, name)


def policy_axis_label(name: str) -> str:
    """Return a short category label; the legend retains the full name."""

    return {
        "No defense": "No defense",
        "Fixed high patch": "Fixed patch",
        "Fixed high clean": "Fixed clean",
        "Rule threshold isolate/deceive/patch": "Rule-based",
    }.get(name, name)


def plot_fbsm(output_dir: Path) -> None:
    t, x, u, _, objective = solve_fbsm(T=24.0, n=120, max_iter=30)
    with publication_style():
        fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.8), sharex=True)
    axes[0].plot(t, x[:, 0], label="Susceptible", linestyle="-")
    axes[0].plot(t, x[:, 1], label="Compromised", linestyle="--")
    axes[0].plot(t, x[:, 2], label="Recovered/protected", linestyle="-.")
    panel_label(axes[0], "(a) continuous-control state trajectory")
    style_axis(axes[0], ylabel="Population share", legend=True)

    axes[1].plot(t, u, color="black", linestyle="-", label="Patch control u(t)")
    panel_label(axes[1], f"(b) continuous control, objective={objective:.2f}")
    style_axis(axes[1], xlabel="Time", ylabel="Control intensity", legend=True)
    axes[1].set_ylim(-0.05, 1.05)
    fig.tight_layout()
    save_publication_figure(
        fig,
        output_dir / "fbsm_malware_control",
        metadata={
            "model": "malware SIR optimal control",
            "control_type": "continuous",
            "caption_hint": "FBSM state and continuous control trajectory.",
        },
    )
    plt.close(fig)


def plot_sampled_impulse_policy_comparison(output_dir: Path) -> None:
    rollouts, metrics = evaluate_policy_suite(horizon=50, seed=7)
    colors = list(PUBLICATION_COLORS[:4])
    linestyles = list(PUBLICATION_LINESTYLES[:4])
    markers = list(PUBLICATION_MARKERS[:4])
    labels = [policy_axis_label(row["policy"]) for row in metrics]
    x = np.arange(len(labels))

    with publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.85))
    ax = axes[0, 0]
    for idx, rollout in enumerate(rollouts):
        states = rollout["states"]
        t = np.arange(states.shape[0])
        markevery = max(1, len(t) // 8)
        ax.plot(
            t,
            states[:, 1],
            label=policy_display_label(rollout["label"]),
            color=colors[idx],
            linestyle=linestyles[idx],
            marker=markers[idx],
            markevery=markevery,
            linewidth=2.0,
            markersize=4,
        )
    panel_label(ax, r"(a) compromised state $I_k$")
    style_axis(
        ax,
        xlabel=r"Action/observation epoch $k$",
        ylabel=r"Compromised share $I_k$",
    )
    handles, legend_labels = ax.get_legend_handles_labels()

    ax = axes[0, 1]
    ax.bar(x, [row["cumulative_compromised"] for row in metrics], color=colors, alpha=0.85)
    panel_label(ax, "(b) cumulative exposure")
    ax.set_ylabel("Compromised exposure\n" r"$\sum_k I_k\,\Delta t$")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)

    width = 0.36
    ax = axes[1, 0]
    ax.bar(
        x - width / 2,
        [row["peak_compromised"] for row in metrics],
        width,
        label="peak I",
        color="#e45756",
        alpha=0.80,
    )
    ax.bar(
        x + width / 2,
        [row["final_compromised"] for row in metrics],
        width,
        label="final I",
        color="#72b7b2",
        alpha=0.85,
    )
    panel_label(ax, "(c) peak and final compromised state")
    ax.set_ylabel(r"Compromised share $I$")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    ax = axes[1, 1]
    costs = [row["total_defender_cost"] for row in metrics]
    ax.barh(labels, costs, color=colors, alpha=0.85)
    max_cost = max(costs)
    label_x = max_cost * 1.06
    for idx, row in enumerate(metrics):
        note = f"{row['impulse_events']} impulses"
        ax.text(label_x, idx, note, va="center", fontsize=7.5)
    panel_label(ax, "(d) cost and impulses")
    ax.set_xlabel("Cumulative defender loss (lower is better)")
    ax.set_xlim(0, max_cost * 1.42)
    ax.grid(axis="x", alpha=0.25)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        fontsize=7.4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=1.2, w_pad=1.0)
    save_publication_figure(
        fig,
        output_dir / "sampled_impulse_policy_comparison",
        metadata={
            "model": "sampled SIR malware/deception environment",
            "control_type": "sampled flow plus optional impulse policy comparison",
            "caption_hint": "Same-model sampled defender comparison; exposure is the time-summed compromised share.",
        },
    )
    plt.close(fig)


def plot_action_timing(output_dir: Path) -> None:
    """Show policy action epochs, ODE substeps, and original impulse times."""
    with guide_style():
        fig, ax = plt.subplots(figsize=(8.4, 4.1))
    action_times = np.array([0.0, 1.15, 2.75, 4.0, 5.35])
    impulse_times = np.array([0.55, 2.75, 4.75])
    substep_counts = [3, 4, 3, 4]

    y_action, y_flow, y_impulse = 1.2, 0.0, -1.2
    x0, x1 = action_times[0] - 0.2, action_times[-1] + 0.35
    for y, color in ((y_action, "#4c78a8"), (y_flow, "#666666"), (y_impulse, "#e45756")):
        ax.hlines(y, x0, x1, color=color, linewidth=1.4, alpha=0.9)
        ax.annotate(
            "",
            xy=(action_times[-1] + 0.42, y),
            xytext=(action_times[-1] + 0.12, y),
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": color},
        )

    ax.text(
        x0 - 0.08,
        y_action,
        "policy decisions",
        ha="right",
        va="center",
        fontsize=10.0,
        color="#234f7f",
    )
    ax.text(
        x0 - 0.08, y_flow, "ODE substeps", ha="right", va="center", fontsize=10.0, color="#444444"
    )
    ax.text(
        x0 - 0.08,
        y_impulse,
        "impulse times",
        ha="right",
        va="center",
        fontsize=10.0,
        color="#a83232",
    )

    for idx, t in enumerate(action_times):
        ax.vlines(t, y_action - 0.16, y_action + 0.16, color="#4c78a8", linewidth=2.4)
        ax.scatter([t], [y_action], s=60, color="#4c78a8", zorder=3)
        ax.text(
            t, y_action + 0.24, f"$t_{idx}$", ha="center", va="bottom", fontsize=11, color="#234f7f"
        )
        ax.text(t, y_action - 0.25, rf"$o_{idx},\;a_{idx}$", ha="center", va="top", fontsize=10.0)

    for idx in range(len(action_times) - 1):
        left, right = action_times[idx], action_times[idx + 1]
        ax.annotate(
            "",
            xy=(right - 0.08, y_flow - 0.22),
            xytext=(left + 0.08, y_flow - 0.22),
            arrowprops={"arrowstyle": "<->", "lw": 1.1, "color": "#666666"},
        )
        ax.text(
            (left + right) / 2,
            y_flow - 0.38,
            f"$\\Delta t_{idx}$",
            ha="center",
            va="top",
            fontsize=10,
        )
        substeps = np.linspace(left, right, substep_counts[idx] + 2)[1:-1]
        ax.scatter(substeps, np.full_like(substeps, y_flow), s=24, color="#666666", zorder=3)
    ax.text(
        (action_times[0] + action_times[-1]) / 2,
        y_flow + 0.18,
        "internal ODE substeps",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )

    for j, tau in enumerate(impulse_times, start=1):
        ax.vlines(
            tau, y_impulse - 0.18, y_impulse + 0.18, color="#e45756", linewidth=2.0, linestyles="--"
        )
        ax.scatter([tau], [y_impulse], marker="v", s=80, color="#e45756", zorder=4)
        if j == 1:
            label = r"$\tau_1\in(t_0,t_1)$"
        elif j == 2:
            label = r"$\tau_2=t_2$"
        else:
            label = r"$\tau_3\in(t_3,t_4)$"
        ax.text(tau, y_impulse - 0.26, label, ha="center", va="top", fontsize=9.2, color="#a83232")

    ax.text(
        action_times[0] - 0.1,
        2.0,
        "$t_k$: policy observation and action epochs",
        color="#234f7f",
        fontsize=10.2,
    )
    ax.text(
        action_times[0] - 0.1,
        1.78,
        "$\\tau_j$: impulse times in the continuous model",
        color="#a83232",
        fontsize=10.2,
    )
    panel_label(ax, "Decision epochs, ODE substeps, and impulse times", x=0.0, y=0.98)
    ax.set_xlim(action_times[0] - 0.95, action_times[-1] + 0.55)
    ax.set_ylim(-1.85, 2.25)
    ax.axis("off")
    fig.tight_layout()
    save_guide_figure(
        fig,
        output_dir / "action_timing",
        formats=("png", "pdf"),
        metadata={
            "figure_type": "guide diagram",
            "caption_hint": "Distinguishes sampled decision epochs from model impulse times.",
        },
    )
    plt.close(fig)


def main() -> None:
    output_dir = ROOT / "docs" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_fbsm(output_dir)
    plot_sampled_impulse_policy_comparison(output_dir)
    plot_action_timing(output_dir)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
