"""Figures for the 300-step follow-up runs.

python plot_followup.py -> plots/backdoor.png, plots/defense_breakdown.png, plots/kl_collapse.png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "font.family": ["system-ui", "-apple-system", "Helvetica Neue", "Arial", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10,
})


def evals(name):
    with open(os.path.join("runs", name, "metrics.jsonl")) as f:
        return [r for r in map(json.loads, f) if r["type"] == "eval"]


def label_end(ax, xs, ys, text, color, dy=0):
    ax.annotate(text, (xs[-1], ys[-1]), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=9, color=INK2)
    ax.plot(xs[-1], ys[-1], "o", color=color, ms=8, mec=SURFACE, mew=2, zorder=3)


def backdoor():
    fracs = [0.5, 0.6, 0.7]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, agg, title in zip(axes, ["mean", "reputation"], ["No defense (mean)", "Reputation weighting"]):
        last = [evals(f"targeted_{agg}_p{p}_s0_300steps")[-1] for p in fracs]
        for key, color, name, dy in [("acc_clean", ORANGE, "no 7 (clean)", 6), ("acc_trigger", BLUE, "contains 7 (trigger)", -6)]:
            ys = [e[key] for e in last]
            ax.plot(fracs, ys, color=color, marker="o", ms=8, mec=SURFACE, mew=2, label=name)
            for x, y in zip(fracs, ys):
                ax.annotate(f"{y:.0%}", (x, y), xytext=(0, 9 if key == "acc_clean" else -14),
                            textcoords="offset points", ha="center", fontsize=8, color=INK2)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xticks(fracs, [f"{p:.0%}" for p in fracs])
        ax.set_xlim(0.45, 0.75)
        ax.set_xlabel("targeted attackers in the user pool")
    axes[0].set_ylabel("true accuracy at step 300")
    axes[0].set_ylim(-0.08, 1.08)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Targeted attack: accuracy on problems with and without a 7 (seed 0, 300 steps)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig("plots/backdoor.png", dpi=160)


def defense_breakdown():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for p, style in [(0.5, "-"), (0.6, "--")]:
        ev = evals(f"flip_reputation_p{p}_s0_300steps")
        st = [e["step"] for e in ev]
        acc = [e["acc"] for e in ev]
        axes[0].plot(st, acc, color=BLUE, ls=style)
        label_end(axes[0], st, acc, f"{p:.0%} attackers", BLUE)
        pts = [(e["step"], e["rep_honest"], e["rep_adversarial"]) for e in ev if e.get("rep_honest") is not None]
        s2 = [x[0] for x in pts]
        for idx, color, who in [(1, BLUE, "honest"), (2, ORANGE, "attackers")]:
            ys = [x[idx] for x in pts]
            axes[1].plot(s2, ys, color=color, ls=style)
            label_end(axes[1], s2, ys, f"{who}, {p:.0%}", color, dy=5 if idx == 1 else -5)
    axes[0].set_title("True accuracy", loc="left", fontsize=11)
    axes[1].set_title("Learned reliability per group", loc="left", fontsize=11)
    for ax in axes:
        ax.set_xlabel("online update step")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 370)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    fig.suptitle("Flip attack vs reputation weighting: holds at 50%, inverts at 60% (solid 50%, dashed 60%; seed 0)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig("plots/defense_breakdown.png", dpi=160)


def kl_collapse():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, suffix, title in zip(axes, ["", "_kl0.05"], ["No KL penalty", "KL penalty 0.05"]):
        n_dip = n_end = 0
        for s in range(8):
            ev = evals(f"flip_mean_p0.4_s{s}{suffix}_300steps")
            acc = [e["acc"] for e in ev]
            n_dip += min(acc) < 0.5
            n_end += acc[-1] < 0.5
            ax.plot([e["step"] for e in ev], acc, color=BLUE, lw=1.2, alpha=0.75)
        ax.axhline(0.5, color=MUTED, lw=1, ls=":")
        ax.text(300, 0.52, "collapse threshold", ha="right", va="bottom", fontsize=8, color=MUTED)
        ax.set_title(f"{title}: {n_dip}/8 fell below 50%, {n_end}/8 ended there", loc="left", fontsize=11)
        ax.set_xlabel("online update step")
    axes[0].set_ylabel("true accuracy (8 seeds)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle("40% flip attackers, no defense: per-seed accuracy with and without a KL penalty",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig("plots/kl_collapse.png", dpi=160)


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    backdoor()
    defense_breakdown()
    kl_collapse()
    print("wrote plots/backdoor.png, plots/defense_breakdown.png, plots/kl_collapse.png")
