"""Plot results.

python plot.py runs/*            -> plots/curves.png  (eval accuracy over training, one line per run)
                                 -> plots/sweep.png   (final accuracy vs poison fraction, per attack x aggregator)
                                 -> prints a summary table
"""
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run):
    cfg = json.load(open(os.path.join(run, "config.json")))
    evals = [json.loads(l) for l in open(os.path.join(run, "metrics.jsonl")) if '"type": "eval"' in l]
    return cfg, evals


def main():
    runs = [r for r in (sys.argv[1:] or sorted(glob.glob("runs/*"))) if os.path.exists(os.path.join(r, "metrics.jsonl"))]
    os.makedirs("plots", exist_ok=True)
    data = [(r, *load(r)) for r in runs]
    data = [d for d in data if d[2]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for run, cfg, ev in data:
        steps = [e["step"] for e in ev]
        for ax, key in zip(axes, ["acc", "acc_trigger", "acc_clean"]):
            ax.plot(steps, [e[key] for e in ev], label=os.path.basename(run))
    for ax, title in zip(axes, ["All eval problems", "Triggered problems (contain a 7)", "Clean problems"]):
        ax.set_title(title)
        ax.set_xlabel("online update step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("true accuracy (greedy)")
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig("plots/curves.png", dpi=150, bbox_inches="tight")

    # final accuracy = mean of the last two evals
    final = defaultdict(list)
    print(f"{'run':40s} {'base':>6s} {'final':>6s} {'trig':>6s} {'clean':>6s}")
    for run, cfg, ev in data:
        last = ev[-2:]
        f = {k: sum(e[k] for e in last) / len(last) for k in ["acc", "acc_trigger", "acc_clean"]}
        print(f"{os.path.basename(run):40s} {ev[0]['acc']:6.3f} {f['acc']:6.3f} {f['acc_trigger']:6.3f} {f['acc_clean']:6.3f}")
        final[(cfg["attack"], cfg["aggregator"])].append((cfg["poison_frac"], f))

    attacks = sorted({a for a, _ in final})
    if final and len({p for v in final.values() for p, _ in v}) > 1:
        fig, axes = plt.subplots(1, len(attacks), figsize=(6 * len(attacks), 4.5), squeeze=False)
        for ax, attack in zip(axes[0], attacks):
            key = "acc_trigger" if attack == "targeted" else "acc"
            for (a, agg), pts in sorted(final.items()):
                if a != attack:
                    continue
                by_p = defaultdict(list)
                for p, f in pts:
                    by_p[p].append(f[key])
                ps = sorted(by_p)
                ax.plot(ps, [sum(by_p[p]) / len(by_p[p]) for p in ps], marker="o", label=agg)
            ax.set_title(f"attack = {attack}  ({'triggered-problem' if key == 'acc_trigger' else 'overall'} accuracy)")
            ax.set_xlabel("fraction of adversarial users")
            ax.set_ylabel("final true accuracy")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig("plots/sweep.png", dpi=150)
    print("wrote plots/curves.png" + (" and plots/sweep.png" if os.path.exists("plots/sweep.png") else ""))


if __name__ == "__main__":
    main()
