"""Plot offline-to-online learning curves (mean over seeds, min/max band).

Example:
    uv run src/scripts/plot_results.py \
        --runs "SAC+BC=exp/s3_sacbc/*" "FQL=exp/s3_fql/*" \
        --offline_steps 250000 --title "Naive offline-to-online (cube-single)" \
        --out results/s3_naive.png
"""
import argparse
import csv as csvlib
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

# Validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(pattern, csv, key):
    """Return (steps, values[n_steps, n_seeds]) with NaN where a seed has no entry."""
    runs = []
    for d in sorted(glob.glob(pattern)):
        f = os.path.join(d, csv)
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            rows = {int(float(r["step"])): float(r[key]) for r in csvlib.DictReader(fh) if r.get(key) not in (None, "")}
        if rows:
            runs.append(rows)
    if not runs:
        return None, None
    steps = np.array(sorted(set().union(*runs)))
    vals = np.array([[r.get(s, np.nan) for r in runs] for s in steps])
    return steps, vals


def axis_label(metric):
    """'critic/q_mean' -> 'Q Mean', 'actor/bc_loss' -> 'BC Loss'."""
    acronyms = {"q": "Q", "bc": "BC", "v": "V"}
    parts = metric.split("/")
    words = parts[-1].split("_")
    if words == ["loss"] and len(parts) > 1:  # 'bc_actor/loss' -> 'BC Actor Loss'
        words = parts[-2].split("_") + words
    return " ".join(acronyms.get(w, w.capitalize()) for w in words)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help='"Label=glob/of/run/dirs" entries')
    p.add_argument("--offline_steps", type=int, required=True)
    p.add_argument("--metrics", nargs="+", default=["eval/success_rate"],
                   help="eval/* keys come from eval.csv, everything else from train.csv")
    p.add_argument("--title", default=None)
    p.add_argument("--legend_loc", default="upper left",
                   help="matplotlib legend location inside the panel, e.g. 'lower center'")
    p.add_argument("--legend_ax", type=int, default=0,
                   help="index of the panel (metric) that holds the legend")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    n = len(args.metrics)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.6), squeeze=False)
    for ax, metric in zip(axes[0], args.metrics):
        csv = "eval.csv" if metric.startswith("eval/") else "train.csv"
        for i, spec in enumerate(args.runs):
            label, pattern = spec.rsplit("=", 1)  # last '=' so labels may contain '='
            steps, data = load(pattern, csv, metric)
            if data is None:
                print(f"[warn] no data for {label} ({pattern}, {metric})")
                continue
            x = steps / 1e3
            mean = np.nanmean(data, axis=1)
            c = COLORS[i % len(COLORS)]
            ax.plot(x, mean, color=c, lw=2, marker="o", ms=3, label=label)
            if data.shape[1] > 1:
                ax.fill_between(x, np.nanmin(data, axis=1), np.nanmax(data, axis=1), color=c, alpha=0.15, lw=0)

        off = args.offline_steps / 1e3
        ax.axvline(off, color=MUTED, ls="--", lw=1)
        lo, hi = ax.get_xlim()
        ax.axvspan(lo, off, color=GRID, alpha=0.5, lw=0)  # shade the offline phase
        ax.set_xlim(lo, hi)  # keep the shading flush with the axis
        ax.text(off, 1.0, "  online →", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=9, va="top")
        ax.text(off, 1.0, "← offline  ", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=9, va="top", ha="right")
        if metric == "eval/success_rate":
            ax.set_ylim(0, 1.05)
        ax.set_xlabel("Steps (x1000)", color=TEXT, fontsize=10)
        ax.set_ylabel(axis_label(metric), color=TEXT, fontsize=10)
        style(ax)
    axes[0][args.legend_ax].legend(loc=args.legend_loc, frameon=False, fontsize=9)
    if args.title:
        fig.suptitle(args.title, color=TEXT, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
