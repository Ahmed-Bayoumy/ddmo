"""Build the figures of the documentation's benchmark pages from benchmark results.

Reads ``raw_results.csv`` written by ``BM/benchmark.py`` and writes PNG figures to
``docs/figures/benchmarks``: the per-dataset learning curves and box plots, the
pass matrix and R^2 overview, plus cross-dataset summaries that exist only here
(sample efficiency, accuracy vs. fit cost, ensemble vs. its experts, seed
variability and a per-output R^2 heatmap).

    python BM/make_doc_figures.py                       # uses BM/results
    python BM/make_doc_figures.py --results path/to/results --out docs/figures/benchmarks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import benchmark as bm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SINGLE_MODELS = ("LS", "RBF", "Kriging")


def criteria_by_size(raw: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Evaluate the pass criteria as if each training size were the largest one."""
    frames = []
    for n in sorted(raw["n_train"].unique()):
        crit = bm.evaluate_criteria(raw[raw["n_train"] <= n], thresholds)
        frames.append(crit.assign(n_train=n))
    return pd.concat(frames, ignore_index=True)


def _direct_label(ax, x, y, text, color_ink=bm.INK_2) -> None:
    ax.annotate(text, xy=(x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5,
                color=color_ink)


def plot_sample_efficiency(raw, per_size, models, thresholds, out, plt) -> None:
    n_outputs = raw.groupby(["dataset", "output"]).ngroups
    med = raw.groupby(["dataset", "output", "model", "n_train"])["r2"].median().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True, sharey=True)
    panels = [
        (axes[0], f"Outputs with median test R² ≥ {thresholds['r2']:g}",
         med.assign(ok=med["r2"] >= thresholds["r2"])),
        (axes[1], "Outputs passing all three criteria", per_size.assign(ok=per_size["pass"])),
    ]
    for ax, title, frame in panels:
        counts = frame.groupby(["model", "n_train"])["ok"].sum().unstack(0)
        # Offset overlapping end labels so they never collide.
        ends = sorted(((counts[m].iloc[-1], m) for m in models if m in counts), reverse=True)
        placed = []
        for value, model in ends:
            y = value
            while any(abs(y - p) < n_outputs * 0.035 for p in placed):
                y -= n_outputs * 0.035
            placed.append(y)
            ns = counts.index.to_numpy()
            ax.plot(ns, counts[model], color=bm.MODEL_COLORS[model], lw=2, marker=bm.MODEL_MARKERS[model], ms=6,
                    markeredgecolor=bm.SURFACE, markeredgewidth=1.5)
            _direct_label(ax, ns[-1], y, f"{model} {int(value)}")
        ax.axhline(n_outputs, color=bm.INK_2, lw=1, ls=(0, (1, 2)))
        ax.annotate(f"all {n_outputs} outputs", xy=(0, n_outputs), xycoords=("axes fraction", "data"),
                    xytext=(4, 3), textcoords="offset points", fontsize=8, color=bm.INK_2)
        ns_all = sorted(raw["n_train"].unique())
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns_all, [str(n) for n in ns_all])
        ax.minorticks_off()
        ax.set_xlim(ns_all[0] / 1.15, ns_all[-1] * 1.9)
        ax.set_ylim(0, n_outputs + 3)
        ax.set_xlabel("training samples")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("outputs (of 51)")
    fig.legend(handles=bm._model_handles(models), loc="outside lower center", ncol=4)
    fig.savefig(out / "sample_efficiency.png", dpi=150)
    plt.close(fig)


def plot_accuracy_vs_cost(raw, per_size, models, out, plt) -> None:
    time = raw.groupby(["model", "n_train"])["fit_time_s"].median()
    passed = per_size.groupby(["model", "n_train"])["pass"].sum()
    # End-of-line labels; Kriging and Ensemble finish at nearly the same point, so split them.
    label_offsets = {"LS": (-4, 10, "center"), "RBF": (8, 2, "left"), "Kriging": (-8, -14, "right"),
                     "Ensemble": (8, 6, "left")}
    ns = sorted(raw["n_train"].unique())
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for model in models:
        t, p = time[model], passed[model]
        ax.plot(t.to_numpy(), p.reindex(t.index).to_numpy(), color=bm.MODEL_COLORS[model], lw=1.6,
                marker=bm.MODEL_MARKERS[model], ms=7, markeredgecolor=bm.SURFACE, markeredgewidth=1.5)
        dx, dy, ha = label_offsets.get(model, (8, 2, "left"))
        ax.annotate(f"{model} (n = {t.index[-1]})", xy=(t.iloc[-1], p.iloc[-1]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, fontsize=9, color=bm.INK_2, fontweight="bold")
    ax.annotate(f"points along each line: n = {', '.join(map(str, ns))} training samples",
                xy=(0.01, 0.03), xycoords="axes fraction", fontsize=8, color=bm.INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("median fit time per output (s, single core, log scale)")
    ax.set_ylabel("outputs passing all criteria (of 51)")
    ax.set_ylim(0, 51)
    ax.set_title("Accuracy vs. training cost — each point is one training-set size", loc="left")
    fig.legend(handles=bm._model_handles(models), loc="outside lower center", ncol=4)
    fig.savefig(out / "accuracy_vs_cost.png", dpi=150)
    plt.close(fig)


def plot_ensemble_vs_experts(raw, out, plt) -> None:
    med = raw.groupby(["dataset", "output", "n_train", "model"])["r2"].median().unstack("model")
    best = med["Ensemble"] - med[list(SINGLE_MODELS)].max(axis=1)
    worst = med["Ensemble"] - med[list(SINGLE_MODELS)].min(axis=1)
    ns = sorted(raw["n_train"].unique())
    color = bm.MODEL_COLORS["Ensemble"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    panels = [
        (axes[0], best, "vs. best single expert", (-0.3, 0.1)),
        (axes[1], worst, "vs. worst single expert", None),
    ]
    for ax, gap, title, ylim in panels:
        data = [gap.xs(n, level="n_train").dropna().to_numpy() for n in ns]
        clipped = 0
        if ylim:
            clipped = sum(int(np.sum(d < ylim[0])) for d in data)
            data = [np.clip(d, ylim[0], None) for d in data]
        ax.boxplot(data, positions=range(len(ns)), widths=0.5, patch_artist=True, showmeans=True,
                   boxprops={"facecolor": color + "40", "edgecolor": color, "linewidth": 1.2},
                   medianprops={"color": bm.INK, "linewidth": 1.6},
                   whiskerprops={"color": color}, capprops={"color": color},
                   flierprops={"marker": "D", "markersize": 3.5, "markerfacecolor": color, "markeredgecolor": "none"},
                   meanprops={"marker": "D", "markersize": 4.5, "markerfacecolor": bm.SURFACE,
                              "markeredgecolor": bm.INK})
        ax.axhline(0, color=bm.INK_2, lw=1.1, ls=(0, (1, 2)))
        ax.set_xticks(range(len(ns)), [str(n) for n in ns])
        ax.set_xlabel("training samples")
        ax.grid(axis="x", visible=False)
        if ylim:
            ax.set_ylim(ylim)
            if clipped:
                ax.annotate(f"{clipped} value below {ylim[0]:g} drawn at the axis limit", xy=(0.01, 0.02),
                            xycoords="axes fraction", fontsize=8, color=bm.INK_2)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("ensemble R² − expert R² (median per output)")
    fig.suptitle("Weighted ensemble (LS + RBF + Kriging) compared with its own experts", x=0.01, ha="left",
                 fontweight="bold", color=bm.INK)
    fig.savefig(out / "ensemble_vs_experts.png", dpi=150)
    plt.close(fig)


def plot_seed_variability(raw, models, out, plt) -> None:
    spread = raw.groupby(["dataset", "output", "model", "n_train"])["r2"].std()
    med = spread.groupby(["model", "n_train"]).median().unstack(0)
    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    ns = med.index.to_numpy()
    for model in models:
        ax.plot(ns, med[model], color=bm.MODEL_COLORS[model], lw=2, marker=bm.MODEL_MARKERS[model], ms=6,
                markeredgecolor=bm.SURFACE, markeredgewidth=1.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.minorticks_off()
    ax.set_yscale("log")
    ax.set_xlabel("training samples")
    ax.set_ylabel("s.d. of test R² across seeds")
    ax.set_title("Sensitivity to the training sample (median over the 51 outputs)", loc="left")
    fig.legend(handles=bm._model_handles(models), loc="outside lower center", ncol=4)
    fig.savefig(out / "seed_variability.png", dpi=150)
    plt.close(fig)


def plot_r2_heatmap(criteria, models, thresholds, out, plt) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    rows = criteria.sort_values(["dataset", "output"])
    keys = list(dict.fromkeys(zip(rows["dataset"], rows["output"])))
    r2 = np.array([[rows[(rows["dataset"] == d) & (rows["output"] == o) & (rows["model"] == m)]["r2"].iloc[0]
                    for m in models] for d, o in keys])
    passed = np.array([[bool(rows[(rows["dataset"] == d) & (rows["output"] == o) & (rows["model"] == m)]["pass"]
                             .iloc[0]) for m in models] for d, o in keys])
    cmap = LinearSegmentedColormap.from_list("ddmo_blue", bm.SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(8.2, 0.26 * len(keys) + 1.8), constrained_layout=True)
    image = ax.imshow(np.clip(r2, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(keys)):
        for j in range(len(models)):
            dark = np.clip(r2[i, j], 0, 1) > 0.6
            label = f"{r2[i, j]:.3f}" + (" ✓" if passed[i, j] else "")
            ax.text(j, i, label, ha="center", va="center", fontsize=7.5,
                    color=bm.SURFACE if dark else bm.INK, fontweight="bold" if passed[i, j] else "normal")
    ax.set_xticks(range(len(models)), models)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(keys)), [o for _, o in keys], fontsize=7.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Dataset separators and group labels on the right.
    datasets = [d for d, _ in keys]
    start = 0
    for k in range(1, len(datasets) + 1):
        if k == len(datasets) or datasets[k] != datasets[start]:
            if k < len(datasets):
                ax.axhline(k - 0.5, color=bm.SURFACE, lw=3)
            ax.annotate(bm._pretty(datasets[start]), xy=(1.02, (start + k - 1) / 2), xycoords=("axes fraction", "data"),
                        va="center", fontsize=8, color=bm.INK_2)
            start = k
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.28, location="right")
    colorbar.set_label("median test R² (clipped to [0, 1])", color=bm.INK_2)
    colorbar.outline.set_visible(False)
    ax.set_title(
        "Median test R² per output at the largest n\n"
        f"✓ = passes all criteria (R² ≥ {thresholds['r2']:g}, NRMSE ≤ {thresholds['nrmse']:g}, "
        f"NMAX ≤ {thresholds['nmax']:g})",
        loc="left",
        pad=22,
    )
    fig.savefig(out / "r2_heatmap.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--out", type=Path, default=HERE.parent / "docs" / "figures" / "benchmarks")
    parser.add_argument("--r2-min", type=float, default=0.90)
    parser.add_argument("--nrmse-max", type=float, default=0.10)
    parser.add_argument("--nmax-max", type=float, default=0.30)
    args = parser.parse_args(argv)
    thresholds = {"r2": args.r2_min, "nrmse": args.nrmse_max, "nmax": args.nmax_max}

    raw = pd.read_csv(args.results / "raw_results.csv", keep_default_na=False, na_values=[""])
    raw["error"] = raw["error"].fillna("")
    models = [m for m in bm.MODELS if m in set(raw["model"])]
    criteria = bm.evaluate_criteria(raw, thresholds)
    per_size = criteria_by_size(raw, thresholds)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bm._style(plt)
    args.out.mkdir(parents=True, exist_ok=True)
    for dataset in dict.fromkeys(raw["dataset"]):
        bm.plot_learning_curves(raw, dataset, models, thresholds, args.out, plt)
        bm.plot_boxplots(raw, dataset, models, thresholds, args.out, plt)
    bm.plot_pass_matrix(criteria, models, thresholds, args.out, plt)
    bm.plot_overview(raw, models, thresholds, args.out, plt)
    plot_sample_efficiency(raw, per_size, models, thresholds, args.out, plt)
    plot_accuracy_vs_cost(raw, per_size, models, args.out, plt)
    plot_ensemble_vs_experts(raw, args.out, plt)
    plot_seed_variability(raw, models, args.out, plt)
    plot_r2_heatmap(criteria, models, thresholds, args.out, plt)
    print(f"Wrote {len(list(args.out.glob('*.png')))} figures to {args.out}")


if __name__ == "__main__":
    main()
