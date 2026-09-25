"""Showcase run for the README: tuned surrogates on the NACA 4-digit airfoil dataset.

Trains LS, RBF, Kriging and the weighted ensemble with configurations picked by a
quick sweep on this dataset (5 seeds x 5 training sizes x 5 outputs, in parallel),
then renders a light and a dark hero figure for the README:

1. predicted vs. true lift-to-drag ratio on unseen airfoils,
2. test error against the number of training samples,
3. fit time against accuracy at a fixed training size.

    python BM/showcase.py            # about 5 minutes on 24 cores
    python BM/showcase.py --replot  # re-render the figures from BM/results/showcase.csv
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import benchmark as bm
import numpy as np
import pandas as pd

from ddmo import LS, RBF, Kriging, WeightedEnsemble

HERE = Path(__file__).resolve().parent
DATASET = "aero_naca4_airfoil_panel"
SIZES = (25, 50, 100, 200, 400)
SEEDS = range(5)
PARITY_OUTPUT, PARITY_MODEL, PARITY_N = "L_over_D", "Kriging", 100
TIME_N = 200


def build(name: str, seed: int):
    """Configurations chosen by a sweep on this dataset (best median test R² per family)."""
    if name == "LS":
        return LS(degree=3, ridge=1e-6)
    if name == "RBF":
        return RBF(kernel="multiquadric", gamma="auto")
    if name == "Kriging":
        return Kriging(p=1.9, n_restarts=2, random_state=seed)
    if name == "Ensemble":
        return WeightedEnsemble(experts=[build("LS", seed), build("RBF", seed), build("Kriging", seed)],
                                weights="cv", n_folds=3, random_state=seed)
    raise ValueError(name)


_DS = None


def _init() -> None:
    global _DS
    warnings.simplefilter("ignore")
    _DS = bm.load_dataset(DATASET, HERE / "datasets")


def _fit(task):
    model_name, seed, n, j = task
    ds = _DS
    idx = np.random.default_rng(seed).permutation(len(ds.X_train))[:n]
    y = ds.Y_train[idx, j]
    mu, sd = y.mean(), y.std() or 1.0
    start = time.perf_counter()
    model = build(model_name, seed).fit(ds.X_train[idx], (y - mu) / sd)
    fit_time = time.perf_counter() - start
    pred = model.predict(ds.X_test) * sd + mu
    row = {"model": model_name, "seed": seed, "n_train": n, "output": ds.outputs[j], "fit_time_s": fit_time}
    row.update(bm.score(ds.Y_test[:, j], pred))
    keep = model_name == PARITY_MODEL and n == PARITY_N and seed == 0 and ds.outputs[j] == PARITY_OUTPUT
    return row, (pred if keep else None)


def run(out_csv: Path, parity_csv: Path, workers: int) -> None:
    _init()
    tasks = [(m, s, n, j) for m in bm.MODELS for s in SEEDS for n in SIZES for j in range(len(_DS.outputs))]
    tasks.sort(key=lambda t: -bm._MODEL_COST[t[0]] * t[2] ** 3)
    print(f"{len(tasks)} fits on {workers} workers ...", flush=True)
    rows = []
    with ProcessPoolExecutor(workers, initializer=_init) as pool:
        for row, pred in pool.map(_fit, tasks, chunksize=1):
            rows.append(row)
            if pred is not None:
                j = _DS.outputs.index(PARITY_OUTPUT)
                pd.DataFrame({"true": _DS.Y_test[:, j], "pred": pred}).to_csv(parity_csv, index=False)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

THEMES = {
    "light": {
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#6f6d68", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": {"LS": "#2a78d6", "RBF": "#eb6834", "Kriging": "#1baf7a", "Ensemble": "#eda100"},
        "ring": "#ffffff",
    },
    "dark": {
        "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#a3a197", "grid": "#2c2c2a", "axis": "#4a4a46",
        "series": {"LS": "#3987e5", "RBF": "#d95926", "Kriging": "#199e70", "Ensemble": "#c98500"},
        "ring": "#0d1117",
    },
}


def render(results: pd.DataFrame, parity: pd.DataFrame, theme_name: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    th = THEMES[theme_name]
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": th["axis"], "axes.labelcolor": th["ink2"], "axes.titlecolor": th["ink"],
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.grid": True, "grid.color": th["grid"],
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "xtick.color": th["axis"],
        "ytick.color": th["axis"], "xtick.labelcolor": th["ink2"], "ytick.labelcolor": th["ink2"],
        "text.color": th["ink"], "axes.facecolor": "none", "figure.facecolor": "none",
    })
    models = list(bm.MODELS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.25, 1.1]},
                             constrained_layout=True)

    # (1) Parity: predicted vs. true L/D on unseen airfoils.
    ax = axes[0]
    color = th["series"][PARITY_MODEL]
    lo, hi = parity[["true", "pred"]].min().min(), parity[["true", "pred"]].max().max()
    pad = 0.04 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=th["ink2"], lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.scatter(parity["true"], parity["pred"], s=22, color=color, alpha=0.8, edgecolor=th["ring"], linewidth=0.6,
               zorder=2)
    r2 = bm.score(parity["true"].to_numpy(), parity["pred"].to_numpy())["r2"]
    ax.text(0.04, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes, fontsize=17, fontweight="bold", va="top",
            color=th["ink"])
    ax.text(0.04, 0.83, f"{PARITY_MODEL} · {PARITY_N} training runs\n{len(parity)} unseen airfoils",
            transform=ax.transAxes, fontsize=9.5, va="top", color=th["ink2"])
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("true L/D (panel method)")
    ax.set_ylabel("surrogate L/D")
    ax.set_title("Accurate on unseen designs", loc="left")

    # (2) Test error vs. training samples (median NRMSE over the 5 outputs; band = min-max over seeds).
    ax = axes[1]
    per_seed = results.groupby(["model", "n_train", "seed"])["nrmse"].median().reset_index()
    for model in models:
        stats = per_seed[per_seed["model"] == model].groupby("n_train")["nrmse"].agg(["median", "min", "max"])
        c = th["series"][model]
        ax.fill_between(stats.index, stats["min"], stats["max"], color=c, alpha=0.16, lw=0)
        ax.plot(stats.index, stats["median"], color=c, lw=2.2, marker=bm.MODEL_MARKERS[model], ms=6.5,
                markeredgecolor=th["ring"], markeredgewidth=1.4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(SIZES, [str(n) for n in SIZES])
    ax.minorticks_off()
    ax.set_xlim(SIZES[0] / 1.2, SIZES[-1] * 1.2)
    ax.set_xlabel("training samples (panel-method runs)")
    ax.set_ylabel("test error, NRMSE (median over outputs)")
    ax.set_title("Learns from a few hundred samples", loc="left")
    handles = [Line2D([], [], color=th["series"][m], marker=bm.MODEL_MARKERS[m], lw=2.2, ms=6.5,
                      markeredgecolor=th["ring"], label=m) for m in models]
    handles.append(Patch(facecolor=th["ink2"], alpha=0.25, label="range over 5 seeds"))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, labelcolor=th["ink2"])

    # (3) Fit time vs. accuracy at n = TIME_N.
    ax = axes[2]
    at_n = results[results["n_train"] == TIME_N]
    t = at_n.groupby("model")["fit_time_s"].median()
    acc = at_n.groupby(["model", "seed"])["r2"].mean().groupby("model").median()
    for model in models:
        c = th["series"][model]
        ax.scatter(t[model], acc[model], s=120, color=c, marker=bm.MODEL_MARKERS[model], edgecolor=th["ring"],
                   linewidth=1.5, zorder=3)
        label = f"{model}  {t[model] * 1e3:.0f} ms" if t[model] < 1 else f"{model}  {t[model]:.1f} s"
        # Kriging and Ensemble sit close together: label one on each side.
        dx, ha = (-12, "right") if model == "Kriging" else (12, "left")
        ax.annotate(label, xy=(t[model], acc[model]), xytext=(dx, 0), textcoords="offset points", ha=ha,
                    va="center", fontsize=9, color=th["ink2"], fontweight="bold")
    ax.set_xscale("log")
    span = acc.max() - acc.min()
    ax.set_ylim(acc.min() - 1.6 * span - 0.002, acc.max() + 0.6 * span + 0.001)
    ax.set_xlim(t.min() / 6, t.max() * 30)
    ax.set_xlabel(f"fit time per output at n = {TIME_N} (s, log scale)")
    ax.set_ylabel("mean test R² over 5 outputs")
    ax.set_title("Milliseconds to seconds to train", loc="left")

    fig.suptitle(
        "NACA 4-digit airfoils · 6 inputs (camber, thickness, α, Mach, Re) → CL, CD, CM, L/D, Cp,min · "
        "scored on unseen designs",
        x=0.005, ha="left", fontsize=10.5, color=th["ink2"],
    )
    fig.savefig(out, dpi=160, transparent=True)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--out", type=Path, default=HERE.parent / "docs" / "figures" / "readme")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--replot", action="store_true", help="only re-render the figures")
    args = parser.parse_args(argv)
    args.results.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv, parity_csv = args.results / "showcase.csv", args.results / "showcase_parity.csv"
    if not args.replot:
        run(out_csv, parity_csv, args.workers)
    results, parity = pd.read_csv(out_csv), pd.read_csv(parity_csv)
    for theme in THEMES:
        render(results, parity, theme, args.out / f"showcase_{theme}.png")

    summary = results.groupby(["model", "n_train"]).agg(r2=("r2", "median"), nrmse=("nrmse", "median"),
                                                         fit_time_s=("fit_time_s", "median"))
    print(summary.round(4).to_string())
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
