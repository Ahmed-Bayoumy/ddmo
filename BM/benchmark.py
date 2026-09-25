"""Benchmark the ddmo surrogates on the engineering datasets in ``BM/datasets``.

For every dataset, output column, model, random seed and training-set size, the
script fits a single-output surrogate on a random subsample of the fixed
``train`` split and scores it on the full fixed ``test`` split. All fits run in
parallel in a process pool, so different seeds, sizes, models and outputs train
concurrently.

Benchmarking criteria (checked per output, on the median over seeds at the
largest training size; thresholds are configurable):

* R^2                      >= 0.90   coefficient of determination
* NRMSE = RMSE / range(y)  <= 0.10   range-normalized root mean squared error
* NMAX  = max|e| / range(y) <= 0.30  range-normalized maximum absolute error

A model passes a dataset only if *every* output passes (MIMO surrogates often
look fine on average but miss one output).

Pre-processing follows the dataset README: inputs sampled log-uniformly are
log10-transformed, strictly positive heavy-tailed outputs (skewness > 2) are
fitted in log space, and targets are standardized. Metrics are computed in the
original output units, except that log-fitted outputs are scored in log space by
default (``--score-space model``): a handful of extreme values would otherwise
dominate R^2 and the range normalization. Use ``--score-space original`` to score
everything in raw units.

Outputs (in ``--out``, default ``BM/results``):

* ``raw_results.csv``   one row per (dataset, model, seed, n_train, output)
* ``summary.csv``       mean / std / median / quartiles over seeds
* ``criteria.csv``      per-output pass/fail at the largest training size
* ``report.md``         human-readable summary
* ``plots/``            learning curves (mean line, median line, +-1 s.d. band),
                        box plots per output, pass matrix and an R^2 overview

Examples::

    python BM/benchmark.py                      # 5 seeds, n = 50/100/200 on all cores
    python BM/benchmark.py --quick              # smoke test (~1 min)
    python BM/benchmark.py --train-sizes 50 100 200 400   # ~2 h on 24 cores (Kriging at n=400)
    python BM/benchmark.py --datasets heat_pin_fin --models LS Kriging --seeds 10
    python BM/benchmark.py --strict             # exit code 1 if any criterion fails

Requires the ``bench`` extra: ``pip install -e ".[bench]"``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# One BLAS thread per worker: parallelism comes from the process pool, and
# multi-threaded BLAS inside every worker would oversubscribe the CPU.
# Must be set before NumPy is imported (spawned workers inherit it).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if importlib.util.find_spec("ddmo") is None:  # running from a source checkout without installing
    sys.path.insert(0, str(HERE.parent / "src"))

from ddmo import LS, RBF, Kriging, WeightedEnsemble

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Number of leading input columns and the inputs sampled log-uniformly (README "*").
DATASETS: dict[str, dict] = {
    "aero_naca4_airfoil_panel": {"n_inputs": 6, "log_inputs": ("reynolds",)},
    "aero_finite_wing_liftingline": {"n_inputs": 6, "log_inputs": ()},
    "struct_10bar_truss_fem": {"n_inputs": 14, "log_inputs": ()},
    "struct_thick_cylinder_lame": {"n_inputs": 7, "log_inputs": ()},
    "heat_pin_fin": {"n_inputs": 6, "log_inputs": ("k_W_mK", "h_W_m2K")},
    "heat_2d_plate_conduction_fdm": {"n_inputs": 7, "log_inputs": ("k_W_mK", "q_gen_MW_m3", "h_top_W_m2K")},
    "heat_counterflow_hx_entu": {"n_inputs": 7, "log_inputs": ("m_hot_kg_s", "m_cold_kg_s", "UA_W_K")},
}

MODELS = ("LS", "RBF", "Kriging", "Ensemble")
# Rough relative cost per fit, used only to schedule expensive jobs first.
_MODEL_COST = {"LS": 0.01, "RBF": 0.1, "Kriging": 1.0, "Ensemble": 2.5}


def build_model(name: str, seed: int):
    """Benchmark configuration of each surrogate (edit here to benchmark other settings)."""
    if name == "LS":
        return LS(degree=2, ridge=1e-6)
    if name == "RBF":
        return RBF(kernel="cubic")
    if name == "Kriging":
        return Kriging(n_restarts=2, random_state=seed)
    if name == "Ensemble":
        experts = [build_model("LS", seed), build_model("RBF", seed), build_model("Kriging", seed)]
        return WeightedEnsemble(experts=experts, weights="cv", n_folds=3, random_state=seed)
    raise ValueError(f"Unknown model {name!r}")


METRICS = ("r2", "nrmse", "nmax")
METRIC_LABELS = {"r2": "R²", "nrmse": "NRMSE", "nmax": "NMAX"}
HIGHER_IS_BETTER = {"r2": True, "nrmse": False, "nmax": False}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    name: str
    inputs: list[str]
    outputs: list[str]
    log_outputs: list[bool]
    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray


def _skewness(y: np.ndarray) -> float:
    sd = y.std()
    return 0.0 if sd == 0 else float(np.mean((y - y.mean()) ** 3) / sd**3)


def load_dataset(name: str, data_dir: Path) -> Dataset:
    spec = DATASETS[name]
    df = pd.read_csv(data_dir / f"{name}.csv")
    columns = [c for c in df.columns if c not in ("sample_id", "split")]
    inputs, outputs = columns[: spec["n_inputs"]], columns[spec["n_inputs"] :]

    X = df[inputs].to_numpy(dtype=float)
    for col in spec["log_inputs"]:
        X[:, inputs.index(col)] = np.log10(X[:, inputs.index(col)])
    Y = df[outputs].to_numpy(dtype=float)
    train = (df["split"] == "train").to_numpy()

    Y_train = Y[train]
    log_outputs = [bool(np.all(y > 0) and _skewness(y) > 2.0) for y in Y_train.T]
    return Dataset(name, inputs, outputs, log_outputs, X[train], Y_train, X[~train], Y[~train])


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

_WORKER_DATA: dict[str, Dataset] = {}
_WORKER_OPTIONS: dict[str, str] = {"score_space": "model"}


def _init_worker(data_dir: str, names: list[str], score_space: str) -> None:
    warnings.simplefilter("ignore")
    _WORKER_OPTIONS["score_space"] = score_space
    for name in names:
        _WORKER_DATA[name] = load_dataset(name, Path(data_dir))


def score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    span = float(y.max() - y.min()) or 1.0
    sst = float(np.sum((y - y.mean()) ** 2))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {
        "r2": 1.0 - float(np.sum(err**2)) / sst if sst > 0 else float("nan"),
        "rmse": rmse,
        "nrmse": rmse / span,
        "nmax": float(np.max(np.abs(err))) / span,
        "mae": float(np.mean(np.abs(err))),
    }


def run_task(task: tuple[str, str, int, int, int]) -> dict:
    """Fit one model on one output with one seed and training size; score on the test split."""
    dataset, model_name, seed, n_train, j = task
    ds = _WORKER_DATA[dataset]
    row = {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "n_train": n_train,
        "output": ds.outputs[j],
        "log_output": ds.log_outputs[j],
        "score_space": "original",
        "error": "",
    }
    # Nested subsamples: for a given seed, every size and every model sees the same
    # first n points of one permutation, so comparisons across models are paired.
    idx = np.random.default_rng(seed).permutation(len(ds.X_train))[:n_train]
    y = ds.Y_train[idx, j]
    t = np.log(y) if ds.log_outputs[j] else y
    mu, sd = t.mean(), t.std() or 1.0

    start = time.perf_counter()
    try:
        model = build_model(model_name, seed).fit(ds.X_train[idx], (t - mu) / sd)
        pred = model.predict(ds.X_test) * sd + mu
        if not np.all(np.isfinite(pred)):
            raise FloatingPointError("non-finite predictions")
        y_test = ds.Y_test[:, j]
        if ds.log_outputs[j] and _WORKER_OPTIONS["score_space"] == "model" and np.all(y_test > 0):
            row["score_space"] = "log"
            row.update(score(np.log(y_test), pred))
        else:
            if ds.log_outputs[j]:
                pred = np.exp(np.clip(pred, -700.0, 700.0))
            row.update(score(y_test, pred))
    except Exception as exc:  # noqa: BLE001 - a failed fit is a benchmark result, not a crash
        row.update(dict.fromkeys(("r2", "rmse", "nrmse", "nmax", "mae"), float("nan")))
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["fit_time_s"] = time.perf_counter() - start
    return row


def run_benchmark(datasets: dict[str, Dataset], args, data_dir: Path) -> pd.DataFrame:
    tasks = []
    for name, ds in datasets.items():
        sizes = [n for n in args.train_sizes if n <= len(ds.X_train)]
        for model in args.models:
            for seed in args.seeds:
                for n in sizes:
                    for j in range(len(ds.outputs)):
                        tasks.append((name, model, seed, n, j))
    # Longest jobs first keeps all workers busy until the end.
    tasks.sort(key=lambda t: -_MODEL_COST[t[1]] * t[3] ** 3 * len(datasets[t[0]].inputs))

    print(f"Running {len(tasks)} fits on {args.workers} worker processes ...", flush=True)
    rows, start = [], time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(str(data_dir), list(datasets), args.score_space)
    ) as pool:
        futures = [pool.submit(run_task, t) for t in tasks]
        report_every = max(1, len(futures) // 20)
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if done % report_every == 0 or done == len(futures):
                elapsed = time.perf_counter() - start
                eta = elapsed / done * (len(futures) - done)
                print(f"  {done:>6}/{len(futures)} fits  elapsed {elapsed:7.1f}s  eta {eta:7.1f}s", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation and criteria
# ---------------------------------------------------------------------------


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    long = raw.melt(
        id_vars=["dataset", "model", "seed", "n_train", "output"],
        value_vars=[*METRICS, "rmse", "mae", "fit_time_s"],
        var_name="metric",
    )
    grouped = long.groupby(["dataset", "model", "n_train", "output", "metric"])["value"]
    return grouped.agg(
        mean="mean",
        std="std",
        median="median",
        q25=lambda v: v.quantile(0.25),
        q75=lambda v: v.quantile(0.75),
        min="min",
        max="max",
        n_seeds="count",
    ).reset_index()


def evaluate_criteria(raw: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    n_max = raw.groupby("dataset")["n_train"].max().rename("n_max")
    at_max = raw.join(n_max, on="dataset").query("n_train == n_max")
    med = at_max.groupby(["dataset", "model", "output"])[list(METRICS)].median().reset_index()
    failed_fits = at_max.groupby(["dataset", "model", "output"])["error"].apply(lambda e: int((e != "").sum()))
    med = med.join(failed_fits.rename("failed_fits"), on=["dataset", "model", "output"])
    med = med.join(n_max, on="dataset")

    med["pass_r2"] = med["r2"] >= thresholds["r2"]
    med["pass_nrmse"] = med["nrmse"] <= thresholds["nrmse"]
    med["pass_nmax"] = med["nmax"] <= thresholds["nmax"]
    med["pass"] = med[["pass_r2", "pass_nrmse", "pass_nmax"]].all(axis=1) & (med["failed_fits"] == 0)
    return med


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# Categorical slots 1-4 of the validated reference palette (fixed order: color follows
# the model, never its rank). Two slots are below 3:1 contrast on the surface, so
# every chart also carries a legend and a distinct marker per model.
MODEL_COLORS = {"LS": "#2a78d6", "RBF": "#eb6834", "Kriging": "#1baf7a", "Ensemble": "#eda100"}
MODEL_MARKERS = {"LS": "o", "RBF": "s", "Kriging": "^", "Ensemble": "D"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "font.size": 9.5,
        }
    )


def _pretty(name: str) -> str:
    return name.replace("_", " ")


def _threshold_line(ax, metric: str, thresholds: dict[str, float]) -> None:
    value = thresholds[metric]
    sign = "≥" if HIGHER_IS_BETTER[metric] else "≤"
    ax.axhline(value, color=INK_2, lw=1.1, ls=(0, (1, 2)), zorder=1)
    ax.annotate(
        f"pass {sign} {value:g}",
        xy=(1.0, value),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK_2,
    )


def _dataset_level(raw: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Per (model, seed, n_train): metrics averaged over the dataset's outputs."""
    sub = raw[raw["dataset"] == dataset]
    return sub.groupby(["model", "seed", "n_train"])[list(METRICS)].mean().reset_index()


def _style_legend_handles():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    return [
        Line2D([], [], color=INK_2, lw=2, label="mean"),
        Line2D([], [], color=INK_2, lw=1.2, ls="--", label="median"),
        Patch(facecolor=INK_2, alpha=0.18, label="mean ± 1 s.d."),
    ]


def _model_handles(models):
    from matplotlib.lines import Line2D

    return [
        Line2D([], [], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m], lw=2, ms=6, label=m) for m in models
    ]


def _learning_panel(ax, level: pd.DataFrame, metric: str, models, thresholds) -> None:
    for model in models:
        stats = level[level["model"] == model].groupby("n_train")[metric].agg(["mean", "std", "median"])
        if stats.empty:
            continue
        ns = stats.index.to_numpy()
        color = MODEL_COLORS[model]
        lo, hi = stats["mean"] - stats["std"].fillna(0), stats["mean"] + stats["std"].fillna(0)
        if not HIGHER_IS_BETTER[metric]:
            lo = lo.clip(lower=stats["mean"].min() * 1e-2)  # keep the band positive on a log axis
        ax.fill_between(ns, lo, hi, color=color, alpha=0.16, lw=0, zorder=2)
        ax.plot(ns, stats["mean"], color=color, lw=2, marker=MODEL_MARKERS[model], ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        ax.plot(ns, stats["median"], color=color, lw=1.2, ls="--", zorder=3)
    _threshold_line(ax, metric, thresholds)
    ax.set_xscale("log", base=2)
    ns_all = sorted(level["n_train"].unique())
    ax.set_xticks(ns_all, [str(n) for n in ns_all])
    ax.minorticks_off()
    ax.set_xlabel("training samples")
    if HIGHER_IS_BETTER[metric]:
        low = np.nanmin(level[metric]) if len(level) else 0.0
        ax.set_ylim(max(low - 0.05, -1.0), 1.02)
    else:
        ax.set_yscale("log")


def plot_learning_curves(raw, dataset, models, thresholds, out: Path, plt) -> None:
    level = _dataset_level(raw, dataset)
    n_outputs = raw.loc[raw["dataset"] == dataset, "output"].nunique()
    n_seeds = raw["seed"].nunique()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        _learning_panel(ax, level, metric, models, thresholds)
        ax.set_title(f"{METRIC_LABELS[metric]} (mean over {n_outputs} outputs)", loc="left")
    fig.suptitle(f"{_pretty(dataset)} — learning curves over {n_seeds} seeds", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.legend(handles=_model_handles(models) + _style_legend_handles(), loc="outside lower center", ncol=7)
    fig.savefig(out / f"learning_curves_{dataset}.png", dpi=150)
    plt.close(fig)


def plot_boxplots(raw, dataset, models, thresholds, out: Path, plt) -> None:
    from matplotlib.lines import Line2D

    sub = raw[raw["dataset"] == dataset]
    n_max = sub["n_train"].max()
    sub = sub[sub["n_train"] == n_max]
    outputs = list(dict.fromkeys(sub["output"]))
    width = 0.8 / len(models)
    fig, axes = plt.subplots(3, 1, figsize=(max(9.0, 0.75 * len(outputs) * len(models) / 2 + 3), 10),
                             constrained_layout=True, sharex=True)
    for ax, metric in zip(axes, METRICS):
        clipped = 0
        for k, model in enumerate(models):
            data = [sub[(sub["model"] == model) & (sub["output"] == o)][metric].dropna().to_numpy() for o in outputs]
            if metric == "r2":
                clipped += sum(int(np.sum(d < -1.0)) for d in data)
                data = [np.clip(d, -1.0, None) for d in data]
            positions = np.arange(len(outputs)) + (k - (len(models) - 1) / 2) * width
            color = MODEL_COLORS[model]
            ax.boxplot(
                [d if d.size else [np.nan] for d in data],
                positions=positions,
                widths=width * 0.8,
                patch_artist=True,
                showmeans=True,
                boxprops={"facecolor": color + "40", "edgecolor": color, "linewidth": 1.2},
                medianprops={"color": INK, "linewidth": 1.6},
                whiskerprops={"color": color, "linewidth": 1.1},
                capprops={"color": color, "linewidth": 1.1},
                flierprops={"marker": MODEL_MARKERS[model], "markersize": 3.5, "markerfacecolor": color,
                            "markeredgecolor": "none", "alpha": 0.7},
                meanprops={"marker": "D", "markersize": 4.5, "markerfacecolor": SURFACE, "markeredgecolor": INK},
            )
        _threshold_line(ax, metric, thresholds)
        title = f"{METRIC_LABELS[metric]} per output"
        if metric == "r2":
            ax.set_ylim(top=1.02)
            if clipped:
                title += f"  ({clipped} values below −1 drawn at −1)"
        else:
            ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[-1].set_xticks(np.arange(len(outputs)), outputs, rotation=35, ha="right")
    extra = [
        Line2D([], [], color=INK, lw=1.6, label="median"),
        Line2D([], [], color=INK, marker="D", ls="", markerfacecolor=SURFACE, ms=5, label="mean"),
    ]
    fig.suptitle(f"{_pretty(dataset)} — test-set metrics at n = {n_max}, spread over seeds", x=0.01,
                 ha="left", fontweight="bold", color=INK)
    fig.legend(handles=_model_handles(models) + extra, loc="outside lower center", ncol=6)
    fig.savefig(out / f"boxplots_{dataset}.png", dpi=150)
    plt.close(fig)


def plot_pass_matrix(criteria, models, thresholds, out: Path, plt) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    datasets = list(dict.fromkeys(criteria["dataset"]))
    frac = np.full((len(models), len(datasets)), np.nan)
    text = [["" for _ in datasets] for _ in models]
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            rows = criteria[(criteria["model"] == model) & (criteria["dataset"] == dataset)]
            if len(rows):
                frac[i, j] = rows["pass"].mean()
                text[i][j] = f"{int(rows['pass'].sum())}/{len(rows)}" + ("  ✓" if rows["pass"].all() else "")
    cmap = LinearSegmentedColormap.from_list("ddmo_blue", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(1.7 * len(datasets) + 2.5, 0.7 * len(models) + 2.4), constrained_layout=True)
    image = ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(models)):
        for j in range(len(datasets)):
            dark = not np.isnan(frac[i, j]) and frac[i, j] > 0.55
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color=SURFACE if dark else INK)
    ax.set_xticks(range(len(datasets)), [_pretty(d) for d in datasets], rotation=25, ha="right")
    ax.set_yticks(range(len(models)), models)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("fraction of outputs passing", color=INK_2)
    colorbar.outline.set_visible(False)
    ax.set_title(
        f"Outputs passing all criteria (median over seeds, largest n): R² ≥ {thresholds['r2']:g}, "
        f"NRMSE ≤ {thresholds['nrmse']:g}, NMAX ≤ {thresholds['nmax']:g}",
        loc="left",
    )
    fig.savefig(out / "pass_matrix.png", dpi=150)
    plt.close(fig)


def plot_overview(raw, models, thresholds, out: Path, plt) -> None:
    datasets = list(dict.fromkeys(raw["dataset"]))
    n_cols = min(4, len(datasets))
    n_rows = int(np.ceil(len(datasets) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.1 * n_rows + 0.8),
                             constrained_layout=True, squeeze=False)
    for ax, dataset in zip(axes.flat, datasets):
        _learning_panel(ax, _dataset_level(raw, dataset), "r2", models, thresholds)
        ax.set_title(_pretty(dataset), loc="left", fontsize=9.5)
    for ax in list(axes.flat)[len(datasets):]:
        ax.set_visible(False)
    fig.suptitle("Test R² (mean over outputs) vs training samples — all datasets", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.legend(handles=_model_handles(models) + _style_legend_handles(), loc="outside lower center", ncol=7)
    fig.savefig(out / "overview_r2.png", dpi=150)
    plt.close(fig)


def make_plots(raw, criteria, models, thresholds, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    models = [m for m in models if m in set(raw["model"])]
    for dataset in dict.fromkeys(raw["dataset"]):
        plot_learning_curves(raw, dataset, models, thresholds, plots, plt)
        plot_boxplots(raw, dataset, models, thresholds, plots, plt)
    plot_pass_matrix(criteria, models, thresholds, plots, plt)
    plot_overview(raw, models, thresholds, plots, plt)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(raw, criteria, args, thresholds, elapsed: float, out: Path) -> bool:
    lines = [
        "# ddmo benchmark report",
        "",
        f"- datasets: {raw['dataset'].nunique()}, models: {', '.join(args.models)}",
        f"- seeds: {list(args.seeds)}, training sizes: {sorted(raw['n_train'].unique().tolist())}",
        (
            f"- fits: {len(raw)} ({int((raw['error'] != '').sum())} failed), wall time {elapsed:.0f} s on "
            f"{args.workers} workers"
        ),
        (
            f"- criteria (median over seeds at the largest training size): R² ≥ {thresholds['r2']:g}, "
            f"NRMSE ≤ {thresholds['nrmse']:g}, NMAX ≤ {thresholds['nmax']:g}"
        ),
        (
            "- scored in log space (heavy-tailed outputs): "
            f"{', '.join(sorted(raw.loc[raw['score_space'] == 'log', 'output'].unique())) or 'none'}"
        ),
        "",
        "## Pass matrix (outputs passing / outputs)",
        "",
        "| dataset | n | " + " | ".join(args.models) + " |",
        "|---|---|" + "---|" * len(args.models),
    ]
    for dataset, rows in criteria.groupby("dataset", sort=False):
        cells = []
        for model in args.models:
            r = rows[rows["model"] == model]
            mark = "✅" if len(r) and r["pass"].all() else "❌"
            cells.append(f"{mark} {int(r['pass'].sum())}/{len(r)}")
        lines.append(f"| {dataset} | {int(rows['n_max'].iloc[0])} | " + " | ".join(cells) + " |")

    failures = criteria[~criteria["pass"]]
    lines += ["", "## Failing outputs", ""]
    if failures.empty:
        lines.append("None — every model passes every output on every dataset.")
    else:
        lines += ["| dataset | model | output | R² | NRMSE | NMAX | failed fits |", "|---|---|---|---|---|---|---|"]
        for _, r in failures.iterrows():
            lines.append(
                f"| {r['dataset']} | {r['model']} | {r['output']} | {r['r2']:.3f} | {r['nrmse']:.3f} | "
                f"{r['nmax']:.3f} | {r['failed_fits']} |"
            )
    lines += ["", "Plots are in `plots/`; raw and aggregated numbers in the CSV files next to this report.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return failures.empty


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=HERE / "datasets")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", type=int, default=5, help="number of random seeds (0 .. N-1)")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[50, 100, 200],
                        help="training-set sizes (adding 400 makes Kriging/Ensemble runs take hours)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--r2-min", type=float, default=0.90)
    parser.add_argument("--nrmse-max", type=float, default=0.10)
    parser.add_argument("--nmax-max", type=float, default=0.30)
    parser.add_argument("--score-space", choices=("model", "original"), default="model",
                        help="score log-fitted heavy-tailed outputs in log space (model) or raw units")
    parser.add_argument("--quick", action="store_true", help="3 seeds, sizes 50 and 100 (smoke test)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit with status 1 if any criterion fails")
    args = parser.parse_args(argv)
    if args.quick:
        args.seeds, args.train_sizes = 3, [50, 100]
    args.seeds = list(range(args.seeds))
    args.train_sizes = sorted(set(args.train_sizes))
    return args


def main(argv=None) -> int:
    # Windows consoles default to cp1252, which cannot print symbols such as "≥".
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args(argv)
    thresholds = {"r2": args.r2_min, "nrmse": args.nrmse_max, "nmax": args.nmax_max}
    args.out.mkdir(parents=True, exist_ok=True)

    datasets = {name: load_dataset(name, args.data_dir) for name in args.datasets}
    start = time.perf_counter()
    raw = run_benchmark(datasets, args, args.data_dir)
    elapsed = time.perf_counter() - start

    raw = raw.sort_values(["dataset", "model", "n_train", "seed", "output"]).reset_index(drop=True)
    criteria = evaluate_criteria(raw, thresholds)
    raw.to_csv(args.out / "raw_results.csv", index=False)
    summarize(raw).to_csv(args.out / "summary.csv", index=False)
    criteria.to_csv(args.out / "criteria.csv", index=False)
    all_pass = write_report(raw, criteria, args, thresholds, elapsed, args.out)
    if not args.no_plots:
        make_plots(raw, criteria, args.models, thresholds, args.out)

    print(f"\nFinished in {elapsed:.0f}s. Report: {args.out / 'report.md'}")
    print((args.out / "report.md").read_text(encoding="utf-8").split("## Failing outputs")[0])
    return 1 if (args.strict and not all_pass) else 0


if __name__ == "__main__":
    sys.exit(main())
