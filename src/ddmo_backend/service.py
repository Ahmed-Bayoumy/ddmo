from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ddmo import LS, RBF, Kriging, WeightedEnsemble
from ddmo.metrics import mae, mape, mse, r2, rmse


@dataclass(frozen=True)
class ModelOption:
    key: str
    label: str


_DEFAULT_COMPOSITE_WEIGHTS = {
    "test_rmse": 0.30,
    "test_mae": 0.15,
    "test_r2": 0.20,
    "predicted_r2_test": 0.15,
    "train_r2": 0.05,
    "predicted_r2_train": 0.10,
    "generalization_rmse_gap": 0.05,
}


def available_models() -> list[ModelOption]:
    return [
        ModelOption("ls", "LS (Polynomial Least Squares)"),
        ModelOption("rbf", "RBF (Radial Basis Function)"),
        ModelOption("kriging", "Kriging"),
        ModelOption("ensemble", "Weighted Ensemble (LS + RBF + Kriging)"),
    ]


def decode_uploaded_csv(contents: str) -> pd.DataFrame:
    if not contents:
        raise ValueError("No uploaded content was provided.")
    try:
        _, payload = contents.split(",", 1)
        raw = base64.b64decode(payload)
        return pd.read_csv(io.StringIO(raw.decode("utf-8")))
    except Exception as exc:  # pragma: no cover - exercised via Dash runtime
        raise ValueError("Could not parse uploaded CSV content.") from exc


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_test = max(1, round(test_ratio * X.shape[0]))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    if train_idx.size == 0:
        raise ValueError("Training set is empty; reduce test_ratio.")

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def build_model(model_key: str, params: dict) -> object:
    if model_key == "ls":
        return LS(
            degree=int(params.get("ls_degree", 2)),
            ridge=float(params.get("ls_ridge", 1e-6)),
            lasso=float(params.get("ls_lasso", 0.0)),
        )
    if model_key == "rbf":
        return RBF(
            kernel=str(params.get("rbf_kernel", "gaussian")),
            gamma=float(params.get("rbf_gamma", 1.0)),
            regularization=float(params.get("rbf_reg", 1e-10)),
        )
    if model_key == "kriging":
        return Kriging(
            p=float(params.get("kriging_p", 2.0)),
            nugget=float(params.get("kriging_nugget", 1e-10)),
        )
    if model_key == "ensemble":
        experts = [
            LS(degree=int(params.get("ls_degree", 2)), ridge=float(params.get("ls_ridge", 1e-6))),
            RBF(
                kernel=str(params.get("rbf_kernel", "gaussian")),
                gamma=float(params.get("rbf_gamma", 1.0)),
                regularization=float(params.get("rbf_reg", 1e-10)),
            ),
            Kriging(
                p=float(params.get("kriging_p", 2.0)),
                nugget=float(params.get("kriging_nugget", 1e-10)),
            ),
        ]
        return WeightedEnsemble(experts=experts, weights="cv")

    raise ValueError(f"Unsupported model key: {model_key}")


def evaluate_model(
    model: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float | np.ndarray]:
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    train_mean = float(np.mean(y_train))

    # External predicted R^2 for holdout data using the training-mean baseline.
    sse_test = float(np.sum((y_test - pred_test) ** 2))
    sst_test = float(np.sum((y_test - train_mean) ** 2))
    predicted_r2_test = 1.0 if sst_test == 0.0 and sse_test == 0.0 else (0.0 if sst_test == 0.0 else 1.0 - sse_test / sst_test)

    return {
        "model": model,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "train_r2": float(r2(y_train, pred_train)),
        "test_r2": float(r2(y_test, pred_test)),
        "predicted_r2_test": float(predicted_r2_test),
        "train_mae": float(mae(y_train, pred_train)),
        "test_mae": float(mae(y_test, pred_test)),
        "train_mape": float(mape(y_train, pred_train)),
        "test_mape": float(mape(y_test, pred_test)),
        "train_rmse": float(rmse(y_train, pred_train)),
        "test_rmse": float(rmse(y_test, pred_test)),
        "train_mse": float(mse(y_train, pred_train)),
        "test_mse": float(mse(y_test, pred_test)),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
    }


def _predicted_r2_train_cv(
    model_key: str,
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_folds: int = 5,
    random_state: int = 0,
) -> float:
    n_samples = X_train.shape[0]
    n_folds = min(max(2, n_folds), n_samples)
    if n_folds < 2:
        return float("nan")

    rng = np.random.default_rng(random_state)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)

    y_oof = np.zeros(n_samples, dtype=float)
    for val_idx in folds:
        train_idx = np.setdiff1d(idx, val_idx)
        model = build_model(model_key, params)
        model.fit(X_train[train_idx], y_train[train_idx])
        y_oof[val_idx] = model.predict(X_train[val_idx])

    return float(r2(y_train, y_oof))


def _normalized_penalty(values: list[float], *, higher_is_better: bool) -> list[float]:
    arr = np.asarray(values, dtype=float)
    min_v = float(np.nanmin(arr))
    max_v = float(np.nanmax(arr))
    if max_v == min_v:
        return [0.0] * len(values)

    scaled = (arr - min_v) / (max_v - min_v)
    if higher_is_better:
        scaled = 1.0 - scaled
    return [float(v) for v in scaled]


def _apply_composite_ranking(
    results: list[dict[str, float | int | str | np.ndarray]],
    metric_weights: dict[str, float] | None,
) -> None:
    weights = dict(_DEFAULT_COMPOSITE_WEIGHTS)
    if metric_weights:
        for key, value in metric_weights.items():
            if key in weights:
                weights[key] = max(0.0, float(value))

    weight_sum = sum(weights.values())
    if weight_sum <= 0.0:
        weights = dict(_DEFAULT_COMPOSITE_WEIGHTS)
        weight_sum = sum(weights.values())
    weights = {k: v / weight_sum for k, v in weights.items()}

    penalties: dict[str, list[float]] = {
        "test_rmse": _normalized_penalty([float(r["test_rmse"]) for r in results], higher_is_better=False),
        "test_mae": _normalized_penalty([float(r["test_mae"]) for r in results], higher_is_better=False),
        "test_r2": _normalized_penalty([float(r["test_r2"]) for r in results], higher_is_better=True),
        "predicted_r2_test": _normalized_penalty(
            [float(r["predicted_r2_test"]) for r in results], higher_is_better=True
        ),
        "train_r2": _normalized_penalty([float(r["train_r2"]) for r in results], higher_is_better=True),
        "predicted_r2_train": _normalized_penalty(
            [float(r["predicted_r2_train"]) for r in results], higher_is_better=True
        ),
        "generalization_rmse_gap": _normalized_penalty(
            [abs(float(r["generalization_rmse_gap"])) for r in results], higher_is_better=False
        ),
    }

    for i, row in enumerate(results):
        composite = 0.0
        for metric, weight in weights.items():
            composite += weight * penalties[metric][i]
        row["composite_score"] = float(composite)

    results.sort(
        key=lambda r: (
            float(r["composite_score"]),
            float(r["test_rmse"]),
            -float(r["test_r2"]),
        )
    )


def evaluate_models(
    model_keys: list[str],
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ranking_mode: str = "standard",
    metric_weights: dict[str, float] | None = None,
) -> list[dict[str, float | int | str | np.ndarray]]:
    if not model_keys:
        raise ValueError("Please select at least one model.")

    results: list[dict[str, float | int | str | np.ndarray]] = []
    for model_key in model_keys:
        model = build_model(model_key, params)
        out = evaluate_model(model, X_train, y_train, X_test, y_test)
        out["predicted_r2_train"] = _predicted_r2_train_cv(
            model_key,
            params,
            X_train,
            y_train,
            random_state=int(params.get("random_state", 0)),
        )
        out["model_key"] = model_key
        out["generalization_rmse_gap"] = float(out["test_rmse"] - out["train_rmse"])
        out["generalization_r2_gap"] = float(out["train_r2"] - out["test_r2"])
        results.append(out)

    if ranking_mode == "weighted_composite":
        _apply_composite_ranking(results, metric_weights)
    else:
        # Standard ranking: prefer lower test RMSE/MAE and higher test/predicted R^2.
        results.sort(
            key=lambda r: (
                float(r["test_rmse"]),
                float(r["test_mae"]),
                -float(r["test_r2"]),
                -float(r["predicted_r2_test"]),
                abs(float(r["generalization_rmse_gap"])),
            )
        )

    for rank, row in enumerate(results, start=1):
        row["rank"] = rank
    return results
