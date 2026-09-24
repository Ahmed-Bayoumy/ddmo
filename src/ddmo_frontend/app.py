from __future__ import annotations

import re
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme, Trim
from scipy.linalg import LinAlgError

from ddmo import save_model
from ddmo_backend.service import (
    available_models,
    decode_uploaded_csv,
    evaluate_models,
    split_train_test,
)

_FONT = "Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
_TRAIN_COLOR = "#6366f1"
_TEST_COLOR = "#f97316"
_IDEAL_COLOR = "#94a3b8"
_NUMBER_FORMAT = Format(precision=4, scheme=Scheme.decimal_or_exponent, trim=Trim.yes)
# Fitted models from recent training runs, kept server-side so they can be exported
# without refitting. Keyed by a per-run id held in the browser (``run-id`` store).
_MAX_CACHED_RUNS = 8
_RUNS: OrderedDict[str, dict] = OrderedDict()
_RANKING_NOTES = {
    "standard": "Lexicographic: test RMSE, then test MAE, then test R² and predicted R².",
    "weighted_composite": "Normalized weighted penalty over train/test metrics. Lower score is better.",
}


def _parse_float(v, fallback):
    try:
        return float(v)
    except (TypeError, ValueError):
        return fallback


def _parse_int(v, fallback):
    try:
        return int(v)
    except (TypeError, ValueError):
        return fallback


def _short_model_labels() -> dict[str, str]:
    return {item.key: item.label.split(" (")[0] for item in available_models()}


def _fmt(value) -> str:
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return "-"


def _cache_run(run: dict) -> str:
    run_id = uuid.uuid4().hex
    _RUNS[run_id] = run
    while len(_RUNS) > _MAX_CACHED_RUNS:
        _RUNS.popitem(last=False)
    return run_id


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_") or "target"


# ---------- Layout building blocks ----------


def _field(label, component, hint=None) -> html.Div:
    children = [html.Label(label, className="field-label"), component]
    if hint:
        children.append(html.Span(hint, className="field-hint"))
    return html.Div(children, className="field")


def _number(id_, value, step=None) -> dcc.Input:
    return dcc.Input(id=id_, type="number", value=value, step=step, className="input")


def _section(step, title, children) -> html.Div:
    return html.Div(
        [html.H3([html.Span(step, className="step"), title], className="section-title"), *children],
        className="section",
    )


def _collapsible(title, children, tag=None, open_=False) -> html.Details:
    summary = [title]
    if tag:
        summary.append(html.Span(tag, className="tag"))
    return html.Details(
        [html.Summary(summary), html.Div(children, className="collapsible-body")],
        className="collapsible",
        open=open_,
    )


def _card(title, subtitle, children) -> html.Div:
    return html.Div(
        [html.H3(title, className="card-title"), html.P(subtitle, className="card-subtitle"), *children],
        className="card",
    )


def _alert(message, kind="warning") -> html.Div:
    icon = "⚠" if kind == "warning" else "✕"
    return html.Div([html.Span(icon), html.Span(message)], className=f"alert alert-{kind}")


def _kpi(label, value, hint=None, accent=False) -> html.Div:
    children = [html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value")]
    if hint:
        children.append(html.Div(hint, className="kpi-hint"))
    return html.Div(children, className="kpi kpi-accent" if accent else "kpi")


def _empty_state() -> html.Div:
    return html.Div(
        html.Div(
            [
                html.Div("◆", className="empty-icon"),
                html.H2("Ready when you are", className="empty-title"),
                html.P(
                    "Upload a CSV, pick a target and the models to compare, then hit Train. "
                    "The leaderboard and diagnostics will appear here.",
                    className="empty-text",
                ),
                html.Div(
                    [
                        html.Span("1 · Upload data", className="meta-chip"),
                        html.Span("2 · Choose models", className="meta-chip"),
                        html.Span("3 · Train & compare", className="meta-chip"),
                    ],
                    className="empty-steps",
                ),
            ],
            className="empty-state",
        ),
        className="card",
    )


def _table_style() -> dict:
    return {
        "style_as_list_view": True,
        "style_table": {"overflowX": "auto", "borderRadius": "10px", "border": "1px solid #e4e8f0"},
        "style_cell": {
            "fontFamily": _FONT,
            "fontSize": "13px",
            "padding": "10px 14px",
            "borderBottom": "1px solid #eef1f6",
            "color": "#0f172a",
            "textAlign": "right",
            "whiteSpace": "nowrap",
        },
        "style_cell_conditional": [
            {"if": {"column_id": c}, "textAlign": "left", "fontWeight": 600} for c in ("Rank", "Model", "Metric")
        ],
        "style_header": {
            "backgroundColor": "#f8fafc",
            "color": "#64748b",
            "fontWeight": 700,
            "fontSize": "11.5px",
            "textTransform": "uppercase",
            "letterSpacing": "0.04em",
            "borderBottom": "1px solid #e4e8f0",
        },
        "style_data_conditional": [
            {"if": {"row_index": "odd"}, "backgroundColor": "#fbfcfe"},
            {"if": {"filter_query": "{Rank} = 1"}, "backgroundColor": "#eef2ff", "color": "#3730a3", "fontWeight": 600},
            {"if": {"state": "active"}, "backgroundColor": "#e0e7ff", "border": "1px solid #c7d2fe"},
        ],
    }


def _columns(names, text_columns=("Model", "Metric")) -> list[dict]:
    return [
        {"name": c, "id": c} if c in text_columns else {"name": c, "id": c, "type": "numeric", "format": _NUMBER_FORMAT}
        for c in names
    ]


# ---------- Figures ----------


def _placeholder_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        annotations=[
            {
                "text": message,
                "showarrow": False,
                "font": {"family": _FONT, "size": 14, "color": "#94a3b8"},
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
            }
        ],
    )
    return fig


def _style_fig(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title={"text": f"<b>{title}</b>", "x": 0.02, "font": {"size": 15, "color": "#0f172a"}},
        font={"family": _FONT, "color": "#334155", "size": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        hoverlabel={"font": {"family": _FONT}},
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    axis_style = {"gridcolor": "#eef1f6", "zerolinecolor": "#e2e8f0", "linecolor": "#cbd5e1", "showline": True}
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def _markers(color: str) -> dict:
    return {"size": 9, "color": color, "opacity": 0.8, "line": {"width": 1, "color": "white"}}


def _graph(id_, message) -> html.Div:
    return html.Div(
        dcc.Graph(id=id_, figure=_placeholder_fig(message), config={"displaylogo": False}),
        className="card graph-card",
    )


# ---------- Page sections ----------


def _export_card() -> html.Div:
    return _card(
        "Export trained model",
        "Download a fitted model as a .pkl file and reuse it with ddmo.load_model() without retraining.",
        [
            html.Div(
                [
                    dcc.Dropdown(
                        id="export-model",
                        placeholder="Train models first",
                        clearable=False,
                        className="dropdown",
                    ),
                    html.Button(
                        "⤓  Download .pkl",
                        id="export-btn",
                        n_clicks=0,
                        disabled=True,
                        className="secondary-btn",
                    ),
                ],
                className="export-row",
            ),
            html.Div(id="export-status"),
        ],
    )


def _guide() -> html.Details:
    items = [
        ("R² (train / test)", "Coefficient of determination on each split. Higher is better."),
        (
            "Predicted R² (test)",
            (
                "External holdout R² using the training-mean baseline: "
                "1 − SSE_test / SST_test(train-mean). Higher is better."
            ),
        ),
        (
            "Predicted R² (train)",
            (
                "Out-of-fold cross-validated R² computed on the training set. "
                "Higher means better expected generalization."
            ),
        ),
        ("RMSE, MAE, MSE", "Error metrics where lower is better."),
        ("MAPE (%)", "Mean absolute percentage error (rows with zero target are ignored). Lower is better."),
        (
            "Generalization gaps",
            (
                "RMSE gap = test RMSE − train RMSE, R² gap = train R² − test R². "
                "Smaller absolute gaps indicate less overfitting."
            ),
        ),
        (
            "Standard ranking",
            "Sort by lower test RMSE, then lower test MAE, then higher test R² and predicted R² (test).",
        ),
        (
            "Weighted composite ranking",
            "Computes a normalized weighted penalty score over train/test metrics; lower composite score is better.",
        ),
    ]
    return html.Details(
        [
            html.Summary("ⓘ  Metric and ranking guide"),
            html.Div(
                html.Div(
                    [html.Div([html.B(name), html.Span(text)], className="guide-item") for name, text in items],
                    className="guide-grid",
                ),
                className="collapsible-body",
            ),
        ],
        className="collapsible card guide",
        style={"padding": 0},
    )


def _sidebar() -> html.Aside:
    model_options = [{"label": label, "value": key} for key, label in _short_model_labels().items()]
    rbf_kernels = ["gaussian", "linear", "cubic", "multiquadric", "inverse_multiquadric", "thin_plate"]

    data_section = _section(
        "1",
        "Data",
        [
            dcc.Upload(
                id="upload-data",
                children=html.Div(
                    [
                        html.Div("⇪", className="dropzone-icon"),
                        html.Div(["Drop a CSV or ", html.A("browse")], className="dropzone-title"),
                        html.Div("One row per sample, numeric columns", className="dropzone-hint"),
                    ],
                    className="dropzone",
                ),
                multiple=False,
            ),
            html.Div(id="upload-status", style={"marginBottom": "12px"}),
            _field("Target column", dcc.Dropdown(id="target-column", placeholder="Select target", className="dropdown")),
            _field(
                "Feature columns",
                dcc.Dropdown(id="feature-columns", multi=True, placeholder="All other columns", className="dropdown"),
                hint="Leave empty to use every column except the target.",
            ),
            html.Div(
                [
                    _field("Test ratio", _number("test-ratio", 0.25, 0.05)),
                    _field("Random state", _number("random-state", 0)),
                ],
                className="field-grid",
            ),
        ],
    )

    models_section = _section(
        "2",
        "Models",
        [
            dcc.Checklist(
                id="model-keys",
                options=model_options,
                value=["ls", "rbf", "kriging", "ensemble"],
                className="chip-list",
                labelClassName="chip",
                style={"marginBottom": "14px"},
            ),
            _collapsible(
                "Least squares",
                html.Div(
                    [
                        _field("Degree", _number("ls-degree", 2)),
                        _field("Ridge", _number("ls-ridge", 1e-6, 1e-6)),
                        _field("Lasso", _number("ls-lasso", 0.0, 0.01)),
                    ],
                    className="field-grid",
                ),
                tag="LS",
            ),
            _collapsible(
                "Radial basis function",
                [
                    _field(
                        "Kernel",
                        dcc.Dropdown(
                            id="rbf-kernel",
                            options=[{"label": k, "value": k} for k in rbf_kernels],
                            value="gaussian",
                            clearable=False,
                            className="dropdown",
                        ),
                    ),
                    html.Div(
                        [
                            _field("Gamma", _number("rbf-gamma", 1.0, 0.1)),
                            _field("Regularization", _number("rbf-reg", 1e-10, 1e-10)),
                        ],
                        className="field-grid",
                    ),
                ],
                tag="RBF",
            ),
            _collapsible(
                "Kriging",
                html.Div(
                    [
                        _field("p", _number("kriging-p", 2.0, 0.1)),
                        _field("Nugget", _number("kriging-nugget", 1e-10, 1e-10)),
                    ],
                    className="field-grid",
                ),
                tag="GP",
            ),
        ],
    )

    ranking_section = _section(
        "3",
        "Ranking",
        [
            dcc.RadioItems(
                id="ranking-mode",
                options=[
                    {"label": "Standard", "value": "standard"},
                    {"label": "Weighted composite", "value": "weighted_composite"},
                ],
                value="standard",
                className="segmented",
                labelClassName="segment",
            ),
            html.P(id="ranking-note", className="field-hint", style={"margin": "8px 0 12px"}),
            html.Div(
                _collapsible(
                    "Composite weights",
                    html.Div(
                        [
                            _field("Test RMSE", _number("w-test-rmse", 0.30, 0.01)),
                            _field("Test MAE", _number("w-test-mae", 0.15, 0.01)),
                            _field("Test R²", _number("w-test-r2", 0.20, 0.01)),
                            _field("Pred. R² (test)", _number("w-pred-r2-test", 0.15, 0.01)),
                            _field("Train R²", _number("w-train-r2", 0.05, 0.01)),
                            _field("Pred. R² (train)", _number("w-pred-r2-train", 0.10, 0.01)),
                            _field("RMSE gap", _number("w-gap-rmse", 0.05, 0.01)),
                        ],
                        className="field-grid",
                    ),
                    open_=True,
                ),
                id="weights-panel",
            ),
        ],
    )

    return html.Aside(
        [
            html.Div([data_section, models_section, ranking_section], className="sidebar-scroll"),
            html.Div(
                html.Button("▶  Train and evaluate", id="train-btn", n_clicks=0, className="train-btn"),
                className="sidebar-footer",
            ),
        ],
        className="sidebar",
    )


def _header() -> html.Header:
    return html.Header(
        [
            html.Div("D", className="brand-mark"),
            html.Div(
                [
                    html.H1("DDMO Surrogate Studio", className="brand-title"),
                    html.P("Train, compare and diagnose data-driven surrogate models", className="brand-subtitle"),
                ]
            ),
            html.Span("LS · RBF · Kriging · Ensemble", className="header-badge"),
        ],
        className="app-header",
    )


def _layout() -> html.Div:
    results = html.Div(
        [
            html.Div(_empty_state(), id="model-summary"),
            _export_card(),
            _card(
                "Model leaderboard",
                "Every selected model, ranked with the chosen strategy. The top model is highlighted.",
                [dash_table.DataTable(id="ranking-table", sort_action="native", **_table_style())],
            ),
            html.Div(
                [
                    _graph("fit-graph", "Predicted vs actual will appear here"),
                    _graph("residual-graph", "Residuals will appear here"),
                ],
                className="card-grid",
            ),
            html.Div(
                [
                    _card(
                        "Best model metrics",
                        "Train vs test fit quality for the top-ranked model.",
                        [dash_table.DataTable(id="metrics-table", **_table_style())],
                    ),
                    _graph("comparison-graph", "Model comparison will appear here"),
                ],
                className="card-grid",
            ),
        ],
        className="main",
    )

    return html.Div(
        [
            _header(),
            html.Div(
                [
                    _sidebar(),
                    html.Main(
                        [
                            dcc.Store(id="dataset-store"),
                            dcc.Store(id="run-id"),
                            dcc.Download(id="model-download"),
                            dcc.Loading(results, type="dot", color=_TRAIN_COLOR),
                            _guide(),
                        ],
                        className="main",
                    ),
                ],
                className="app-shell",
            ),
        ]
    )


def _summary(out, ranking_mode, feature_columns, target_column, labels) -> html.Div:
    mode_label = "Weighted composite" if ranking_mode == "weighted_composite" else "Standard"
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(f"Best model · rank #{out['rank']}", className="hero-eyebrow"),
                            html.H2(labels.get(out["model_key"], out["model_key"]), className="hero-model"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span(["Target ", html.B(target_column)], className="meta-chip"),
                            html.Span(
                                ["Features ", html.B(len(feature_columns))],
                                className="meta-chip",
                                title=", ".join(feature_columns),
                            ),
                            html.Span(["Ranking ", html.B(mode_label)], className="meta-chip"),
                        ],
                        className="hero-meta",
                    ),
                ],
                className="hero",
            ),
            html.Div(
                [
                    _kpi("Test R²", _fmt(out["test_r2"]), f"train {_fmt(out['train_r2'])}", accent=True),
                    _kpi("Test RMSE", _fmt(out["test_rmse"]), f"train {_fmt(out['train_rmse'])}"),
                    _kpi("Test MAE", _fmt(out["test_mae"]), f"MAPE {_fmt(out['test_mape'])}%"),
                    _kpi("Predicted R²", _fmt(out["predicted_r2_test"]), f"CV train {_fmt(out['predicted_r2_train'])}"),
                    _kpi("Samples", f"{out['n_train']} / {out['n_test']}", "train / test"),
                ],
                className="kpi-row",
            ),
        ],
        className="card",
    )


def create_app() -> Dash:
    app = Dash(
        __name__,
        external_stylesheets=[
            "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
        ],
    )
    app.title = "DDMO Surrogate Studio"
    app.layout = _layout()

    @callback(
        Output("ranking-note", "children"),
        Output("weights-panel", "style"),
        Input("ranking-mode", "value"),
    )
    def toggle_ranking_mode(ranking_mode):
        mode = ranking_mode or "standard"
        return _RANKING_NOTES[mode], {"display": "block" if mode == "weighted_composite" else "none"}

    @callback(
        Output("dataset-store", "data"),
        Output("upload-status", "children"),
        Output("target-column", "options"),
        Output("feature-columns", "options"),
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
        prevent_initial_call=True,
    )
    def parse_upload(contents, filename):
        if not contents:
            return None, html.Span("✕  No file uploaded.", className="pill pill-error"), [], []
        try:
            df = decode_uploaded_csv(contents)
        except ValueError as exc:
            return None, html.Span(f"✕  {exc}", className="pill pill-error"), [], []

        options = [{"label": c, "value": c} for c in df.columns]
        status = html.Span(
            f"✓  {filename} · {df.shape[0]} rows × {df.shape[1]} cols",
            className="pill pill-success",
        )
        return df.to_json(date_format="iso", orient="split"), status, options, options

    @callback(
        Output("model-summary", "children"),
        Output("ranking-table", "data"),
        Output("ranking-table", "columns"),
        Output("fit-graph", "figure"),
        Output("residual-graph", "figure"),
        Output("comparison-graph", "figure"),
        Output("metrics-table", "data"),
        Output("metrics-table", "columns"),
        Output("run-id", "data"),
        Output("export-model", "options"),
        Output("export-model", "value"),
        Output("export-btn", "disabled"),
        Input("train-btn", "n_clicks"),
        State("dataset-store", "data"),
        State("target-column", "value"),
        State("feature-columns", "value"),
        State("model-keys", "value"),
        State("ranking-mode", "value"),
        State("test-ratio", "value"),
        State("random-state", "value"),
        State("ls-degree", "value"),
        State("ls-ridge", "value"),
        State("ls-lasso", "value"),
        State("rbf-gamma", "value"),
        State("rbf-reg", "value"),
        State("rbf-kernel", "value"),
        State("kriging-p", "value"),
        State("kriging-nugget", "value"),
        State("w-test-rmse", "value"),
        State("w-test-mae", "value"),
        State("w-test-r2", "value"),
        State("w-pred-r2-test", "value"),
        State("w-train-r2", "value"),
        State("w-pred-r2-train", "value"),
        State("w-gap-rmse", "value"),
        prevent_initial_call=True,
    )
    def train_and_evaluate(
        _n_clicks,
        data_json,
        target_column,
        feature_columns,
        model_keys,
        ranking_mode,
        test_ratio,
        random_state,
        ls_degree,
        ls_ridge,
        ls_lasso,
        rbf_gamma,
        rbf_reg,
        rbf_kernel,
        kriging_p,
        kriging_nugget,
        w_test_rmse,
        w_test_mae,
        w_test_r2,
        w_pred_r2_test,
        w_train_r2,
        w_pred_r2_train,
        w_gap_rmse,
    ):
        def fail(message, kind="warning"):
            fig = _placeholder_fig("No results yet")
            return _alert(message, kind), [], [], fig, fig, fig, [], [], None, [], None, True

        if not data_json:
            return fail("Please upload a dataset first.")
        if not target_column:
            return fail("Please select a target column.")
        if not model_keys:
            return fail("Select at least one model to compare.")

        frame = pd.read_json(data_json, orient="split")
        if not feature_columns:
            feature_columns = [c for c in frame.columns if c != target_column]
        if not feature_columns:
            return fail("Select at least one feature column.")

        X = frame[feature_columns].to_numpy(dtype=float)
        y = frame[target_column].to_numpy(dtype=float)
        ranking_mode = ranking_mode or "standard"

        params = {
            "ls_degree": _parse_int(ls_degree, 2),
            "ls_ridge": _parse_float(ls_ridge, 1e-6),
            "ls_lasso": _parse_float(ls_lasso, 0.0),
            "rbf_gamma": _parse_float(rbf_gamma, 1.0),
            "rbf_reg": _parse_float(rbf_reg, 1e-10),
            "rbf_kernel": rbf_kernel or "gaussian",
            "kriging_p": _parse_float(kriging_p, 2.0),
            "kriging_nugget": _parse_float(kriging_nugget, 1e-10),
            "random_state": _parse_int(random_state, 0),
        }
        metric_weights = {
            "test_rmse": _parse_float(w_test_rmse, 0.30),
            "test_mae": _parse_float(w_test_mae, 0.15),
            "test_r2": _parse_float(w_test_r2, 0.20),
            "predicted_r2_test": _parse_float(w_pred_r2_test, 0.15),
            "train_r2": _parse_float(w_train_r2, 0.05),
            "predicted_r2_train": _parse_float(w_pred_r2_train, 0.10),
            "generalization_rmse_gap": _parse_float(w_gap_rmse, 0.05),
        }

        try:
            X_train, X_test, y_train, y_test = split_train_test(
                X,
                y,
                test_ratio=_parse_float(test_ratio, 0.25),
                random_state=_parse_int(random_state, 0),
            )
            ranked = evaluate_models(
                model_keys,
                params,
                X_train,
                y_train,
                X_test,
                y_test,
                ranking_mode=ranking_mode,
                metric_weights=metric_weights,
            )
            out = ranked[0]
        except (LinAlgError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - runtime path
            return fail(f"Training failed: {exc}", kind="error")

        labels = _short_model_labels()
        ranking_data = [
            {
                "Rank": row["rank"],
                "Model": labels.get(row["model_key"], row["model_key"]),
                "Test RMSE": row["test_rmse"],
                "Test MAE": row["test_mae"],
                "Test MAPE %": row["test_mape"],
                "Test R2": row["test_r2"],
                "Predicted R2 (test)": row["predicted_r2_test"],
                "Train RMSE": row["train_rmse"],
                "Train R2": row["train_r2"],
                "Predicted R2 (train)": row["predicted_r2_train"],
                "RMSE gap (test-train)": row["generalization_rmse_gap"],
                "R2 gap (train-test)": row["generalization_r2_gap"],
                "Composite score": row.get("composite_score", "-"),
            }
            for row in ranked
        ]
        ranking_columns = _columns(ranking_data[0])

        fit_fig = go.Figure()
        fit_fig.add_trace(
            go.Scatter(x=y_train, y=out["pred_train"], mode="markers", name="Train", marker=_markers(_TRAIN_COLOR))
        )
        fit_fig.add_trace(
            go.Scatter(x=y_test, y=out["pred_test"], mode="markers", name="Test", marker=_markers(_TEST_COLOR))
        )
        all_vals = np.concatenate([y_train, y_test, out["pred_train"], out["pred_test"]])
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        fit_fig.add_trace(
            go.Scatter(
                x=[vmin, vmax],
                y=[vmin, vmax],
                mode="lines",
                name="Ideal",
                line={"dash": "dash", "color": _IDEAL_COLOR, "width": 1.5},
            )
        )
        _style_fig(fit_fig, "Predicted vs actual", "Actual", "Predicted")

        residual_fig = go.Figure()
        residual_fig.add_trace(
            go.Scatter(
                x=out["pred_train"],
                y=y_train - out["pred_train"],
                mode="markers",
                name="Train",
                marker=_markers(_TRAIN_COLOR),
            )
        )
        residual_fig.add_trace(
            go.Scatter(
                x=out["pred_test"],
                y=y_test - out["pred_test"],
                mode="markers",
                name="Test",
                marker=_markers(_TEST_COLOR),
            )
        )
        residual_fig.add_hline(y=0.0, line_dash="dash", line_color=_IDEAL_COLOR)
        _style_fig(residual_fig, "Residuals", "Predicted", "Actual − predicted")

        model_names = [labels.get(row["model_key"], row["model_key"]) for row in ranked]
        comparison_fig = go.Figure()
        comparison_fig.add_trace(
            go.Bar(
                x=model_names,
                y=[row["train_rmse"] for row in ranked],
                name="Train RMSE",
                marker_color=_TRAIN_COLOR,
            )
        )
        comparison_fig.add_trace(
            go.Bar(
                x=model_names,
                y=[row["test_rmse"] for row in ranked],
                name="Test RMSE",
                marker_color=_TEST_COLOR,
                text=[f"R² {_fmt(row['test_r2'])}" for row in ranked],
                textposition="outside",
                cliponaxis=False,
            )
        )
        _style_fig(comparison_fig, "RMSE by model (ranked)", "", "RMSE")
        comparison_fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
        comparison_fig.update_traces(marker_line_width=0, selector={"type": "bar"})

        metrics_data = [
            {"Metric": "R2", "Train": out["train_r2"], "Test": out["test_r2"]},
            {"Metric": "Predicted R2", "Train": out["predicted_r2_train"], "Test": out["predicted_r2_test"]},
            {"Metric": "MAE", "Train": out["train_mae"], "Test": out["test_mae"]},
            {"Metric": "MAPE (%)", "Train": out["train_mape"], "Test": out["test_mape"]},
            {"Metric": "RMSE", "Train": out["train_rmse"], "Test": out["test_rmse"]},
            {"Metric": "MSE", "Train": out["train_mse"], "Test": out["test_mse"]},
            {"Metric": "Samples", "Train": out["n_train"], "Test": out["n_test"]},
            {"Metric": "Features", "Train": out["n_features"], "Test": out["n_features"]},
        ]
        columns = _columns(["Metric", "Train", "Test"])

        summary = _summary(out, ranking_mode, feature_columns, target_column, labels)

        trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        run_id = _cache_run(
            {
                row["model_key"]: {
                    "model": row["model"],
                    "feature_names": list(feature_columns),
                    "target_name": target_column,
                    "metadata": {
                        "model_key": row["model_key"],
                        "model_label": labels.get(row["model_key"], row["model_key"]),
                        "rank": row["rank"],
                        "ranking_mode": ranking_mode,
                        "trained_at": trained_at,
                        "hyperparameters": params,
                        "test_ratio": _parse_float(test_ratio, 0.25),
                        "metrics": {
                            k: v for k, v in row.items() if isinstance(v, (int, float)) and k != "rank"
                        },
                    },
                }
                for row in ranked
            }
        )
        export_options = [
            {"label": f"#{row['rank']} · {labels.get(row['model_key'], row['model_key'])}", "value": row["model_key"]}
            for row in ranked
        ]

        return (
            summary,
            ranking_data,
            ranking_columns,
            fit_fig,
            residual_fig,
            comparison_fig,
            metrics_data,
            columns,
            run_id,
            export_options,
            out["model_key"],
            False,
        )

    @callback(
        Output("model-download", "data"),
        Output("export-status", "children"),
        Input("export-btn", "n_clicks"),
        State("run-id", "data"),
        State("export-model", "value"),
        prevent_initial_call=True,
    )
    def download_model(_n_clicks, run_id, model_key):
        entry = _RUNS.get(run_id, {}).get(model_key)
        if entry is None:
            return None, html.Span("✕  Model is no longer in memory. Train again to export it.", className="pill pill-error")

        filename = f"ddmo_{model_key}_{_slug(entry['target_name'])}.pkl"
        payload = dcc.send_bytes(
            lambda buffer: save_model(
                entry["model"],
                buffer,
                feature_names=entry["feature_names"],
                target_name=entry["target_name"],
                metadata=entry["metadata"],
            ),
            filename,
        )
        return payload, html.Span(f"✓  Saved {filename}", className="pill pill-success")

    return app


def main() -> None:
    app = create_app()
    app.run(debug=True)


if __name__ == "__main__":
    main()
