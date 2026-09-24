# Backend Technical Documentation

This page documents the backend implementation in `src/ddmo_backend/service.py` and the core contracts it relies on.

## Service responsibilities

The backend service exposes four primary responsibilities:

- model catalog creation via `available_models()`,
- uploaded CSV decoding via `decode_uploaded_csv()`,
- randomized train/test splitting via `split_train_test()`,
- model construction and evaluation via `build_model()`, `evaluate_model()`, and `evaluate_models()`.

## Data ingestion

The upload decoder expects Dash-style `contents` strings consisting of a metadata prefix and a base64 payload. The decoder:

1. splits the string on the first comma,
2. decodes the base64 payload,
3. parses the result as UTF-8 CSV using pandas,
4. raises a `ValueError` when decoding fails.

```{code-block} python
from ddmo_backend.service import decode_uploaded_csv

df = decode_uploaded_csv(contents)
```

## Train/test split contract

`split_train_test(X, y, test_ratio, random_state)` shuffles sample indices with NumPy's random generator and enforces:

- $0 < \rho < 1$ for the test ratio,
- at least one sample in the test partition,
- at least one sample remaining in the training partition.

The returned tuple is

$$
(X_{\mathrm{train}}, X_{\mathrm{test}}, y_{\mathrm{train}}, y_{\mathrm{test}}).
$$

```{algorithm}
:name: alg-train-test-split

\begin{algorithm}
\caption{Randomized train/test splitting}
\begin{algorithmic}[1]
\Require $X \in \mathbb{R}^{n \times d}$, $y \in \mathbb{R}^n$, $\rho \in (0, 1)$, seed $s$
\Ensure $(X_{\mathrm{train}}, X_{\mathrm{test}}, y_{\mathrm{train}}, y_{\mathrm{test}})$
\State $\pi \sim \mathcal{U}(S_n)$ \Comment{random permutation, seeded with $s$}
\State $n_{\mathrm{test}} \gets \max\bigl(1, \operatorname{round}(\rho n)\bigr)$
\State $I_{\mathrm{test}} \gets \{\pi_1, \dots, \pi_{n_{\mathrm{test}}}\}, \quad I_{\mathrm{train}} \gets \{\pi_{n_{\mathrm{test}}+1}, \dots, \pi_n\}$
\If{$I_{\mathrm{train}} = \emptyset$}
    \State \textbf{raise} ValueError
\EndIf
\State \Return $\bigl(X_{I_{\mathrm{train}}}, X_{I_{\mathrm{test}}}, y_{I_{\mathrm{train}}}, y_{I_{\mathrm{test}}}\bigr)$
\end{algorithmic}
\end{algorithm}
```

## Model factory

`build_model()` maps a UI model key to a configured estimator instance:

- `ls` maps to {py:mod}`ddmo.models.ls`,
- `rbf` maps to {py:mod}`ddmo.models.rbf`,
- `kriging` maps to {py:mod}`ddmo.models.kriging`,
- `ensemble` constructs a `WeightedEnsemble` around LS, RBF, and Kriging experts.

This function is the backend's control point for exposing hyperparameters through the dashboard.

## Evaluation outputs

`evaluate_model()` fits one model and returns a dictionary containing:

- fitted estimator object,
- train and test predictions,
- `train_r2`, `test_r2`, and `predicted_r2_test`,
- `train_mae`, `test_mae`, `train_mape`, `test_mape`,
- `train_rmse`, `test_rmse`, `train_mse`, `test_mse`,
- dataset dimensions for reporting.

The backend defines predicted test $R^2$ against the training mean baseline:

$$
R^2_{\mathrm{pred,test}} = 1 - \frac{\sum_i (y_i^{\mathrm{test}} - \widehat{y}_i^{\mathrm{test}})^2}{\sum_i (y_i^{\mathrm{test}} - \bar{y}_{\mathrm{train}})^2}.
$$

## Cross-validated predicted training $R^2$

`_predicted_r2_train_cv()` computes out-of-fold predictions on the training set and then evaluates $R^2$ on those stitched predictions. This approximates how well each model generalizes before seeing the holdout set.

```{algorithm}
:name: alg-cv-predicted-r2

\begin{algorithm}
\caption{Out-of-fold predicted $R^2$ on the training set}
\begin{algorithmic}[1]
\Require $X \in \mathbb{R}^{n \times d}$, $y \in \mathbb{R}^n$, training map $\mathcal{A}_m$ with hyperparameters $\Theta$, $K = 5$, seed $s$
\Ensure $R^2_{\mathrm{pred,train}}$
\State $K \gets \min\bigl(\max(2, K),\, n\bigr)$
\State $\pi \sim \mathcal{U}(S_n)$; $\;(I_1, \dots, I_K) \gets$ near-equal contiguous blocks of $\pi$
\For{$k = 1, \dots, K$}
    \State $\widehat{f}^{(-k)} \gets \mathcal{A}_m(\Theta)\bigl(X_{\bar{I}_k}, y_{\bar{I}_k}\bigr), \quad \bar{I}_k = \{1, \dots, n\} \setminus I_k$
    \State $\widehat{y}^{\,\mathrm{oof}}_{I_k} \gets \widehat{f}^{(-k)}\bigl(X_{I_k}\bigr)$
\EndFor
\State \Return $1 - \dfrac{\lVert y - \widehat{y}^{\,\mathrm{oof}} \rVert_2^2}{\lVert y - \bar{y}\mathbf{1} \rVert_2^2}$
\end{algorithmic}
\end{algorithm}
```

## Ranking modes

`evaluate_models()` supports two ranking strategies:

- `standard`: lexicographic ordering by test RMSE, then test MAE, then descending test and predicted $R^2$,
- `weighted_composite`: a normalized penalty aggregation over several train/test diagnostics.

For the weighted composite score, if normalized penalties are denoted by $p_k \in [0,1]$ and weights by $\alpha_k$ with $\sum_k \alpha_k = 1$, then

$$
\operatorname{score}_{\mathrm{composite}} = \sum_k \alpha_k p_k.
$$

Lower values are better.

## Persistence and export

The frontend uses `ddmo.save_model()` to serialize fitted estimators into a `pickle`-based envelope containing:

- format name and version,
- `ddmo` version,
- timestamp,
- model class name,
- fitted estimator,
- feature names, target name, and metadata.

:::{warning}
Because deserializing pickle files can execute arbitrary code, only trusted model files should be loaded.
:::

## Failure modes worth noting

- Ill-conditioned kriging or RBF systems may raise `LinAlgError`.
- Invalid split ratios raise `ValueError`.
- Unsupported model keys raise `ValueError`.
- Empty or malformed uploads raise `ValueError`.

These exceptions are the correct boundary for frontend error reporting and should not be silently swallowed in downstream integrations.