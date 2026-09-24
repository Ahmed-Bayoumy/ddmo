# User Guide

## What `ddmo` solves

Many optimization loops rely on expensive analyses such as CFD, FEM, or laboratory experiments. `ddmo` replaces repeated calls to the expensive function with a surrogate model fitted to sampled data:

$$
f : \mathbb{R}^d \to \mathbb{R}, \qquad
\widehat{f} \approx f.
$$

Given a design matrix

$$
X = \begin{bmatrix}
x^{(1)} \\
\vdots \\
x^{(n)}
\end{bmatrix} \in \mathbb{R}^{n \times d},
\qquad
y = \begin{bmatrix}
f(x^{(1)}) \\
\vdots \\
f(x^{(n)})
\end{bmatrix} \in \mathbb{R}^{n},
$$

the package fits a model $\widehat{f}$ that can be queried cheaply for predictions, gradients, and quality metrics.

## Installation

### Core package

```bash
pip install -e .
```

### Dashboard and docs extras

```bash
pip install -e ".[ui,docs,test]"
```

## Basic workflow

1. Prepare an input matrix `X` with shape `(n_samples, n_features)`.
2. Prepare a target vector `y` with shape `(n_samples,)`.
3. Fit one of the surrogate models.
4. Evaluate the model on held-out data.
5. Persist the fitted surrogate when needed.

```python
import numpy as np
from ddmo import Kriging

rng = np.random.default_rng(0)
X = rng.uniform(-1.0, 1.0, size=(40, 2))
y = np.sin(3.0 * X[:, 0]) + X[:, 1] ** 2

model = Kriging().fit(X, y)
mean, std = model.predict(np.array([[0.2, -0.4]]), return_std=True)
grad = model.predict_gradient(np.array([[0.2, -0.4]]))
```

## Backend execution flow

The dashboard backend orchestrates data loading, splitting, fitting, and ranking. At a high level, the workflow is:

```{algorithm}
:name: alg-backend-workflow

\begin{algorithm}
\caption{Backend training and evaluation workflow}
\begin{algorithmic}[1]
\Require $D \in \mathbb{R}^{N \times q}$, feature columns $F$, target column $t$, models $\mathcal{M}$, hyperparameters $\Theta$, $\rho \in (0, 1)$, seed $s$, ranking weights $\alpha$
\Ensure $\mathcal{M}$ ordered by rank, fitted surrogates $\bigl(\widehat{f}_m\bigr)_{m \in \mathcal{M}}$
\State $X \gets D_{:,F}, \quad y \gets D_{:,t}$
\State $(X_{\mathrm{tr}}, X_{\mathrm{te}}, y_{\mathrm{tr}}, y_{\mathrm{te}}) \gets$ \Call{Split}{$X, y, \rho, s$}
\For{$m \in \mathcal{M}$}
    \State $\widehat{f}_m \gets \mathcal{A}_m(\Theta)(X_{\mathrm{tr}}, y_{\mathrm{tr}})$
    \State $e^{\mathrm{tr}} \gets y_{\mathrm{tr}} - \widehat{f}_m(X_{\mathrm{tr}}), \quad e^{\mathrm{te}} \gets y_{\mathrm{te}} - \widehat{f}_m(X_{\mathrm{te}})$
    \State $\mathrm{RMSE}^{\mathrm{te}}_m \gets \lVert e^{\mathrm{te}} \rVert_2 / \sqrt{n_{\mathrm{te}}}, \quad \mathrm{MAE}^{\mathrm{te}}_m \gets \lVert e^{\mathrm{te}} \rVert_1 / n_{\mathrm{te}}$
    \State $R^2_m \gets 1 - \dfrac{\lVert e^{\mathrm{te}} \rVert_2^2}{\lVert y_{\mathrm{te}} - \bar{y}_{\mathrm{te}}\mathbf{1} \rVert_2^2}, \quad R^2_{\mathrm{pred},m} \gets 1 - \dfrac{\lVert e^{\mathrm{te}} \rVert_2^2}{\lVert y_{\mathrm{te}} - \bar{y}_{\mathrm{tr}}\mathbf{1} \rVert_2^2}$
    \State $\Delta_m \gets \mathrm{RMSE}^{\mathrm{te}}_m - \mathrm{RMSE}^{\mathrm{tr}}_m, \quad R^2_{\mathrm{cv},m} \gets$ \Call{OutOfFoldR2}{$m, X_{\mathrm{tr}}, y_{\mathrm{tr}}$}
\EndFor
\If{ranking is weighted composite}
    \State $p_k(m) \gets \dfrac{v_k(m) - \min_{m'} v_k(m')}{\max_{m'} v_k(m') - \min_{m'} v_k(m')}$ \Comment{use $1 - p_k$ if higher is better}
    \State sort $\mathcal{M}$ by $S_m = \sum_k \alpha_k\, p_k(m) \,/\, \sum_k \alpha_k$ ascending
\Else
    \State sort $\mathcal{M}$ lexicographically by $\bigl(\mathrm{RMSE}^{\mathrm{te}}_m,\ \mathrm{MAE}^{\mathrm{te}}_m,\ -R^2_m,\ -R^2_{\mathrm{pred},m},\ |\Delta_m|\bigr)$
\EndIf
\State \Return $\mathcal{M}$, $\bigl(\widehat{f}_m\bigr)_{m \in \mathcal{M}}$
\end{algorithmic}
\end{algorithm}
```

## Running the dashboard

```bash
ddmo-dashboard
```

The dashboard allows you to upload CSV data, choose feature and target columns, compare multiple models, and export the best fitted model for downstream use.

## Persisting a trained model

```python
from ddmo import save_model, load_model

save_model(model, "kriging_model.pkl", feature_names=["x0", "x1"], target_name="y")
bundle = load_model("kriging_model.pkl")
prediction = bundle.predict(X)
```

## When to choose each model

- `LS`: smooth global trends, response surfaces, and sparse polynomial structure.
- `RBF`: exact interpolation with fast fitting and flexible kernels.
- `Kriging`: interpolation plus uncertainty estimation.
- `WeightedEnsemble`: blended predictions when no single surrogate family dominates.