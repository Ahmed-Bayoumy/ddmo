# Tutorials

## Tutorial 1: Compare surrogate models on sampled data

This tutorial trains several models on the same dataset and compares their holdout accuracy.

```python
import numpy as np
from ddmo_backend.service import evaluate_models, split_train_test

rng = np.random.default_rng(42)
X = rng.uniform(-1.0, 1.0, size=(80, 2))
y = np.sin(2.5 * X[:, 0]) + 0.5 * X[:, 1] ** 2

X_train, X_test, y_train, y_test = split_train_test(X, y, test_ratio=0.2, random_state=7)
results = evaluate_models(
    ["ls", "rbf", "kriging", "ensemble"],
    params={"random_state": 7},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)
```

```{algorithm}
:name: alg-model-comparison

\begin{algorithm}
\caption{Model comparison tutorial workflow}
\begin{algorithmic}[1]
\Require $X \in \mathbb{R}^{n \times d}$, $y \in \mathbb{R}^n$, $\mathcal{M} = \{\text{LS}, \text{RBF}, \text{Kriging}, \text{Ensemble}\}$, $\rho = 0.2$
\Ensure $\mathcal{M}$ sorted by holdout error
\State $(X_{\mathrm{tr}}, X_{\mathrm{te}}, y_{\mathrm{tr}}, y_{\mathrm{te}}) \gets$ \Call{Split}{$X, y, \rho$}
\For{$m \in \mathcal{M}$}
    \State $\widehat{f}_m \gets \mathcal{A}_m(X_{\mathrm{tr}}, y_{\mathrm{tr}})$
    \State $e_m \gets y_{\mathrm{te}} - \widehat{f}_m(X_{\mathrm{te}})$
    \State $\mathrm{RMSE}_m \gets \lVert e_m \rVert_2 / \sqrt{n_{\mathrm{te}}}, \quad R^2_m \gets 1 - \lVert e_m \rVert_2^2 \,/\, \lVert y_{\mathrm{te}} - \bar{y}_{\mathrm{te}}\mathbf{1} \rVert_2^2$
\EndFor
\State \Return $\operatorname{argsort}_{m \in \mathcal{M}}\, \mathrm{RMSE}_m$
\end{algorithmic}
\end{algorithm}
```

## Tutorial 2: Feature screening before surrogate fitting

When input columns are redundant, screen them before fitting the surrogate:

```python
import numpy as np
from ddmo import CollinearityFilter, Kriging

rng = np.random.default_rng(1)
a, b, c = rng.normal(size=(3, 100))
X = np.column_stack([a, b, a + b, c])
y = a - c

filter_ = CollinearityFilter(method="vif").fit(X)
model = Kriging().fit(filter_.transform(X), y)
```

## Tutorial 3: Train and export from the dashboard backend

The backend can train a model and package it for reuse:

```python
from ddmo import save_model
from ddmo_backend.service import build_model

model = build_model("kriging", {"kriging_p": 2.0, "kriging_nugget": 1e-10})
model.fit(X_train, y_train)
save_model(model, "trained_model.pkl", feature_names=["x1", "x2"], target_name="response")
```

## Tutorial 4: Reuse a persisted model

```python
from ddmo import load_model

bundle = load_model("trained_model.pkl")
y_pred = bundle.predict(X_test)
```

## Tutorial 5: Use gradients in an optimizer

```python
import numpy as np
from scipy.optimize import minimize
from ddmo import Kriging

model = Kriging().fit(X_train, y_train)

result = minimize(
    lambda x: model.predict(x[None, :])[0],
    x0=np.zeros(X_train.shape[1]),
    jac=lambda x: model.predict_gradient(x[None, :])[0],
)
```

This workflow is especially useful when the surrogate replaces a high-cost deterministic simulation.