# ddmo

[![Linux](https://github.com/Ahmed-Bayoumy/ddmo/actions/workflows/lx-build-and-tests.yml/badge.svg?branch=DEV)](https://github.com/Ahmed-Bayoumy/ddmo/actions/workflows/lx-build-and-tests.yml)
[![macOS](https://github.com/Ahmed-Bayoumy/ddmo/actions/workflows/macos-build-and-pytest.yml/badge.svg?branch=DEV)](https://github.com/Ahmed-Bayoumy/ddmo/actions/workflows/macos-build-and-pytest.yml)
[![Windows](https://github.com/Ahmed-Bayoumy/ddmo/actions/workflows/win-build-and-pytest.yml/badge.svg?branch=DEV)](https://github.com/Ahmed-Bayoumy/ddmo/actions/workflows/win-build-and-pytest.yml)

**Data-Driven Models for Optimization** — lightweight surrogate models built on NumPy and SciPy.

`ddmo` fits cheap approximations of expensive functions (simulations, experiments) from a
set of samples, so an optimizer can query the surrogate instead of the true function.
All models share one scikit-learn-style interface: `fit(X, y)`, `predict(X)`, `score(X, y)`.

| Model | Class | Use it when |
|---|---|---|
| Polynomial least squares | `LS` | You want a smooth global trend or a response surface (degree 1–3), or the data is noisy. |
| Radial basis functions | `RBF` | You want an interpolant that passes through every sample and is fast to fit. |
| Kriging (Gaussian process) | `Kriging` | You also need an uncertainty estimate, e.g. for expected improvement. |
| Weighted ensemble | `WeightedEnsemble` | You want to average several models, optionally weighted by cross-validation error. |

## Installation

Requires Python 3.9+, NumPy and SciPy.

```bash
pip install git+https://github.com/Ahmed-Bayoumy/ddmo.git
```

For development:

```bash
git clone https://github.com/Ahmed-Bayoumy/ddmo.git
cd ddmo
pip install -e ".[test]"
pytest
```

## Quick start

```python
import numpy as np
from ddmo import Kriging

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=(40, 2))          # 40 samples, 2 inputs
y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2

model = Kriging().fit(X, y)

X_new = np.array([[0.2, -0.4], [0.9, 0.9]])
mean, std = model.predict(X_new, return_std=True)
print(mean, std)
print("R^2 on training data:", model.score(X, y))
```

`X` is always a 2-D array of shape `(n_samples, n_features)`; `y` is 1-D (a single output).
Inputs are standardized internally using the training data (`normalize=True` by default),
and NaN or infinite values are rejected.

## Models

### `LS` — polynomial least squares

```python
from ddmo import LS

LS(degree=2)              # full quadratic: 1, x0, x1, x0^2, x0*x1, x1^2
LS(degree=3, ridge=1e-3)  # cubic with ridge regularization
```

- `degree`: total polynomial degree, including interaction terms. `degree=0` fits a constant.
- `ridge`: L2 penalty. Basis columns are scaled first so every term is penalized equally;
  the intercept is never penalized.

The model is solved with `numpy.linalg.lstsq`, so it stays stable when there are fewer
samples than terms (it returns the minimum-norm solution).

### `RBF` — radial basis function interpolation

```python
from ddmo import RBF

RBF(kernel="cubic")                   # no shape parameter to tune
RBF(kernel="gaussian", gamma="auto")  # shape parameter chosen by leave-one-out CV
```

The interpolant is `s(x) = Σ wᵢ φ(γ‖x − xᵢ‖) + q(x)`, where `q` is a polynomial tail.

| `kernel` | φ(r) | Default tail degree |
|---|---|---|
| `gaussian` | exp(−(γr)²) | 0 |
| `inverse_multiquadric` | 1 / √(1 + (γr)²) | 0 |
| `multiquadric` | √(1 + (γr)²) | 0 |
| `linear` | r | 0 |
| `cubic` | r³ | 1 |
| `thin_plate` | r² log r | 1 |

- `gamma`: shape parameter (larger = more localized). Ignored by `linear`, `cubic` and
  `thin_plate`. `"auto"` selects it with Rippa's closed-form leave-one-out error, skipping
  values whose linear system is ill-conditioned.
- `degree`: polynomial tail degree. `None` uses the default above; each kernel has a
  minimum degree that guarantees a unique solution (1 for `cubic`/`thin_plate`), and
  `degree=-1` (no tail) is allowed only for `gaussian` and `inverse_multiquadric`.
- `regularization`: added to the kernel diagonal. Increase it to smooth noisy data.

### `Kriging` — ordinary kriging

```python
from ddmo import Kriging

model = Kriging().fit(X, y)          # theta estimated by maximum likelihood
model.theta_                         # one correlation parameter per input dimension
mean, std = model.predict(X_new, return_std=True)
```

Correlation: `R(x, x') = exp(−Σₖ θₖ |xₖ − x'ₖ|^p)`.

- `theta`: `None` (default) estimates one θₖ per dimension by maximizing the concentrated
  likelihood (multi-start L-BFGS-B within `theta_bounds`). Pass a number or an array to fix it.
  A small fitted θₖ means the model found input `k` to have little influence.
- `p`: smoothness exponent, `0 < p ≤ 2` (`2` = Gaussian).
- `nugget`: diagonal jitter. It is increased automatically if the correlation matrix is not
  numerically positive definite.
- `n_restarts`, `random_state`: control the likelihood optimization.

`predict(X, return_std=True)` returns the kriging standard error, which is zero at the
training points and grows away from them. Fitted values include `beta_` (the constant mean)
and `sigma2_` (the process variance).

### `WeightedEnsemble` — model averaging

```python
from ddmo import LS, RBF, WeightedEnsemble

ens = WeightedEnsemble(experts=[LS(degree=2), RBF(kernel="cubic")], weights="cv").fit(X, y)
ens.weights_   # normalized weights
ens.cv_mse_    # cross-validation MSE of each expert (weights="cv" only)
```

- `weights`: `None`/`"uniform"`, `"cv"` (proportional to 1 / k-fold CV error), or a list of
  non-negative numbers, one per expert.
- `n_folds`, `random_state`: control the cross-validation.

The weights are the same everywhere in the input space (there is no gating network).
The experts are fitted in place, so you can inspect them after fitting.

## Gradients

Every model provides the analytic gradient of its prediction with respect to the inputs,
which you can pass to gradient-based optimizers:

```python
from scipy.optimize import minimize

model = Kriging().fit(X, y)

grad = model.predict_gradient(X_new)   # shape (n_samples, n_features), in original input units

res = minimize(
    lambda x: model.predict(x[None, :])[0],
    x0=np.zeros(2),
    jac=lambda x: model.predict_gradient(x[None, :])[0],
    bounds=[(-1, 1), (-1, 1)],
)
```

Gradients account for the internal input normalization, so they are always with respect to
the inputs as you passed them. The `linear` and `thin_plate` RBF kernels, and Kriging with
`p <= 1`, are not differentiable exactly at training points; the gradient contribution there
is taken as 0.

## Metrics

```python
from ddmo import metrics

metrics.mse(y_true, y_pred)
metrics.rmse(y_true, y_pred)
metrics.r2(y_true, y_pred)
```

`model.score(X, y)` returns R².

## Name aliases

`LinearSurrogate`, `RBFSurrogate`, `KrigingSurrogate`, `MixtureOfExperts` and `MOE` are
aliases of `LS`, `RBF`, `Kriging` and `WeightedEnsemble`. The RBF kernel names
`multiquadratic`, `inverse_multiquadratic` and `absolute` are deprecated in favour of
`multiquadric`, `inverse_multiquadric` and `linear`.

## Limitations

- Single-output models only.
- Gradients are available for the prediction, but not for the Kriging standard error.
- Kriging and RBF build dense `n × n` matrices, so they suit up to a few thousand samples.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
