import warnings

import numpy as np
import pytest

from ddmo import metrics
from ddmo.models import (
    LS,
    MOE,
    RBF,
    Kriging,
    KrigingSurrogate,
    LinearSurrogate,
    RBFSurrogate,
    WeightedEnsemble,
)


def branin_like(X):
    return np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + X[:, 0] * X[:, 1]


@pytest.fixture
def data_2d():
    rng = np.random.default_rng(42)
    X_train = rng.uniform(-1, 1, size=(60, 2))
    X_test = rng.uniform(-0.9, 0.9, size=(200, 2))
    return X_train, branin_like(X_train), X_test, branin_like(X_test)


# ---------------------------------------------------------------- base / metrics


def test_predict_is_independent_of_call_order():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = X[:, 0] ** 2
    for make in (LS, RBF, Kriging):
        model = make().fit(X, y)
        batch = model.predict(X)
        single = np.array([model.predict(x[None, :])[0] for x in X])
        assert np.allclose(batch, single)


def test_input_validation():
    model = LS().fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="features"):
        model.predict([[0.0, 1.0]])
    with pytest.raises(ValueError, match="NaN"):
        LS().fit([[0.0], [np.nan]], [0.0, 1.0])
    with pytest.raises(ValueError, match="1D"):
        LS().fit([[0.0], [1.0]], [[0.0, 1.0], [1.0, 2.0]])
    with pytest.raises(RuntimeError, match="not been fitted"):
        LS().predict([[0.0]])


def test_score_is_r2():
    X = np.linspace(0, 1, 10)[:, None]
    y = 2 * X[:, 0] + 1
    assert LS().fit(X, y).score(X, y) == pytest.approx(1.0)


def test_metrics():
    y, p = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])
    assert metrics.mse(y, p) == pytest.approx(4 / 3)
    assert metrics.rmse(y, p) == pytest.approx(np.sqrt(4 / 3))
    assert metrics.r2(y, p) == pytest.approx(1 - 4 / 2)
    assert metrics.r2([2.0, 2.0], [2.0, 2.0]) == 1.0


def test_aliases():
    assert LinearSurrogate is LS
    assert RBFSurrogate is RBF
    assert KrigingSurrogate is Kriging
    assert MOE is WeightedEnsemble


# ---------------------------------------------------------------- least squares


def test_ls_recovers_linear_function_exactly():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = 2 * X[:, 0] + 1
    model = LS().fit(X, y)
    assert np.allclose(model.predict([[10.0]]), [21.0])


def test_ls_degree_zero_is_constant():
    X = np.random.default_rng(0).random((20, 3))
    y = X.sum(axis=1)
    model = LS(degree=0).fit(X, y)
    assert model.coefficients_.size == 1
    assert np.allclose(model.predict(X), y.mean())


def test_ls_quadratic_includes_interactions(data_2d):
    X, _, X_test, _ = data_2d

    def f(Z):
        return 1 + Z[:, 0] - 2 * Z[:, 1] + 3 * Z[:, 0] * Z[:, 1] + Z[:, 1] ** 2

    model = LS(degree=2).fit(X, f(X))
    assert len(model.terms_) == 6
    assert np.allclose(model.predict(X_test), f(X_test), atol=1e-10)


def test_ls_underdetermined_does_not_crash():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0])
    assert np.all(np.isfinite(LS(degree=3).fit(X, y).predict(X)))
    assert np.all(np.isfinite(LS(degree=3, ridge=1e-3).fit(X, y).predict(X)))


def test_ls_ridge_shrinks_but_keeps_intercept():
    X = np.linspace(-1, 1, 30)[:, None]
    y = 5.0 + 3.0 * X[:, 0]
    heavy = LS(ridge=1e6).fit(X, y)
    assert np.allclose(heavy.predict(X), 5.0, atol=1e-3)


# ---------------------------------------------------------------- RBF


@pytest.mark.parametrize(
    "kernel", ["gaussian", "multiquadric", "inverse_multiquadric", "linear", "cubic", "thin_plate"]
)
def test_rbf_interpolates_and_generalizes(kernel, data_2d):
    X, y, X_test, y_test = data_2d
    model = RBF(kernel=kernel, gamma="auto").fit(X, y)
    assert np.allclose(model.predict(X), y, atol=1e-5)
    assert metrics.r2(y_test, model.predict(X_test)) > 0.95


@pytest.mark.parametrize("kernel", ["cubic", "thin_plate", "multiquadric", "linear"])
def test_rbf_polynomial_reproduction(kernel):
    # With a degree-1 tail, a linear function is reproduced everywhere, including far away.
    X = np.random.default_rng(1).random((15, 2))

    def f(Z):
        return 1 + 2 * Z[:, 0] - Z[:, 1]

    model = RBF(kernel=kernel, degree=1, regularization=0.0).fit(X, f(X))
    far = np.array([[5.0, 5.0], [-3.0, 2.0]])
    assert np.allclose(model.predict(far), f(far), atol=1e-6)


def test_rbf_gaussian_far_field_tends_to_mean_not_zero():
    X = np.linspace(0, 1, 10)[:, None]
    y = 100 + np.sin(6 * X[:, 0])
    far = RBF(gamma=2.0).fit(X, y).predict([[50.0]])[0]
    assert far == pytest.approx(y.mean(), rel=0.02)


def test_rbf_requires_minimum_tail_degree():
    X = np.random.default_rng(2).random((10, 2))
    with pytest.raises(ValueError, match="degree >= 1"):
        RBF(kernel="thin_plate", degree=0).fit(X, X[:, 0])
    RBF(kernel="gaussian", degree=-1).fit(X, X[:, 0])


def test_rbf_deprecated_kernel_names():
    X = np.linspace(0, 1, 5)[:, None]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = RBF(kernel="multiquadratic").fit(X, X[:, 0])
    assert model.kernel_ == "multiquadric"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    with pytest.raises(ValueError, match="Unsupported kernel"):
        RBF(kernel="nope").fit(X, X[:, 0])


# ---------------------------------------------------------------- kriging


def test_kriging_interpolates_and_generalizes(data_2d):
    X, y, X_test, y_test = data_2d
    model = Kriging().fit(X, y)
    assert np.allclose(model.predict(X), y, atol=1e-5)
    assert metrics.r2(y_test, model.predict(X_test)) > 0.99


def test_kriging_mle_detects_irrelevant_dimension():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, size=(40, 2))
    y = np.sin(3 * X[:, 0])  # x1 is irrelevant
    model = Kriging().fit(X, y)
    assert model.theta_[0] > 10 * model.theta_[1]


def test_kriging_std_zero_at_data_and_grows_away():
    X = np.linspace(0, 1, 6)[:, None]
    y = np.sin(4 * X[:, 0])
    model = Kriging().fit(X, y)
    _, std_train = model.predict(X, return_std=True)
    _, std_mid = model.predict([[0.1]], return_std=True)
    _, std_far = model.predict([[100.0]], return_std=True)
    assert np.all(std_train < 1e-3)
    assert 0 < std_mid[0] < std_far[0]
    assert std_far[0] == pytest.approx(
        np.sqrt(model.sigma2_ * (1 + 1 / model._ones_rinv_ones)), rel=1e-6
    )


def test_kriging_fixed_theta_and_validation():
    X = np.linspace(0, 1, 5)[:, None]
    model = Kriging(theta=2.0).fit(X, X[:, 0])
    assert np.allclose(model.theta_, [2.0])
    with pytest.raises(ValueError, match="p must"):
        Kriging(p=3.0).fit(X, X[:, 0])


def test_kriging_handles_near_duplicate_points():
    X = np.array([[0.0], [1e-9], [0.5], [1.0]])
    y = np.array([0.0, 0.0, 0.25, 1.0])
    model = Kriging(theta=1.0, nugget=0.0).fit(X, y)
    assert np.all(np.isfinite(model.predict(X)))


# ---------------------------------------------------------------- ensemble


def test_ensemble_equals_weighted_average_of_experts(data_2d):
    X, y, X_test, _ = data_2d
    ls, rbf = LS(degree=2), RBF(kernel="cubic")
    model = MOE(experts=[ls, rbf], weights=[1.0, 3.0]).fit(X, y)
    expected = 0.25 * ls.predict(X_test) + 0.75 * rbf.predict(X_test)
    assert np.allclose(model.predict(X_test), expected)
    assert np.allclose(model.predict(X_test[:1]), expected[:1])
    assert model.n_features_in_ == 2


def test_ensemble_cv_weights_prefer_better_expert(data_2d):
    X, y, X_test, y_test = data_2d
    model = MOE(experts=[LS(degree=0), RBF(kernel="cubic")], weights="cv").fit(X, y)
    assert model.weights_[1] > 0.9
    assert metrics.r2(y_test, model.predict(X_test)) > 0.95


def test_ensemble_weight_validation():
    X = np.linspace(0, 1, 5)[:, None]
    with pytest.raises(ValueError, match="2 weights for 1 experts"):
        MOE(experts=[LS()], weights=[1.0, 2.0]).fit(X, X[:, 0])
    with pytest.raises(ValueError, match="non-negative"):
        MOE(experts=[LS(), LS()], weights=[1.0, -1.0]).fit(X, X[:, 0])
    with pytest.raises(ValueError, match="At least one"):
        MOE().fit(X, X[:, 0])
