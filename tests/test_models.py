import warnings

import numpy as np
import pytest

from ddmo import CollinearityFilter, feature_selection, metrics
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


def test_kriging_interpolates_with_well_conditioned_theta(data_2d):
    X, y, _, _ = data_2d
    model = Kriging(theta=10.0).fit(X, y)
    assert np.allclose(model.predict(X), y, atol=1e-6)


def test_kriging_mle_generalizes(data_2d):
    # For smooth data the likelihood favours a nearly flat correlation, so R is close to
    # singular and the nugget costs ~1e-5 of interpolation accuracy (platform-dependent).
    X, y, X_test, y_test = data_2d
    model = Kriging().fit(X, y)
    assert np.allclose(model.predict(X), y, atol=1e-3)
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


# ---------------------------------------------------------------- gradients


def finite_difference_gradient(model, X, h=1e-4):
    # Fourth-order central differences: truncation error O(h^4) without pushing h into
    # the range where round-off from ill-conditioned models dominates.
    grad = np.zeros_like(X)
    for k in range(X.shape[1]):
        step = np.zeros(X.shape[1])
        step[k] = h * max(1.0, np.abs(X[:, k]).max())
        f = [model.predict(X + j * step) for j in (-2, -1, 1, 2)]
        grad[:, k] = (f[0] - 8 * f[1] + 8 * f[2] - f[3]) / (12 * step[k])
    return grad


@pytest.fixture
def data_scaled():
    # Columns on very different scales to exercise the normalization chain rule.
    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, size=(40, 2)) * np.array([1.0, 100.0])
    y = np.sin(3 * X[:, 0]) + (X[:, 1] / 100) ** 2 + X[:, 0] * X[:, 1] / 100
    X_eval = rng.uniform(-0.9, 0.9, size=(15, 2)) * np.array([1.0, 100.0])
    return X, y, X_eval


@pytest.mark.parametrize(
    "make",
    [
        lambda: LS(degree=3),
        lambda: LS(degree=2, ridge=1e-2, normalize=False),
        lambda: LS(degree=2, lasso=1e-2, ridge=1e-3),
        lambda: RBF(kernel="gaussian", gamma=2.0),
        lambda: RBF(kernel="multiquadric", gamma=1.5),
        lambda: RBF(kernel="inverse_multiquadric", gamma=1.5),
        lambda: RBF(kernel="linear"),
        lambda: RBF(kernel="cubic", degree=2),
        lambda: RBF(kernel="thin_plate"),
        lambda: RBF(kernel="cubic", normalize=False),
        lambda: Kriging(),
        lambda: Kriging(theta=[3.0, 0.5], p=1.5),
        lambda: Kriging(theta=[3.0, 3e-4], normalize=False),
        lambda: MOE(experts=[LS(degree=2), RBF(kernel="cubic"), Kriging()], weights="cv"),
    ],
)
def test_gradient_matches_finite_differences(make, data_scaled):
    X, y, X_eval = data_scaled
    model = make().fit(X, y)
    grad = model.predict_gradient(X_eval)
    assert grad.shape == X_eval.shape
    fd = finite_difference_gradient(model, X_eval)
    scale = np.abs(fd).max(axis=0)
    assert np.allclose(grad / scale, fd / scale, atol=1e-5)


def test_ls_gradient_is_exact():
    X = np.random.default_rng(0).random((20, 2)) * 10

    def f(Z):
        return 1 + 2 * Z[:, 0] - Z[:, 1] + 0.5 * Z[:, 0] * Z[:, 1]

    grad = LS(degree=2).fit(X, f(X)).predict_gradient([[1.0, 2.0], [3.0, -1.0]])
    assert np.allclose(grad, [[2 + 0.5 * 2, -1 + 0.5 * 1], [2 + 0.5 * -1, -1 + 0.5 * 3]])


def test_gradient_at_training_points_is_finite():
    X = np.random.default_rng(3).random((12, 2))
    y = X[:, 0] ** 2 - X[:, 1]
    for model in (RBF(kernel="linear"), RBF(kernel="thin_plate"), Kriging(theta=1.0, p=1.0)):
        assert np.all(np.isfinite(model.fit(X, y).predict_gradient(X)))


def test_ensemble_gradient_is_weighted_sum():
    X = np.random.default_rng(4).random((25, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1]
    ls, rbf = LS(degree=2), RBF(kernel="cubic")
    model = MOE(experts=[ls, rbf], weights=[1.0, 3.0]).fit(X, y)
    expected = 0.25 * ls.predict_gradient(X) + 0.75 * rbf.predict_gradient(X)
    assert np.allclose(model.predict_gradient(X), expected)


def test_gradient_validation():
    with pytest.raises(RuntimeError, match="not been fitted"):
        LS().predict_gradient([[0.0]])
    model = LS().fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="features"):
        model.predict_gradient([[0.0, 1.0]])


# ---------------------------------------------------------------- lasso option


@pytest.fixture
def sparse_data():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(80, 8))
    y = 3 * X[:, 0] - 2 * X[:, 3] + 0.01 * rng.normal(size=80)
    return X, y


def test_ls_lasso_recovers_sparse_support(sparse_data):
    X, y = sparse_data
    model = LS(lasso=5.0).fit(X, y)
    assert list(model.selected_features_) == [0, 3]
    assert model.score(X, y) > 0.99


def test_ls_lasso_auto_keeps_informative_features(sparse_data):
    X, y = sparse_data
    model = LS(lasso="auto").fit(X, y)
    assert {0, 3} <= set(model.selected_features_)
    assert model.lasso_ in model.lasso_path_
    assert model.score(X, y) > 0.99


def test_ls_lasso_satisfies_kkt_conditions(sparse_data):
    X, y = sparse_data
    lasso, ridge = 20.0, 0.5
    model = LS(lasso=lasso, ridge=ridge, normalize=False, tol=1e-12).fit(X, y)
    A = model._design(X)
    w = model.coefficients_[1:]
    corr = (A[:, 1:] - A[:, 1:].mean(axis=0)).T @ (y - A @ model.coefficients_) - ridge * w
    nz = w != 0
    assert np.allclose(corr[nz], lasso * np.sign(w[nz]), atol=1e-6)
    assert np.all(np.abs(corr[~nz]) <= lasso + 1e-6)


def test_ls_lasso_zero_matches_plain_least_squares(sparse_data):
    X, y = sparse_data
    assert np.allclose(LS(lasso=0.0).fit(X, y).predict(X), LS().fit(X, y).predict(X))


def test_ls_elastic_net_approaches_ridge_as_lasso_vanishes(sparse_data):
    X, y = sparse_data
    ridge = LS(ridge=5.0).fit(X, y).predict(X)
    enet = LS(ridge=5.0, lasso=1e-9, tol=1e-12).fit(X, y).predict(X)
    assert np.allclose(ridge, enet, atol=1e-6)


def test_ls_large_lasso_gives_constant_model(sparse_data):
    X, y = sparse_data
    model = LS(lasso=1e6).fit(X, y)
    assert model.selected_features_.size == 0
    assert np.allclose(model.predict(X), y.mean())


def test_ls_lasso_with_duplicated_inputs_matches_deduplicated_fit():
    # With exact duplicates the lasso optimum is not unique (any split between the copies
    # is optimal), but the fitted function is. CollinearityFilter removes such columns.
    rng = np.random.default_rng(5)
    x = rng.normal(size=(60, 2))
    X = np.column_stack([x[:, 0], x[:, 0], x[:, 1]])  # column 1 duplicates column 0
    y = 2 * x[:, 0] + x[:, 1]
    dup = LS(lasso=1.0, tol=1e-12).fit(X, y)
    ref = LS(lasso=1.0, tol=1e-12).fit(x, y)
    assert np.allclose(dup.predict(X), ref.predict(x), atol=1e-8)
    assert dup.coefficients_[1] + dup.coefficients_[2] == pytest.approx(ref.coefficients_[1])
    assert list(CollinearityFilter(method="correlation").fit(X).selected_features_) == [0, 2]


def test_ls_lasso_selects_interaction_term():
    X = np.random.default_rng(6).uniform(-1, 1, size=(80, 3))
    y = X[:, 0] * X[:, 1]
    model = LS(degree=2, lasso=5.0).fit(X, y)
    nonzero = [t for t, c in zip(model.terms_[1:], model.coefficients_[1:]) if c != 0]
    assert nonzero == [(0, 1)]
    assert list(model.selected_features_) == [0, 1]


def test_ls_lasso_validation():
    X = np.random.default_rng(0).random((10, 2))
    with pytest.raises(ValueError, match="lasso must be"):
        LS(lasso=-1.0).fit(X, X[:, 0])
    with pytest.raises(ValueError, match="lasso must be"):
        LS(lasso="cv").fit(X, X[:, 0])


# ---------------------------------------------------------------- feature selection


@pytest.fixture
def collinear_X():
    rng = np.random.default_rng(8)
    x0, x1, x3 = rng.normal(size=(3, 100))
    # column 2 is an exact combination of 0 and 1; column 4 is constant
    return np.column_stack([x0, x1, x0 + x1, x3, np.full(100, 2.0)])


def test_variance_inflation_factors(collinear_X):
    vif = feature_selection.variance_inflation_factors(collinear_X)
    assert np.all(np.isinf(vif[:3]))
    assert vif[3] < 1.2
    assert np.isnan(vif[4])
    independent = np.random.default_rng(9).normal(size=(200, 3))
    assert np.all(feature_selection.variance_inflation_factors(independent) < 1.1)


def test_vif_filter_drops_combination_and_constant(collinear_X):
    assert list(feature_selection.vif_filter(collinear_X)) == [0, 1, 3]


def test_correlation_filter_drops_near_duplicates():
    rng = np.random.default_rng(10)
    x0, x2 = rng.normal(size=(2, 100))
    X = np.column_stack([x0, 2 * x0 + 1e-3 * rng.normal(size=100), x2])
    assert list(feature_selection.correlation_filter(X, threshold=0.99)) == [0, 2]
    assert list(feature_selection.correlation_filter(X, threshold=1.0)) == [0, 1, 2]


def test_lasso_select(sparse_data):
    X, y = sparse_data
    assert list(feature_selection.lasso_select(X, y, lasso=5.0)) == [0, 3]
    with pytest.raises(ValueError, match="positive"):
        feature_selection.lasso_select(X, y, lasso=0.0)


def test_collinearity_filter_pipeline(collinear_X):
    y = collinear_X[:, 0] - collinear_X[:, 3]
    filt = CollinearityFilter(method="vif").fit(collinear_X)
    assert list(filt.selected_features_) == [0, 1, 3]
    assert list(filt.dropped_features_) == [2, 4]
    Xs = filt.transform(collinear_X)
    assert Xs.shape == (100, 3)
    model = LS().fit(Xs, y)
    assert model.score(filt.transform(collinear_X), y) == pytest.approx(1.0)

    corr = CollinearityFilter(method="correlation").fit(collinear_X)
    assert 4 not in corr.selected_features_
    with pytest.raises(ValueError, match="features"):
        filt.transform(collinear_X[:, :3])
    with pytest.raises(ValueError, match="method"):
        CollinearityFilter(method="pca").fit(collinear_X)
    with pytest.raises(RuntimeError, match="not been fitted"):
        CollinearityFilter().transform(collinear_X)
