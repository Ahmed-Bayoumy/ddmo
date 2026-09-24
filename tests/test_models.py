import numpy as np

from ddmo.models import MOE, KrigingSurrogate, LinearSurrogate, RBFSurrogate


def test_linear_surrogate_fit_predict():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([1.0, 3.0, 5.0, 7.0], dtype=float)

    model = LinearSurrogate()
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert np.allclose(pred, y, atol=1e-6)


def test_rbf_surrogate_fit_predict():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0.0, 1.0, 4.0, 9.0], dtype=float)

    model = RBFSurrogate(gamma=0.5, regularization=1e-8)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert np.allclose(pred, y, atol=1e-2)


def test_kriging_surrogate_fit_predict():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0.0, 1.0, 4.0, 9.0], dtype=float)

    model = KrigingSurrogate(theta=1.0, p=2.0, nugget=1e-8)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert np.allclose(pred, y, atol=1e-1)


def test_moe_fit_predict():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0.0, 1.0, 4.0, 9.0], dtype=float)

    experts = [LinearSurrogate(), RBFSurrogate(gamma=0.5, regularization=1e-8)]
    model = MOE(experts=experts)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))
