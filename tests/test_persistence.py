import io
import pickle

import numpy as np
import pytest

from ddmo import LS, RBF, Kriging, ModelBundle, WeightedEnsemble, load_model, save_model


@pytest.fixture
def data_2d():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(40, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    return X, y, rng.uniform(-1, 1, size=(25, 2))


@pytest.mark.parametrize(
    "make",
    [
        lambda: LS(degree=2),
        lambda: LS(degree=3, lasso=0.01),
        lambda: RBF(kernel="cubic"),
        lambda: Kriging(),
        lambda: WeightedEnsemble(experts=[LS(), RBF(), Kriging()], weights="cv"),
    ],
)
def test_round_trip_reproduces_predictions(tmp_path, data_2d, make):
    X, y, X_new = data_2d
    model = make().fit(X, y)
    path = tmp_path / "model.pkl"

    save_model(model, path, feature_names=["a", "b"], target_name="f", metadata={"note": "hi"})
    bundle = load_model(path)

    assert isinstance(bundle, ModelBundle)
    assert type(bundle.model) is type(model)
    np.testing.assert_array_equal(bundle.predict(X_new), model.predict(X_new))
    assert bundle.feature_names == ["a", "b"]
    assert bundle.target_name == "f"
    assert bundle.metadata == {"note": "hi"}
    assert bundle.created_at


def test_round_trip_through_file_object(data_2d):
    X, y, X_new = data_2d
    model = Kriging().fit(X, y)
    buffer = io.BytesIO()
    save_model(model, buffer)
    buffer.seek(0)

    bundle = load_model(buffer)
    np.testing.assert_array_equal(bundle.predict_gradient(X_new), model.predict_gradient(X_new))


def test_predict_selects_named_columns(tmp_path, data_2d):
    pd = pytest.importorskip("pandas")
    X, y, X_new = data_2d
    model = LS().fit(X, y)
    save_model(model, tmp_path / "m.pkl", feature_names=["a", "b"])
    bundle = load_model(tmp_path / "m.pkl")

    frame = pd.DataFrame({"b": X_new[:, 1], "extra": 0.0, "a": X_new[:, 0]})
    np.testing.assert_array_equal(bundle.predict(frame), model.predict(X_new))
    with pytest.raises(ValueError, match="missing feature"):
        bundle.predict(frame.drop(columns="a"))


def test_save_rejects_unfitted_model_and_bad_feature_names(tmp_path, data_2d):
    X, y, _ = data_2d
    with pytest.raises(ValueError, match="fitted"):
        save_model(LS(), tmp_path / "m.pkl")
    with pytest.raises(ValueError, match="feature names"):
        save_model(LS().fit(X, y), tmp_path / "m.pkl", feature_names=["only_one"])


def test_load_rejects_foreign_pickle(tmp_path):
    path = tmp_path / "other.pkl"
    path.write_bytes(pickle.dumps({"hello": "world"}))
    with pytest.raises(ValueError, match="not a ddmo model"):
        load_model(path)
