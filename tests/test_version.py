from pathlib import Path

import pytest

import ddmo

tomllib = pytest.importorskip("tomllib")  # stdlib from Python 3.11

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_package_version_matches_pyproject():
    with PYPROJECT.open("rb") as fh:
        project_version = tomllib.load(fh)["project"]["version"]
    assert ddmo.__version__ == project_version
