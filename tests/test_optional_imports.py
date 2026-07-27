"""Tests that graph_weather.data degrades gracefully without optional heavy deps.

pandas, xarray, anemoi.datasets, and nnja-ai are not declared dependencies of the
graph_weather package, so a plain `pip install graph_weather` may not have them
available. Importing graph_weather.data should not crash in that case - each
optional dataset class should just fall back to None.
"""

import builtins
import importlib
import sys

import pytest

OPTIONAL_MODULES = ("pandas", "xarray", "anemoi.datasets", "nnja_ai")


@pytest.fixture
def hide_optional_deps(monkeypatch):
    """Make imports of the optional heavy deps raise ImportError, like a fresh install."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in OPTIONAL_MODULES or any(name.startswith(mod + ".") for mod in OPTIONAL_MODULES):
            raise ImportError(f"No module named '{name}' (blocked for test)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Drop any already-imported copies so the blocked import actually gets exercised.
    for name in list(sys.modules):
        if name.split(".")[0] in ("graph_weather",) or name in OPTIONAL_MODULES:
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_data_package_imports_without_optional_deps(hide_optional_deps):
    """graph_weather.data should import cleanly with pandas/xarray/anemoi/nnja-ai missing."""
    data = importlib.import_module("graph_weather.data")

    assert data.AnemoiDataset is None
    assert data.SensorDataset is None
    assert data.WeatherStationReader is None
