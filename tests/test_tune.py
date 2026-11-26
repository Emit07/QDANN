"""The guards around saved models: provenance, no silent overwrite, the GEE/MPC check."""

import json
import types

import pytest
import shapely

from qdann import mpc, predict, tune, weather
from test_train import table


def _save(tmp_path, monkeypatch, name: str):
    path = tmp_path / name
    table(n_counties=6, years=range(2016, 2019)).to_parquet(path)
    source = weather.read(path)
    monkeypatch.setattr(weather, "DATA_DIR", tmp_path)
    # filtering keeps the stamp, as gate.py and arms.py filter before saving
    return source, tune.save(source[source["year"] != 2016], "maize", 0.1, 2, 0.3)


def test_a_saved_model_names_its_table_and_is_never_overwritten(tmp_path, monkeypatch):
    source, saved = _save(tmp_path, monkeypatch, "source_mpc_maize.parquet")
    provenance = json.loads(saved.with_suffix(".json").read_text())["provenance"]
    assert provenance["inputs"][0] == source.attrs["source"]
    assert len(source.attrs["source"]["sha256"]) == 64
    assert len(provenance["git_commit"]) == 40  # the repo's, wherever DATA_DIR points
    with pytest.raises(FileExistsError):
        tune.save(source, "maize", 0.1, 2, 0.3)
    predict.load("maize")  # an MPC table passes


def test_loading_a_model_trained_on_the_gee_table_fails(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, "source_maize.parquet")
    with pytest.raises(SystemExit, match="AMBIGUITIES #27"):
        predict.load("maize")
    predict.load("maize", check=False)  # on purpose, for a comparison


def test_an_unstamped_table_warns_rather_than_crashes(caplog):
    for saved in ({}, {"provenance": {"inputs": ["unknown: not read with weather.read"]}}):
        predict.check_reader(saved)
    assert caplog.text.count("cannot confirm") == 2


def test_the_resume_key_changes_with_the_source_table(tmp_path):
    frame = table(n_counties=4, years=range(2017, 2019))
    keys = []
    for name, rows in (("a", frame), ("b", frame.iloc[1:])):
        rows.to_parquet(tmp_path / f"{name}.parquet")
        job = tune._jobs(weather.read(tmp_path / f"{name}.parquet"), "qdann", 0.1, 2)[0]
        keys.append(tune._key(job))
    assert keys[0] != keys[1]


def test_a_year_without_cdl_tiles_raises(monkeypatch):
    # MPC's CDL stops at 2021; an empty list would give every county an empty crop mask
    search = types.SimpleNamespace(items=lambda: [])
    monkeypatch.setattr(mpc, "catalog", lambda: types.SimpleNamespace(search=lambda **_: search))
    with pytest.raises(ValueError, match="2022"):
        mpc.cdl_tiles(year=2022, region=shapely.box(-94, 42, -93, 43))
