"""Automated checks for the Week 2 v2 submission and run artefacts.

Run after `python week2_v2.py` (set IAA_OUT to check another output folder):
    python -m pytest tests/test_week2_v2.py -q
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EM = ROOT / "data" / "emerging"
OUT = Path(os.environ.get("IAA_OUT", ROOT / "outputs_week2_v2"))
LINES = ["motor", "buildings", "contents"]


@pytest.fixture(scope="module")
def sub():
    return pd.read_csv(OUT / "submission_week2_v2.csv")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((OUT / "run_manifest_v2.json").read_text())


@pytest.fixture(scope="module")
def test_rows():
    return pd.concat([pd.read_parquet(EM / f"test_{l}.parquet").assign(line=l) for l in LINES], ignore_index=True)


def test_format(sub):
    assert list(sub.columns) == ["row_id", "prediction"]
    assert sub.row_id.is_unique


def test_rows_match_sample_and_test_files(sub, test_rows):
    sample = pd.read_parquet(EM / "sample_submission.parquet")
    assert len(sub) == len(sample) == len(test_rows)
    assert set(sub.row_id) == set(sample.row_id) == set(test_rows.row_id)


def test_predictions_valid(sub):
    assert sub.prediction.notna().all() and np.isfinite(sub.prediction).all()
    assert (sub.prediction >= 0).all()
    assert sub.prediction.max() < 100 * sub.prediction.median()


@pytest.mark.parametrize("line", LINES)
def test_level_within_band(manifest, line):
    """2025 cost per exposure vs 2023-24 experience at 2025 prices: same band the run escalates on."""
    assert 0.80 <= manifest["levels"][line]["ratio"] <= 1.35


@pytest.mark.parametrize("line", LINES)
def test_predictions_scale_with_exposure(sub, test_rows, line):
    t = test_rows[test_rows.line == line].merge(sub, on="row_id")
    assert t[t.exposure < 0.25].prediction.mean() < t[t.exposure > 0.95].prediction.mean()


def test_motor_new_business_costs_more(sub, test_rows):
    t = test_rows[test_rows.line == "motor"].merge(sub, on="row_id")
    s = t.groupby("new_business")[["prediction", "exposure"]].sum()
    assert s.prediction[True] / s.exposure[True] > s.prediction[False] / s.exposure[False]


def test_data_unchanged_since_run(manifest):
    for name, md5 in manifest["data_md5"].items():
        assert hashlib.md5((EM / name).read_bytes()).hexdigest() == md5, f"{name} changed since the run"


def test_full_run(manifest):
    assert not manifest["smoke"], "submission came from a smoke (reduced) run"
    assert manifest["validation"] and manifest["validation"]["score"] > 0.15


def test_severity_development_sane(manifest):
    """IBNER factors: only upward, bounded, only on the immature years; motor bodily injury needs it."""
    dev = manifest.get("severity_development")
    if dev is None:
        pytest.skip("run without severity development")
    d = pd.DataFrame(dev)
    assert d.factor.between(1.0, 1.6).all() and set(d.loc[d.factor > 1, "loss_year"]) <= {2023, 2024}
    bi = d[(d.line == "motor") & (d.peril == "bodily_injury")].set_index("loss_year").factor
    assert bi[2024] > 1.1


def test_excess_profile_sane(manifest):
    """Large-loss allocation: relativities are non-negative and the known large-loss perils carry it."""
    xs = manifest.get("excess_profile")
    if not xs or not manifest["config"].get("xs_peril"):
        pytest.skip("flat large-loss loading")
    assert all(v >= 0 for d in xs.values() for v in d.values())
    assert xs["motor"]["bodily_injury"] > 1.5 and xs["buildings"]["fire"] > 1.5 and xs["contents"]["flood"] > 1.5


def test_config_sane(manifest):
    cfg = manifest["config"]
    for line in LINES:
        lc = cfg["layer"][line]
        assert 0 <= lc["w_peril"] <= 1 and 0 <= lc["w_a"] <= 1 and 0 <= lc["w_resid"] <= 1
        assert 0 <= lc.get("w_pool", 0) <= 1
        assert (1 - lc["w_a"]) * (1 - lc.get("w_pool", 0)) >= 0.5, "the transfer layer must keep most of the weight"
        assert lc["nb_mode"] in ("none", "line", "peril")


def test_kaggle_copy_matches_run_config(manifest):
    """The Kaggle script must carry the configuration of the run that wrote the submission (no re-selection
    on Kaggle). Re-export with IAA_MODE=export and the same IAA_OUT after any change to CFG."""
    k = ROOT / "week2_v2_kaggle.py"
    if not k.exists():
        pytest.skip("export not run yet")
    src = k.read_text()
    ns = {}
    a, b = src.index("CFG = {"), src.index("WEATHER_PERIL_TERMS = ")
    exec(src[a:b], {"np": np}, ns)
    assert ns["CFG"] == manifest["config"]
    nb = k.with_suffix(".ipynb")
    if nb.exists():
        assert "".join(json.loads(nb.read_text())["cells"][-1]["source"]) == src, "notebook differs from the .py copy"
