"""Automated checks for the Week 2 (emerging market) submission.

Run after `python week2_emerging.py`:
    python -m pytest tests/test_week2.py -q
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EM = ROOT / "data" / "emerging"
OUT = ROOT / "outputs_week2"
LINES = {"M": "motor", "B": "buildings", "C": "contents"}


@pytest.fixture(scope="module")
def sub():
    return pd.read_csv(OUT / "submission_week2.csv")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((OUT / "run_manifest.json").read_text())


@pytest.fixture(scope="module")
def test_rows():
    return pd.concat([pd.read_parquet(EM / f"test_{l}.parquet").assign(line=l) for l in LINES.values()],
                     ignore_index=True)


# ---------------------------------------------------------------- submission format
def test_header_and_types(sub):
    assert list(sub.columns) == ["row_id", "prediction"]
    assert pd.api.types.is_float_dtype(sub.prediction)


def test_rows_match_sample_and_test_files(sub, test_rows):
    sample = pd.read_parquet(EM / "sample_submission.parquet")
    assert sub.row_id.is_unique
    assert len(sub) == len(sample) == len(test_rows)
    assert set(sub.row_id) == set(sample.row_id) == set(test_rows.row_id)


def test_predictions_valid(sub):
    assert sub.prediction.notna().all()
    assert np.isfinite(sub.prediction).all()
    assert (sub.prediction >= 0).all()


# ---------------------------------------------------------------- actuarial sense checks
def _train_pure_premium(line):
    """Raw incurred net of recoveries per exposure year, 2023-24 (no development, no trend)."""
    pol = pd.read_parquet(EM / f"policies_{line}.parquet")
    expo = pol.exposure.where(pol.exposure <= 1, 1.0).sum()
    cl = pd.read_parquet(EM / "claims.parquet").query("line == @line and status != 'repudiated'")
    return (cl.incurred_to_date - cl.recoveries).clip(lower=0).sum() / expo


@pytest.mark.parametrize("prefix,line", LINES.items())
def test_line_level_is_plausible(sub, test_rows, prefix, line):
    """2025 cost per exposure within 0.8x-1.6x of raw 2023-24 experience (development, trend and
    the new-business mix shift together should not move it further than that)."""
    t = test_rows[test_rows.line == line].merge(sub, on="row_id")
    pp_2025 = t.prediction.sum() / t.exposure.sum()
    ratio = pp_2025 / _train_pure_premium(line)
    assert 0.8 <= ratio <= 1.6, f"{line}: 2025 level is {ratio:.2f}x raw experience"


@pytest.mark.parametrize("line", LINES.values())
def test_predictions_scale_with_exposure(sub, test_rows, line):
    t = test_rows[test_rows.line == line].merge(sub, on="row_id")
    short, full = t[t.exposure < 0.25], t[t.exposure > 0.95]
    assert short.prediction.mean() < full.prediction.mean()


def test_motor_new_business_loaded(sub, test_rows):
    t = test_rows[test_rows.line == "motor"].merge(sub, on="row_id")
    sums = t.groupby("new_business")[["prediction", "exposure"]].sum()
    rate = sums.prediction / sums.exposure
    assert rate[True] > rate[False], "motor new business should cost more than renewals"


def test_no_extreme_outliers(sub):
    assert sub.prediction.max() < 100 * sub.prediction.median()


# ---------------------------------------------------------------- governance / reproducibility
def test_data_unchanged_since_run(manifest):
    """Submission must have been produced from the data currently in data/emerging."""
    for name, md5 in manifest["data_md5"].items():
        assert hashlib.md5((EM / name).read_bytes()).hexdigest() == md5, f"{name} changed since the run"


def test_leaky_feature_excluded(manifest):
    assert "customer_service_contacts" in manifest["leak_features_dropped"]


def test_blend_weights_valid(manifest):
    for line, w in manifest["blend_weights"].items():
        assert abs(sum(w.values()) - 1) < 1e-6 and min(w.values()) >= 0, line


def test_new_business_calibration_bounded(manifest):
    for line, f in manifest["nb_calibration"].items():
        assert all(0.8 <= v <= 1.25 for v in f.values()), line


def test_full_run_not_smoke(manifest):
    assert not manifest["smoke"], "submission was produced by a smoke (reduced) run"


def test_qa_log_records_known_issues():
    qa = pd.read_csv(OUT / "qa_log.csv")
    em = qa[qa.market == "emerging"]
    issues = " ".join(em.issue)
    for expected in ["exposure", "sum_insured", "construction_type", "occupation", "outside any policy period"]:
        assert expected in issues, f"QA log missing: {expected}"
