"""Tests for sentinelview.processing.ml_detector."""

import tempfile
import os

import numpy as np
import pandas as pd
import pytest

from sentinelview.processing.ml_detector import (
    FLIGHT_FEATURES,
    VESSEL_FEATURES,
    MLAnomalyDetector,
)


def _make_vessel_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "computed_speed_kts": rng.uniform(0, 20, n),
            "heading_delta": rng.uniform(0, 30, n),
            "time_gap_s": rng.uniform(60, 600, n),
            "distance_km": rng.uniform(0, 50, n),
            "loitering": rng.integers(0, 2, n),
            "sog_vs_computed_delta": rng.uniform(0, 5, n),
        }
    )


def _make_flight_df(n: int = 100, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "computed_speed_kts": rng.uniform(200, 500, n),
            "heading_delta": rng.uniform(0, 15, n),
            "time_gap_s": rng.uniform(10, 60, n),
            "distance_km": rng.uniform(5, 200, n),
            "altitude_change_ft": rng.uniform(-500, 500, n),
            "descent_rate_fpm": rng.uniform(-1000, 1000, n),
        }
    )


# ── initialisation ────────────────────────────────────────────────────────────

def test_default_mode_is_vessel():
    det = MLAnomalyDetector()
    assert det.mode == "vessel"
    assert det._features == VESSEL_FEATURES


def test_flight_mode_uses_flight_features():
    det = MLAnomalyDetector(mode="flight")
    assert det._features == FLIGHT_FEATURES


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        MLAnomalyDetector(mode="submarine")


def test_not_fitted_by_default():
    assert MLAnomalyDetector().fitted is False


# ── fit / score ───────────────────────────────────────────────────────────────

def test_fit_returns_self():
    det = MLAnomalyDetector()
    df = _make_vessel_df()
    result = det.fit(df)
    assert result is det


def test_fit_sets_fitted_flag():
    det = MLAnomalyDetector()
    det.fit(_make_vessel_df())
    assert det.fitted is True


def test_score_raises_if_not_fitted():
    det = MLAnomalyDetector()
    with pytest.raises(RuntimeError):
        det.score(_make_vessel_df())


def test_score_adds_expected_columns():
    det = MLAnomalyDetector()
    df = _make_vessel_df()
    det.fit(df)
    out = det.score(df)
    assert "ml_raw_score" in out.columns
    assert "ml_anomaly_score" in out.columns
    assert "ml_flag" in out.columns


def test_ml_anomaly_score_in_range():
    det = MLAnomalyDetector()
    df = _make_vessel_df()
    det.fit(df)
    out = det.score(df)
    assert out["ml_anomaly_score"].between(0.0, 1.0).all()


def test_ml_flag_is_binary():
    det = MLAnomalyDetector()
    df = _make_vessel_df()
    det.fit(df)
    out = det.score(df)
    assert set(out["ml_flag"].unique()).issubset({0, 1})


def test_score_with_nan_does_not_raise():
    det = MLAnomalyDetector()
    df = _make_vessel_df()
    det.fit(df)
    df_nan = df.copy()
    df_nan.loc[0, "computed_speed_kts"] = float("nan")
    out = det.score(df_nan)
    assert len(out) == len(df_nan)


def test_flight_mode_score():
    det = MLAnomalyDetector(mode="flight")
    df = _make_flight_df()
    det.fit(df)
    out = det.score(df)
    assert "ml_anomaly_score" in out.columns


# ── save / load ───────────────────────────────────────────────────────────────

def test_save_load_roundtrip():
    det = MLAnomalyDetector(contamination=0.1)
    df = _make_vessel_df()
    det.fit(df)
    original_scores = det.score(df)["ml_anomaly_score"].values

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
        path = fh.name
    try:
        det.save(path)
        loaded = MLAnomalyDetector.load(path)
        assert loaded.fitted is True
        assert loaded.mode == "vessel"
        loaded_scores = loaded.score(df)["ml_anomaly_score"].values
        np.testing.assert_allclose(original_scores, loaded_scores)
    finally:
        os.unlink(path)
