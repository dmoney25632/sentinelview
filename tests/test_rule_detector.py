"""Tests for sentinelview.processing.rule_detector."""

import pandas as pd
import pytest

from sentinelview.processing.rule_detector import (
    apply_flight_rules,
    apply_vessel_rules,
    ensemble_score,
)


def _vessel_row(**kwargs) -> pd.DataFrame:
    """Build a minimal single-row vessel DataFrame with sensible defaults."""
    defaults = {
        "time_gap_s": 100.0,
        "distance_km": 1.0,
        "heading_delta": 5.0,
        "sog_vs_computed_delta": 0.5,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


def _flight_row(**kwargs) -> pd.DataFrame:
    """Build a minimal single-row flight DataFrame with sensible defaults."""
    defaults = {
        "descent_rate_fpm": 0.0,
        "distance_km": 10.0,
        "time_gap_s": 120.0,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ── vessel rules ──────────────────────────────────────────────────────────────

def test_dark_vessel_flag_fires():
    df = apply_vessel_rules(_vessel_row(time_gap_s=8000))
    assert df["dark_vessel_flag"].iloc[0] == 1


def test_dark_vessel_flag_no_fire():
    df = apply_vessel_rules(_vessel_row(time_gap_s=3600))
    assert df["dark_vessel_flag"].iloc[0] == 0


def test_vessel_jump_flag_fires():
    df = apply_vessel_rules(_vessel_row(distance_km=200, time_gap_s=60))
    assert df["jump_flag"].iloc[0] == 1


def test_vessel_jump_flag_no_fire_slow():
    # distance OK but time_gap too large
    df = apply_vessel_rules(_vessel_row(distance_km=200, time_gap_s=400))
    assert df["jump_flag"].iloc[0] == 0


def test_vessel_sharp_turn_flag_fires():
    df = apply_vessel_rules(_vessel_row(heading_delta=45))
    assert df["sharp_turn_flag"].iloc[0] == 1


def test_vessel_speed_spoof_flag_fires():
    df = apply_vessel_rules(_vessel_row(sog_vs_computed_delta=15))
    assert df["speed_spoof_flag"].iloc[0] == 1


def test_vessel_rule_score_range():
    df = apply_vessel_rules(_vessel_row(time_gap_s=8000, distance_km=200))
    assert 0.0 <= df["rule_score"].iloc[0] <= 1.0


# ── flight rules ──────────────────────────────────────────────────────────────

def test_extreme_descent_flag_fires():
    df = apply_flight_rules(_flight_row(descent_rate_fpm=-4000))
    assert df["extreme_descent_flag"].iloc[0] == 1


def test_extreme_descent_flag_no_fire():
    df = apply_flight_rules(_flight_row(descent_rate_fpm=-1000))
    assert df["extreme_descent_flag"].iloc[0] == 0


def test_flight_jump_flag_fires():
    df = apply_flight_rules(_flight_row(distance_km=600, time_gap_s=30))
    assert df["jump_flag"].iloc[0] == 1


def test_flight_rule_score_range():
    df = apply_flight_rules(_flight_row(descent_rate_fpm=-4000, distance_km=600, time_gap_s=30))
    assert 0.0 <= df["rule_score"].iloc[0] <= 1.0


# ── ensemble_score ────────────────────────────────────────────────────────────

def test_ensemble_score_threat_in_range():
    df = pd.DataFrame([{"ml_anomaly_score": 0.5, "rule_score": 0.5}])
    result = ensemble_score(df)
    assert 0.0 <= result["threat_score"].iloc[0] <= 1.0


def test_ensemble_score_is_anomalous_when_high():
    # threat_score = 0.6*0.9 + 0.4*0.9 = 0.9 > 0.6
    df = pd.DataFrame([{"ml_anomaly_score": 0.9, "rule_score": 0.9}])
    result = ensemble_score(df)
    assert result["is_anomalous"].iloc[0] == 1


def test_ensemble_score_not_anomalous_when_low():
    # threat_score = 0.6*0.1 + 0.4*0.1 = 0.1 <= 0.6
    df = pd.DataFrame([{"ml_anomaly_score": 0.1, "rule_score": 0.1}])
    result = ensemble_score(df)
    assert result["is_anomalous"].iloc[0] == 0


def test_ensemble_score_custom_weights():
    df = pd.DataFrame([{"ml_anomaly_score": 0.8, "rule_score": 0.0}])
    result = ensemble_score(df, ml_weight=1.0, rule_weight=0.0)
    assert pytest.approx(result["threat_score"].iloc[0]) == 0.8
