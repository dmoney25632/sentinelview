"""Tests for sentinelview.processing.feature_engineer."""

import pandas as pd
import pytest

from sentinelview.processing.feature_engineer import (
    engineer_vessel_features,
    haversine,
    normalize_timestamps,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def vessel_df() -> pd.DataFrame:
    """Three vessels with three pings each.

    Vessel A: slow, large time gap → loitering = 1
    Vessel B: normal speed
    Vessel C: heading wrap test (350 → 10)
    """
    rows = [
        # mmsi, timestamp (UTC), lat, lon, sog, cog, heading
        # Vessel A – stationary / very slow
        ("A", pd.Timestamp("2024-01-01 00:00:00", tz="UTC"), 0.0, 0.0, 0.5, 0, 0),
        ("A", pd.Timestamp("2024-01-01 00:10:00", tz="UTC"), 0.001, 0.001, 0.5, 0, 0),
        ("A", pd.Timestamp("2024-01-01 00:20:00", tz="UTC"), 0.002, 0.002, 0.5, 0, 0),
        # Vessel B – cruising
        ("B", pd.Timestamp("2024-01-01 00:00:00", tz="UTC"), 10.0, 20.0, 12.0, 90, 90),
        ("B", pd.Timestamp("2024-01-01 01:00:00", tz="UTC"), 10.0, 20.2, 12.0, 90, 90),
        ("B", pd.Timestamp("2024-01-01 02:00:00", tz="UTC"), 10.0, 20.4, 12.0, 90, 90),
        # Vessel C – heading wrap (350 → 10)
        ("C", pd.Timestamp("2024-01-01 00:00:00", tz="UTC"), 5.0, 5.0, 5.0, 350, 350),
        ("C", pd.Timestamp("2024-01-01 01:00:00", tz="UTC"), 5.1, 5.1, 5.0, 10, 10),
        ("C", pd.Timestamp("2024-01-01 02:00:00", tz="UTC"), 5.2, 5.2, 5.0, 30, 30),
    ]
    return pd.DataFrame(rows, columns=["mmsi", "timestamp", "latitude", "longitude", "sog", "cog", "heading"])


# ---------------------------------------------------------------------------
# haversine
# ---------------------------------------------------------------------------

def test_haversine_one_degree_longitude() -> None:
    """One degree of longitude on the equator should be ≈111.19 km."""
    result = haversine(0.0, 0.0, 0.0, 1.0)
    assert abs(result - 111.19) < 0.02


def test_haversine_zero_distance() -> None:
    assert haversine(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0)


def test_haversine_symmetric() -> None:
    d1 = haversine(48.8566, 2.3522, 51.5074, -0.1278)
    d2 = haversine(51.5074, -0.1278, 48.8566, 2.3522)
    assert d1 == pytest.approx(d2)


# ---------------------------------------------------------------------------
# engineer_vessel_features – column count
# ---------------------------------------------------------------------------

def test_engineer_vessel_features_column_count(vessel_df: pd.DataFrame) -> None:
    """Output should contain all original columns plus the 8 new feature columns."""
    original_cols = set(vessel_df.columns)
    new_feature_cols = {
        "prev_lat", "prev_lon", "prev_time",
        "distance_km", "time_gap_s", "computed_speed_kts",
        "heading_delta", "sog_vs_computed_delta", "loitering",
    }
    result = engineer_vessel_features(vessel_df)
    assert set(result.columns) >= original_cols | new_feature_cols


def test_engineer_vessel_features_row_count(vessel_df: pd.DataFrame) -> None:
    """First ping per vessel is dropped, so 9 pings → 6 rows."""
    result = engineer_vessel_features(vessel_df)
    assert len(result) == 6


# ---------------------------------------------------------------------------
# loitering flag
# ---------------------------------------------------------------------------

def test_loitering_flag_vessel_a(vessel_df: pd.DataFrame) -> None:
    """Vessel A is stationary (<1.5 kts) with time_gap_s > 300 → loitering = 1."""
    result = engineer_vessel_features(vessel_df)
    vessel_a = result[result["mmsi"] == "A"]
    assert (vessel_a["loitering"] == 1).all()


def test_loitering_flag_vessel_b(vessel_df: pd.DataFrame) -> None:
    """Vessel B is cruising → loitering = 0."""
    result = engineer_vessel_features(vessel_df)
    vessel_b = result[result["mmsi"] == "B"]
    assert (vessel_b["loitering"] == 0).all()


# ---------------------------------------------------------------------------
# heading_delta wrap
# ---------------------------------------------------------------------------

def test_heading_delta_wrap(vessel_df: pd.DataFrame) -> None:
    """Heading change from 350° to 10° should be 20°, not 340°."""
    result = engineer_vessel_features(vessel_df)
    # First remaining ping for vessel C is the transition 350→10
    vessel_c = result[result["mmsi"] == "C"].reset_index(drop=True)
    assert vessel_c.loc[0, "heading_delta"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# normalize_timestamps
# ---------------------------------------------------------------------------

def test_normalize_timestamps_integer() -> None:
    """Unix epoch integers should be converted to UTC datetime64."""
    df = pd.DataFrame({"timestamp": [0, 3600]})
    out = normalize_timestamps(df)
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])
    assert str(out["timestamp"].dt.tz) == "UTC"
    assert out["timestamp"].iloc[0] == pd.Timestamp("1970-01-01 00:00:00", tz="UTC")


def test_normalize_timestamps_string() -> None:
    """ISO 8601 strings should be converted to UTC datetime64."""
    df = pd.DataFrame({"timestamp": ["2024-06-01T12:00:00Z", "2024-06-01T13:00:00+00:00"]})
    out = normalize_timestamps(df)
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])
    assert str(out["timestamp"].dt.tz) == "UTC"


def test_normalize_timestamps_custom_col() -> None:
    """Custom column name should be respected."""
    df = pd.DataFrame({"ts": [0]})
    out = normalize_timestamps(df, col="ts")
    assert pd.api.types.is_datetime64_any_dtype(out["ts"])
