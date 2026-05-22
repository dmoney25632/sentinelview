"""Feature engineering pipeline for raw ingested data."""

import math

import numpy as np
import pandas as pd

_EARTH_RADIUS_KM = 6371.0
_KNOTS_PER_KMH = 1.852


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance between two points in kilometres.

    Args:
        lat1: Latitude of the first point in decimal degrees.
        lon1: Longitude of the first point in decimal degrees.
        lat2: Latitude of the second point in decimal degrees.
        lon2: Longitude of the second point in decimal degrees.

    Returns:
        Distance in kilometres.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return _EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


def normalize_timestamps(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """Ensure a timestamp column is pandas datetime64 in UTC.

    Handles both Unix epoch integers (seconds since 1970-01-01) and ISO 8601
    strings.

    Args:
        df: Input DataFrame containing the column to normalise.
        col: Name of the column to normalise.  Defaults to ``"timestamp"``.

    Returns:
        DataFrame with the column replaced by timezone-aware UTC datetime64
        values.
    """
    series = df[col]
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        df = df.copy()
        df[col] = pd.to_datetime(series, unit="s", utc=True)
    else:
        df = df.copy()
        df[col] = pd.to_datetime(series, utc=True)
    return df


def _heading_delta(h1: pd.Series, h2: pd.Series) -> pd.Series:
    """Compute the minimum absolute angular difference between two heading series.

    The result is always in the range [0, 180].

    Args:
        h1: Previous heading values in degrees.
        h2: Current heading values in degrees.

    Returns:
        Absolute angular difference capped at 180 degrees.
    """
    diff = (h2 - h1) % 360
    return diff.where(diff <= 180, 360 - diff)


def engineer_vessel_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features for AIS vessel ping data.

    The function sorts by ``mmsi`` then ``timestamp``, groups by vessel, and
    computes inter-ping kinematics.  Rows where no previous ping exists (first
    ping per vessel) are dropped.

    Input columns expected: ``mmsi``, ``timestamp`` (datetime), ``latitude``,
    ``longitude``, ``sog``, ``cog``, ``heading``.

    New columns added:
        - ``prev_lat``, ``prev_lon``: position at previous ping.
        - ``prev_time``: timestamp of previous ping.
        - ``distance_km``: Haversine distance from previous ping in km.
        - ``time_gap_s``: seconds between consecutive pings.
        - ``computed_speed_kts``: speed inferred from position delta in knots.
        - ``heading_delta``: minimum absolute heading change (0-180 degrees).
        - ``sog_vs_computed_delta``: ``|sog - computed_speed_kts|``.
        - ``loitering``: 1 when ``computed_speed_kts < 1.5`` and
          ``time_gap_s > 300``, else 0.

    Args:
        df: Raw AIS DataFrame.

    Returns:
        Feature-enriched DataFrame with first-ping rows removed.
    """
    df = df.sort_values(["mmsi", "timestamp"]).copy()

    grp = df.groupby("mmsi", sort=False)
    df["prev_lat"] = grp["latitude"].shift(1)
    df["prev_lon"] = grp["longitude"].shift(1)
    df["prev_time"] = grp["timestamp"].shift(1)

    # Drop first-ping rows (no previous position)
    df = df.dropna(subset=["prev_lat"]).copy()

    df["distance_km"] = df.apply(
        lambda r: haversine(r["prev_lat"], r["prev_lon"], r["latitude"], r["longitude"]),
        axis=1,
    )

    df["time_gap_s"] = (df["timestamp"] - df["prev_time"]).dt.total_seconds()

    time_gap_hours = df["time_gap_s"] / 3600.0
    df["computed_speed_kts"] = (df["distance_km"] / time_gap_hours.replace(0, np.nan)) / _KNOTS_PER_KMH

    prev_heading = grp["heading"].shift(1).loc[df.index]
    df["heading_delta"] = _heading_delta(prev_heading, df["heading"])

    df["sog_vs_computed_delta"] = (df["sog"] - df["computed_speed_kts"]).abs()

    df["loitering"] = (
        (df["computed_speed_kts"] < 1.5) & (df["time_gap_s"] > 300)
    ).astype(int)

    return df.reset_index(drop=True)


def engineer_flight_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features for ADS-B flight ping data.

    The function sorts by ``icao24`` then ``timestamp``, groups by aircraft,
    and computes inter-ping kinematics.  Rows where no previous ping exists
    (first ping per aircraft) are dropped.

    Input columns expected: ``icao24``, ``timestamp`` (datetime),
    ``latitude``, ``longitude``, ``velocity``, ``true_track``,
    ``baro_altitude``, ``vertical_rate``.

    New columns added:
        - ``prev_lat``, ``prev_lon``: position at previous ping.
        - ``prev_time``: timestamp of previous ping.
        - ``distance_km``: Haversine distance from previous ping in km.
        - ``time_gap_s``: seconds between consecutive pings.
        - ``computed_speed_kts``: speed inferred from position delta in knots.
        - ``heading_delta``: minimum absolute heading change (0-180 degrees).
        - ``altitude_change_ft``: change in barometric altitude in feet.
        - ``descent_rate_fpm``: vertical rate converted to feet per minute.
        - ``squawk_7700``: placeholder column of zeros (int).

    Args:
        df: Raw ADS-B DataFrame.

    Returns:
        Feature-enriched DataFrame with first-ping rows removed.
    """
    _METRES_TO_FEET = 3.28084
    _MPS_TO_FPM = 196.850394

    df = df.sort_values(["icao24", "timestamp"]).copy()

    grp = df.groupby("icao24", sort=False)
    df["prev_lat"] = grp["latitude"].shift(1)
    df["prev_lon"] = grp["longitude"].shift(1)
    df["prev_time"] = grp["timestamp"].shift(1)
    prev_altitude = grp["baro_altitude"].shift(1)
    prev_track = grp["true_track"].shift(1)

    df = df.dropna(subset=["prev_lat"]).copy()

    prev_altitude = prev_altitude.loc[df.index]
    prev_track = prev_track.loc[df.index]

    df["distance_km"] = df.apply(
        lambda r: haversine(r["prev_lat"], r["prev_lon"], r["latitude"], r["longitude"]),
        axis=1,
    )

    df["time_gap_s"] = (df["timestamp"] - df["prev_time"]).dt.total_seconds()

    time_gap_hours = df["time_gap_s"] / 3600.0
    df["computed_speed_kts"] = (df["distance_km"] / time_gap_hours.replace(0, np.nan)) / _KNOTS_PER_KMH

    df["heading_delta"] = _heading_delta(prev_track, df["true_track"])

    df["altitude_change_ft"] = (df["baro_altitude"] - prev_altitude) * _METRES_TO_FEET

    df["descent_rate_fpm"] = df["vertical_rate"] * _MPS_TO_FPM

    df["squawk_7700"] = 0

    return df.reset_index(drop=True)
