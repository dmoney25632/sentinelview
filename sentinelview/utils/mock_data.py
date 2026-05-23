"""Mock data generators for offline development and testing.

Provides :func:`generate_mock_vessels` and :func:`generate_mock_flights`,
which return realistic DataFrames that match the schemas expected by the
SentinelView ingestion clients.  A handful of rows contain deliberately
anomalous values (large time gaps, impossible position jumps, etc.) so that
the detection pipeline has something to flag.
"""

import time

import numpy as np
import pandas as pd

_RNG_SEED = 42


def generate_mock_vessels(n: int = 50) -> pd.DataFrame:
    """Generate a realistic mock AIS vessel DataFrame.

    Produces *n* rows spread across global coordinates.  Approximately 5–10
    rows contain anomalous values designed to trigger the rule-based detector:
    large time gaps (> 7 200 s), impossible position jumps, and large
    speed-over-ground vs computed-speed discrepancies.

    Args:
        n: Total number of vessel rows to generate.  Must be ≥ 10.

    Returns:
        DataFrame with columns: ``mmsi``, ``timestamp``, ``latitude``,
        ``longitude``, ``sog``, ``cog``, ``heading``, ``navstat``,
        ``callsign``.
    """
    rng = np.random.default_rng(_RNG_SEED)
    now = int(time.time())

    mmsi_pool = [str(200_000_000 + i) for i in range(n)]

    # Normal pings – two consecutive readings per MMSI so the feature
    # engineer has consecutive pairs to work with.
    lats = rng.uniform(-60.0, 70.0, n).tolist()
    lons = rng.uniform(-170.0, 170.0, n).tolist()
    sogs = rng.uniform(0.0, 22.0, n).tolist()
    cogs = rng.uniform(0.0, 360.0, n).tolist()
    headings = rng.uniform(0.0, 360.0, n).tolist()
    navstats = rng.integers(0, 8, n).tolist()
    timestamps = [now - rng.integers(0, 3600) for _ in range(n)]

    callsign_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    callsigns = [
        "".join(rng.choice(list(callsign_chars), 6).tolist()) for _ in range(n)
    ]

    rows = []
    for i in range(n):
        rows.append(
            {
                "mmsi": mmsi_pool[i],
                "timestamp": timestamps[i],
                "latitude": lats[i],
                "longitude": lons[i],
                "sog": round(sogs[i], 1),
                "cog": round(cogs[i], 1),
                "heading": round(headings[i], 0),
                "navstat": int(navstats[i]),
                "callsign": callsigns[i],
            }
        )

    # ── Inject anomalous rows (duplicate MMSIs with suspicious values) ──
    anomalous_count = min(8, n // 6)
    for j in range(anomalous_count):
        base = rows[j]
        anomaly_type = j % 4

        if anomaly_type == 0:
            # Large time gap – dark vessel (> 7 200 s)
            anon = dict(base)
            anon["timestamp"] = base["timestamp"] - rng.integers(10_000, 30_000)
        elif anomaly_type == 1:
            # Impossible position jump – teleport ~2 000 km in < 5 min
            anon = dict(base)
            anon["latitude"] = base["latitude"] + rng.uniform(15.0, 20.0)
            anon["longitude"] = base["longitude"] + rng.uniform(15.0, 20.0)
            anon["timestamp"] = base["timestamp"] - 250
        elif anomaly_type == 2:
            # Speed spoofing – reported SOG far from computed speed
            anon = dict(base)
            anon["sog"] = round(float(base["sog"]) + rng.uniform(25.0, 40.0), 1)
            anon["timestamp"] = base["timestamp"] - 600
        else:
            # Sharp turn (heading delta > 30°)
            anon = dict(base)
            anon["heading"] = (float(base["heading"]) + 90.0) % 360.0
            anon["timestamp"] = base["timestamp"] - 120

        rows.append(anon)

    df = pd.DataFrame(rows)
    return df[
        ["mmsi", "timestamp", "latitude", "longitude", "sog", "cog", "heading", "navstat", "callsign"]
    ].reset_index(drop=True)


def generate_mock_flights(n: int = 50) -> pd.DataFrame:
    """Generate a realistic mock ADS-B flight DataFrame.

    Produces *n* rows spread across global coordinates.  Approximately 5–10
    rows contain anomalous values designed to trigger the rule-based detector:
    extreme descent rates and impossible position jumps within short time gaps.

    Args:
        n: Total number of flight rows to generate.  Must be ≥ 10.

    Returns:
        DataFrame with columns: ``icao24``, ``timestamp``, ``latitude``,
        ``longitude``, ``velocity``, ``true_track``, ``baro_altitude``,
        ``vertical_rate``.
    """
    rng = np.random.default_rng(_RNG_SEED + 1)
    now = int(time.time())

    hex_chars = "0123456789abcdef"
    icao_pool = [
        "".join(rng.choice(list(hex_chars), 6).tolist()) for _ in range(n)
    ]

    lats = rng.uniform(-60.0, 70.0, n).tolist()
    lons = rng.uniform(-170.0, 170.0, n).tolist()
    velocities = rng.uniform(100.0, 900.0, n).tolist()
    tracks = rng.uniform(0.0, 360.0, n).tolist()
    altitudes = rng.uniform(1000.0, 12_500.0, n).tolist()
    vertical_rates = rng.uniform(-15.0, 15.0, n).tolist()
    timestamps = [now - rng.integers(0, 3600) for _ in range(n)]

    rows = []
    for i in range(n):
        rows.append(
            {
                "icao24": icao_pool[i],
                "timestamp": timestamps[i],
                "latitude": lats[i],
                "longitude": lons[i],
                "velocity": round(velocities[i], 1),
                "true_track": round(tracks[i], 1),
                "baro_altitude": round(altitudes[i], 0),
                "vertical_rate": round(vertical_rates[i], 2),
            }
        )

    # ── Inject anomalous rows ────────────────────────────────────────────
    anomalous_count = min(8, n // 6)
    for j in range(anomalous_count):
        base = rows[j]
        anomaly_type = j % 2

        if anomaly_type == 0:
            # Extreme descent (> 3 000 fpm ≈ > 15.24 m/s negative)
            anon = dict(base)
            anon["vertical_rate"] = round(-rng.uniform(16.0, 25.0), 2)
            anon["timestamp"] = base["timestamp"] - 300
        else:
            # Impossible position jump (> 500 km in < 60 s)
            anon = dict(base)
            anon["latitude"] = base["latitude"] + rng.uniform(4.0, 6.0)
            anon["longitude"] = base["longitude"] + rng.uniform(4.0, 6.0)
            anon["timestamp"] = base["timestamp"] - 30

        rows.append(anon)

    df = pd.DataFrame(rows)
    return df[
        ["icao24", "timestamp", "latitude", "longitude", "velocity", "true_track", "baro_altitude", "vertical_rate"]
    ].reset_index(drop=True)
