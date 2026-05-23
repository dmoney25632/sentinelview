"""OpenSky Network ADS-B flight data ingestion client."""

import logging
import os
from typing import Optional

import pandas as pd
from pyopensky.rest import REST

from sentinelview.utils.mock_data import generate_mock_flights

_log = logging.getLogger(__name__)

# Bounding boxes as (min_lat, max_lat, min_lon, max_lon)
_BBOX_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "persian_gulf": (23.0, 30.5, 48.0, 60.0),
    "south_china_sea": (0.0, 25.0, 99.0, 122.0),
    "strait_of_hormuz": (23.5, 27.5, 55.0, 60.0),
    "black_sea": (40.0, 47.0, 27.0, 42.0),
    "red_sea": (11.0, 30.0, 31.0, 44.0),
}

_LIVE_STATES_COLUMNS = [
    "icao24",
    "callsign",
    "origin_country",
    "longitude",
    "latitude",
    "baro_altitude",
    "velocity",
    "true_track",
    "vertical_rate",
    "on_ground",
    "timestamp",
]

_TRAJECTORY_COLUMNS = [
    "time",
    "latitude",
    "longitude",
    "baro_altitude",
    "true_track",
    "on_ground",
]


def bbox_from_region(region: str) -> tuple[float, float, float, float]:
    """Return a bounding box tuple for a named geopolitical region.

    Args:
        region: One of "persian_gulf", "south_china_sea", "strait_of_hormuz",
            "black_sea", or "red_sea".

    Returns:
        Tuple of (min_lat, max_lat, min_lon, max_lon).

    Raises:
        ValueError: If the region name is not recognised.
    """
    if region not in _BBOX_REGIONS:
        raise ValueError(
            f"Unknown region '{region}'. Valid regions: {sorted(_BBOX_REGIONS)}"
        )
    return _BBOX_REGIONS[region]


class OpenSkyClient:
    """Client for fetching live ADS-B flight data from the OpenSky Network REST API."""

    def __init__(self) -> None:
        """Initialize the OpenSky REST client.

        Credentials are read from OPENSKY_USERNAME and OPENSKY_PASSWORD
        environment variables (or from the pyopensky config file when present).
        """
        self.rest = REST()

    def get_live_states(self, bbox: Optional[tuple] = None) -> pd.DataFrame:
        """Fetch current ADS-B state vectors from OpenSky.

        Args:
            bbox: Optional bounding box as (min_lat, max_lat, min_lon, max_lon).

        Returns:
            DataFrame with columns: icao24, callsign, origin_country, longitude,
            latitude, baro_altitude, velocity, true_track, vertical_rate,
            on_ground, timestamp. Returns an empty DataFrame when the API
            returns no data.
        """
        if os.getenv("SENTINELVIEW_MOCK", "").lower() == "true":
            _log.info("SENTINELVIEW_MOCK=true — returning mock flight data.")
            return generate_mock_flights()

        try:
            bounds = None
            if bbox is not None:
                min_lat, max_lat, min_lon, max_lon = bbox
                # pyopensky expects (west, south, east, north)
                bounds = (min_lon, min_lat, max_lon, max_lat)

            result = self.rest.states(bounds=bounds)

            if result is None or result.empty:
                return pd.DataFrame(columns=_LIVE_STATES_COLUMNS)

            return result.rename(
                columns={
                    "altitude": "baro_altitude",
                    "onground": "on_ground",
                    "groundspeed": "velocity",
                    "track": "true_track",
                }
            )[_LIVE_STATES_COLUMNS]

        except Exception as exc:
            _log.warning("Failed to fetch live states from OpenSky: %s", exc)
            return pd.DataFrame(columns=_LIVE_STATES_COLUMNS)

    def get_trajectory(self, icao24: str) -> pd.DataFrame:
        """Fetch historical track waypoints for a single aircraft.

        Args:
            icao24: 24-bit ICAO address of the aircraft (hex string).

        Returns:
            DataFrame with columns: time, latitude, longitude, baro_altitude,
            true_track, on_ground. Returns an empty DataFrame on error.
        """
        try:
            result = self.rest.tracks(icao24=icao24)

            if result is None or result.empty:
                return pd.DataFrame(columns=_TRAJECTORY_COLUMNS)

            return result.rename(
                columns={
                    "timestamp": "time",
                    "altitude": "baro_altitude",
                    "track": "true_track",
                    "onground": "on_ground",
                }
            )[_TRAJECTORY_COLUMNS]

        except Exception as exc:
            _log.warning(
                "Failed to fetch trajectory for %s from OpenSky: %s", icao24, exc
            )
            return pd.DataFrame(columns=_TRAJECTORY_COLUMNS)
