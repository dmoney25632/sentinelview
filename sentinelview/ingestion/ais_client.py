"""AISHub AIS maritime vessel data ingestion client."""

import logging
import os
from typing import Optional

import pandas as pd
import requests

_log = logging.getLogger(__name__)

_AIS_API_URL = "https://data.aishub.net/ws.php"

_VESSEL_COLUMNS = [
    "mmsi",
    "name",
    "latitude",
    "longitude",
    "sog",
    "cog",
    "heading",
    "navstat",
    "callsign",
    "destination",
    "imo",
]

# Mapping from AISHub JSON keys to our normalised lowercase column names.
_KEY_MAP = {
    "MMSI": "mmsi",
    "NAME": "name",
    "LATITUDE": "latitude",
    "LONGITUDE": "longitude",
    "SOG": "sog",
    "COG": "cog",
    "HEADING": "heading",
    "NAVSTAT": "navstat",
    "CALLSIGN": "callsign",
    "DESTINATION": "destination",
    "IMO": "imo",
}


class AISHubClient:
    """Client for fetching live AIS maritime vessel data from AISHub."""

    def __init__(self) -> None:
        """Initialize the AISHub client.

        Credentials are read from the AISHUB_USERNAME environment variable.
        """
        self.username = os.getenv("AISHUB_USERNAME", "")

    def get_vessels(
        self,
        mmsi: Optional[str] = None,
        bbox: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """Fetch current AIS vessel positions from AISHub.

        Args:
            mmsi: Optional MMSI filter (single vessel).
            bbox: Optional bounding box as (min_lat, max_lat, min_lon, max_lon).

        Returns:
            DataFrame with columns: mmsi, name, latitude, longitude, sog, cog,
            heading, navstat, callsign, destination, imo. Returns an empty
            DataFrame on error.
        """
        try:
            params: dict = {
                "username": self.username,
                "format": 1,
                "output": "json",
                "compress": 0,
            }
            if mmsi is not None:
                params["mmsi"] = mmsi
            if bbox is not None:
                min_lat, max_lat, min_lon, max_lon = bbox
                params["latmin"] = min_lat
                params["latmax"] = max_lat
                params["lonmin"] = min_lon
                params["lonmax"] = max_lon

            response = requests.get(_AIS_API_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            # AISHub returns [header_dict, [vessel_list]]
            if not isinstance(data, list) or len(data) < 2:
                return pd.DataFrame(columns=_VESSEL_COLUMNS)

            vessel_list = data[1]
            if not vessel_list:
                return pd.DataFrame(columns=_VESSEL_COLUMNS)

            records = [
                {_KEY_MAP[k]: v for k, v in vessel.items() if k in _KEY_MAP}
                for vessel in vessel_list
            ]
            df = pd.DataFrame(records)
            for col in _VESSEL_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df[_VESSEL_COLUMNS]

        except Exception as exc:
            _log.warning("Failed to fetch vessels from AISHub: %s", exc)
            return pd.DataFrame(columns=_VESSEL_COLUMNS)
