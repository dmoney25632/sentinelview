"""ACLED conflict event data ingestion client."""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

_log = logging.getLogger(__name__)

_ACLED_API_URL = "https://acleddata.com/api/acled/read"

_EVENT_COLUMNS = [
    "event_id_cnty",
    "event_date",
    "event_type",
    "country",
    "latitude",
    "longitude",
    "fatalities",
    "notes",
]


class ACLEDClient:
    """Client for fetching conflict event data from ACLED."""

    def __init__(self) -> None:
        """Initialize the ACLED client.

        Credentials are read from the ACLED_EMAIL and ACLED_KEY environment
        variables.
        """
        self.email = os.getenv("ACLED_EMAIL", "")
        self.key = os.getenv("ACLED_KEY", "")

    def get_recent_events(
        self,
        days_back: int = 7,
        country: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch recent conflict events from ACLED.

        Args:
            days_back: Number of days into the past to query (default 7).
            country: Optional ISO country name filter.

        Returns:
            DataFrame with columns: event_id_cnty, event_date, event_type,
            country, latitude, longitude, fatalities, notes. Returns an empty
            DataFrame on error.
        """
        try:
            since_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime(
                "%Y-%m-%d"
            )
            params: dict = {
                "key": self.key,
                "email": self.email,
                "event_date": since_date,
                "event_date_where": ">=",
                "limit": 500,
            }
            if country is not None:
                params["country"] = country

            response = requests.get(_ACLED_API_URL, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()

            rows = payload.get("data", [])
            if not rows:
                return pd.DataFrame(columns=_EVENT_COLUMNS)

            df = pd.DataFrame(rows)
            for col in _EVENT_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df[_EVENT_COLUMNS]

        except Exception as exc:
            _log.warning("Failed to fetch events from ACLED: %s", exc)
            return pd.DataFrame(columns=_EVENT_COLUMNS)
