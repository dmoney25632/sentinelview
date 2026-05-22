"""NOAA weather context data ingestion client."""

import logging
from typing import Optional

import requests

_log = logging.getLogger(__name__)

_NOAA_BASE_URL = "https://api.weather.gov"


class NOAAClient:
    """Client for fetching weather context from the NOAA Weather API."""

    def __init__(self) -> None:
        """Initialize the NOAA client. No credentials are required."""
        self.base_url = _NOAA_BASE_URL

    def get_point_forecast(
        self, lat: float, lon: float
    ) -> Optional[list[dict]]:
        """Fetch hourly weather forecast for the next 3 hours at a given location.

        Args:
            lat: Latitude of the point (US coordinates only).
            lon: Longitude of the point (US coordinates only).

        Returns:
            List of dicts with keys: temperature, windSpeed, shortForecast,
            startTime for the next 3 forecast periods. Returns None on error
            or for non-US coordinates.
        """
        try:
            headers = {"User-Agent": "SentinelView/1.0"}
            points_url = f"{self.base_url}/points/{lat},{lon}"
            points_resp = requests.get(points_url, headers=headers, timeout=10)

            if points_resp.status_code == 404:
                _log.info(
                    "NOAA does not cover coordinates (%s, %s) — non-US location.",
                    lat,
                    lon,
                )
                return None

            points_resp.raise_for_status()
            forecast_hourly_url = (
                points_resp.json()
                .get("properties", {})
                .get("forecastHourly")
            )
            if not forecast_hourly_url:
                return None

            forecast_resp = requests.get(
                forecast_hourly_url, headers=headers, timeout=10
            )
            forecast_resp.raise_for_status()

            periods = (
                forecast_resp.json()
                .get("properties", {})
                .get("periods", [])
            )
            result = [
                {
                    "temperature": p.get("temperature"),
                    "windSpeed": p.get("windSpeed"),
                    "shortForecast": p.get("shortForecast"),
                    "startTime": p.get("startTime"),
                }
                for p in periods[:3]
            ]
            return result if result else None

        except Exception as exc:
            _log.warning("Failed to fetch forecast from NOAA: %s", exc)
            return None
