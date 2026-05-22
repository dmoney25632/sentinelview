"""Tests for sentinelview.ingestion.noaa_client."""

import pytest
from unittest.mock import MagicMock, patch

from sentinelview.ingestion.noaa_client import NOAAClient

_POINTS_RESPONSE = {
    "properties": {
        "forecastHourly": "https://api.weather.gov/gridpoints/OKX/32,34/forecast/hourly"
    }
}

_FORECAST_PERIOD = {
    "temperature": 72,
    "windSpeed": "10 mph",
    "shortForecast": "Mostly Cloudy",
    "startTime": "2023-10-01T14:00:00-04:00",
}

_FORECAST_RESPONSE = {
    "properties": {
        "periods": [_FORECAST_PERIOD, _FORECAST_PERIOD, _FORECAST_PERIOD, _FORECAST_PERIOD]
    }
}


@pytest.fixture
def client() -> NOAAClient:
    """NOAAClient instance (no credentials required)."""
    return NOAAClient()


class TestGetPointForecast:
    """Tests for NOAAClient.get_point_forecast."""

    def test_returns_three_periods_on_success(self, client: NOAAClient) -> None:
        """Successful response returns exactly 3 forecast period dicts."""
        points_resp = MagicMock()
        points_resp.status_code = 200
        points_resp.json.return_value = _POINTS_RESPONSE
        points_resp.raise_for_status.return_value = None

        forecast_resp = MagicMock()
        forecast_resp.json.return_value = _FORECAST_RESPONSE
        forecast_resp.raise_for_status.return_value = None

        with patch(
            "sentinelview.ingestion.noaa_client.requests.get",
            side_effect=[points_resp, forecast_resp],
        ):
            result = client.get_point_forecast(40.71, -74.01)

        assert result is not None
        assert len(result) == 3
        for period in result:
            assert "temperature" in period
            assert "windSpeed" in period
            assert "shortForecast" in period
            assert "startTime" in period

    def test_returns_none_on_404_non_us_location(self, client: NOAAClient) -> None:
        """Returns None gracefully for non-US coordinates (404 from points endpoint)."""
        points_resp = MagicMock()
        points_resp.status_code = 404

        with patch(
            "sentinelview.ingestion.noaa_client.requests.get",
            return_value=points_resp,
        ):
            result = client.get_point_forecast(51.5, -0.1)  # London

        assert result is None

    def test_returns_none_on_network_error(self, client: NOAAClient) -> None:
        """Returns None (not raised) on network error."""
        with patch(
            "sentinelview.ingestion.noaa_client.requests.get",
            side_effect=RuntimeError("network failure"),
        ):
            result = client.get_point_forecast(40.71, -74.01)

        assert result is None

    def test_returns_none_when_forecast_hourly_url_missing(
        self, client: NOAAClient
    ) -> None:
        """Returns None when the points response has no forecastHourly URL."""
        points_resp = MagicMock()
        points_resp.status_code = 200
        points_resp.json.return_value = {"properties": {}}
        points_resp.raise_for_status.return_value = None

        with patch(
            "sentinelview.ingestion.noaa_client.requests.get",
            return_value=points_resp,
        ):
            result = client.get_point_forecast(40.71, -74.01)

        assert result is None

    def test_returns_none_when_periods_empty(self, client: NOAAClient) -> None:
        """Returns None when the forecast has an empty periods list."""
        points_resp = MagicMock()
        points_resp.status_code = 200
        points_resp.json.return_value = _POINTS_RESPONSE
        points_resp.raise_for_status.return_value = None

        forecast_resp = MagicMock()
        forecast_resp.json.return_value = {"properties": {"periods": []}}
        forecast_resp.raise_for_status.return_value = None

        with patch(
            "sentinelview.ingestion.noaa_client.requests.get",
            side_effect=[points_resp, forecast_resp],
        ):
            result = client.get_point_forecast(40.71, -74.01)

        assert result is None
