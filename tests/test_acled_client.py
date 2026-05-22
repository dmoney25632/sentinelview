"""Tests for sentinelview.ingestion.acled_client."""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from sentinelview.ingestion.acled_client import ACLEDClient

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

_SAMPLE_EVENT = {
    "event_id_cnty": "IRQ2023-001",
    "event_date": "2023-10-01",
    "event_type": "Explosion/Remote violence",
    "country": "Iraq",
    "latitude": "33.34",
    "longitude": "44.40",
    "fatalities": "2",
    "notes": "IED detonated near checkpoint.",
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> ACLEDClient:
    """ACLEDClient with credentials set via environment variables."""
    monkeypatch.setenv("ACLED_EMAIL", "test@example.com")
    monkeypatch.setenv("ACLED_KEY", "testkey123")
    return ACLEDClient()


class TestGetRecentEvents:
    """Tests for ACLEDClient.get_recent_events."""

    def test_returns_dataframe_with_correct_columns_on_success(
        self, client: ACLEDClient
    ) -> None:
        """Successful response returns DataFrame with expected columns."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [_SAMPLE_EVENT]}
        mock_resp.raise_for_status.return_value = None

        with patch("sentinelview.ingestion.acled_client.requests.get", return_value=mock_resp):
            result = client.get_recent_events()

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == _EVENT_COLUMNS
        assert len(result) == 1
        assert result.iloc[0]["event_id_cnty"] == "IRQ2023-001"

    def test_returns_empty_dataframe_on_exception(
        self, client: ACLEDClient
    ) -> None:
        """Empty DataFrame returned (not raised) on network error."""
        with patch(
            "sentinelview.ingestion.acled_client.requests.get",
            side_effect=RuntimeError("timeout"),
        ):
            result = client.get_recent_events()

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == _EVENT_COLUMNS

    def test_returns_empty_dataframe_when_data_is_empty_list(
        self, client: ACLEDClient
    ) -> None:
        """Empty DataFrame returned when API returns no events."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status.return_value = None

        with patch("sentinelview.ingestion.acled_client.requests.get", return_value=mock_resp):
            result = client.get_recent_events()

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_empty_dataframe_on_missing_data_key(
        self, client: ACLEDClient
    ) -> None:
        """Empty DataFrame returned when response lacks a 'data' key."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status.return_value = None

        with patch("sentinelview.ingestion.acled_client.requests.get", return_value=mock_resp):
            result = client.get_recent_events()

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_country_filter_passed_in_params(self, client: ACLEDClient) -> None:
        """Country filter is forwarded to the HTTP request."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status.return_value = None

        with patch(
            "sentinelview.ingestion.acled_client.requests.get", return_value=mock_resp
        ) as mock_get:
            client.get_recent_events(country="Iraq")

        params = mock_get.call_args[1]["params"]
        assert params.get("country") == "Iraq"

    def test_days_back_affects_event_date_param(self, client: ACLEDClient) -> None:
        """event_date param reflects the days_back argument."""
        from datetime import datetime, timedelta, timezone

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status.return_value = None

        expected_date = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")

        with patch(
            "sentinelview.ingestion.acled_client.requests.get", return_value=mock_resp
        ) as mock_get:
            client.get_recent_events(days_back=14)

        params = mock_get.call_args[1]["params"]
        assert params.get("event_date") == expected_date
