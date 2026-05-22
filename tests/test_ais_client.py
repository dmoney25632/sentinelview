"""Tests for sentinelview.ingestion.ais_client."""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from sentinelview.ingestion.ais_client import AISHubClient

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

_SAMPLE_RESPONSE = [
    {"ID": 1},  # header dict
    [
        {
            "MMSI": "123456789",
            "NAME": "TEST VESSEL",
            "LATITUDE": 25.0,
            "LONGITUDE": 55.0,
            "SOG": 10.5,
            "COG": 270.0,
            "HEADING": 268,
            "NAVSTAT": 0,
            "CALLSIGN": "ABCD1",
            "DESTINATION": "DUBAI",
            "IMO": 9000001,
        }
    ],
]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> AISHubClient:
    """AISHubClient with AISHUB_USERNAME set."""
    monkeypatch.setenv("AISHUB_USERNAME", "testuser")
    return AISHubClient()


class TestGetVessels:
    """Tests for AISHubClient.get_vessels."""

    def test_returns_dataframe_with_correct_columns_on_success(
        self, client: AISHubClient
    ) -> None:
        """Successful response returns DataFrame with expected columns."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = _SAMPLE_RESPONSE
        mock_resp.raise_for_status.return_value = None

        with patch("sentinelview.ingestion.ais_client.requests.get", return_value=mock_resp):
            result = client.get_vessels()

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == _VESSEL_COLUMNS
        assert len(result) == 1
        assert result.iloc[0]["mmsi"] == "123456789"

    def test_returns_empty_dataframe_on_http_error(
        self, client: AISHubClient
    ) -> None:
        """Empty DataFrame returned (not raised) on HTTP error."""
        with patch(
            "sentinelview.ingestion.ais_client.requests.get",
            side_effect=RuntimeError("connection refused"),
        ):
            result = client.get_vessels()

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == _VESSEL_COLUMNS

    def test_returns_empty_dataframe_when_vessel_list_empty(
        self, client: AISHubClient
    ) -> None:
        """Empty DataFrame returned when API returns an empty vessel list."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"ID": 1}, []]
        mock_resp.raise_for_status.return_value = None

        with patch("sentinelview.ingestion.ais_client.requests.get", return_value=mock_resp):
            result = client.get_vessels()

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == _VESSEL_COLUMNS

    def test_returns_empty_dataframe_on_malformed_response(
        self, client: AISHubClient
    ) -> None:
        """Empty DataFrame returned when response format is unexpected."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "bad request"}
        mock_resp.raise_for_status.return_value = None

        with patch("sentinelview.ingestion.ais_client.requests.get", return_value=mock_resp):
            result = client.get_vessels()

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_passes_mmsi_param(self, client: AISHubClient) -> None:
        """MMSI filter is forwarded to the HTTP request."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"ID": 1}, []]
        mock_resp.raise_for_status.return_value = None

        with patch(
            "sentinelview.ingestion.ais_client.requests.get", return_value=mock_resp
        ) as mock_get:
            client.get_vessels(mmsi="123456789")

        call_kwargs = mock_get.call_args
        params = call_kwargs[1]["params"] if "params" in call_kwargs[1] else call_kwargs[0][1]
        assert params.get("mmsi") == "123456789"

    def test_passes_bbox_params(self, client: AISHubClient) -> None:
        """Bounding box is forwarded as latmin/latmax/lonmin/lonmax params."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"ID": 1}, []]
        mock_resp.raise_for_status.return_value = None

        with patch(
            "sentinelview.ingestion.ais_client.requests.get", return_value=mock_resp
        ) as mock_get:
            client.get_vessels(bbox=(23.0, 30.5, 48.0, 60.0))

        params = mock_get.call_args[1]["params"]
        assert params["latmin"] == 23.0
        assert params["latmax"] == 30.5
        assert params["lonmin"] == 48.0
        assert params["lonmax"] == 60.0
