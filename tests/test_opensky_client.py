"""Tests for sentinelview.ingestion.opensky_client."""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from sentinelview.ingestion.opensky_client import OpenSkyClient, bbox_from_region


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> OpenSkyClient:
    """Create an OpenSkyClient whose REST instance is fully mocked."""
    with patch("sentinelview.ingestion.opensky_client.REST") as mock_rest_cls:
        mock_rest_cls.return_value = MagicMock()
        client = OpenSkyClient()
    return client


# ---------------------------------------------------------------------------
# OpenSkyClient.get_live_states
# ---------------------------------------------------------------------------


class TestGetLiveStates:
    """Tests for OpenSkyClient.get_live_states."""

    def test_returns_empty_dataframe_when_api_returns_none(
        self, mock_client: OpenSkyClient
    ) -> None:
        """Empty DataFrame with correct columns when states() returns None."""
        mock_client.rest.states.return_value = None

        result = mock_client.get_live_states()

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == [
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

    def test_returns_empty_dataframe_when_api_returns_empty_df(
        self, mock_client: OpenSkyClient
    ) -> None:
        """Empty DataFrame with correct columns when states() returns empty DataFrame."""
        mock_client.rest.states.return_value = pd.DataFrame()

        result = mock_client.get_live_states()

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_empty_dataframe_on_exception(
        self, mock_client: OpenSkyClient
    ) -> None:
        """Empty DataFrame returned (not raised) on network error."""
        mock_client.rest.states.side_effect = RuntimeError("network error")

        result = mock_client.get_live_states()

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_passes_converted_bounds_to_rest(
        self, mock_client: OpenSkyClient
    ) -> None:
        """bbox (min_lat, max_lat, min_lon, max_lon) is converted to pyopensky bounds."""
        mock_client.rest.states.return_value = None

        mock_client.get_live_states(bbox=(23.0, 30.5, 48.0, 60.0))

        mock_client.rest.states.assert_called_once_with(
            bounds=(48.0, 23.0, 60.0, 30.5)
        )


# ---------------------------------------------------------------------------
# bbox_from_region
# ---------------------------------------------------------------------------


class TestBboxFromRegion:
    """Tests for the bbox_from_region helper function."""

    @pytest.mark.parametrize(
        "region",
        [
            "persian_gulf",
            "south_china_sea",
            "strait_of_hormuz",
            "black_sea",
            "red_sea",
        ],
    )
    def test_valid_regions_return_four_element_tuple(self, region: str) -> None:
        """All supported regions return a (min_lat, max_lat, min_lon, max_lon) tuple."""
        result = bbox_from_region(region)

        assert isinstance(result, tuple)
        assert len(result) == 4

        min_lat, max_lat, min_lon, max_lon = result
        assert min_lat < max_lat, "min_lat must be less than max_lat"
        assert min_lon < max_lon, "min_lon must be less than max_lon"

    def test_invalid_region_raises_value_error(self) -> None:
        """Unknown region names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown region"):
            bbox_from_region("narnia")

    def test_invalid_region_error_mentions_name(self) -> None:
        """ValueError message includes the offending region name."""
        bad_region = "atlantis"
        with pytest.raises(ValueError, match=bad_region):
            bbox_from_region(bad_region)
