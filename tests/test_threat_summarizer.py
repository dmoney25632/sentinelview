"""Tests for sentinelview.intelligence.threat_summarizer."""

import numpy as np
import pandas as pd

from sentinelview.intelligence.threat_summarizer import MockThreatSummarizer


def _make_anomaly_df(n: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "mmsi": [str(100000000 + i) for i in range(n)],
            "latitude": rng.uniform(-90, 90, n),
            "longitude": rng.uniform(-180, 180, n),
            "threat_score": rng.uniform(0.0, 1.0, n),
            "ml_flag": rng.integers(0, 2, n),
            "loitering_flag": rng.integers(0, 2, n),
        }
    )


def test_mock_summarize_returns_nonempty_string():
    mock = MockThreatSummarizer()
    df = _make_anomaly_df()
    result = mock.summarize(df)
    assert isinstance(result, str)
    assert len(result) > 0


def test_mock_summarize_batch_returns_list_of_dicts():
    mock = MockThreatSummarizer()
    df = _make_anomaly_df(n=5)
    results = mock.summarize_batch(df, max_items=3)
    assert isinstance(results, list)
    assert len(results) == 3
    for item in results:
        assert "id" in item
        assert "threat_score" in item
        assert "summary" in item
        assert isinstance(item["summary"], str)
        assert len(item["summary"]) > 0
