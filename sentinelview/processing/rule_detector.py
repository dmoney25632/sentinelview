"""Rule-based anomaly detection for geospatial OSINT signals."""

import pandas as pd


def apply_vessel_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rule-based anomaly flags to AIS vessel data.

    Adds the following integer flag columns and a normalised score:

    - ``dark_vessel_flag``: ``1`` when ``time_gap_s > 7200`` (>2 h gap).
    - ``jump_flag``: ``1`` when ``distance_km > 100`` AND
      ``time_gap_s < 300``.
    - ``sharp_turn_flag``: ``1`` when ``heading_delta > 30``.
    - ``speed_spoof_flag``: ``1`` when ``sog_vs_computed_delta > 10``.
    - ``rule_score``: float, sum of the four flags divided by 4
      (range ``[0.0, 1.0]``).

    Args:
        df: Feature-enriched AIS DataFrame (output of
            :func:`~sentinelview.processing.feature_engineer.engineer_vessel_features`).

    Returns:
        Copy of *df* with the new flag and score columns appended.
    """
    out = df.copy()
    out["dark_vessel_flag"] = (out["time_gap_s"] > 7200).astype(int)
    out["jump_flag"] = ((out["distance_km"] > 100) & (out["time_gap_s"] < 300)).astype(int)
    out["sharp_turn_flag"] = (out["heading_delta"] > 30).astype(int)
    out["speed_spoof_flag"] = (out["sog_vs_computed_delta"] > 10).astype(int)
    out["rule_score"] = (
        out["dark_vessel_flag"]
        + out["jump_flag"]
        + out["sharp_turn_flag"]
        + out["speed_spoof_flag"]
    ) / 4.0
    return out


def apply_flight_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rule-based anomaly flags to ADS-B flight data.

    Adds the following integer flag columns and a normalised score:

    - ``extreme_descent_flag``: ``1`` when ``descent_rate_fpm < -3000``.
    - ``jump_flag``: ``1`` when ``distance_km > 500`` AND
      ``time_gap_s < 60``.
    - ``rule_score``: float, sum of the two flags divided by 2
      (range ``[0.0, 1.0]``).

    Args:
        df: Feature-enriched ADS-B DataFrame (output of
            :func:`~sentinelview.processing.feature_engineer.engineer_flight_features`).

    Returns:
        Copy of *df* with the new flag and score columns appended.
    """
    out = df.copy()
    out["extreme_descent_flag"] = (out["descent_rate_fpm"] < -3000).astype(int)
    out["jump_flag"] = ((out["distance_km"] > 500) & (out["time_gap_s"] < 60)).astype(int)
    out["rule_score"] = (out["extreme_descent_flag"] + out["jump_flag"]) / 2.0
    return out


def ensemble_score(
    df: pd.DataFrame,
    ml_weight: float = 0.6,
    rule_weight: float = 0.4,
) -> pd.DataFrame:
    """Combine ML and rule scores into a single threat score.

    The function expects ``ml_anomaly_score`` and ``rule_score`` columns to
    already be present in *df* (added by :class:`~sentinelview.processing.ml_detector.MLAnomalyDetector`
    and one of the ``apply_*_rules`` functions respectively).

    New columns added:

    - ``threat_score``: weighted combination
      ``ml_weight * ml_anomaly_score + rule_weight * rule_score``.
    - ``is_anomalous``: ``1`` when ``threat_score > 0.6``, else ``0``.

    Args:
        df: DataFrame with ``ml_anomaly_score`` and ``rule_score`` columns.
        ml_weight: Weight applied to the ML anomaly score.
        rule_weight: Weight applied to the rule-based score.

    Returns:
        Copy of *df* with ``threat_score`` and ``is_anomalous`` appended.
    """
    out = df.copy()
    out["threat_score"] = ml_weight * out["ml_anomaly_score"] + rule_weight * out["rule_score"]
    out["is_anomalous"] = (out["threat_score"] > 0.6).astype(int)
    return out
