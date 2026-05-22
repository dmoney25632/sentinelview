"""FastAPI application entry-point for the SentinelView dashboard."""

from dotenv import load_dotenv

load_dotenv()

import logging
from typing import Union

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from sentinelview.dashboard.map_renderer import build_map
from sentinelview.ingestion.ais_client import AISHubClient
from sentinelview.ingestion.opensky_client import OpenSkyClient
from sentinelview.intelligence.threat_summarizer import (
    MockThreatSummarizer,
    ThreatSummarizer,
    get_summarizer,
)
from sentinelview.processing.feature_engineer import (
    engineer_flight_features,
    engineer_vessel_features,
    normalize_timestamps,
)
from sentinelview.processing.ml_detector import (
    FLIGHT_FEATURES,
    VESSEL_FEATURES,
    MLAnomalyDetector,
)
from sentinelview.processing.rule_detector import (
    apply_flight_rules,
    apply_vessel_rules,
    ensemble_score,
)

_log = logging.getLogger(__name__)

app = FastAPI(title="SentinelView", version="0.1.0")

vessel_detector = MLAnomalyDetector(mode="vessel")
flight_detector = MLAnomalyDetector(mode="flight")
summarizer: Union[ThreatSummarizer, MockThreatSummarizer] = get_summarizer()


def _ensure_timestamp(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """Add a UTC timestamp column to *df* when it is absent.

    Args:
        df: Input DataFrame.
        col: Name of the timestamp column.

    Returns:
        DataFrame with ``col`` present as datetime64[ns, UTC].
    """
    if col not in df.columns:
        df = df.copy()
        df[col] = pd.Timestamp.now(tz="UTC")
    return normalize_timestamps(df, col)


def _fill_missing_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Fill any missing feature columns with zeros.

    Args:
        df: DataFrame that may be missing some feature columns.
        feature_cols: List of column names that must be present.

    Returns:
        Copy of *df* with all *feature_cols* filled (defaulting to 0).
    """
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0.0
    return out


def _run_vessel_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the full vessel anomaly-detection pipeline.

    Runs feature engineering, ML scoring (fitting on the first call), rule
    detection, and ensemble scoring.  Handles the case where the raw DataFrame
    has only one ping per vessel (no consecutive rows) by working with the raw
    data and default feature values.

    Args:
        df: Raw AIS vessel DataFrame from :class:`~sentinelview.ingestion.ais_client.AISHubClient`.

    Returns:
        Enriched DataFrame with ``threat_score`` and ``is_anomalous`` columns.
    """
    df = _ensure_timestamp(df)

    engineered = engineer_vessel_features(df)

    if engineered.empty:
        _log.info("No consecutive vessel pings; using raw DataFrame with default features.")
        engineered = _fill_missing_features(df, VESSEL_FEATURES)
        engineered = _fill_missing_features(
            engineered,
            ["dark_vessel_flag", "jump_flag", "sharp_turn_flag", "speed_spoof_flag"],
        )
        engineered["rule_score"] = 0.0

    if not vessel_detector.fitted:
        vessel_detector.fit(engineered)

    scored = vessel_detector.score(engineered)

    if "rule_score" not in scored.columns:
        scored = apply_vessel_rules(scored)

    return ensemble_score(scored)


def _run_flight_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the full flight anomaly-detection pipeline.

    Args:
        df: Raw ADS-B DataFrame from :class:`~sentinelview.ingestion.opensky_client.OpenSkyClient`.

    Returns:
        Enriched DataFrame with ``threat_score`` and ``is_anomalous`` columns.
    """
    df = _ensure_timestamp(df)

    engineered = engineer_flight_features(df)

    if engineered.empty:
        _log.info("No consecutive flight pings; using raw DataFrame with default features.")
        engineered = _fill_missing_features(df, FLIGHT_FEATURES)
        engineered = _fill_missing_features(
            engineered, ["extreme_descent_flag", "jump_flag"]
        )
        engineered["rule_score"] = 0.0

    if not flight_detector.fitted:
        flight_detector.fit(engineered)

    scored = flight_detector.score(engineered)

    if "rule_score" not in scored.columns:
        scored = apply_flight_rules(scored)

    return ensemble_score(scored)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Render the vessel anomaly dashboard map.

    Fetches live AIS vessel positions, runs the full anomaly-detection
    pipeline, generates an AI intelligence brief, and returns an interactive
    Folium HTML map.

    Returns:
        HTMLResponse containing the rendered map.
    """
    raw = AISHubClient().get_vessels()
    if raw.empty:
        return HTMLResponse("<h1>No vessel data available.</h1>")

    result = _run_vessel_pipeline(raw)
    anomalies = result[result["is_anomalous"] == 1]
    summary = summarizer.summarize(anomalies) if not anomalies.empty else "No anomalies detected."
    return HTMLResponse(build_map(result, summary, mode="vessel"))


@app.get("/flights", response_class=HTMLResponse)
async def flights() -> HTMLResponse:
    """Render the flight anomaly dashboard map.

    Fetches live ADS-B state vectors from OpenSky, runs the full
    anomaly-detection pipeline, and returns an interactive Folium HTML map.

    Returns:
        HTMLResponse containing the rendered map.
    """
    raw = OpenSkyClient().get_live_states()
    if raw.empty:
        return HTMLResponse("<h1>No flight data available.</h1>")

    result = _run_flight_pipeline(raw)
    anomalies = result[result["is_anomalous"] == 1]
    summary = summarizer.summarize(anomalies) if not anomalies.empty else "No anomalies detected."
    return HTMLResponse(build_map(result, summary, mode="flight"))


@app.get("/api/anomalies", response_class=JSONResponse)
async def api_anomalies() -> JSONResponse:
    """Return anomalous vessel detections as JSON.

    Runs the vessel pipeline and returns all rows where ``is_anomalous == 1``.

    Returns:
        JSONResponse containing a list of anomalous vessel records.
    """
    raw = AISHubClient().get_vessels()
    if raw.empty:
        return JSONResponse([])

    result = _run_vessel_pipeline(raw)
    anomalies = result[result["is_anomalous"] == 1]
    return JSONResponse(anomalies.to_dict(orient="records"))


@app.get("/api/status", response_class=JSONResponse)
async def api_status() -> JSONResponse:
    """Return detector status and application version.

    Returns:
        JSONResponse with ``vessel_detector_fitted``, ``flight_detector_fitted``,
        and ``version`` keys.
    """
    return JSONResponse(
        {
            "vessel_detector_fitted": vessel_detector.fitted,
            "flight_detector_fitted": flight_detector.fitted,
            "version": "0.1.0",
        }
    )


@app.get("/health", response_class=JSONResponse)
async def health() -> JSONResponse:
    """Health-check endpoint.

    Returns:
        JSONResponse with ``{"status": "ok"}``.
    """
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("sentinelview.dashboard.app:app", host="0.0.0.0", port=8000, reload=True)
