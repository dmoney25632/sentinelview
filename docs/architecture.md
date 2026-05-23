# SentinelView — System Architecture

## Overview

SentinelView is structured as a four-layer pipeline: **ingestion → processing → intelligence →
dashboard**. Each layer is a standalone Python package (`sentinelview/<layer>/`) with a clear
public interface. FastAPI wires the layers together at request time; no persistent message
bus or queue is required for the default single-node deployment.

---

## Repository Layout

```
sentinelview/
├── sentinelview/               # Application package
│   ├── ingestion/              # API clients (one module per data source)
│   │   ├── opensky_client.py   # ADS-B flight data via pyopensky REST
│   │   ├── ais_client.py       # AIS vessel data via AISHub JSON API
│   │   ├── acled_client.py     # Conflict events via ACLED REST API
│   │   └── noaa_client.py      # Weather context via NOAA Weather API
│   ├── processing/             # Feature engineering and anomaly detection
│   │   ├── feature_engineer.py # Kinematic derivatives (speed, heading, etc.)
│   │   ├── ml_detector.py      # scikit-learn Isolation Forest wrapper
│   │   └── rule_detector.py    # Deterministic rule flags + ensemble scorer
│   ├── intelligence/           # LLM-based threat summarisation
│   │   └── threat_summarizer.py # AWS Bedrock (Claude) via LangChain
│   ├── dashboard/              # HTTP layer
│   │   ├── app.py              # FastAPI application and pipeline orchestration
│   │   └── map_renderer.py     # Folium map builder
│   └── utils/
│       └── mock_data.py        # Synthetic data generators for offline use
├── tests/                      # pytest unit tests (one file per module)
├── notebooks/                  # Exploratory Jupyter notebooks
├── docs/                       # This documentation
├── .env.example                # Template for required environment variables
├── requirements.txt            # Python dependencies
└── Makefile                    # Developer workflow shortcuts
```

---

## Layer Descriptions

### 1. Ingestion (`sentinelview/ingestion/`)

Each client encapsulates a single external API. Clients read credentials from environment
variables and always return a typed pandas `DataFrame` (or `None` for the NOAA point-forecast
helper). On any network or parsing error the client logs a warning and returns an empty DataFrame
with the correct schema, ensuring the downstream pipeline never crashes due to a missing data
source.

| Module | Class | Data returned |
|---|---|---|
| `opensky_client.py` | `OpenSkyClient` | ADS-B state vectors: position, altitude, speed, heading, vertical rate |
| `ais_client.py` | `AISHubClient` | AIS vessel reports: position, speed-over-ground, course, navstatus, MMSI |
| `acled_client.py` | `ACLEDClient` | Conflict events: date, type, country, lat/lon, fatalities, notes |
| `noaa_client.py` | `NOAAClient` | Hourly point forecasts: temperature, wind speed, short forecast text |

All clients respect `SENTINELVIEW_MOCK=true` (OpenSky and AIS) to return deterministic
synthetic DataFrames during development and CI.

### 2. Processing (`sentinelview/processing/`)

#### Feature Engineering (`feature_engineer.py`)

Raw position pings alone are not informative enough for anomaly detection. The feature
engineering step sorts pings by entity identifier and timestamp, then computes
*inter-ping kinematic deltas* using a group-wise `shift(1)` approach:

**Vessel features** (from `engineer_vessel_features`):

| Feature | Formula | Anomaly Signal |
|---|---|---|
| `distance_km` | Haversine(prev → current position) | Large jump |
| `time_gap_s` | `(current_ts − prev_ts).total_seconds()` | Long silence |
| `computed_speed_kts` | `(distance_km / time_gap_h) / 1.852` | Speed inconsistency |
| `heading_delta` | Min absolute angular diff (0–180°) | Sharp manoeuvre |
| `sog_vs_computed_delta` | `|AIS SOG − computed_speed_kts|` | Transponder spoofing |
| `loitering` | 1 when speed < 1.5 kts AND gap > 5 min | Suspicious hovering |

**Flight features** (from `engineer_flight_features`) extend the vessel set with:

| Feature | Formula | Anomaly Signal |
|---|---|---|
| `altitude_change_ft` | `(current − prev altitude) × 3.28084` | Rapid altitude loss |
| `descent_rate_fpm` | `vertical_rate × 196.85` | Emergency descent |

Rows that lack a preceding ping (first observation per entity) are dropped; no features can
be computed without a baseline.

#### ML Detector (`ml_detector.py`)

`MLAnomalyDetector` wraps a `StandardScaler` and `IsolationForest(n_estimators=200,
contamination=0.05)` from scikit-learn. The class is mode-aware: `mode="vessel"` and
`mode="flight"` each select a different feature column list. On the first live request the
model is fitted in-place on the incoming data batch (`fit` then `score`). Subsequent requests
re-use the fitted model. The output columns are:

- `ml_raw_score` — negated `score_samples` (higher = more anomalous).
- `ml_anomaly_score` — min-max normalised to `[0, 1]`.
- `ml_flag` — 1 when the model predicts `−1` (outlier), else 0.

Models can be serialised to disk via `save(path)` and reloaded via `MLAnomalyDetector.load(path)`
for deployment scenarios with pre-trained weights.

#### Rule Detector (`rule_detector.py`)

The rule functions (`apply_vessel_rules`, `apply_flight_rules`) add deterministic binary flags
per row, then normalise the flag count to a `rule_score` in `[0, 1]`:

**Vessel rules:**

| Flag | Condition | Rationale |
|---|---|---|
| `dark_vessel_flag` | `time_gap_s > 7200` | >2 h AIS silence — likely transponder off |
| `jump_flag` | distance > 100 km AND gap < 5 min | Physically impossible movement |
| `sharp_turn_flag` | `heading_delta > 30°` | Sudden evasive manoeuvre |
| `speed_spoof_flag` | `sog_vs_computed_delta > 10 kts` | Reported speed does not match kinematics |

**Flight rules:**

| Flag | Condition | Rationale |
|---|---|---|
| `extreme_descent_flag` | `descent_rate_fpm < −3000` | Uncontrolled or emergency descent |
| `jump_flag` | distance > 500 km AND gap < 60 s | Impossible ADS-B position jump |

The `ensemble_score` function combines both scores:
`threat_score = 0.6 × ml_anomaly_score + 0.4 × rule_score`. Entities with
`threat_score > 0.6` are labelled `is_anomalous = 1`.

### 3. Intelligence (`sentinelview/intelligence/`)

`ThreatSummarizer` builds a LangChain `PromptTemplate | BedrockLLM` chain. On each call to
`summarize`, the top-5 anomalous rows (by `threat_score`) are serialised as a plaintext table
and injected into the prompt. Claude returns a 3–5 sentence analyst brief that identifies the
anomaly type, approximate location, threat level (LOW/MEDIUM/HIGH), and a recommended follow-up
action.

`MockThreatSummarizer` is a drop-in replacement that returns a realistic hardcoded brief when
AWS credentials are absent. `get_summarizer()` auto-selects the correct implementation based on
the presence of `AWS_ACCESS_KEY_ID` in the environment.

### 4. Dashboard (`sentinelview/dashboard/`)

#### FastAPI Application (`app.py`)

The application exposes four routes:

| Route | Response | Description |
|---|---|---|
| `GET /` | HTML | Vessel anomaly map (AIS pipeline) |
| `GET /flights` | HTML | Flight anomaly map (ADS-B pipeline) |
| `GET /api/anomalies` | JSON | Anomalous vessel records |
| `GET /api/status` | JSON | Detector fit status and version |
| `GET /health` | JSON | Health check |

The pipeline helpers `_run_vessel_pipeline` and `_run_flight_pipeline` handle the edge case
where the live dataset contains only one ping per entity (no consecutive pairs): in that
scenario the feature engineering step returns an empty DataFrame, and the pipeline falls back
to raw data with all kinematic features defaulted to zero.

#### Map Renderer (`map_renderer.py`)

`build_map` constructs a dark-themed CartoDB map with three overlays:

1. **HeatMap** — density layer over all entity positions.
2. **Red circle markers** (radius 12) — anomalous entities with a structured HTML popup
   showing the entity ID, threat score, and active flags.
3. **Blue circle markers** (radius 4) — normal entities.
4. **Floating HTML panel** — the AI intelligence brief, positioned fixed bottom-right with
   a green monospaced terminal aesthetic.

---

## Data Flow Sequence

```
HTTP GET /
     │
     ▼
AISHubClient.get_vessels()
     │  → raw DataFrame [mmsi, latitude, longitude, sog, ...]
     ▼
_ensure_timestamp()          ← add/normalise UTC timestamp column
     │
     ▼
engineer_vessel_features()   ← compute distance, speed, heading delta, loitering
     │  → feature DataFrame (first ping per vessel dropped)
     ▼
MLAnomalyDetector.score()    ← IsolationForest → ml_anomaly_score, ml_flag
     │
     ▼
apply_vessel_rules()         ← dark_vessel, jump, sharp_turn, speed_spoof flags
     │
     ▼
ensemble_score()             ← threat_score, is_anomalous columns
     │
     ▼
ThreatSummarizer.summarize() ← top-5 anomalies → Claude → intelligence brief
     │
     ▼
build_map()                  ← Folium HTML with markers + AI panel
     │
     ▼
HTMLResponse → browser
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENSKY_USERNAME` | No | OpenSky registered account username (higher rate limit) |
| `OPENSKY_PASSWORD` | No | OpenSky registered account password |
| `AISHUB_USERNAME` | Yes | AISHub account username |
| `ACLED_EMAIL` | Yes | Email address registered with ACLED |
| `ACLED_KEY` | Yes | ACLED API key |
| `AWS_REGION` | Yes (Bedrock) | AWS region for Bedrock (e.g. `us-east-1`) |
| `AWS_ACCESS_KEY_ID` | Yes (Bedrock) | AWS access key ID |
| `AWS_SECRET_ACCESS_KEY` | Yes (Bedrock) | AWS secret access key |
| `BEDROCK_MODEL_ID` | No | Bedrock model ID (default: `anthropic.claude-3-sonnet-20240229-v1:0`) |
| `SENTINELVIEW_MOCK` | No | Set to `true` to use synthetic data — no external calls are made |

---

## Testing

Tests live in `tests/` and mirror the source module structure. Each test module covers
a single source module. Mock data is provided by `sentinelview/utils/mock_data.py`.
Run the full suite with:

```bash
make test
# or: pytest tests/
```
