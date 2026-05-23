# SentinelView

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-IsolationForest-F7931E?logo=scikitlearn&logoColor=white)
![AWS Bedrock](https://img.shields.io/badge/AWS%20Bedrock-Claude-FF9900?logo=amazonaws&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Real-time geospatial OSINT intelligence dashboard for maritime, airspace, and conflict monitoring.**

---

## What This Does

SentinelView continuously ingests live data from four open-source intelligence streams — ADS-B
flight transponders via the OpenSky Network, AIS maritime vessel beacons via AISHub, armed
conflict events from ACLED, and weather context from NOAA — then fuses them into a unified
geospatial picture. Each track is passed through a feature engineering pipeline that computes
kinematic derivatives (computed speed, heading delta, loitering indicators, altitude rate) before
being scored by an ensemble of an Isolation Forest model and a deterministic rule-based detector.
Flagged anomalies are automatically summarised by Claude (via AWS Bedrock) into structured
intelligence briefs. The resulting interactive Folium map — with heat-map density overlays, red
anomaly markers, and a floating AI brief panel — is served in real time through a FastAPI backend
accessible on `localhost:8000`.

---

## Why It Matters

The commercial defence analogy for this capability is Palantir Foundry / Quiver, which fuses
multi-domain sensor data into operator dashboards, and Anduril Lattice, which builds a persistent
common operating picture from heterogeneous sensor feeds. SentinelView is an open-source analog
that demonstrates the same multi-source ingestion → ML anomaly scoring → LLM synthesis pipeline
using entirely public APIs and permissive licenses. It is purpose-built as a portfolio artefact
and research testbed for analysts, engineers, and security researchers who want a transparent,
hackable implementation of these ideas without vendor lock-in.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                               │
│                                                                     │
│  OpenSky Network   AISHub (AIS)   ACLED (Conflicts)   NOAA (Wx)    │
│  (ADS-B flights)   (vessels)      (events/fatalities) (forecasts)  │
└────────┬──────────────┬──────────────┬──────────────────┬──────────┘
         │              │              │                  │
         ▼              ▼              ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                            │
│  Haversine distance · time-gap · computed speed (kts) ·            │
│  heading delta · loitering flag · altitude change · descent rate   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
┌─────────────────────┐          ┌─────────────────────────┐
│  Isolation Forest   │          │  Rule-Based Detector    │
│  (ML anomaly score  │          │  dark vessel · jump ·   │
│   0-1, fitted on    │          │  sharp turn · speed     │
│   live data)        │          │  spoof · extreme descent│
└──────────┬──────────┘          └──────────┬──────────────┘
           │   ml_anomaly_score (×0.6)       │   rule_score (×0.4)
           └───────────────┬────────────────┘
                           ▼
              ┌────────────────────────┐
              │    Ensemble Score      │
              │  threat_score > 0.6   │
              │  → is_anomalous = 1   │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │    AWS Bedrock         │
              │  (Claude via LangChain)│
              │  → Intelligence Brief  │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │    Folium Map          │
              │  Heat-map + markers +  │
              │  floating AI panel     │
              │  served via FastAPI    │
              └────────────────────────┘
```

---

## Stack

| Component | Technology | Purpose |
|---|---|---|
| Web framework | FastAPI + uvicorn | REST API and HTML dashboard endpoints |
| Flight data | pyopensky (OpenSky Network) | Live ADS-B state vectors |
| Vessel data | requests → AISHub JSON API | Live AIS position reports |
| Conflict data | requests → ACLED REST API | Geolocated armed conflict events |
| Weather data | requests → NOAA Weather API | Hourly point forecasts (US coverage) |
| Feature engineering | pandas + numpy | Kinematic derivatives per entity track |
| ML anomaly detection | scikit-learn IsolationForest | Unsupervised outlier scoring |
| Rule-based detection | Pure Python | Deterministic flag logic (dark vessel, jump, etc.) |
| Intelligence synthesis | langchain-aws + boto3 (Bedrock / Claude) | AI intelligence briefs |
| Visualisation | Folium | Interactive HTML map with heat-map and markers |
| Configuration | python-dotenv | Credential management via `.env` file |
| Testing | pytest | Unit tests for all pipeline modules |

---

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/dmoney25632/sentinelview.git
   cd sentinelview
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure credentials**
   ```bash
   cp .env.example .env
   # Edit .env and fill in your API keys (see docs/data_sources.md)
   ```

4. **Run the server**
   ```bash
   python -m sentinelview.dashboard.app
   # or: uvicorn sentinelview.dashboard.app:app --reload
   ```

5. **Open the dashboard**
   Navigate to [http://localhost:8000](http://localhost:8000) for the vessel map or
   [http://localhost:8000/flights](http://localhost:8000/flights) for the flight map.

> **Tip — mock mode:** Set `SENTINELVIEW_MOCK=true` in your `.env` to run the full pipeline
> with synthetic data when live API credentials are unavailable.

---

## Data Sources

| Source | Data Type | Update Frequency | Free Tier |
|---|---|---|---|
| [OpenSky Network](https://opensky-network.org/) | ADS-B state vectors (position, altitude, speed, heading) | ~10 s (anonymous), ~5 s (registered) | Yes — registered accounts get higher rate limits |
| [AISHub](https://www.aishub.net/) | AIS vessel positions (MMSI, SOG, COG, navstat) | ~30–60 s aggregated feed | Yes — free accounts with receiver contribution or application |
| [ACLED](https://acleddata.com/) | Geolocated armed conflict events and fatalities | Daily | Yes — academic and non-profit accounts via application |
| [NOAA Weather API](https://www.weather.gov/documentation/services-web-api) | Hourly point forecasts (temperature, wind, conditions) | Hourly | Yes — completely free, no key required |
| [AWS Bedrock](https://aws.amazon.com/bedrock/) | Claude LLM inference for intelligence summaries | On-demand | No — standard AWS pay-per-token pricing applies |

---

## Anomaly Detection Methods

| Method | Type | What It Catches |
|---|---|---|
| **Isolation Forest** | Unsupervised ML (scikit-learn) | Statistical outliers in kinematic feature space — vessels or aircraft whose speed, heading change, altitude behaviour, or inter-ping gap deviate significantly from the fleet norm |
| **Dark Vessel Rule** | Rule-based | AIS signal gaps longer than 2 hours, suggesting deliberate transponder suppression |
| **Jump Rule** | Rule-based | Impossible position jumps — distance > 100 km in under 5 minutes for vessels, or > 500 km in under 60 seconds for aircraft |
| **Sharp Turn Rule** | Rule-based | Heading change greater than 30° between consecutive vessel pings |
| **Speed Spoof Rule** | Rule-based | Discrepancy > 10 kts between the AIS-reported speed-over-ground and the speed computed from consecutive position fixes |
| **Extreme Descent Rule** | Rule-based | Aircraft vertical rate below −3 000 ft/min (uncontrolled or emergency descent) |
| **Ensemble Score** | Weighted combination | Blends ML score (60 %) and rule score (40 %) into a single `threat_score`; entities above 0.6 are flagged `is_anomalous = 1` |

---

## Use Cases

- **Maritime domain awareness** — monitor vessel traffic across strategic chokepoints (Strait of
  Hormuz, South China Sea, Red Sea) and surface dark vessels, position-spoofing, or loitering
  behaviour in near real time.
- **Sanctions vessel monitoring** — cross-reference AIS anomalies (signal loss, identity changes,
  ship-to-ship transfers) with known sanctions-evasion patterns to generate leads for further
  investigation.
- **Airspace deconfliction** — detect extreme descents, impossible position jumps, and divergence
  between reported and computed speed for aircraft operating in contested or restricted airspace.
- **Geopolitical early warning** — overlay ACLED conflict events on vessel and flight tracks to
  correlate kinematic anomalies with ground-truth escalation signals for situational awareness.

---

## Limitations

- **API rate limits** — OpenSky anonymous access is rate-limited to one request per 10 seconds;
  AISHub free accounts impose similar constraints. High-frequency polling will result in HTTP 429
  errors. The `SENTINELVIEW_MOCK=true` flag bypasses all external calls for development.
- **NOAA coverage is US-only** — the NOAA Weather API (`api.weather.gov`) returns a 404 for any
  coordinates outside the contiguous United States. Weather context is silently skipped for
  non-US locations; international weather integration requires a third-party provider such as
  Open-Meteo.
- **Cold-start model fitting** — the Isolation Forest is fitted on the first live data batch it
  receives rather than on a pre-trained corpus. With small datasets (< 100 tracks) the model may
  lack statistical power and produce noisy scores until enough data has accumulated across requests.
- **No persistent storage** — entity state is held in memory for the duration of a single request;
  there is no database backend. Historical trend analysis and cross-request anomaly correlation
  require adding a persistence layer (e.g., PostgreSQL + PostGIS or a time-series store).

---

## Roadmap

- [ ] PostgreSQL + PostGIS persistence layer for cross-request track history
- [ ] Pre-trained Isolation Forest model serialised from a historical AIS/ADS-B corpus
- [ ] Open-Meteo integration for global weather context beyond US waters
- [ ] Sanctions list cross-reference (UN, OFAC) with automated MMSI lookups
- [ ] WebSocket endpoint for live map push updates without full page reload
- [ ] Docker Compose deployment with Nginx reverse proxy
- [ ] ACLED conflict overlay rendered as a choropleth layer on the dashboard map
- [ ] Alert webhook (Slack / PagerDuty) for high-severity anomaly notifications
- [ ] Ship identity change detection (MMSI recycling, flag-hopping)
- [ ] CI/CD pipeline with GitHub Actions, coverage reporting, and container image publishing

---

## License

[MIT](LICENSE)
