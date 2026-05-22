# SentinelView — Copilot Global Instructions

## What This Project Is
SentinelView is a real-time geospatial OSINT intelligence dashboard built in Python.
It ingests live ADS-B flight data (OpenSky Network), AIS maritime vessel data (AISHub),
conflict events (ACLED), and weather context (NOAA), runs multi-method anomaly detection
(Isolation Forest + rule-based), and generates AI threat summaries via AWS Bedrock (Claude).
A FastAPI backend serves a Folium interactive map dashboard.

## Tech Stack
Python 3.11+, FastAPI, uvicorn, pyopensky, requests, pandas, numpy,
scikit-learn IsolationForest, langchain-aws, boto3, folium, python-dotenv, pytest

## Project Structure
sentinelview/
├── sentinelview/
│   ├── ingestion/        # API clients for each data source
│   ├── processing/       # Feature engineering + anomaly detection
│   ├── intelligence/     # LLM threat summarization
│   └── dashboard/        # FastAPI app + Folium map renderer
├── tests/
├── notebooks/
└── .github/copilot-instructions.md

## Coding Standards
- Type hints on ALL function signatures
- Google-style docstrings on every class and public method
- All credentials read from env vars via os.getenv() — NEVER hardcoded
- All external API calls wrapped in try/except with informative error messages
- Functions max ~40 lines, single-purpose
- Ask clarifying questions before implementing if anything is unclear
