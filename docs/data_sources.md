# SentinelView — Data Sources & API Setup

This document covers account registration, credential configuration, and API-specific notes
for every external data source used by SentinelView.

---

## Table of Contents

1. [OpenSky Network (ADS-B flights)](#1-opensky-network-ads-b-flights)
2. [AISHub (AIS maritime vessels)](#2-aishub-ais-maritime-vessels)
3. [ACLED (Armed conflict events)](#3-acled-armed-conflict-events)
4. [NOAA Weather API (Weather context)](#4-noaa-weather-api-weather-context)
5. [AWS Bedrock / Claude (Intelligence summaries)](#5-aws-bedrock--claude-intelligence-summaries)

---

## 1. OpenSky Network (ADS-B flights)

**What it provides:** Live and historical ADS-B state vectors — position (lat/lon), barometric
altitude, velocity, heading, vertical rate, and on-ground status — for commercial and general
aviation aircraft worldwide.

**Rate limits:**
- **Anonymous:** 400 API credits per day, one request per 10 seconds.
- **Registered (free):** 4 000 credits/day, one request per 5 seconds.
- Requests for geographic bounding boxes larger than 25° × 25° may be rejected.

### Setup

1. Create a free account at [opensky-network.org](https://opensky-network.org/).
2. Verify your email address.
3. Add your credentials to `.env`:
   ```
   OPENSKY_USERNAME=your_username
   OPENSKY_PASSWORD=your_password
   ```

SentinelView uses the `pyopensky` library's `REST` client, which reads credentials from these
environment variables (or from `~/.config/pyopensky/secret.conf` if you prefer the pyopensky
config file approach).

### Notes

- If credentials are absent, the REST client falls back to anonymous access with the stricter
  rate limit.
- `SENTINELVIEW_MOCK=true` bypasses the OpenSky API entirely and returns synthetic flight data.
- The `bbox_from_region` helper in `opensky_client.py` provides pre-defined bounding boxes for
  the Persian Gulf, South China Sea, Strait of Hormuz, Black Sea, and Red Sea.

---

## 2. AISHub (AIS maritime vessels)

**What it provides:** Aggregated real-time AIS (Automatic Identification System) position reports
from a global network of shore-based and satellite receivers. Reports include MMSI, vessel name,
IMO number, position, speed-over-ground, course-over-ground, heading, navigational status,
callsign, and destination.

**Rate limits:**
- Free accounts: typically one full-feed request per minute.
- Requests without a geographic bounding box return the entire global feed (large payload).

### Setup

1. Register at [aishub.net/register](https://www.aishub.net/register).
   - Free accounts are available. AISHub encourages users with AIS receivers to contribute their
     data feed in exchange for API access, but a receiver is not strictly required.
2. After approval, note your username from the account settings page.
3. Add the credential to `.env`:
   ```
   AISHUB_USERNAME=your_aishub_username
   ```

### Notes

- The AISHub JSON endpoint returns a two-element array: `[header_dict, [vessel_list]]`. The
  `AISHubClient` handles this structure automatically.
- Use the `bbox` parameter to `get_vessels()` to restrict the query to a geographic region and
  reduce response size: `bbox=(min_lat, max_lat, min_lon, max_lon)`.
- `SENTINELVIEW_MOCK=true` returns synthetic vessel data without contacting AISHub.

---

## 3. ACLED (Armed conflict events)

**What it provides:** Geolocated, dated records of armed conflict events globally — battles,
explosions, protests, strategic developments, and civilian targeting — with event type,
country, latitude/longitude, fatality count, and descriptive notes. Data is updated daily.

**Rate limits:**
- The REST API is rate-limited per registered API key; typical limits are generous for
  non-commercial research use (thousands of requests per day).
- The default query in SentinelView fetches up to 500 events per request.

### Setup

1. Register for an account at [acleddata.com/register](https://acleddata.com/register/).
   - ACLED provides free access for academic, policy, non-profit, and journalistic use.
   - Commercial or government use may require a separate agreement.
2. After registration, your API key will be emailed to you. You can also retrieve it from
   the ACLED dashboard under **Account → API Access**.
3. Add credentials to `.env`:
   ```
   ACLED_EMAIL=you@example.com
   ACLED_KEY=your_acled_api_key
   ```

### Notes

- The `ACLEDClient.get_recent_events(days_back=7)` method defaults to the last 7 days. Adjust
  `days_back` for broader historical sweeps.
- Supply `country="Ukraine"` (or any ISO country name) to filter results geographically.
- ACLED event data is most useful as a contextual overlay — correlate vessel or flight
  anomalies with nearby conflict events to assess geopolitical risk.

---

## 4. NOAA Weather API (Weather context)

**What it provides:** Hourly weather forecasts for points within the contiguous United States,
including temperature, wind speed and direction, and a short textual forecast. No authentication
is required.

**Rate limits:**
- No formal rate limit is published; NOAA requests that all clients include a `User-Agent`
  header (SentinelView includes `User-Agent: SentinelView/1.0` automatically).
- Excessive automated polling may result in temporary IP throttling.

### Setup

No API key or account is required. The NOAA Weather API is completely free and unauthenticated.
No `.env` entries are needed.

### Notes

- **US coverage only.** Coordinates outside the contiguous United States (CONUS) return HTTP
  404 from the `/points/{lat},{lon}` endpoint. `NOAAClient.get_point_forecast` returns `None`
  for these locations silently.
- For international weather context, consider integrating
  [Open-Meteo](https://open-meteo.com/) (free, global, no key required) as a drop-in
  replacement.
- Forecast data is fetched in two steps: first a `/points/` metadata call to resolve the grid
  office, then a `/gridpoints/.../forecast/hourly` call. Both use a 10-second timeout.

---

## 5. AWS Bedrock / Claude (Intelligence summaries)

**What it provides:** Managed access to foundation models (including Anthropic Claude) via
the AWS Bedrock API. SentinelView uses Claude to generate 3–5 sentence structured intelligence
briefs from anomaly data.

**Cost:** AWS Bedrock is billed per input/output token. At current Claude Sonnet pricing
(approximately $3 / 1M input tokens, $15 / 1M output tokens), generating a single brief
(~500 input + ~200 output tokens) costs less than $0.01. There is no free tier.

### Setup

1. **Enable AWS Bedrock and model access:**
   - Sign in to the [AWS Console](https://console.aws.amazon.com/).
   - Navigate to **Amazon Bedrock → Model access**.
   - Request access to `Anthropic Claude 3 Sonnet` (or whichever model you wish to use).
   - Access is typically approved within minutes.

2. **Create an IAM user or role:**
   - Navigate to **IAM → Users → Create user**.
   - Attach the managed policy `AmazonBedrockFullAccess` (or a more restrictive custom
     policy that allows `bedrock:InvokeModel` on the specific model ARN).
   - Under **Security credentials**, create an access key and note the key ID and secret.

3. **Add credentials to `.env`:**
   ```
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
   AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
   BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
   ```

4. **Verify access:**
   ```bash
   python - <<'EOF'
   import boto3, os
   from dotenv import load_dotenv
   load_dotenv()
   client = boto3.client("bedrock", region_name=os.getenv("AWS_REGION", "us-east-1"))
   models = client.list_foundation_models()
   print([m["modelId"] for m in models["modelSummaries"] if "claude" in m["modelId"]])
   EOF
   ```

### Notes

- If `AWS_ACCESS_KEY_ID` is absent from the environment, `get_summarizer()` automatically
  returns `MockThreatSummarizer`, which produces a realistic hardcoded brief with no API call.
  This is the recommended path for development and CI.
- The default model is `anthropic.claude-3-sonnet-20240229-v1:0`. Override with
  `BEDROCK_MODEL_ID` to use Claude Haiku (cheaper) or Claude Opus (higher quality).
- Bedrock model IDs and availability vary by region. Check
  [the Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
  for the current list.

---

## Mock Mode

Set `SENTINELVIEW_MOCK=true` in your `.env` to run the full pipeline — feature engineering,
anomaly detection, LLM summarisation (via `MockThreatSummarizer`), and map rendering — without
making any external API calls. This is the recommended mode for development, offline use, and CI.

```bash
echo "SENTINELVIEW_MOCK=true" >> .env
python -m sentinelview.dashboard.app
```
