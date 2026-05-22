"""Folium interactive map renderer for geospatial intelligence overlays."""

import html

import folium
import folium.plugins
import pandas as pd

_ANOMALY_COLOR = "#FF4444"
_NORMAL_COLOR = "#00aaff"


def _get_id_value(row: pd.Series, mode: str) -> str:
    """Return the entity identifier for a DataFrame row.

    Args:
        row: A single row from the enriched DataFrame.
        mode: ``"vessel"`` uses ``mmsi``; ``"flight"`` uses ``icao24``.

    Returns:
        String representation of the entity ID, or ``"N/A"`` when absent.
    """
    id_col = "mmsi" if mode == "vessel" else "icao24"
    val = row.get(id_col, "N/A")
    return "N/A" if val is None else str(val)


def _safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    """Return a float value from *row[col]*, falling back to *default*.

    Args:
        row: A single row from the enriched DataFrame.
        col: Column name to read.
        default: Value returned when the column is missing or NaN.

    Returns:
        Float value, or *default*.
    """
    val = row.get(col)
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _anomaly_popup_html(row: pd.Series, mode: str) -> str:
    """Build an HTML popup string for an anomalous entity marker.

    Args:
        row: A single row from the enriched DataFrame.
        mode: ``"vessel"`` or ``"flight"``.

    Returns:
        HTML string suitable for use in a :class:`folium.Popup`.
    """
    entity_id = html.escape(_get_id_value(row, mode))
    threat = _safe_float(row, "threat_score")
    dark_flag = int(_safe_float(row, "dark_vessel_flag"))
    jump_flag = int(_safe_float(row, "jump_flag"))
    spoof_flag = int(_safe_float(row, "speed_spoof_flag"))
    loitering = int(_safe_float(row, "loitering"))

    return (
        "<div style='"
        "background:#1a1a2e;color:#e0e0e0;font-family:monospace;"
        "padding:10px;border-radius:6px;min-width:220px;"
        "'>"
        f"<b style='color:{_ANOMALY_COLOR}'>⚠️ ANOMALY DETECTED</b><br/>"
        f"<b>ID:</b> {entity_id}<br/>"
        f"<b>Threat Score:</b> {threat:.3f}<br/>"
        f"<b>Dark / Gap Flag:</b> {dark_flag}<br/>"
        f"<b>Jump Flag:</b> {jump_flag}<br/>"
        f"<b>Speed Spoof:</b> {spoof_flag}<br/>"
        f"<b>Loitering:</b> {loitering}"
        "</div>"
    )


def _intelligence_panel_html(ai_summary: str) -> str:
    """Build the floating AI intelligence brief HTML element.

    Args:
        ai_summary: Plain-text summary produced by the threat summarizer.

    Returns:
        HTML string for a fixed-position panel injected into the map.
    """
    escaped = html.escape(ai_summary).replace("\n", "<br/>")
    return (
        "<div style='"
        "position:fixed;"
        "bottom:30px;right:20px;"
        "z-index:9999;"
        "background:#0d0d0d;"
        "color:#00ff88;"
        "font-family:monospace;"
        "font-size:12px;"
        "padding:14px 16px;"
        "border:1px solid #00ff88;"
        "border-radius:6px;"
        "max-width:400px;"
        "box-shadow:0 0 12px rgba(0,255,136,0.3);"
        "'>"
        "<b style='font-size:13px;letter-spacing:1px;'>"
        "🤖 AI INTELLIGENCE BRIEF"
        "</b>"
        "<hr style='border-color:#00ff88;margin:6px 0;'/>"
        f"<span>{escaped}</span>"
        "</div>"
    )


def build_map(df: pd.DataFrame, ai_summary: str, mode: str = "vessel") -> str:
    """Render an interactive Folium map with anomaly overlays and AI summary.

    Creates a dark-themed world map with:

    * A heat-map layer covering all entity positions.
    * Red circle markers (radius 12) for anomalous entities with a detailed
      HTML popup and tooltip.
    * Blue circle markers (radius 4) for non-anomalous entities.
    * A fixed-position HTML panel (bottom-right) showing the AI summary.

    Args:
        df: Enriched DataFrame containing at least ``latitude``, ``longitude``,
            and ``is_anomalous`` columns.  The columns ``threat_score``,
            ``dark_vessel_flag``, ``jump_flag``, ``speed_spoof_flag``, and
            ``loitering`` are used in popups when present.
        ai_summary: Intelligence brief text from the threat summarizer.
        mode: ``"vessel"`` (default) or ``"flight"``.  Controls which ID
            column (``mmsi`` / ``icao24``) appears in popups.

    Returns:
        Fully rendered HTML string for the interactive map.
    """
    m = folium.Map(location=[20, 0], zoom_start=3, tiles="CartoDB dark_matter")

    # ── Heat-map of all positions ──────────────────────────────────────────
    heat_data = (
        df[["latitude", "longitude"]]
        .dropna()
        .values.tolist()
    )
    if heat_data:
        folium.plugins.HeatMap(heat_data, radius=8, blur=12).add_to(m)

    # ── Per-entity markers ─────────────────────────────────────────────────
    for _, row in df.iterrows():
        lat = row.get("latitude")
        lon = row.get("longitude")
        if lat is None or lon is None:
            continue
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue

        is_anomalous = int(_safe_float(row, "is_anomalous")) == 1

        if is_anomalous:
            threat = _safe_float(row, "threat_score")
            folium.CircleMarker(
                location=[lat, lon],
                radius=12,
                color=_ANOMALY_COLOR,
                fill=True,
                fill_opacity=0.85,
                popup=folium.Popup(_anomaly_popup_html(row, mode), max_width=300),
                tooltip=f"⚠️ ANOMALY — Score: {threat:.2f}",
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=_NORMAL_COLOR,
                fill=True,
                fill_opacity=0.4,
            ).add_to(m)

    # ── Floating AI intelligence panel ────────────────────────────────────
    m.get_root().html.add_child(
        folium.Element(_intelligence_panel_html(ai_summary))
    )

    return m.get_root().render()
