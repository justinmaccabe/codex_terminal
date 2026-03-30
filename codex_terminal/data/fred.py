from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Dict, Iterable

import pandas as pd
import requests
import streamlit as st

from codex_terminal.config.settings import get_settings


FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
REQUEST_HEADERS = {
    "User-Agent": "codex_terminal/1.0 (streamlit; market research terminal)",
    "Accept": "text/csv,application/json,text/plain,*/*",
}


@dataclass
class FredSeries:
    series_id: str
    label: str


@dataclass(frozen=True)
class FredStatus:
    ok: bool
    loaded: int
    total: int
    source: str
    message: str


DEFAULT_FRED_SERIES = [
    FredSeries("INDPRO", "Industrial Production"),
    FredSeries("UNRATE", "Unemployment Rate"),
    FredSeries("CPIAUCSL", "CPI"),
    FredSeries("FEDFUNDS", "Fed Funds"),
    FredSeries("T10Y2Y", "10Y-2Y Curve"),
    FredSeries("BAMLH0A0HYM2", "High Yield OAS"),
]


@st.cache_data(ttl=60 * 60 * 6)
def fetch_fred_series(series_id: str) -> pd.Series:
    settings = get_settings()
    if settings.fred_api_key:
        try:
            response = requests.get(
                FRED_API_URL,
                params={
                    "series_id": series_id,
                    "api_key": settings.fred_api_key,
                    "file_type": "json",
                },
                headers=REQUEST_HEADERS,
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            observations = payload.get("observations", [])
            if observations:
                frame = pd.DataFrame(observations)
                out = pd.Series(
                    pd.to_numeric(frame["value"], errors="coerce").values,
                    index=pd.to_datetime(frame["date"], errors="coerce"),
                    name=series_id,
                ).dropna()
                if not out.empty:
                    return out
        except Exception:
            pass

    response = requests.get(
        FRED_GRAPH_URL,
        params={"id": series_id},
        headers=REQUEST_HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    if frame.empty or "DATE" not in frame.columns or series_id not in frame.columns:
        return pd.Series(name=series_id, dtype=float)
    series = frame.rename(columns={"DATE": "date"}).assign(
        date=lambda df: pd.to_datetime(df["date"], errors="coerce"),
        value=lambda df: pd.to_numeric(df[series_id], errors="coerce"),
    )
    out = series.set_index("date")["value"].dropna()
    out.name = series_id
    return out


def fetch_fred_bundle(series_list: Iterable[FredSeries] = DEFAULT_FRED_SERIES) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for item in series_list:
        try:
            out[item.series_id] = fetch_fred_series(item.series_id)
        except Exception:
            out[item.series_id] = pd.Series(name=item.series_id, dtype=float)
    return out


def infer_fred_status(
    bundle: Dict[str, pd.Series],
    series_list: Iterable[FredSeries] = DEFAULT_FRED_SERIES,
) -> FredStatus:
    items = list(series_list)
    total = len(items)
    loaded = sum(1 for item in items if not bundle.get(item.series_id, pd.Series(dtype=float)).empty)
    settings = get_settings()
    source = "FRED API" if settings.fred_api_key else "FRED public endpoint"

    if total == 0:
        return FredStatus(
            ok=False,
            loaded=0,
            total=0,
            source=source,
            message="No FRED series configured.",
        )

    if loaded == total:
        return FredStatus(
            ok=True,
            loaded=loaded,
            total=total,
            source=source,
            message=f"Macro data loaded ({loaded}/{total}) via {source}.",
        )

    if loaded > 0:
        return FredStatus(
            ok=False,
            loaded=loaded,
            total=total,
            source=source,
            message=f"Partial macro coverage ({loaded}/{total}) via {source}.",
        )

    guidance = (
        "No macro series loaded. Verify outbound access to FRED"
        if settings.fred_api_key
        else "No macro series loaded. Add FRED_API_KEY to Streamlit secrets or verify public FRED access."
    )
    return FredStatus(
        ok=False,
        loaded=0,
        total=total,
        source=source,
        message=guidance,
    )
