from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import re

import pandas as pd
import requests


@dataclass(frozen=True)
class VanguardFund:
    ticker: str
    label: str
    target_year: int


VANGUARD_TARGET_FUNDS = [
    VanguardFund("VTINX", "Target Retirement Income", 2010),
    VanguardFund("VTWNX", "Target Retirement 2020", 2020),
    VanguardFund("VTTVX", "Target Retirement 2025", 2025),
    VanguardFund("VTHRX", "Target Retirement 2030", 2030),
    VanguardFund("VTTHX", "Target Retirement 2035", 2035),
    VanguardFund("VFORX", "Target Retirement 2040", 2040),
    VanguardFund("VTIVX", "Target Retirement 2045", 2045),
    VanguardFund("VFIFX", "Target Retirement 2050", 2050),
    VanguardFund("VFFVX", "Target Retirement 2055", 2055),
    VanguardFund("VTTSX", "Target Retirement 2060", 2060),
    VanguardFund("VLXVX", "Target Retirement 2065", 2065),
]


def infer_vanguard_target_fund(age_years: int, retirement_age: int = 65, current_year: int | None = None) -> VanguardFund:
    if current_year is None:
        current_year = date.today().year
    years_to_retirement = max(retirement_age - age_years, 0)
    target_year = current_year + years_to_retirement
    return min(VANGUARD_TARGET_FUNDS, key=lambda fund: abs(fund.target_year - target_year))


def _candidate_urls(ticker: str) -> list[str]:
    return [
        f"https://investor.vanguard.com/investment-products/mutual-funds/profile/{ticker}",
        f"https://investor.vanguard.com/investment-products/mutual-funds/profile/{ticker}?tab=performance",
        f"https://investor.vanguard.com/investment-products/mutual-funds/profile/{ticker}#performance-fees",
    ]


def _extract_history_from_json_blob(blob: str) -> pd.Series:
    date_price_patterns = [
        r'"date"\s*:\s*"(?P<date>[^"]+)"[^{}]{0,120}?"(?:price|nav|value|close)"\s*:\s*"?(?P<value>-?\d+(?:\.\d+)?)"?',
        r'"(?:price|nav|value|close)"\s*:\s*"?(?P<value>-?\d+(?:\.\d+)?)"?[^{}]{0,120}?"date"\s*:\s*"(?P<date>[^"]+)"',
    ]
    matches: list[tuple[str, float]] = []
    for pattern in date_price_patterns:
        for match in re.finditer(pattern, blob, flags=re.IGNORECASE):
            try:
                matches.append((match.group("date"), float(match.group("value"))))
            except Exception:
                continue
    if not matches:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(matches, columns=["date", "value"])
    out = pd.Series(
        pd.to_numeric(frame["value"], errors="coerce").values,
        index=pd.to_datetime(frame["date"], errors="coerce"),
        name="nav",
    ).dropna()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _extract_history_from_html_tables(html: str) -> pd.Series:
    try:
        tables = pd.read_html(html)
    except Exception:
        return pd.Series(dtype=float)
    for table in tables:
        columns = [str(col).strip().lower() for col in table.columns]
        if not any("date" in col for col in columns):
            continue
        if not any(any(word in col for word in ["nav", "price", "share"]) for col in columns):
            continue
        date_col = next((col for col in table.columns if "date" in str(col).lower()), None)
        value_col = next(
            (col for col in table.columns if any(word in str(col).lower() for word in ["nav", "price", "share"])),
            None,
        )
        if date_col is None or value_col is None:
            continue
        out = pd.Series(
            pd.to_numeric(table[value_col], errors="coerce").values,
            index=pd.to_datetime(table[date_col], errors="coerce"),
            name="nav",
        ).dropna()
        out = out[~out.index.duplicated(keep="last")].sort_index()
        if not out.empty:
            return out
    return pd.Series(dtype=float)


def _extract_history_from_html(html: str) -> pd.Series:
    # First try raw script/json-like blobs.
    script_blobs = re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.IGNORECASE | re.DOTALL)
    for blob in script_blobs:
        series = _extract_history_from_json_blob(blob)
        if len(series) >= 5:
            return series
        # Try to decode embedded JSON fragments opportunistically.
        candidate_json = re.findall(r"(\{.*?\}|\[.*?\])", blob, flags=re.DOTALL)
        for candidate in candidate_json[:50]:
            try:
                loaded = json.loads(candidate)
            except Exception:
                continue
            flat = json.dumps(loaded)
            series = _extract_history_from_json_blob(flat)
            if len(series) >= 5:
                return series

    # Fall back to HTML tables if Vanguard renders performance history as a table.
    return _extract_history_from_html_tables(html)


def fetch_vanguard_benchmark_history(ticker: str, start: str, end: str) -> pd.Series:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    for url in _candidate_urls(ticker):
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
        except Exception:
            continue
        series = _extract_history_from_html(response.text)
        if series.empty:
            continue
        series.name = ticker
        filtered = series.loc[start:end]
        if not filtered.empty:
            return filtered
    return pd.Series(name=ticker, dtype=float)
