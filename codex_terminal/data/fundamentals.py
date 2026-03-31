from __future__ import annotations

from io import StringIO
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; codex_terminal/1.0; +https://github.com/justinmaccabe/codex_terminal)"
}


def _safe_float(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _parse_percent_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace("%", "", regex=False).str.replace("†", "", regex=False).str.strip()
    numeric = pd.to_numeric(cleaned, errors="coerce")
    return numeric / 100.0


@st.cache_data(ttl=6 * 60 * 60)
def fetch_fundamental_snapshots(tickers: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in list(dict.fromkeys(tickers)):
        try:
            info = yf.Ticker(ticker).get_info() or {}
        except Exception:
            info = {}
        trailing_pe = _safe_float(info.get("trailingPE"))
        forward_pe = _safe_float(info.get("forwardPE"))
        earnings_growth = _safe_float(info.get("earningsQuarterlyGrowth"))
        profit_margins = _safe_float(info.get("profitMargins"))
        roe = _safe_float(info.get("returnOnEquity"))
        dividend_yield = _safe_float(info.get("dividendYield"))
        rows.append(
            {
                "Ticker": ticker,
                "Name": info.get("shortName") or info.get("longName") or ticker,
                "Quote Type": info.get("quoteType"),
                "Trailing PE": trailing_pe,
                "Forward PE": forward_pe,
                "Trailing Earnings Yield": 1.0 / trailing_pe if trailing_pe and trailing_pe > 0 else np.nan,
                "Forward Earnings Yield": 1.0 / forward_pe if forward_pe and forward_pe > 0 else np.nan,
                "Earnings Growth": earnings_growth,
                "Profit Margin": profit_margins,
                "Return On Equity": roe,
                "Dividend Yield": dividend_yield,
            }
        )
    return pd.DataFrame(rows)


def _fetch_multpl_table(url: str) -> pd.DataFrame:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    if not tables:
        return pd.DataFrame()
    frame = tables[0].copy()
    if frame.empty or len(frame.columns) < 2:
        return pd.DataFrame()
    frame.columns = ["Date", "Value"]
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame.dropna(subset=["Date"])


@st.cache_data(ttl=12 * 60 * 60)
def fetch_sp500_earnings_history() -> pd.DataFrame:
    try:
        earnings = _fetch_multpl_table("https://www.multpl.com/s-p-500-earnings/table/by-month")
        yield_frame = _fetch_multpl_table("https://www.multpl.com/s-p-500-earnings-yield/table/by-month")
    except Exception:
        return pd.DataFrame()

    if earnings.empty:
        return pd.DataFrame()

    earnings["EPS"] = pd.to_numeric(earnings["Value"], errors="coerce")
    earnings = earnings.drop(columns=["Value"]).set_index("Date").sort_index()

    if not yield_frame.empty:
        yield_frame["Earnings Yield"] = _parse_percent_series(yield_frame["Value"])
        yield_frame = yield_frame.drop(columns=["Value"]).set_index("Date").sort_index()
        merged = earnings.join(yield_frame, how="left")
    else:
        merged = earnings

    return merged.dropna(how="all")
