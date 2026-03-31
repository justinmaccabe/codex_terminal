from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, Optional

import pandas as pd
import streamlit as st
import yfinance as yf


@dataclass
class DataStatus:
    ok: bool
    message: str = ""


@st.cache_data(ttl=60 * 60)
def fetch_price_history(
    tickers: Iterable[str],
    start: str = "2010-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    ticker_list = list(dict.fromkeys(tickers))
    if not ticker_list:
        return pd.DataFrame()

    if end is None:
        end = date.today().isoformat()

    try:
        data = yf.download(
            ticker_list,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()
    if data.empty:
        return pd.DataFrame()

    if len(ticker_list) == 1:
        single = ticker_list[0]
        if "Close" in data.columns:
            out = data[["Close"]].rename(columns={"Close": single})
            return out.dropna(how="all")
        return pd.DataFrame()

    close_frames: Dict[str, pd.Series] = {}
    for ticker in ticker_list:
        if ticker in data and "Close" in data[ticker]:
            close_frames[ticker] = data[ticker]["Close"]

    if not close_frames:
        return pd.DataFrame()
    return pd.DataFrame(close_frames).dropna(how="all")


@st.cache_data(ttl=15 * 60)
def fetch_intraday_history(
    tickers: Iterable[str],
    period: str = "5d",
    interval: str = "30m",
) -> pd.DataFrame:
    ticker_list = list(dict.fromkeys(tickers))
    if not ticker_list:
        return pd.DataFrame()

    try:
        data = yf.download(
            ticker_list,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
            prepost=False,
        )
    except Exception:
        return pd.DataFrame()
    if data.empty:
        return pd.DataFrame()

    if len(ticker_list) == 1:
        single = ticker_list[0]
        if "Close" in data.columns:
            out = data[["Close"]].rename(columns={"Close": single})
            return out.dropna(how="all")
        return pd.DataFrame()

    close_frames: Dict[str, pd.Series] = {}
    for ticker in ticker_list:
        if ticker in data and "Close" in data[ticker]:
            close_frames[ticker] = data[ticker]["Close"]

    if not close_frames:
        return pd.DataFrame()
    return pd.DataFrame(close_frames).dropna(how="all")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    return prices.sort_index().pct_change().dropna(how="all")


def latest_available(prices: pd.DataFrame) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    return prices.ffill().iloc[-1].dropna()


def infer_status(prices: pd.DataFrame, requested: Iterable[str]) -> DataStatus:
    requested_list = list(dict.fromkeys(requested))
    if prices.empty:
        return DataStatus(ok=False, message="No price history returned from Yahoo Finance.")
    available = set(prices.columns)
    missing = [ticker for ticker in requested_list if ticker not in available]
    if missing:
        return DataStatus(
            ok=False,
            message=f"Partial market data coverage. Missing: {', '.join(missing)}",
        )
    return DataStatus(ok=True, message="Market data loaded.")
