from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


CORE_BRIEF_TICKERS = ["SPY", "TLT", "TIP", "PDBC", "GLDM", "WTMF", "VWO", "VGIT"]


def leadership_table(screener: pd.DataFrame) -> pd.DataFrame:
    if screener.empty:
        return pd.DataFrame()
    out = screener[["1M", "3M", "6M", "Trend", "Composite Score", "Stance"]].copy()
    out["1W Proxy"] = screener["1M"] / 4.0
    return out.sort_values("Composite Score", ascending=False)


def what_changed_table(screener: pd.DataFrame) -> pd.DataFrame:
    if screener.empty:
        return pd.DataFrame()
    out = screener.copy()
    out["1W Proxy"] = out["1M"] / 4.0
    out["Momentum Delta"] = out["1M"] - out["3M"] / 3.0
    out["Trend Break Risk"] = np.where((out["Trend"] < 1.0) & (out["Composite Percentile"] > 0.6), "Yes", "No")
    return out[["1W Proxy", "1M", "3M", "Momentum Delta", "Trend", "Trend Break Risk", "Stance"]].sort_values(
        "Momentum Delta", ascending=False
    )


def cross_asset_signal_table(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    available = [ticker for ticker in CORE_BRIEF_TICKERS if ticker in prices.columns]
    if not available:
        return pd.DataFrame()
    clean = prices[available].ffill().dropna(how="all")
    if len(clean) < 65:
        return pd.DataFrame()
    current_21 = clean.iloc[-1] / clean.iloc[-22] - 1
    prior_21 = clean.iloc[-22] / clean.iloc[-43] - 1 if len(clean) > 43 else current_21 * np.nan
    trend = (clean.iloc[-1] > clean.rolling(50).mean().iloc[-1]).astype(int)
    out = pd.DataFrame(
        {
            "1M": current_21,
            "Prev 1M": prior_21,
            "1M Delta": current_21 - prior_21,
            "Trend On": trend,
        }
    )
    return out


def rolling_corr_series(returns: pd.DataFrame, left: str, right: str, window: int = 63) -> pd.Series:
    if left not in returns.columns or right not in returns.columns:
        return pd.Series(dtype=float)
    pair = returns[[left, right]].dropna()
    if pair.empty:
        return pd.Series(dtype=float)
    return pair[left].rolling(window).corr(pair[right]).dropna()


def correlation_snapshot(returns: pd.DataFrame, tickers: Iterable[str], current_window: int = 63, prior_window: int = 126) -> tuple[pd.DataFrame, pd.DataFrame]:
    use = [ticker for ticker in tickers if ticker in returns.columns]
    if len(use) < 2:
        return pd.DataFrame(), pd.DataFrame()
    current = returns[use].tail(current_window).corr()
    prior = returns[use].tail(prior_window).head(max(prior_window - current_window, 1)).corr() if len(returns) >= prior_window else current * np.nan
    delta = current - prior
    return current, delta
