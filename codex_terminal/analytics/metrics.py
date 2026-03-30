from __future__ import annotations

from math import sqrt
from typing import Dict

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def total_return(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return (1 + series).prod() - 1


def annualized_return(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    periods = len(series)
    return (1 + total_return(series)) ** (TRADING_DAYS / periods) - 1


def annualized_volatility(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return series.std() * sqrt(TRADING_DAYS)


def sharpe_ratio(series: pd.Series, risk_free: float = 0.0) -> float:
    vol = annualized_volatility(series)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (annualized_return(series) - risk_free) / vol


def max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    wealth = (1 + series).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1
    return drawdown.min()


def downside_deviation(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    downside = np.minimum(series, 0)
    return np.sqrt((downside**2).mean()) * sqrt(TRADING_DAYS)


def sortino_ratio(series: pd.Series, risk_free: float = 0.0) -> float:
    dd = downside_deviation(series)
    if dd == 0 or np.isnan(dd):
        return np.nan
    return (annualized_return(series) - risk_free) / dd


def summary_stats(series: pd.Series) -> Dict[str, float]:
    return {
        "CAGR": annualized_return(series),
        "Volatility": annualized_volatility(series),
        "Sharpe": sharpe_ratio(series),
        "Sortino": sortino_ratio(series),
        "Max Drawdown": max_drawdown(series),
    }


def rolling_total_return(series: pd.Series, window: int = 63) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    return (1 + series).rolling(window).apply(np.prod, raw=True) - 1
