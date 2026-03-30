from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from codex_terminal.analytics.metrics import summary_stats


STRESS_WINDOWS: Dict[str, tuple[str, str]] = {
    "GFC": ("2008-09-01", "2009-03-31"),
    "COVID Crash": ("2020-02-15", "2020-04-30"),
    "Inflation Shock": ("2022-01-01", "2022-10-31"),
}


@dataclass
class OptimizationResult:
    weights: pd.DataFrame
    stats: Dict[str, float]
    leverage_to_spy_vol: float
    stress_table: pd.DataFrame


def equal_weight_portfolio(tickers: Iterable[str]) -> pd.DataFrame:
    names = list(dict.fromkeys(tickers))
    if not names:
        return pd.DataFrame(columns=["ticker", "weight"])
    weight = 1.0 / len(names)
    return pd.DataFrame({"ticker": names, "weight": [weight] * len(names)})


def inverse_vol_weights(asset_returns: pd.DataFrame, tickers: Iterable[str], lookback: int = 63) -> pd.DataFrame:
    names = [ticker for ticker in dict.fromkeys(tickers) if ticker in asset_returns.columns]
    if not names:
        return pd.DataFrame(columns=["ticker", "weight"])
    vol = asset_returns[names].tail(lookback).std()
    if vol.empty or vol.fillna(0).sum() <= 0:
        return equal_weight_portfolio(names)
    adjusted = vol.clip(lower=1e-6)
    inv = (1.0 / adjusted).replace([np.inf, -np.inf], np.nan).fillna(0)
    if inv.sum() == 0:
        return equal_weight_portfolio(names)
    weights = inv / inv.sum()
    return pd.DataFrame({"ticker": weights.index, "weight": weights.values})


def score_portfolio(
    portfolio_returns: pd.Series,
    asset_returns: pd.DataFrame | None = None,
    weights: np.ndarray | None = None,
) -> float:
    stats = summary_stats(portfolio_returns)
    sharpe = stats["Sharpe"] if not np.isnan(stats["Sharpe"]) else -1.0
    sortino = stats["Sortino"] if not np.isnan(stats["Sortino"]) else -1.0
    drawdown = stats["Max Drawdown"] if not np.isnan(stats["Max Drawdown"]) else -0.5
    cagr = stats["CAGR"] if not np.isnan(stats["CAGR"]) else -0.25

    stress_penalty = 0.0
    worst_stress = np.nan
    if asset_returns is not None and "SPY" in asset_returns:
        stress = compute_stress_table(portfolio_returns, asset_returns["SPY"].dropna())
        if not stress.empty:
            worst_stress = stress["Portfolio Return"].min()
            if not pd.isna(worst_stress):
                stress_penalty = abs(min(worst_stress, 0.0)) * 1.3

    concentration_penalty = 0.0
    correlation_penalty = 0.0
    if weights is not None and len(weights) > 0:
        concentration_penalty = float(np.square(weights).sum()) * 0.55
        if asset_returns is not None and asset_returns.shape[1] > 1:
            corr = asset_returns.corr().fillna(0.0).values
            correlation_penalty = float(weights @ corr @ weights) * 0.18

    drawdown_penalty = abs(min(drawdown, 0.0)) * 1.6
    downside_bonus = max(cagr, -0.25) * 0.35

    return (
        sharpe
        + 0.55 * sortino
        + downside_bonus
        - drawdown_penalty
        - stress_penalty
        - concentration_penalty
        - correlation_penalty
    )


def random_search_optimize(
    asset_returns: pd.DataFrame,
    tickers: Iterable[str],
    trials: int = 2500,
    seed: int = 7,
) -> pd.DataFrame:
    names = [ticker for ticker in dict.fromkeys(tickers) if ticker in asset_returns.columns]
    if not names:
        return pd.DataFrame(columns=["ticker", "weight"])
    rng = np.random.default_rng(seed)
    data = asset_returns[names].dropna(how="all")
    if data.empty:
        return equal_weight_portfolio(names)

    best_score = -np.inf
    best_weights = None
    for _ in range(trials):
        w = rng.dirichlet(np.ones(len(names)))
        port = data.mul(w, axis=1).sum(axis=1)
        score = score_portfolio(port, data, w)
        if score > best_score:
            best_score = score
            best_weights = w

    if best_weights is None:
        return equal_weight_portfolio(names)
    return pd.DataFrame({"ticker": names, "weight": best_weights})


def compute_portfolio_series(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.Series:
    if weights.empty or asset_returns.empty:
        return pd.Series(dtype=float)
    aligned = weights.set_index("ticker")["weight"]
    available = [ticker for ticker in aligned.index if ticker in asset_returns.columns]
    if not available:
        return pd.Series(dtype=float)
    return asset_returns[available].mul(aligned.loc[available], axis=1).sum(axis=1)


def compute_stress_table(portfolio_returns: pd.Series, spy_returns: pd.Series) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for name, (start, end) in STRESS_WINDOWS.items():
        port_slice = portfolio_returns.loc[start:end]
        spy_slice = spy_returns.loc[start:end]
        rows.append(
            {
                "Window": name,
                "Portfolio Return": (1 + port_slice).prod() - 1 if not port_slice.empty else np.nan,
                "SPY Return": (1 + spy_slice).prod() - 1 if not spy_slice.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def leverage_to_match_spy_vol(portfolio_returns: pd.Series, spy_returns: pd.Series) -> float:
    port_vol = summary_stats(portfolio_returns)["Volatility"]
    spy_vol = summary_stats(spy_returns)["Volatility"]
    if np.isnan(port_vol) or np.isnan(spy_vol) or port_vol <= 0:
        return np.nan
    return spy_vol / port_vol
