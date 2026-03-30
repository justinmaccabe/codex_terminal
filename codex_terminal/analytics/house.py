from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from codex_terminal.analytics.metrics import summary_stats
from codex_terminal.analytics.portfolio import (
    compute_stress_table,
    inverse_vol_weights,
    leverage_to_match_spy_vol,
    random_search_optimize,
)
from codex_terminal.config.universe import universe_by_ticker
from codex_terminal.portfolio.compare import portfolio_returns


STRATEGIC_CORE_WEIGHTS: Dict[str, float] = {
    "VTI": 0.14,
    "VBR": 0.09,
    "VEA": 0.10,
    "IVLU": 0.05,
    "VWO": 0.08,
    "AVES": 0.04,
    "VGIT": 0.11,
    "TLT": 0.08,
    "TIP": 0.08,
    "VNQ": 0.05,
    "PDBC": 0.07,
    "GLDM": 0.05,
    "WTMF": 0.05,
    "SGOV": 0.01,
}


TACTICAL_MAP = {
    "Overweight": 1.22,
    "Watchlist": 1.10,
    "Neutral": 1.00,
    "Underweight": 0.85,
    "Avoid": 0.70,
}


@dataclass
class HousePortfolioModel:
    mode: str
    holdings: pd.DataFrame
    series: pd.Series
    stats: Dict[str, float]
    stress_table: pd.DataFrame
    vol_target_leverage: float
    target_vol_series: pd.Series
    target_vol_stats: Dict[str, float]
    diagnostics: List[str]


HOUSE_BENCHMARK_MODES = [
    "Strategic + Tactical",
    "Max Sharpe",
    "Risk Parity",
    "Blend",
]


def _score_tilt_rows(screener: pd.DataFrame) -> pd.DataFrame:
    known = universe_by_ticker()
    rows = []
    for ticker, strategic_weight in STRATEGIC_CORE_WEIGHTS.items():
        stance = "Neutral"
        percentile = 0.5
        macro_score = 0.0
        tactical_score = 0.0
        composite_score = 0.0
        if ticker in screener.index:
            stance = str(screener.at[ticker, "Stance"])
            percentile = float(screener.at[ticker, "Composite Percentile"])
            macro_score = float(screener.at[ticker, "Macro Score"])
            tactical_score = float(screener.at[ticker, "Tactical Score"])
            composite_score = float(screener.at[ticker, "Composite Score"])

        asset = known[ticker]
        tactical_multiplier = TACTICAL_MAP.get(stance, 1.0) if asset.tactical_overlay else 1.0
        percentile_tilt = 1.0 + 0.18 * (percentile - 0.5)
        macro_tilt = 1.0 + 0.08 * np.clip(macro_score, -1.0, 1.0)
        style_bias = 1.0
        if asset.style in {"Value", "Size + Value"}:
            style_bias += 0.03
        if asset.style in {"Trend", "Momentum"} and percentile > 0.65:
            style_bias += 0.04

        raw_weight = strategic_weight * tactical_multiplier * percentile_tilt * macro_tilt * style_bias
        floor_weight = strategic_weight * (0.55 if asset.tactical_overlay else 0.80)
        cap_weight = max(strategic_weight * (1.65 if asset.tactical_overlay else 1.20), 0.03)
        bounded_weight = min(max(raw_weight, floor_weight), cap_weight)

        rows.append(
            {
                "ticker": ticker,
                "strategic_weight": strategic_weight,
                "stance": stance,
                "composite_percentile": percentile,
                "tactical_score": tactical_score,
                "macro_score": macro_score,
                "composite_score": composite_score,
                "raw_weight": raw_weight,
                "bounded_weight": bounded_weight,
            }
        )
    return pd.DataFrame(rows)


def _normalize_holdings(frame: pd.DataFrame) -> pd.DataFrame:
    holdings = frame.copy()
    holdings["weight"] = holdings["weight"] / holdings["weight"].sum()
    holdings["tilt_vs_strategic"] = holdings["weight"] - holdings["strategic_weight"]
    return holdings.sort_values("weight", ascending=False).reset_index(drop=True)


def _build_mode_holdings(
    asset_returns: pd.DataFrame,
    screener: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    base = _score_tilt_rows(screener)
    tickers = list(STRATEGIC_CORE_WEIGHTS.keys())

    if mode == "Strategic + Tactical":
        base["weight"] = base["bounded_weight"]
        return _normalize_holdings(base.drop(columns=["raw_weight", "bounded_weight"]))

    if mode == "Risk Parity":
        rp = inverse_vol_weights(asset_returns, tickers)
        base = base.drop(columns=["raw_weight", "bounded_weight"])
        merged = base.merge(rp, on="ticker", how="left", suffixes=("", "_new"))
        merged["weight"] = merged["weight_new"].fillna(merged["strategic_weight"])
        merged = merged.drop(columns=["weight_new"])
        return _normalize_holdings(merged)

    if mode == "Max Sharpe":
        ms = random_search_optimize(asset_returns, tickers, trials=3500)
        base = base.drop(columns=["raw_weight", "bounded_weight"])
        merged = base.merge(ms, on="ticker", how="left", suffixes=("", "_new"))
        merged["weight"] = merged["weight_new"].fillna(merged["strategic_weight"])
        merged = merged.drop(columns=["weight_new"])
        return _normalize_holdings(merged)

    # Blend
    tactical = _build_mode_holdings(asset_returns, screener, "Strategic + Tactical")
    max_sharpe = _build_mode_holdings(asset_returns, screener, "Max Sharpe")
    merged = tactical.merge(max_sharpe[["ticker", "weight"]], on="ticker", suffixes=("", "_ms"))
    merged["weight"] = 0.55 * merged["weight"] + 0.45 * merged["weight_ms"]
    merged = merged.drop(columns=["weight_ms"])
    return _normalize_holdings(merged)


def _diagnostics(holdings: pd.DataFrame, stats: Dict[str, float], spy_stats: Dict[str, float]) -> List[str]:
    notes: List[str] = []
    top = holdings.iloc[0]
    if top["weight"] > 0.20:
        notes.append(f"Top sleeve concentration is elevated in {top['ticker']} at {top['weight']:.1%}.")
    if stats.get("Sharpe", np.nan) < spy_stats.get("Sharpe", np.nan):
        notes.append("The house benchmark still trails SPY on Sharpe; leverage is amplifying a weaker engine.")
    if stats.get("Max Drawdown", 0) < spy_stats.get("Max Drawdown", 0):
        notes.append("Drawdown profile is still harsher than SPY despite diversification.")
    if not notes:
        notes.append("Current construction is behaving reasonably versus SPY on the chosen sample.")
    return notes


def build_market_beating_portfolio(
    asset_returns: pd.DataFrame,
    screener: pd.DataFrame,
    spy_returns: pd.Series,
    mode: str = "Strategic + Tactical",
) -> HousePortfolioModel:
    holdings = _build_mode_holdings(asset_returns, screener, mode)

    series = portfolio_returns(holdings[["ticker", "weight"]], asset_returns)
    stats = summary_stats(series) if not series.empty else {}
    spy_stats = summary_stats(spy_returns) if not spy_returns.empty else {}
    leverage = leverage_to_match_spy_vol(series, spy_returns) if not series.empty else np.nan
    leverage = float(np.clip(leverage, 0.75, 1.75)) if not np.isnan(leverage) else np.nan
    target_vol_series = series * leverage if not series.empty and not np.isnan(leverage) else pd.Series(dtype=float)
    target_vol_stats = summary_stats(target_vol_series) if not target_vol_series.empty else {}
    stress_table = compute_stress_table(series, spy_returns) if not series.empty else pd.DataFrame()
    diagnostics = _diagnostics(holdings, target_vol_stats or stats, spy_stats)

    return HousePortfolioModel(
        mode=mode,
        holdings=holdings,
        series=series,
        stats=stats,
        stress_table=stress_table,
        vol_target_leverage=leverage,
        target_vol_series=target_vol_series,
        target_vol_stats=target_vol_stats,
        diagnostics=diagnostics,
    )
