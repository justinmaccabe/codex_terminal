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
    score_portfolio,
)
from codex_terminal.config.universe import universe_by_ticker
from codex_terminal.portfolio.compare import portfolio_returns


STRATEGIC_CORE_WEIGHTS: Dict[str, float] = {
    "VTI": 0.10,
    "VBR": 0.11,
    "VEA": 0.07,
    "IVLU": 0.07,
    "VWO": 0.07,
    "AVES": 0.07,
    "VGIT": 0.07,
    "TLT": 0.07,
    "TIP": 0.08,
    "VNQ": 0.05,
    "PDBC": 0.07,
    "GLDM": 0.05,
    "WTMF": 0.09,
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
    financing_rate: float
    target_vol_series: pd.Series
    target_vol_stats: Dict[str, float]
    net_target_vol_series: pd.Series
    net_target_vol_stats: Dict[str, float]
    research_table: pd.DataFrame
    subperiod_table: pd.DataFrame
    diagnostics: List[str]


HOUSE_BENCHMARK_MODES = [
    "Strategic + Tactical",
    "Max Sharpe",
    "Risk Parity",
    "Blend",
]


SUBPERIOD_WINDOWS: Dict[str, tuple[str, str | None]] = {
    "Post-GFC Expansion": ("2010-01-01", "2019-12-31"),
    "Pandemic / Reopening": ("2020-01-01", "2021-12-31"),
    "Inflation Shock": ("2022-01-01", "2023-12-31"),
    "Recent Tape": ("2024-01-01", None),
}


def summarize_house_modes(
    asset_returns: pd.DataFrame,
    screener: pd.DataFrame,
    spy_returns: pd.Series,
    financing_rate: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for mode in HOUSE_BENCHMARK_MODES:
        model = build_market_beating_portfolio(
            asset_returns,
            screener,
            spy_returns,
            mode=mode,
            financing_rate=financing_rate,
        )
        gross = model.target_vol_stats or {}
        net = model.net_target_vol_stats or model.target_vol_stats or {}
        rows.append(
            {
                "Mode": mode,
                "Gross Sharpe": gross.get("Sharpe"),
                "Net Sharpe": net.get("Sharpe"),
                "Net CAGR": net.get("CAGR"),
                "Net Max Drawdown": net.get("Max Drawdown"),
                "Leverage": model.vol_target_leverage,
            }
        )
    return pd.DataFrame(rows)


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


def _optimize_around_anchor(
    asset_returns: pd.DataFrame,
    anchor: pd.DataFrame,
    seed: int,
    trials: int,
    max_shift: float,
) -> pd.DataFrame:
    names = [ticker for ticker in anchor["ticker"].tolist() if ticker in asset_returns.columns]
    if not names:
        return anchor[["ticker", "strategic_weight", "stance", "composite_percentile", "tactical_score", "macro_score", "composite_score", "weight"]]

    data = asset_returns[names].dropna(how="all")
    if data.empty:
        return anchor

    anchor_weights = anchor.set_index("ticker").loc[names, "weight"].astype(float).values
    rng = np.random.default_rng(seed)
    best_weights = anchor_weights.copy()
    best_score = score_portfolio(data.mul(best_weights, axis=1).sum(axis=1), data, best_weights)

    for _ in range(trials):
        noise = rng.normal(0.0, max_shift, len(names))
        trial = np.clip(anchor_weights + noise, 0.0, None)
        if trial.sum() <= 0:
            continue
        trial = trial / trial.sum()
        if np.max(np.abs(trial - anchor_weights)) > max_shift * 2.2:
            continue
        port = data.mul(trial, axis=1).sum(axis=1)
        score = score_portfolio(port, data, trial)
        if score > best_score:
            best_score = score
            best_weights = trial.copy()

    optimized = anchor.copy()
    optimized["weight"] = optimized["ticker"].map(dict(zip(names, best_weights)))
    optimized["weight"] = optimized["weight"].fillna(optimized["weight"])
    return _normalize_holdings(optimized)


def _apply_financing_drag(series: pd.Series, leverage: float, financing_rate: float) -> pd.Series:
    if series.empty or np.isnan(leverage):
        return pd.Series(dtype=float)
    borrowed = max(leverage - 1.0, 0.0)
    if borrowed <= 0 or financing_rate <= 0:
        return series * leverage
    daily_drag = borrowed * financing_rate / 252.0
    return series * leverage - daily_drag


def _build_research_table(holdings: pd.DataFrame) -> pd.DataFrame:
    known = universe_by_ticker()
    rows = []
    for row in holdings.itertuples(index=False):
        asset = known[row.ticker]
        if asset.diversifier and asset.asset_class in {"Rates", "Real Assets", "Alternatives", "Cash"}:
            role = "Diversifier"
        elif asset.style in {"Value", "Size", "Size + Value"}:
            role = "Structural return sleeve"
        elif asset.style in {"Momentum", "Trend"}:
            role = "Tactical / trend sleeve"
        else:
            role = "Core beta sleeve"

        rationale = (
            "Improves diversification and crisis balance."
            if role == "Diversifier"
            else "Raises long-run expected return through structural tilts."
            if role == "Structural return sleeve"
            else "Responds faster to regime and trend changes."
            if role == "Tactical / trend sleeve"
            else "Keeps the benchmark anchored to broad market participation."
        )
        rows.append(
            {
                "Ticker": row.ticker,
                "Proxy": asset.proxy_description,
                "Role": role,
                "Why It Is Here": rationale,
                "Strategic Weight": row.strategic_weight,
                "Current Weight": row.weight,
                "Tilt": row.tilt_vs_strategic,
                "Stance": row.stance,
            }
        )
    return pd.DataFrame(rows)


def _build_subperiod_table(
    house_series: pd.Series,
    spy_series: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    if house_series.empty or spy_series.empty:
        return pd.DataFrame()
    joined = pd.concat([house_series.rename("house"), spy_series.rename("spy")], axis=1).dropna()
    if joined.empty:
        return pd.DataFrame()
    for label, (start, end) in SUBPERIOD_WINDOWS.items():
        window = joined.loc[start:end] if end else joined.loc[start:]
        if window.empty:
            continue
        house_stats = summary_stats(window["house"])
        spy_stats = summary_stats(window["spy"])
        rows.append(
            {
                "Window": label,
                "House CAGR": house_stats.get("CAGR"),
                "SPY CAGR": spy_stats.get("CAGR"),
                "House Sharpe": house_stats.get("Sharpe"),
                "SPY Sharpe": spy_stats.get("Sharpe"),
                "Sharpe Spread": (house_stats.get("Sharpe") or np.nan) - (spy_stats.get("Sharpe") or np.nan),
                "CAGR Spread": (house_stats.get("CAGR") or np.nan) - (spy_stats.get("CAGR") or np.nan),
                "House Max Drawdown": house_stats.get("Max Drawdown"),
                "SPY Max Drawdown": spy_stats.get("Max Drawdown"),
            }
        )
    return pd.DataFrame(rows)


def _build_mode_holdings(
    asset_returns: pd.DataFrame,
    screener: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    base = _score_tilt_rows(screener)
    tickers = list(STRATEGIC_CORE_WEIGHTS.keys())

    if mode == "Strategic + Tactical":
        base["weight"] = base["bounded_weight"]
        tactical = _normalize_holdings(base.drop(columns=["raw_weight", "bounded_weight"]))
        return _optimize_around_anchor(asset_returns, tactical, seed=21, trials=2200, max_shift=0.035)

    if mode == "Risk Parity":
        rp = inverse_vol_weights(asset_returns, tickers).rename(columns={"weight": "mode_weight"})
        base = base.drop(columns=["raw_weight", "bounded_weight"])
        merged = base.merge(rp, on="ticker", how="left")
        merged["weight"] = merged["mode_weight"].fillna(merged["strategic_weight"])
        merged = merged.drop(columns=["mode_weight"])
        return _normalize_holdings(merged)

    if mode == "Max Sharpe":
        ms = random_search_optimize(asset_returns, tickers, trials=3500).rename(columns={"weight": "mode_weight"})
        base = base.drop(columns=["raw_weight", "bounded_weight"])
        merged = base.merge(ms, on="ticker", how="left")
        merged["weight"] = merged["mode_weight"].fillna(merged["strategic_weight"])
        merged = merged.drop(columns=["mode_weight"])
        return _optimize_around_anchor(asset_returns, _normalize_holdings(merged), seed=33, trials=1600, max_shift=0.025)

    # Blend
    tactical = _build_mode_holdings(asset_returns, screener, "Strategic + Tactical")
    max_sharpe = _build_mode_holdings(asset_returns, screener, "Max Sharpe")
    merged = tactical.merge(max_sharpe[["ticker", "weight"]], on="ticker", suffixes=("", "_ms"))
    merged["weight"] = 0.65 * merged["weight"] + 0.35 * merged["weight_ms"]
    merged = merged.drop(columns=["weight_ms"])
    return _optimize_around_anchor(asset_returns, _normalize_holdings(merged), seed=57, trials=1800, max_shift=0.02)


def _diagnostics(holdings: pd.DataFrame, stats: Dict[str, float], spy_stats: Dict[str, float]) -> List[str]:
    notes: List[str] = []
    top = holdings.iloc[0]
    if top["weight"] > 0.20:
        notes.append(f"Top sleeve concentration is elevated in {top['ticker']} at {top['weight']:.1%}.")
    if stats.get("Sharpe", np.nan) < spy_stats.get("Sharpe", np.nan):
        notes.append("The house benchmark still trails SPY on Sharpe; leverage is amplifying a weaker engine.")
    if stats.get("Max Drawdown", 0) < spy_stats.get("Max Drawdown", 0):
        notes.append("Drawdown profile is still harsher than SPY despite diversification.")
    if holdings["weight"].sum() > 0:
        diversifiers = holdings.loc[holdings["ticker"].map(lambda x: universe_by_ticker()[x].diversifier), "weight"].sum()
        if diversifiers < 0.30:
            notes.append("Diversifier weight is still light; the portfolio may be too equity-adjacent to deserve leverage.")
        else:
            notes.append(f"Diversifiers account for {diversifiers:.1%} of capital, which improves the odds that leverage is scaling a broader return engine.")
    if not notes:
        notes.append("Current construction is behaving reasonably versus SPY on the chosen sample.")
    return notes


def build_market_beating_portfolio(
    asset_returns: pd.DataFrame,
    screener: pd.DataFrame,
    spy_returns: pd.Series,
    mode: str = "Strategic + Tactical",
    financing_rate: float = 0.04,
) -> HousePortfolioModel:
    holdings = _build_mode_holdings(asset_returns, screener, mode)

    series = portfolio_returns(holdings[["ticker", "weight"]], asset_returns)
    stats = summary_stats(series) if not series.empty else {}
    spy_stats = summary_stats(spy_returns) if not spy_returns.empty else {}
    leverage = leverage_to_match_spy_vol(series, spy_returns) if not series.empty else np.nan
    leverage = float(np.clip(leverage, 0.75, 1.75)) if not np.isnan(leverage) else np.nan
    target_vol_series = series * leverage if not series.empty and not np.isnan(leverage) else pd.Series(dtype=float)
    target_vol_stats = summary_stats(target_vol_series) if not target_vol_series.empty else {}
    net_target_vol_series = _apply_financing_drag(series, leverage, financing_rate)
    net_target_vol_stats = summary_stats(net_target_vol_series) if not net_target_vol_series.empty else {}
    stress_table = compute_stress_table(series, spy_returns) if not series.empty else pd.DataFrame()
    diagnostics = _diagnostics(holdings, net_target_vol_stats or target_vol_stats or stats, spy_stats)
    research_table = _build_research_table(holdings)
    subperiod_table = _build_subperiod_table(net_target_vol_series if not net_target_vol_series.empty else target_vol_series, spy_returns)

    return HousePortfolioModel(
        mode=mode,
        holdings=holdings,
        series=series,
        stats=stats,
        stress_table=stress_table,
        vol_target_leverage=leverage,
        financing_rate=financing_rate,
        target_vol_series=target_vol_series,
        target_vol_stats=target_vol_stats,
        net_target_vol_series=net_target_vol_series,
        net_target_vol_stats=net_target_vol_stats,
        research_table=research_table,
        subperiod_table=subperiod_table,
        diagnostics=diagnostics,
    )
