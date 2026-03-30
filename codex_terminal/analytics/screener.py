from __future__ import annotations

import numpy as np
import pandas as pd

from codex_terminal.analytics.macro import regime_fit_scores
from codex_terminal.config.universe import universe_by_ticker


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _structural_signal_frame(columns: list[str]) -> pd.DataFrame:
    known = universe_by_ticker()
    rows = {}
    for ticker in columns:
        asset = known.get(ticker)
        value = 0.0
        carry = 0.0
        expected = 0.0
        if asset is not None:
            if "Value" in asset.style:
                value += 1.0
                expected += 0.4
            if asset.style in {"Cash", "Short Duration", "Intermediate Duration", "Long Duration", "Aggregate"}:
                carry += 0.4
            if asset.style == "Inflation-Linked":
                expected += 0.2
            if asset.style in {"Momentum", "Trend"}:
                expected += 0.2
        rows[ticker] = {
            "value_context": value,
            "carry_context": carry,
            "expected_return_context": expected,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _dynamic_structural_scores(clean: pd.DataFrame) -> pd.DataFrame:
    one_year = clean.pct_change(252).iloc[-1] if len(clean) > 252 else pd.Series(0.0, index=clean.columns)
    three_year = clean.pct_change(756).iloc[-1] if len(clean) > 756 else one_year * 0.0
    drawdown = clean.iloc[-1] / clean.cummax().iloc[-1] - 1
    value_reversion = _zscore((-three_year).fillna(0.0) + (-drawdown).fillna(0.0))
    expected_context = _zscore(((-three_year).fillna(0.0) * 0.6) + (one_year.fillna(0.0) * 0.4))
    return pd.DataFrame(
        {
            "dynamic_value_context": value_reversion,
            "dynamic_expected_return_context": expected_context,
        }
    )


def compute_screener_scores(prices: pd.DataFrame, regime: str = "Unavailable") -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    clean = prices.sort_index().ffill()
    last = clean.iloc[-1]
    ret_21 = clean.iloc[-1] / clean.iloc[-22].replace(0, np.nan) - 1 if len(clean) > 21 else pd.Series(np.nan, index=clean.columns)
    ret_63 = clean.iloc[-1] / clean.iloc[-64].replace(0, np.nan) - 1 if len(clean) > 63 else pd.Series(np.nan, index=clean.columns)
    ret_126 = clean.iloc[-1] / clean.iloc[-127].replace(0, np.nan) - 1 if len(clean) > 126 else pd.Series(np.nan, index=clean.columns)
    ret_252 = clean.iloc[-1] / clean.iloc[-253].replace(0, np.nan) - 1 if len(clean) > 252 else pd.Series(np.nan, index=clean.columns)
    vol_63 = clean.pct_change().rolling(63).std().iloc[-1] * np.sqrt(252)
    ma_50 = clean.rolling(50).mean().iloc[-1]
    ma_200 = clean.rolling(200).mean().iloc[-1]

    absolute_momentum = ret_126.fillna(ret_63)
    relative_momentum = absolute_momentum.rank(pct=True)
    trend = ((last > ma_50).astype(float) + (last > ma_200).astype(float)) / 2.0
    vol_adjusted = (ret_63 / vol_63.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    tactical_block = pd.concat(
        [
            _zscore(absolute_momentum).rename("absolute_momentum"),
            _zscore(relative_momentum).rename("relative_momentum"),
            _zscore(trend).rename("trend"),
            _zscore(vol_adjusted.fillna(0)).rename("vol_adjusted_strength"),
        ],
        axis=1,
    )
    structural_block = _structural_signal_frame(list(prices.columns)).join(_dynamic_structural_scores(clean), how="left")
    structural_block["value_context"] = structural_block[["value_context", "dynamic_value_context"]].mean(axis=1)
    structural_block["expected_return_context"] = structural_block[
        ["expected_return_context", "dynamic_expected_return_context"]
    ].mean(axis=1)
    structural_block = structural_block[["value_context", "carry_context", "expected_return_context"]]
    macro_block = pd.DataFrame({"macro_fit": regime_fit_scores(regime, list(prices.columns))}, index=prices.columns)

    tactical_score = tactical_block.mean(axis=1)
    structural_score = structural_block.mean(axis=1)
    macro_score = macro_block.mean(axis=1)
    composite = 0.6 * tactical_score + 0.3 * structural_score + 0.1 * macro_score
    percentile = composite.rank(pct=True)

    stance = pd.Series("Neutral", index=prices.columns)
    stance[percentile >= 0.8] = "Overweight"
    stance[percentile <= 0.2] = "Avoid"
    stance[(percentile > 0.2) & (percentile < 0.4)] = "Underweight"
    watchlist_mask = (percentile >= 0.65) & (trend > 0) & (relative_momentum < 0.8)
    stance[watchlist_mask & (stance == "Neutral")] = "Watchlist"

    result = pd.DataFrame(
        {
            "1M": ret_21,
            "3M": ret_63,
            "6M": ret_126,
            "12M": ret_252,
            "Absolute Momentum": absolute_momentum,
            "Relative Momentum": relative_momentum,
            "Trend": trend,
            "Vol-Adjusted Strength": vol_adjusted,
            "Tactical Score": tactical_score,
            "Structural Score": structural_score,
            "Macro Score": macro_score,
            "Composite Score": composite,
            "Composite Percentile": percentile,
            "Stance": stance,
        }
    )
    return result.sort_values("Composite Score", ascending=False)
