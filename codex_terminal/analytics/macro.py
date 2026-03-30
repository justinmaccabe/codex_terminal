from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from codex_terminal.config.universe import universe_by_ticker


@dataclass
class MacroRegime:
    regime: str
    growth_direction: str
    inflation_direction: str
    confidence: str
    summary: str


@dataclass
class MacroSnapshot:
    label: str
    latest: float
    trailing_change: float
    direction: str


def _direction(series: pd.Series, lookback: int = 12) -> str:
    if series.empty or len(series.dropna()) < lookback + 1:
        return "Unknown"
    monthly = series.resample("M").last().dropna()
    if len(monthly) < lookback + 1:
        return "Unknown"
    delta = monthly.iloc[-1] - monthly.iloc[-1 - lookback]
    if np.isclose(delta, 0):
        return "Flat"
    return "Up" if delta > 0 else "Down"


def classify_regime(bundle: Dict[str, pd.Series]) -> MacroRegime:
    growth_series = bundle.get("INDPRO", pd.Series(dtype=float))
    inflation_series = bundle.get("CPIAUCSL", pd.Series(dtype=float))

    growth_direction = _direction(growth_series)
    inflation_direction = _direction(inflation_series)

    if growth_direction == "Unknown" or inflation_direction == "Unknown":
        return MacroRegime(
            regime="Unavailable",
            growth_direction=growth_direction,
            inflation_direction=inflation_direction,
            confidence="Low",
            summary="FRED data is unavailable or insufficient to classify the current regime.",
        )

    if growth_direction == "Up" and inflation_direction == "Up":
        regime = "Growth Up / Inflation Up"
    elif growth_direction == "Up" and inflation_direction == "Down":
        regime = "Growth Up / Inflation Down"
    elif growth_direction == "Down" and inflation_direction == "Up":
        regime = "Growth Down / Inflation Up"
    else:
        regime = "Growth Down / Inflation Down"

    confidence = "Medium" if "Flat" in {growth_direction, inflation_direction} else "High"
    summary = (
        f"Growth is {growth_direction.lower()} and inflation is {inflation_direction.lower()} "
        f"versus the trailing monthly baseline."
    )
    return MacroRegime(regime, growth_direction, inflation_direction, confidence, summary)


def macro_snapshots(bundle: Dict[str, pd.Series]) -> Dict[str, MacroSnapshot]:
    labels = {
        "INDPRO": "Industrial Production",
        "UNRATE": "Unemployment Rate",
        "CPIAUCSL": "CPI",
        "FEDFUNDS": "Fed Funds",
        "T10Y2Y": "10Y-2Y Curve",
        "BAMLH0A0HYM2": "High Yield OAS",
    }
    out: Dict[str, MacroSnapshot] = {}
    for series_id, label in labels.items():
        series = bundle.get(series_id, pd.Series(dtype=float))
        if series.empty:
            continue
        monthly = series.resample("ME").last().dropna()
        if monthly.empty:
            continue
        latest = float(monthly.iloc[-1])
        trailing = float(monthly.iloc[-1] - monthly.iloc[-2]) if len(monthly) > 1 else 0.0
        if np.isclose(trailing, 0):
            direction = "Flat"
        else:
            direction = "Up" if trailing > 0 else "Down"
        out[series_id] = MacroSnapshot(label=label, latest=latest, trailing_change=trailing, direction=direction)
    return out


def regime_implications(regime: str) -> Dict[str, list[str]]:
    mapping = {
        "Growth Up / Inflation Up": {
            "favored": ["commodities", "inflation_linked_bonds", "listed_real_estate", "managed_futures"],
            "challenged": ["long_treasuries", "intermediate_treasuries"],
            "narrative": [
                "Inflation-sensitive sleeves typically gain relative support.",
                "Duration-heavy assets usually face more pressure.",
            ],
        },
        "Growth Up / Inflation Down": {
            "favored": ["us_large_cap_beta", "us_momentum", "international_developed_equity"],
            "challenged": ["gold", "broad_commodities"],
            "narrative": [
                "Risk assets tend to benefit when growth holds and inflation cools.",
                "Pure inflation hedges often lose urgency in this regime.",
            ],
        },
        "Growth Down / Inflation Up": {
            "favored": ["managed_futures", "gold", "broad_commodities", "inflation_linked_bonds"],
            "challenged": ["us_large_cap_beta", "us_total_market", "international_developed_equity"],
            "narrative": [
                "This is the most uncomfortable regime for broad equities and duration together.",
                "Diversifiers and inflation hedges matter most here.",
            ],
        },
        "Growth Down / Inflation Down": {
            "favored": ["long_treasuries", "intermediate_treasuries", "cash_ultra_short"],
            "challenged": ["broad_commodities", "listed_real_estate"],
            "narrative": [
                "Duration typically becomes more useful as growth weakens and inflation fades.",
                "Real-asset sleeves tend to lose cyclical support.",
            ],
        },
    }
    return mapping.get(regime, {"favored": [], "challenged": [], "narrative": []})


def regime_fit_scores(regime: str, tickers: list[str]) -> pd.Series:
    known = universe_by_ticker()
    scores = {}
    for ticker in tickers:
        asset = known.get(ticker)
        if asset is None:
            scores[ticker] = 0.0
            continue
        score = 0.0
        if regime == "Growth Up / Inflation Up":
            if asset.asset_class == "Real Assets" or asset.inflation_sensitive:
                score += 1.0
            if asset.asset_class == "Rates" and asset.duration_sensitive:
                score -= 0.5
        elif regime == "Growth Up / Inflation Down":
            if asset.asset_class == "Equity":
                score += 1.0
            if asset.style in {"Momentum", "Beta"}:
                score += 0.25
        elif regime == "Growth Down / Inflation Up":
            if asset.inflation_sensitive:
                score += 0.75
            if asset.asset_class == "Equity":
                score -= 0.75
        elif regime == "Growth Down / Inflation Down":
            if asset.duration_sensitive:
                score += 1.0
            if asset.asset_class == "Equity":
                score -= 0.25
        scores[ticker] = score
    return pd.Series(scores, dtype=float)
