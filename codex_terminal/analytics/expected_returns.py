from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from codex_terminal.config.universe import universe_by_ticker


@dataclass(frozen=True)
class SleeveExpectedReturn:
    ticker: str
    proxy: str
    structural_base: float
    market_adjustment: float
    factor_adjustment: float
    crisis_adjustment: float
    expected_return: float
    confidence: float
    rationale: str


def _latest_value(bundle: Dict[str, pd.Series], series_id: str, default: float = 0.0) -> float:
    series = bundle.get(series_id, pd.Series(dtype=float))
    if series.empty:
        return default
    value = pd.to_numeric(series.dropna(), errors="coerce")
    if value.empty:
        return default
    return float(value.iloc[-1])


def _three_year_total_return(prices: pd.DataFrame, ticker: str) -> float:
    if ticker not in prices.columns or len(prices) < 756:
        return 0.0
    clean = prices[ticker].dropna()
    if len(clean) < 756:
        return 0.0
    return float(clean.iloc[-1] / clean.iloc[-757] - 1)


def build_expected_return_table(
    prices: pd.DataFrame,
    fred_bundle: Dict[str, pd.Series],
) -> pd.DataFrame:
    known = universe_by_ticker()
    fed_funds = _latest_value(fred_bundle, "FEDFUNDS", 4.0) / 100.0
    curve = _latest_value(fred_bundle, "T10Y2Y", 0.0) / 100.0
    hy_oas = _latest_value(fred_bundle, "BAMLH0A0HYM2", 4.0) / 100.0
    inflation = _latest_value(fred_bundle, "CPIAUCSL", 300.0)
    inflation_series = fred_bundle.get("CPIAUCSL", pd.Series(dtype=float))
    inflation_yoy = 0.02
    if not inflation_series.empty and len(inflation_series.dropna()) > 12:
        monthly = inflation_series.resample("ME").last().dropna()
        if len(monthly) > 12 and monthly.iloc[-13] != 0:
            inflation_yoy = float(monthly.iloc[-1] / monthly.iloc[-13] - 1)

    rows: list[SleeveExpectedReturn] = []
    for ticker, asset in known.items():
        trailing = _three_year_total_return(prices, ticker)
        structural_base = 0.04
        market_adjustment = 0.0
        factor_adjustment = 0.0
        crisis_adjustment = 0.0
        confidence = 0.50
        rationale_parts: list[str] = []

        if asset.asset_class == "Equity":
            structural_base = 0.06
            confidence = 0.58
            rationale_parts.append("equity risk premium baseline")
            if asset.style in {"Value", "Size", "Size + Value"}:
                factor_adjustment += 0.015
                confidence += 0.05
                rationale_parts.append("size/value premium")
            if asset.region == "Emerging":
                factor_adjustment += 0.010
                confidence -= 0.02
                rationale_parts.append("emerging premium")
            if trailing < 0:
                market_adjustment += min(abs(trailing) * 0.015, 0.015)
                rationale_parts.append("valuation mean reversion tailwind")
            else:
                market_adjustment -= min(trailing * 0.006, 0.010)
                rationale_parts.append("recent strength trims forward return")
        elif asset.asset_class == "Rates":
            structural_base = fed_funds
            confidence = 0.68
            rationale_parts.append("yield-led fixed-income baseline")
            if asset.style == "Long Duration":
                market_adjustment += max(-curve, 0.0) * 0.50
                crisis_adjustment += 0.004
                rationale_parts.append("long duration helps when growth breaks")
            elif asset.style == "Intermediate Duration":
                market_adjustment += max(fed_funds - 0.02, 0.0) * 0.20
            elif asset.style == "Inflation-Linked":
                structural_base = max(fed_funds - inflation_yoy, 0.0) + inflation_yoy
                market_adjustment += max(inflation_yoy - 0.02, 0.0) * 0.35
                rationale_parts.append("inflation linkage")
            elif asset.style == "Cash":
                structural_base = fed_funds
                confidence = 0.80
        elif asset.asset_class == "Real Assets":
            structural_base = 0.03
            confidence = 0.42
            rationale_parts.append("real asset structural return")
            if asset.style == "Commodities":
                market_adjustment += max(inflation_yoy - 0.02, 0.0) * 0.80
                market_adjustment += hy_oas * 0.08
                crisis_adjustment += 0.002
                rationale_parts.append("inflation and carry support")
            elif asset.style == "Precious Metals":
                structural_base = 0.02
                market_adjustment += max(-curve, 0.0) * 0.25
                crisis_adjustment += 0.006
                rationale_parts.append("stress hedge role")
            elif asset.style == "Real Estate":
                structural_base = 0.05
                market_adjustment -= max(fed_funds - 0.03, 0.0) * 0.30
                rationale_parts.append("rate-sensitive income asset")
        elif asset.asset_class == "Alternatives":
            structural_base = 0.055
            confidence = 0.46
            factor_adjustment += 0.010
            crisis_adjustment += 0.008
            rationale_parts.append("trend/carry alternative return stream")
        elif asset.asset_class == "Cash":
            structural_base = fed_funds
            confidence = 0.82
            rationale_parts.append("cash carry baseline")

        expected_return = structural_base + market_adjustment + factor_adjustment + crisis_adjustment
        confidence = float(np.clip(confidence, 0.20, 0.90))
        rows.append(
            SleeveExpectedReturn(
                ticker=ticker,
                proxy=asset.proxy_description,
                structural_base=structural_base,
                market_adjustment=market_adjustment,
                factor_adjustment=factor_adjustment,
                crisis_adjustment=crisis_adjustment,
                expected_return=expected_return,
                confidence=confidence,
                rationale=", ".join(dict.fromkeys(rationale_parts)) or "baseline estimate",
            )
        )

    frame = pd.DataFrame([row.__dict__ for row in rows])
    if frame.empty:
        return frame
    std = frame["expected_return"].std(ddof=0)
    if std and not np.isclose(std, 0.0):
        frame["expected_return_score"] = (frame["expected_return"] - frame["expected_return"].mean()) / std
    else:
        frame["expected_return_score"] = 0.0
    return frame.sort_values("expected_return", ascending=False).reset_index(drop=True)
