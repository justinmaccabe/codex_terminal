from __future__ import annotations

import numpy as np
import pandas as pd

from codex_terminal.config.universe import universe_by_ticker


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def build_spy_earnings_leash(spy_prices: pd.Series, earnings_history: pd.DataFrame) -> dict[str, object]:
    if spy_prices.empty or earnings_history.empty or "EPS" not in earnings_history.columns:
        return {"chart": pd.DataFrame(), "metrics": {}, "decomposition": pd.DataFrame()}

    monthly_price = spy_prices.dropna().resample("ME").last()
    aligned = pd.concat(
        [
            monthly_price.rename("SPY Price"),
            earnings_history["EPS"].rename("S&P 500 EPS"),
            earnings_history.get("Earnings Yield", pd.Series(dtype=float)).rename("Earnings Yield"),
        ],
        axis=1,
    ).dropna(subset=["SPY Price", "S&P 500 EPS"])

    if aligned.empty or len(aligned) < 24:
        return {"chart": pd.DataFrame(), "metrics": {}, "decomposition": pd.DataFrame()}

    chart = pd.DataFrame(
        {
            "SPY Price Index": aligned["SPY Price"] / aligned["SPY Price"].iloc[0] * 100.0,
            "Earnings Index": aligned["S&P 500 EPS"] / aligned["S&P 500 EPS"].iloc[0] * 100.0,
        },
        index=aligned.index,
    )

    price_growth_1y = aligned["SPY Price"].pct_change(12).iloc[-1]
    earnings_growth_1y = aligned["S&P 500 EPS"].pct_change(12).iloc[-1]
    leash_tension = chart["SPY Price Index"].iloc[-1] / chart["Earnings Index"].iloc[-1] - 1.0
    earnings_yield = aligned["Earnings Yield"].dropna().iloc[-1] if "Earnings Yield" in aligned and aligned["Earnings Yield"].dropna().size else np.nan
    valuation_change = price_growth_1y - earnings_growth_1y if pd.notna(price_growth_1y) and pd.notna(earnings_growth_1y) else np.nan

    decomposition = pd.DataFrame(
        [
            {"Window": "1Y", "Price Growth": price_growth_1y, "Earnings Growth": earnings_growth_1y, "Valuation Change Proxy": valuation_change},
            {
                "Window": "3Y",
                "Price Growth": aligned["SPY Price"].pct_change(36).iloc[-1] if len(aligned) > 36 else np.nan,
                "Earnings Growth": aligned["S&P 500 EPS"].pct_change(36).iloc[-1] if len(aligned) > 36 else np.nan,
                "Valuation Change Proxy": (
                    aligned["SPY Price"].pct_change(36).iloc[-1] - aligned["S&P 500 EPS"].pct_change(36).iloc[-1]
                    if len(aligned) > 36
                    else np.nan
                ),
            },
            {
                "Window": "5Y",
                "Price Growth": aligned["SPY Price"].pct_change(60).iloc[-1] if len(aligned) > 60 else np.nan,
                "Earnings Growth": aligned["S&P 500 EPS"].pct_change(60).iloc[-1] if len(aligned) > 60 else np.nan,
                "Valuation Change Proxy": (
                    aligned["SPY Price"].pct_change(60).iloc[-1] - aligned["S&P 500 EPS"].pct_change(60).iloc[-1]
                    if len(aligned) > 60
                    else np.nan
                ),
            },
        ]
    )

    metrics = {
        "Price Growth 1Y": price_growth_1y,
        "Earnings Growth 1Y": earnings_growth_1y,
        "Valuation Change Proxy 1Y": valuation_change,
        "Leash Tension": leash_tension,
        "Earnings Yield": earnings_yield,
    }
    return {"chart": chart, "metrics": metrics, "decomposition": decomposition}


def build_equity_fundamental_support(fundamental_snapshots: pd.DataFrame) -> pd.DataFrame:
    if fundamental_snapshots.empty:
        return pd.DataFrame()

    known = universe_by_ticker()
    frame = fundamental_snapshots[fundamental_snapshots["Ticker"].isin([ticker for ticker, asset in known.items() if asset.asset_class == "Equity"])].copy()
    if frame.empty:
        return pd.DataFrame()

    valuation_support = frame["Forward Earnings Yield"].fillna(frame["Trailing Earnings Yield"])
    earnings_growth = frame["Earnings Growth"].fillna(0.0)
    quality = frame["Return On Equity"].fillna(frame["Profit Margin"]).fillna(0.0)

    support = 0.50 * _zscore(earnings_growth.fillna(0.0)) + 0.35 * _zscore(valuation_support.fillna(0.0)) + 0.15 * _zscore(quality.fillna(0.0))
    frame["Fundamental Support"] = support
    frame["Support Signal"] = np.where(
        frame["Fundamental Support"] > 0.5,
        "Strong",
        np.where(frame["Fundamental Support"] < -0.5, "Weak", "Mixed"),
    )
    frame["Proxy"] = frame["Ticker"].map(lambda ticker: known[ticker].proxy_description if ticker in known else "unresolved")
    return frame[
        [
            "Ticker",
            "Proxy",
            "Earnings Growth",
            "Trailing PE",
            "Forward PE",
            "Trailing Earnings Yield",
            "Forward Earnings Yield",
            "Profit Margin",
            "Return On Equity",
            "Fundamental Support",
            "Support Signal",
        ]
    ].sort_values("Fundamental Support", ascending=False)


def fundamental_support_map(fundamental_support: pd.DataFrame) -> dict[str, float]:
    if fundamental_support.empty or "Ticker" not in fundamental_support.columns:
        return {}
    return fundamental_support.set_index("Ticker")["Fundamental Support"].fillna(0.0).to_dict()
