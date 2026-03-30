from __future__ import annotations

import pandas as pd


def normalize_portfolio_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = {col.lower().strip(): col for col in frame.columns}
    if "ticker" not in cols or "weight" not in cols:
        raise ValueError("Portfolio upload must include 'ticker' and 'weight' columns.")

    out = frame.rename(columns={cols["ticker"]: "ticker", cols["weight"]: "weight"})[
        ["ticker", "weight"]
    ].copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out = out[out["ticker"] != ""]
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.dropna(subset=["weight"])

    total = out["weight"].sum()
    if total > 1.5:
        out["weight"] = out["weight"] / 100.0
    total = out["weight"].sum()
    if total != 0:
        out["weight"] = out["weight"] / total
    return out.groupby("ticker", as_index=False)["weight"].sum()


def portfolio_returns(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.Series:
    aligned = weights.set_index("ticker")["weight"]
    available = [ticker for ticker in aligned.index if ticker in asset_returns.columns]
    if not available:
        return pd.Series(dtype=float)
    return asset_returns[available].mul(aligned.loc[available], axis=1).sum(axis=1)


def compare_stats(left: pd.Series, right: pd.Series) -> pd.DataFrame:
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if joined.empty:
        return pd.DataFrame()
    correlation = joined["left"].corr(joined["right"])
    diff = (1 + joined["left"]).prod() - (1 + joined["right"]).prod()
    tracking_error = (joined["left"] - joined["right"]).std() * (252**0.5)
    beta_denom = joined["right"].var()
    beta = joined["left"].cov(joined["right"]) / beta_denom if beta_denom and not pd.isna(beta_denom) else float("nan")
    return pd.DataFrame(
        {
            "Metric": ["Correlation", "Beta", "Tracking Error", "Cumulative Return Spread"],
            "Value": [correlation, beta, tracking_error, diff],
        }
    )
