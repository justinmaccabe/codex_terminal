from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class FactorAttribution:
    exposures: pd.DataFrame
    r_squared: float
    annualized_alpha: float


def build_factor_proxy_returns(asset_returns: pd.DataFrame) -> pd.DataFrame:
    factors = pd.DataFrame(index=asset_returns.index)
    if "SPY" in asset_returns:
        factors["MKT"] = asset_returns["SPY"]
    if {"VBR", "VTI"}.issubset(asset_returns.columns):
        factors["SMB"] = asset_returns["VBR"] - asset_returns["VTI"]
    if {"VTV", "VTI"}.issubset(asset_returns.columns):
        factors["HML"] = asset_returns["VTV"] - asset_returns["VTI"]
    if {"MTUM", "VTI"}.issubset(asset_returns.columns):
        factors["MOM"] = asset_returns["MTUM"] - asset_returns["VTI"]
    if {"WTMF", "SPY"}.issubset(asset_returns.columns):
        factors["TF"] = asset_returns["WTMF"] - asset_returns["SPY"]
    return factors.dropna(how="all")


def compute_factor_attribution(
    portfolio_returns: pd.Series,
    asset_returns: pd.DataFrame,
    freq: str = "M",
) -> FactorAttribution | None:
    if portfolio_returns.empty or asset_returns.empty:
        return None

    factor_returns = build_factor_proxy_returns(asset_returns)
    if factor_returns.empty:
        return None

    portfolio = portfolio_returns.dropna()
    if freq == "M":
        portfolio = (1 + portfolio).resample("ME").prod() - 1
        factor_returns = (1 + factor_returns).resample("ME").prod() - 1

    joined = pd.concat([portfolio.rename("portfolio"), factor_returns], axis=1).dropna()
    if joined.empty or len(joined) < 12:
        return None

    y = joined["portfolio"].values
    X = joined[factor_returns.columns].values
    X = np.column_stack([np.ones(len(X)), X])

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residual = y - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    alpha_monthly = float(beta[0])
    annualized_alpha = (1 + alpha_monthly) ** 12 - 1 if not np.isnan(alpha_monthly) else np.nan

    exposures = pd.DataFrame(
        {
            "Factor": ["Alpha"] + list(factor_returns.columns),
            "Exposure": [alpha_monthly] + list(beta[1:]),
        }
    )
    return FactorAttribution(exposures=exposures, r_squared=r_squared, annualized_alpha=annualized_alpha)
