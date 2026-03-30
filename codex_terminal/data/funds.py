from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd
import streamlit as st
import yfinance as yf


@dataclass(frozen=True)
class FundProfile:
    ticker: str
    name: str
    quote_type: str
    currency: str
    country: str
    exchange: str
    category: str
    family: str
    expense_ratio: float | None
    yield_pct: float | None
    total_assets: float | None
    investment_style: str
    summary: str
    is_canadian_mutual_fund: bool


def _normalize_ratio(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric > 1:
        numeric /= 100.0
    return numeric


def _build_profile(ticker: str, info: dict) -> FundProfile:
    quote_type = str(info.get("quoteType") or "Unknown")
    currency = str(info.get("currency") or "Unknown")
    country = str(info.get("country") or "Unknown")
    exchange = str(info.get("exchange") or info.get("fullExchangeName") or "Unknown")
    is_canadian_mutual_fund = (
        quote_type.upper() == "MUTUALFUND"
        and (
            currency.upper() == "CAD"
            or country.lower() == "canada"
            or ".CF" in ticker
        )
    )
    return FundProfile(
        ticker=ticker,
        name=str(
            info.get("longName")
            or info.get("shortName")
            or info.get("displayName")
            or ticker
        ),
        quote_type=quote_type,
        currency=currency,
        country=country,
        exchange=exchange,
        category=str(info.get("category") or info.get("fundCategory") or "Unavailable"),
        family=str(info.get("fundFamily") or info.get("family") or "Unavailable"),
        expense_ratio=_normalize_ratio(
            info.get("annualReportExpenseRatio")
            or info.get("netExpenseRatio")
            or info.get("annualHoldingsTurnover")
        ),
        yield_pct=_normalize_ratio(info.get("yield") or info.get("trailingAnnualDividendYield")),
        total_assets=(
            float(info["totalAssets"])
            if info.get("totalAssets") not in (None, "")
            else None
        ),
        investment_style=str(
            info.get("legalType")
            or info.get("investmentType")
            or info.get("fundInceptionDate")
            or "Unavailable"
        ),
        summary=str(
            info.get("longBusinessSummary")
            or info.get("summary")
            or info.get("fundProfile")
            or "No summary available."
        ),
        is_canadian_mutual_fund=is_canadian_mutual_fund,
    )


@st.cache_data(ttl=60 * 60 * 6)
def fetch_fund_profiles(tickers: Iterable[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for raw_ticker in tickers:
        ticker = str(raw_ticker).upper().strip()
        if not ticker:
            continue
        try:
            info = yf.Ticker(ticker).info or {}
            profile = _build_profile(ticker, info)
        except Exception:
            profile = FundProfile(
                ticker=ticker,
                name=ticker,
                quote_type="Unavailable",
                currency="Unavailable",
                country="Unavailable",
                exchange="Unavailable",
                category="Unavailable",
                family="Unavailable",
                expense_ratio=None,
                yield_pct=None,
                total_assets=None,
                investment_style="Unavailable",
                summary="Profile unavailable from Yahoo Finance.",
                is_canadian_mutual_fund=False,
            )
        rows.append(asdict(profile))
    return pd.DataFrame(rows)
