from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class UniverseAsset:
    ticker: str
    sleeve: str
    provider: str
    asset_class: str
    region: str
    style: str
    diversifier: bool
    strategic_core: bool
    tactical_overlay: bool
    inflation_sensitive: bool
    duration_sensitive: bool
    proxy_description: str
    notes: str


APPROVED_PROVIDERS: List[str] = [
    "Vanguard",
    "iShares",
    "SPDR",
    "Invesco",
    "Schwab",
    "Avantis",
    "Alpha Architect",
    "JPMorgan",
    "AQR",
    "WisdomTree",
    "PIMCO",
]


UNIVERSE: List[UniverseAsset] = [
    UniverseAsset("SPY", "US Large Cap", "SPDR", "Equity", "US", "Beta", False, True, True, False, False, "us_large_cap_beta", "Primary benchmark"),
    UniverseAsset("VTI", "US Total Market", "Vanguard", "Equity", "US", "Beta", False, True, True, False, False, "us_total_market", "Broad US market"),
    UniverseAsset("VB", "US Small Cap", "Vanguard", "Equity", "US", "Size", False, True, True, False, False, "us_small_cap", "US small cap"),
    UniverseAsset("VTV", "US Value", "Vanguard", "Equity", "US", "Value", False, True, True, False, False, "us_value", "US value"),
    UniverseAsset("VBR", "US Small Value", "Vanguard", "Equity", "US", "Size + Value", False, True, True, False, False, "us_small_value", "US small value"),
    UniverseAsset("MTUM", "US Momentum", "iShares", "Equity", "US", "Momentum", False, False, True, False, False, "us_momentum", "Momentum factor proxy"),
    UniverseAsset("VEA", "International Developed", "Vanguard", "Equity", "Developed ex-US", "Beta", False, True, True, False, False, "international_developed_equity", "Developed ex-US"),
    UniverseAsset("VSS", "International Small Cap", "Vanguard", "Equity", "International", "Size", False, True, True, False, False, "international_small_cap", "International small cap"),
    UniverseAsset("IVLU", "International Value", "iShares", "Equity", "International", "Value", False, True, True, False, False, "international_value", "International value"),
    UniverseAsset("VWO", "Emerging Markets", "Vanguard", "Equity", "Emerging", "Beta", False, True, True, False, False, "emerging_markets_equity", "Broad EM"),
    UniverseAsset("AVES", "EM Value", "Avantis", "Equity", "Emerging", "Value", False, True, True, False, False, "emerging_markets_value", "EM value"),
    UniverseAsset("VGSH", "Short Treasuries", "Vanguard", "Rates", "US", "Short Duration", True, True, True, False, True, "short_treasuries", "Cash-adjacent ballast"),
    UniverseAsset("VGIT", "Intermediate Treasuries", "Vanguard", "Rates", "US", "Intermediate Duration", True, True, True, False, True, "intermediate_treasuries", "Core rates"),
    UniverseAsset("TLT", "Long Treasuries", "iShares", "Rates", "US", "Long Duration", True, True, True, False, True, "long_treasuries", "Long duration hedge"),
    UniverseAsset("TIP", "TIPS", "iShares", "Rates", "US", "Inflation-Linked", True, True, True, True, True, "inflation_linked_bonds", "Inflation-linked bonds"),
    UniverseAsset("BND", "Broad Bonds", "Vanguard", "Rates", "US", "Aggregate", True, True, True, False, True, "broad_bonds", "Broad bond proxy"),
    UniverseAsset("VNQ", "REITs", "Vanguard", "Real Assets", "US", "Real Estate", False, True, True, True, False, "listed_real_estate", "Listed real estate"),
    UniverseAsset("PDBC", "Broad Commodities", "Invesco", "Real Assets", "Global", "Commodities", True, True, True, True, False, "broad_commodities", "Broad commodities"),
    UniverseAsset("GLDM", "Gold", "SPDR", "Real Assets", "Global", "Precious Metals", True, True, True, True, False, "gold", "Gold proxy"),
    UniverseAsset("WTMF", "Managed Futures", "WisdomTree", "Alternatives", "Global", "Trend", True, False, True, False, False, "managed_futures", "Managed futures proxy"),
    UniverseAsset("SGOV", "Cash / Ultra-Short", "iShares", "Cash", "US", "Cash", True, True, True, False, True, "cash_ultra_short", "Cash proxy"),
]


BENCHMARK_TICKERS: List[str] = ["SPY"]
VANGUARD_TARGET_RETIREMENT_LABEL = "Vanguard Target Retirement"


def universe_by_ticker() -> Dict[str, UniverseAsset]:
    return {asset.ticker: asset for asset in UNIVERSE}


def tickers() -> List[str]:
    return [asset.ticker for asset in UNIVERSE]
