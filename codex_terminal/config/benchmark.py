from __future__ import annotations

import pandas as pd


def market_beating_portfolio_weights() -> pd.DataFrame:
    # First-pass house benchmark: diversified strategic core with liquid ETF sleeves.
    weights = {
        "VTI": 0.20,
        "VBR": 0.08,
        "VEA": 0.12,
        "VWO": 0.08,
        "VGIT": 0.12,
        "TLT": 0.10,
        "TIP": 0.08,
        "VNQ": 0.06,
        "PDBC": 0.08,
        "GLDM": 0.04,
        "WTMF": 0.04,
    }
    frame = pd.DataFrame({"ticker": list(weights.keys()), "weight": list(weights.values())})
    frame["weight"] = frame["weight"] / frame["weight"].sum()
    return frame
