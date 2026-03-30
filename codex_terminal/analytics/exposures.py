from __future__ import annotations

import pandas as pd

from codex_terminal.config.universe import universe_by_ticker


def classify_holdings(weights: pd.DataFrame) -> pd.DataFrame:
    known = universe_by_ticker()
    frame = weights.copy()
    frame["Proxy"] = frame["ticker"].map(lambda x: known[x].proxy_description if x in known else "unresolved")
    frame["Sleeve"] = frame["ticker"].map(lambda x: known[x].sleeve if x in known else "Unresolved")
    frame["Asset Class"] = frame["ticker"].map(lambda x: known[x].asset_class if x in known else "Unresolved")
    frame["Region"] = frame["ticker"].map(lambda x: known[x].region if x in known else "Unresolved")
    frame["Style"] = frame["ticker"].map(lambda x: known[x].style if x in known else "Unresolved")
    return frame


def summarize_exposures(classified: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for column in ["Asset Class", "Region", "Style", "Sleeve"]:
        grouped = (
            classified.groupby(column, as_index=False)["weight"]
            .sum()
            .sort_values("weight", ascending=False)
            .rename(columns={"weight": "Weight"})
        )
        out[column] = grouped
    return out


def compare_exposure_summary(left: pd.DataFrame, right: pd.DataFrame, column: str) -> pd.DataFrame:
    left_sum = left.groupby(column)["weight"].sum().rename("Left")
    right_sum = right.groupby(column)["weight"].sum().rename("Right")
    joined = pd.concat([left_sum, right_sum], axis=1).fillna(0.0)
    joined["Difference"] = joined["Left"] - joined["Right"]
    return joined.reset_index().sort_values("Difference", ascending=False)
