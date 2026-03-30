from __future__ import annotations

import os
from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class Settings:
    fred_api_key: str | None


def get_settings() -> Settings:
    secret_key = None
    try:
        secret_key = st.secrets.get("FRED_API_KEY")
    except Exception:
        secret_key = None
    return Settings(fred_api_key=secret_key or os.getenv("FRED_API_KEY"))
