from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VanguardBenchmarkState:
    status: str
    latest_nav: Optional[float] = None
    note: str = ""


def degraded_vanguard_state(manual_nav: Optional[float] = None) -> VanguardBenchmarkState:
    if manual_nav is not None:
        return VanguardBenchmarkState(
            status="Manual Latest NAV",
            latest_nav=manual_nav,
            note="Automated Vanguard benchmark fetch unavailable. Using manually entered latest NAV.",
        )
    return VanguardBenchmarkState(
        status="Unavailable",
        note="Automated Vanguard benchmark fetch unavailable. Enter the latest NAV manually to continue current-point comparisons.",
    )
