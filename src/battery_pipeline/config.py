"""Central configuration: folder layout and every threshold used by the pipeline.

Keeping the numbers in one place means a reviewer can see (and challenge) each
business rule without reading the code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- physical plausibility limits (validation rules, silver layer) -----------
SOH_MIN, SOH_MAX = 0.0, 100.0            # state of health, %
TEMP_MIN_C, TEMP_MAX_C = -30.0, 80.0     # battery temperature, °C
VOLTAGE_MIN_V, VOLTAGE_MAX_V = 250.0, 450.0

# --- quality gate -------------------------------------------------------------
# If more than this share of raw rows has to be quarantined, the pipeline stops:
# something is wrong upstream and the output cannot be trusted.
MAX_QUARANTINE_RATE = 0.15

# --- business thresholds (gold layer) -------------------------------------------
ATTENTION_RATE_PCT_30D = 1.5   # SoH lost per 30 days at or above this -> ATTENTION
WATCH_RATE_PCT_30D = 1.0       # ... at or above this -> WATCH
WATCH_TEMP_MAX_C = 45.0        # max temperature at or above this -> WATCH
MIN_DAYS_FOR_TREND = 30        # fewer days of data -> no trend / forecast

# --- forecasting -------------------------------------------------------------
FORECAST_HORIZON_DAYS = 180
BACKTEST_TEST_FRACTION = 0.2


@dataclass(frozen=True)
class Paths:
    """Data-lake layout. raw = bronze, processed = silver, consumers = gold."""

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def quarantine(self) -> Path:
        return self.root / "data" / "quarantine"

    @property
    def consumers(self) -> Path:
        return self.root / "data" / "consumers"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.processed, self.quarantine, self.consumers, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(root: str | Path | None = None) -> Paths:
    """Return the lake paths. Defaults to the repository root."""
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    return Paths(base)
