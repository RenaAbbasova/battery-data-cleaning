"""Silver layer: turn messy multi-manufacturer rows into one trustworthy table.

Design rule: **nothing is silently dropped.** A row is either valid (silver) or it
goes to quarantine with the reason it failed, so the data owner can fix it
upstream and the quality report can count it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import config as cfg
from .config import Paths

# Each manufacturer sends timestamps in its own format.
TS_FORMATS = {"maker_a": "%d/%m/%Y %H:%M", "maker_b": "%Y-%m-%dT%H:%M:%S"}
ID_PATTERN = re.compile(r"^[AB]-\d{3}$")

RAW_COLUMNS = ["vehicle_id", "timestamp", "temp", "temp_unit", "state_of_health_pct", "voltage_v", "source"]
CLEAN_COLUMNS = ["vehicle_id", "timestamp", "temp_c", "state_of_health_pct", "voltage_v", "source"]
DEDUP_KEY = ["vehicle_id", "timestamp", "temp_c", "state_of_health_pct", "voltage_v"]


def load_raw(raw_dir: str | Path) -> pd.DataFrame:
    """Read every raw CSV (one per manufacturer) into a single frame."""
    files = sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw CSV files found in {raw_dir}")
    dtypes = {"vehicle_id": str, "timestamp": str, "temp_unit": str, "source": str}
    df = pd.concat([pd.read_csv(f, dtype=dtypes) for f in files], ignore_index=True)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing columns: {sorted(missing)}")
    return df[RAW_COLUMNS]


def normalize_vehicle_id(ids: pd.Series) -> pd.Series:
    """' b_001' / 'B-001 ' / 'b-001' -> 'B-001'."""
    return ids.astype("string").str.strip().str.upper().str.replace("_", "-", regex=False)


def parse_timestamps(df: pd.DataFrame) -> pd.Series:
    """Parse each row with the format of its own manufacturer; bad values become NaT."""
    parts = []
    for source, fmt in TS_FORMATS.items():
        mask = df["source"] == source
        parts.append(pd.to_datetime(df.loc[mask, "timestamp"], format=fmt, errors="coerce"))
    unknown = ~df["source"].isin(TS_FORMATS)
    parts.append(pd.Series(pd.NaT, index=df.index[unknown], dtype="datetime64[ns]"))
    return pd.concat(parts).reindex(df.index)


def to_celsius(temp: pd.Series, unit: pd.Series) -> pd.Series:
    """Unify temperature to °C. Unknown units become NaN (and are quarantined)."""
    unit = unit.astype("string").str.strip().str.upper()
    value = pd.to_numeric(temp, errors="coerce").astype(float)
    celsius = value.where(unit == "C")
    return celsius.where(unit != "F", (value - 32) / 1.8)


def clean(raw: pd.DataFrame, now: pd.Timestamp | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(silver, quarantine)``.

    ``now`` is injectable so tests are deterministic; by default it is the real clock.
    """
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    df = raw.copy()
    df["vehicle_id"] = normalize_vehicle_id(df["vehicle_id"])
    df["timestamp_parsed"] = parse_timestamps(df)
    df["temp_c"] = to_celsius(df["temp"], df["temp_unit"])
    df["state_of_health_pct"] = pd.to_numeric(df["state_of_health_pct"], errors="coerce")
    df["voltage_v"] = pd.to_numeric(df["voltage_v"], errors="coerce")

    valid_id = df["vehicle_id"].str.match(ID_PATTERN).fillna(False).astype(bool)
    # Order matters: the first rule that fails is the recorded reason.
    rules = [
        ("bad_vehicle_id", ~valid_id),
        ("bad_timestamp", df["timestamp_parsed"].isna()),
        ("future_timestamp", df["timestamp_parsed"] > now),
        ("missing_temperature", df["temp_c"].isna()),
        ("missing_soh", df["state_of_health_pct"].isna()),
        ("soh_out_of_range", ~df["state_of_health_pct"].between(cfg.SOH_MIN, cfg.SOH_MAX)),
        ("temp_out_of_range", ~df["temp_c"].between(cfg.TEMP_MIN_C, cfg.TEMP_MAX_C)),
        ("voltage_out_of_range", ~df["voltage_v"].between(cfg.VOLTAGE_MIN_V, cfg.VOLTAGE_MAX_V)),
    ]
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    for name, failed in rules:
        reason = reason.mask(reason.isna() & failed.fillna(False).astype(bool), name)

    # Exact duplicates among rows that are otherwise valid.
    df["timestamp"] = df["timestamp_parsed"]
    valid_rows = reason.isna()
    is_dup = df[valid_rows].duplicated(subset=DEDUP_KEY, keep="first")
    reason.loc[is_dup[is_dup].index] = "duplicate"

    silver = (
        df.loc[reason.isna(), CLEAN_COLUMNS]
        .sort_values(["vehicle_id", "timestamp"])
        .reset_index(drop=True)
    )
    quarantine = raw.loc[reason.notna()].copy()
    quarantine["quarantine_reason"] = reason[reason.notna()].values
    return silver, quarantine.reset_index(drop=True)


def write_layers(silver: pd.DataFrame, quarantine: pd.DataFrame, paths: Paths) -> None:
    paths.ensure()
    silver.to_csv(paths.processed / "battery_clean.csv", index=False)
    quarantine.to_csv(paths.quarantine / "rejected_rows.csv", index=False)


def read_silver(paths: Paths) -> pd.DataFrame:
    return pd.read_csv(paths.processed / "battery_clean.csv", parse_dates=["timestamp"])
