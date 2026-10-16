"""Anonymized dataset for third parties.

What it does: replaces the vehicle ID with a salted hash, drops every direct
identifier (ID, maker, timestamps) and rounds the numbers.

What it does NOT do (stated on purpose): guarantee anonymity. A hash of a small
ID space can be reversed by brute force if the salt leaks, and rare combinations of
values can still single a vehicle out. A production version would add k-anonymity
checks and keep the salt in a secret manager.
"""
from __future__ import annotations

import hashlib
import os
import warnings

import pandas as pd

DEFAULT_SALT = "dev-only-change-me"
KEEP = ["anon_id", "temp_mean", "temp_max", "temp_range", "days_active",
        "degradation_rate_pct_30d", "status"]


def hash_id(vehicle_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{vehicle_id}".encode("utf-8")).hexdigest()[:12]


def anonymize(features: pd.DataFrame, salt: str | None = None) -> pd.DataFrame:
    salt = salt or os.environ.get("ANON_SALT", DEFAULT_SALT)
    if salt == DEFAULT_SALT:
        warnings.warn("Using the default demo salt; set ANON_SALT for real use.", stacklevel=2)
    out = features.copy()
    out["anon_id"] = [hash_id(v, salt) for v in out["vehicle_id"]]
    out = out[KEEP].copy()
    for col in ("temp_mean", "temp_max", "temp_range"):
        out[col] = out[col].round(1)
    out["degradation_rate_pct_30d"] = out["degradation_rate_pct_30d"].round(2)
    return out.sample(frac=1.0, random_state=0).reset_index(drop=True)   # break the original order
