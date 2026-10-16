"""Gold layer, part 1: daily time series and per-vehicle features.

The time-series idea in one sentence: readings arrive at irregular times, so we
first aggregate them to one value per vehicle per day, then fit a straight line
to state of health over time. The slope is the degradation rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def build_daily(silver: pd.DataFrame) -> pd.DataFrame:
    """One row per vehicle per calendar day (mean of that day's readings).

    Days without readings are kept as NaN so gaps stay visible instead of being
    silently invented.
    """
    daily = (
        silver.set_index("timestamp")
        .groupby("vehicle_id")[["state_of_health_pct", "temp_c", "voltage_v"]]
        .resample("D")
        .mean()
        .reset_index()
        .rename(columns={"timestamp": "date"})
    )
    return daily


def linear_trend(days: np.ndarray, values: np.ndarray) -> dict:
    """Least-squares line through the non-missing points.

    Returns slope (per day), intercept, R² and the residual standard deviation.
    With fewer than ``MIN_DAYS_FOR_TREND`` points there is no trend (all NaN).
    """
    mask = ~np.isnan(values)
    x, y = days[mask], values[mask]
    empty = {"slope_per_day": np.nan, "intercept": np.nan, "r2": np.nan, "resid_std": np.nan, "n_points": int(mask.sum())}
    if len(x) < cfg.MIN_DAYS_FOR_TREND:
        return empty
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "slope_per_day": float(slope),
        "intercept": float(intercept),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "resid_std": float(np.sqrt(ss_res / (len(x) - 2))),
        "n_points": int(mask.sum()),
    }


def classify(rate_30d: float, temp_max: float) -> str:
    """Business rule that turns numbers into a status (thresholds live in config.py)."""
    if np.isnan(rate_30d):
        return "INSUFFICIENT_DATA"
    if rate_30d >= cfg.ATTENTION_RATE_PCT_30D:
        return "ATTENTION"
    if rate_30d >= cfg.WATCH_RATE_PCT_30D or temp_max >= cfg.WATCH_TEMP_MAX_C:
        return "WATCH"
    return "OK"


def vehicle_features(silver: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """One row per vehicle: temperature stats, activity and degradation trend."""
    temp = silver.groupby("vehicle_id")["temp_c"].agg(temp_mean="mean", temp_max="max", temp_min="min")
    temp["temp_range"] = temp["temp_max"] - temp["temp_min"]
    ts = silver.groupby("vehicle_id")["timestamp"].agg(first="min", last="max", n_readings="count")
    ts["days_active"] = (ts["last"] - ts["first"]).dt.days + 1
    source = silver.groupby("vehicle_id")["source"].first()

    rows = []
    for vid, g in daily.groupby("vehicle_id"):
        g = g.sort_values("date")
        days = (g["date"] - g["date"].min()).dt.days.to_numpy(float)
        trend = linear_trend(days, g["state_of_health_pct"].to_numpy(float))
        soh_now = g["state_of_health_pct"].dropna().tail(7).mean()
        rows.append({
            "vehicle_id": vid,
            "soh_current": soh_now,
            "degradation_rate_pct_30d": -trend["slope_per_day"] * 30,   # positive = health lost
            "trend_r2": trend["r2"],
        })
    out = (
        pd.DataFrame(rows).set_index("vehicle_id")
        .join(temp).join(ts[["n_readings", "days_active"]]).join(source.rename("maker"))
    )
    out["status"] = [classify(r, t) for r, t in zip(out["degradation_rate_pct_30d"], out["temp_max"])]
    return out.reset_index().sort_values("degradation_rate_pct_30d", ascending=False).reset_index(drop=True)


def temperature_effect(features: pd.DataFrame) -> dict:
    """Does a hotter vehicle degrade faster? Spearman correlation across the fleet."""
    f = features.dropna(subset=["degradation_rate_pct_30d", "temp_mean"])
    if len(f) < 5:
        return {"spearman_temp_vs_degradation": None, "vehicles": int(len(f))}
    # Spearman = Pearson correlation of the ranks. Computing it this way needs no SciPy
    # and handles ties with average ranks, exactly like scipy.stats.spearmanr.
    rho = f["temp_mean"].rank().corr(f["degradation_rate_pct_30d"].rank())
    return {"spearman_temp_vs_degradation": round(float(rho), 3), "vehicles": int(len(f))}
