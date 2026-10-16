"""Gold layer, part 2: forecast state of health and *prove* the forecast with a backtest.

Model choice on purpose: a straight-line trend. It is explainable in one sentence
("health is falling at X % per month, so in six months it will be about Y"), it
needs no tuning and it gives an honest baseline. Real batteries are not linear
(there can be a 'knee' where degradation accelerates), so the backtest compares
the model against a naive "nothing changes" forecast instead of just trusting it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .features import linear_trend

Z_95 = 1.96


def forecast_vehicle(g: pd.DataFrame, horizon_days: int = cfg.FORECAST_HORIZON_DAYS) -> pd.DataFrame:
    """Extend one vehicle's daily SoH line ``horizon_days`` into the future.

    The band is +-1.96 residual standard deviations. It only reflects scatter around
    the line, not uncertainty about the line itself, so treat it as optimistic.
    """
    g = g.sort_values("date")
    day0 = g["date"].min()
    days = (g["date"] - day0).dt.days.to_numpy(float)
    trend = linear_trend(days, g["state_of_health_pct"].to_numpy(float))
    if np.isnan(trend["slope_per_day"]):
        return pd.DataFrame(columns=["date", "soh_forecast", "soh_lower", "soh_upper"])
    last = g.loc[g["state_of_health_pct"].notna(), "date"].max()
    future = pd.date_range(last + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    x = (future - day0).days.to_numpy(float)
    mid = trend["slope_per_day"] * x + trend["intercept"]
    band = Z_95 * trend["resid_std"]
    return pd.DataFrame({
        "date": future,
        "soh_forecast": mid.clip(0, 100),
        "soh_lower": (mid - band).clip(0, 100),
        "soh_upper": (mid + band).clip(0, 100),
    })


def forecast_fleet(daily: pd.DataFrame, horizon_days: int = cfg.FORECAST_HORIZON_DAYS) -> pd.DataFrame:
    parts = []
    for vid, g in daily.groupby("vehicle_id"):
        f = forecast_vehicle(g, horizon_days)
        if len(f):
            f.insert(0, "vehicle_id", vid)
            parts.append(f)
    cols = ["vehicle_id", "date", "soh_forecast", "soh_lower", "soh_upper"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def backtest(daily: pd.DataFrame, test_fraction: float = cfg.BACKTEST_TEST_FRACTION) -> tuple[pd.DataFrame, dict]:
    """Hold out the last ``test_fraction`` of each vehicle's timeline and predict it.

    The split is by time, never random: a random split would leak the future into
    training. Compared against a naive forecast that repeats the last known level.
    """
    rows = []
    for vid, g in daily.groupby("vehicle_id"):
        g = g.sort_values("date").reset_index(drop=True)
        cut = int(len(g) * (1 - test_fraction))
        train, test = g.iloc[:cut], g.iloc[cut:]
        day0 = g["date"].min()
        tr_days = (train["date"] - day0).dt.days.to_numpy(float)
        trend = linear_trend(tr_days, train["state_of_health_pct"].to_numpy(float))
        actual = test.dropna(subset=["state_of_health_pct"])
        if np.isnan(trend["slope_per_day"]) or len(actual) < 5:
            continue
        x = (actual["date"] - day0).dt.days.to_numpy(float)
        pred = trend["slope_per_day"] * x + trend["intercept"]
        naive = train["state_of_health_pct"].dropna().tail(7).mean()
        y = actual["state_of_health_pct"].to_numpy(float)
        mae_model = float(np.abs(y - pred).mean())
        mae_naive = float(np.abs(y - naive).mean())
        rows.append({"vehicle_id": vid, "mae_model": mae_model, "mae_naive": mae_naive,
                     "model_beats_naive": mae_model < mae_naive})
    per_vehicle = pd.DataFrame(rows, columns=["vehicle_id", "mae_model", "mae_naive", "model_beats_naive"])
    if per_vehicle.empty:
        return per_vehicle, {"vehicles_evaluated": 0}
    summary = {
        "vehicles_evaluated": int(len(per_vehicle)),
        "mean_mae_model_pct_points": round(float(per_vehicle["mae_model"].mean()), 3),
        "mean_mae_naive_pct_points": round(float(per_vehicle["mae_naive"].mean()), 3),
        "share_vehicles_model_beats_naive": round(float(per_vehicle["model_beats_naive"].mean()), 3),
        "test_fraction": test_fraction,
    }
    return per_vehicle, summary
