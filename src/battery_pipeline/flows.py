"""Orchestration with Prefect.

Why an orchestrator instead of a script that calls other scripts: every step is a
*task* with its own state, logs and retry policy; a failed step stops everything
downstream; and the same flow can be put on a schedule without changing its code.

Retries are only used where a failure can be transient (reading/writing files).
The quality gate has no retries on purpose: bad data does not get better by trying
again, it needs a human.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from prefect import flow, get_run_logger, task

from . import anonymize as anon
from . import charts as charts_mod
from . import clean as clean_mod
from . import config as cfg
from . import features as feat
from . import forecast as fc
from . import quality as qual
from . import recommend as rec
from .config import get_paths
from .generate import generate_raw, read_manifest


@task(name="1 generate raw data (bronze)", retries=2, retry_delay_seconds=1)
def t_generate(root, seed, n_vehicles, days, start_date):
    return generate_raw(get_paths(root), seed=seed, n_vehicles=n_vehicles, days=days, start_date=start_date)


@task(name="2 load raw", retries=2, retry_delay_seconds=1)
def t_load(root):
    raw = clean_mod.load_raw(get_paths(root).raw)
    get_run_logger().info("Loaded %d raw rows", len(raw))
    return raw


@task(name="3 clean + validate (silver)")
def t_clean(raw, now):
    silver, quarantine = clean_mod.clean(raw, now=now)
    get_run_logger().info("Silver: %d rows, quarantine: %d rows", len(silver), len(quarantine))
    return silver, quarantine


@task(name="4 write silver + quarantine", retries=2, retry_delay_seconds=1)
def t_write_layers(silver, quarantine, root):
    clean_mod.write_layers(silver, quarantine, get_paths(root))


@task(name="5 quality report + gate")
def t_quality(raw, silver, quarantine, manifest, root):
    report = qual.build_report(raw, silver, quarantine, manifest)
    qual.write_report(report, get_paths(root).reports)
    qual.assert_quality_gate(report)          # raises -> the flow stops here
    return report


@task(name="6 daily time series")
def t_daily(silver):
    return feat.build_daily(silver)


@task(name="7 vehicle features")
def t_features(silver, daily):
    feats = feat.vehicle_features(silver, daily)
    return feats, feat.temperature_effect(feats)


@task(name="8 forecast + backtest")
def t_forecast(daily):
    forecast = fc.forecast_fleet(daily)
    per_vehicle, summary = fc.backtest(daily)
    return forecast, per_vehicle, summary


@task(name="9 write consumers (gold)", retries=2, retry_delay_seconds=1)
def t_consumers(feats, forecast, backtest_per_vehicle, root):
    paths = get_paths(root).ensure()
    c = paths.consumers
    for sub in ("dashboard", "ml_features", "forecast", "recommendations", "anonymized"):
        (c / sub).mkdir(parents=True, exist_ok=True)

    last_forecast = forecast.sort_values("date").groupby("vehicle_id").tail(1)
    dash = feats[["vehicle_id", "status", "soh_current", "degradation_rate_pct_30d", "temp_max"]].merge(
        last_forecast[["vehicle_id", "soh_forecast"]].rename(columns={"soh_forecast": "soh_in_6_months"}),
        on="vehicle_id", how="left",
    )
    dash.round(2).to_csv(c / "dashboard" / "vehicle_summary.csv", index=False)
    feats.round(4).to_csv(c / "ml_features" / "vehicle_features.csv", index=False)
    forecast.round({"soh_forecast": 3, "soh_lower": 3, "soh_upper": 3}).to_csv(c / "forecast" / "soh_forecast.csv", index=False)
    backtest_per_vehicle.round(4).to_csv(c / "forecast" / "backtest_per_vehicle.csv", index=False)
    for lang in ("es", "en"):
        rec.recommend(feats, lang).round(2).to_csv(c / "recommendations" / f"recommendations_{lang}.csv", index=False)
    anon.anonymize(feats).to_csv(c / "anonymized" / "fleet_anonymized.csv", index=False)
    return dash


@task(name="10 charts")
def t_charts(daily, forecast, feats, effect, report, root):
    return charts_mod.make_charts(daily, forecast, feats, effect, report, get_paths(root).reports)


@flow(name="battery-pipeline", log_prints=True)
def battery_pipeline(
    root: str | None = None,
    regenerate: bool = True,
    seed: int = 42,
    n_vehicles: int = 24,
    days: int = 180,
    start_date: str | None = None,
    now: str | None = None,
) -> dict:
    """Run the whole lake: raw -> silver -> gold, with a quality gate in the middle."""
    paths = get_paths(root).ensure()
    root_s = str(paths.root)

    manifest = t_generate(root_s, seed, n_vehicles, days, start_date) if regenerate else read_manifest(paths.raw)
    raw = t_load(root_s)
    silver, quarantine = t_clean(raw, now)
    t_write_layers(silver, quarantine, root_s)
    report = t_quality(raw, silver, quarantine, manifest, root_s)

    daily = t_daily(silver)
    feats, effect = t_features(silver, daily)
    forecast, bt_per_vehicle, bt_summary = t_forecast(daily)
    t_consumers(feats, forecast, bt_per_vehicle, root_s)
    pngs = t_charts(daily, forecast, feats, effect, report, root_s)

    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "rows_raw": report["rows_raw"],
        "rows_clean": report["rows_clean"],
        "rows_quarantined": report["rows_quarantined"],
        "quarantine_rate": report["quarantine_rate"],
        "vehicles": report["vehicles"],
        "status_counts": feats["status"].value_counts().to_dict(),
        "temperature_effect": effect,
        "forecast_backtest": bt_summary,
        "charts": pngs,
    }
    (paths.reports / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Pipeline finished: {summary['rows_clean']} clean rows, {summary['vehicles']} vehicles, "
          f"backtest {bt_summary}")
    return summary


def serve_on_schedule(cron: str = "0 6 * * *") -> None:
    """Keep a process alive that runs the flow on a cron schedule (default: 06:00 daily)."""
    battery_pipeline.serve(name="battery-pipeline-scheduled", cron=cron)
