"""Synthetic raw telemetry from two simulated manufacturers.

The generator is a *test bench*: it produces battery data that behaves like the
real thing (slow state-of-health decay, daily temperature cycle) and then
deliberately breaks it in the ways real multi-brand telemetry breaks. The
pipeline is what is being demonstrated, not the generator.

Built-in relationship (by construction, documented on purpose): vehicles that run
hotter lose state of health faster. The analysis should *recover* it.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Paths

TS_FORMAT_A = "%d/%m/%Y %H:%M"          # maker A: dd/mm/yyyy hh:mm
TS_FORMAT_B = "%Y-%m-%dT%H:%M:%S"       # maker B: ISO 8601


def _simulate_vehicle(rng: np.random.Generator, days: int, start: datetime) -> pd.DataFrame:
    mean_temp = rng.normal(27, 5)
    start_soh = rng.uniform(93, 100)
    # % of state of health lost every 30 days; hotter fleet members degrade faster
    rate_30d = max(0.2, 0.5 + 0.05 * max(mean_temp - 20, 0) + rng.normal(0, 0.1))

    rows = []
    for d in range(days):
        for _ in range(int(np.clip(rng.poisson(2), 0, 4))):
            hour = int(rng.integers(0, 24))
            minute = int(rng.integers(0, 60))
            ts = start + timedelta(days=d, hours=hour, minutes=minute)
            soh = start_soh - rate_30d * (d / 30) + rng.normal(0, 0.25)
            temp_c = mean_temp + 6 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1.5)
            voltage = 345 + 0.45 * (soh - 90) + rng.normal(0, 2.5)
            rows.append((ts, round(temp_c, 2), round(soh, 2), round(voltage, 1)))
    return pd.DataFrame(rows, columns=["ts", "temp_c", "soh", "voltage_v"])


def generate_raw(
    paths: Paths,
    seed: int = 42,
    n_vehicles: int = 24,
    days: int = 180,
    start_date: str | None = None,
) -> dict:
    """Write raw/maker_a.csv and raw/maker_b.csv and return a generation manifest."""
    rng = np.random.default_rng(seed)
    paths.ensure()
    if start_date is None:
        start = (datetime.now() - timedelta(days=days + 2)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        start = datetime.fromisoformat(start_date)

    frames: dict[str, list[pd.DataFrame]] = {"maker_a": [], "maker_b": []}
    for i in range(n_vehicles):
        maker = "maker_a" if i % 2 == 0 else "maker_b"
        sim = _simulate_vehicle(rng, days, start)
        sim["vehicle_no"] = i // 2 + 1
        frames[maker].append(sim)

    injected = {"duplicates": 0, "soh_out_of_range": 0, "temp_out_of_range": 0,
                "missing_soh": 0, "future_timestamp": 0}
    for maker, parts in frames.items():
        df = pd.concat(parts, ignore_index=True)
        n = len(df)

        # -- inject the defects real telemetry has ------------------------------
        idx = rng.choice(n, size=max(1, int(0.010 * n)), replace=False)
        df.loc[idx, "soh"] = rng.uniform(101, 130, size=len(idx)).round(1)
        injected["soh_out_of_range"] += len(idx)

        idx = rng.choice(n, size=max(1, int(0.005 * n)), replace=False)
        df.loc[idx, "temp_c"] = 999.0                       # sentinel value
        injected["temp_out_of_range"] += len(idx)

        idx = rng.choice(n, size=max(1, int(0.010 * n)), replace=False)
        df.loc[idx, "soh"] = np.nan
        injected["missing_soh"] += len(idx)

        idx = rng.choice(n, size=max(1, int(0.003 * n)), replace=False)
        df.loc[idx, "ts"] = df.loc[idx, "ts"] + timedelta(days=400)
        injected["future_timestamp"] += len(idx)

        # -- each maker has its own format ----------------------------------------
        if maker == "maker_a":
            out = pd.DataFrame({
                "vehicle_id": [f"A-{v:03d}" for v in df["vehicle_no"]],
                "timestamp": df["ts"].dt.strftime(TS_FORMAT_A),
                "temp": df["temp_c"],
                "temp_unit": "C",
            })
        else:
            temp_f = df["temp_c"] * 1.8 + 32
            style = rng.choice(["b_{:03d}", " B-{:03d}", "B-{:03d}"], size=n)   # inconsistent IDs
            out = pd.DataFrame({
                "vehicle_id": [s.format(v) for s, v in zip(style, df["vehicle_no"])],
                "timestamp": df["ts"].dt.strftime(TS_FORMAT_B),
                "temp": temp_f.round(2),
                "temp_unit": "F",
            })
        out["state_of_health_pct"] = df["soh"].values
        out["voltage_v"] = df["voltage_v"].values
        out["source"] = maker

        dup_idx = rng.choice(len(out), size=max(1, int(0.02 * len(out))), replace=False)
        out = pd.concat([out, out.iloc[dup_idx]], ignore_index=True)       # exact duplicates
        injected["duplicates"] += len(dup_idx)
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        out.to_csv(paths.raw / f"{maker}.csv", index=False)

    manifest = {
        "seed": seed, "n_vehicles": n_vehicles, "days": days,
        "start_date": start.isoformat(), "injected_defects": injected,
        "note": "SYNTHETIC data. Defects are injected on purpose to test the pipeline.",
    }
    (paths.raw / "_generation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_manifest(raw_dir: Path) -> dict | None:
    f = raw_dir / "_generation_manifest.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
