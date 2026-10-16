"""Read-only REST API over the gold layer.

The pipeline produces the data; this API serves it to whoever consumes it (an app,
a dashboard, another service). It never computes anything: it reads the files the
pipeline wrote, so the numbers an API client sees are exactly the numbers in the
reports.

Honest scope
------------
* Data comes from CSV files, re-read automatically when the pipeline writes new ones.
  At real volume this would sit on a database.
* Authentication is optional: if the environment variable ``API_KEY`` is set, every
  request must send it in the ``X-API-Key`` header. Without it the API is open, which is
  fine for a local demo and wrong for anything real. These endpoints return vehicle IDs
  and would need per-user authorisation in production.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi import Path as PathParam
from pydantic import BaseModel

from . import __version__
from .config import FORECAST_HORIZON_DAYS, get_paths

VEHICLE_ID_PATTERN = r"^[AaBb]-\d{3}$"
Status = Literal["OK", "WATCH", "ATTENTION", "INSUFFICIENT_DATA"]
Lang = Literal["es", "en"]
NO_DATA = "No pipeline output found. Run: python -m battery_pipeline run"


# --------------------------------------------------------------------------- schemas
class VehicleSummary(BaseModel):
    vehicle_id: str
    status: Status
    soh_current: Optional[float] = None
    degradation_rate_pct_30d: Optional[float] = None
    soh_in_6_months: Optional[float] = None
    temp_max: Optional[float] = None


class VehicleHealth(VehicleSummary):
    maker: Optional[str] = None
    temp_mean: Optional[float] = None
    temp_min: Optional[float] = None
    temp_range: Optional[float] = None
    days_active: Optional[int] = None
    n_readings: Optional[int] = None
    trend_r2: Optional[float] = None
    recommendation: Optional[str] = None


class ForecastPoint(BaseModel):
    date: str
    soh_forecast: float
    soh_lower: float
    soh_upper: float


class ForecastResponse(BaseModel):
    vehicle_id: str
    days: int
    points: list[ForecastPoint]


class FleetSummary(BaseModel):
    vehicles: int
    status_counts: dict[str, int]
    mean_degradation_rate_pct_30d: Optional[float]
    fastest_degrading_vehicle: Optional[str]
    slowest_degrading_vehicle: Optional[str]
    forecast_backtest: Optional[dict] = None
    temperature_effect: Optional[dict] = None
    last_run: Optional[str] = None


class ServiceHealth(BaseModel):
    status: str
    data_available: bool
    data_source: str
    last_run: Optional[str] = None
    version: str


# --------------------------------------------------------------------------- data access
class Store:
    """Reads the pipeline outputs; a file is re-read only when it changes on disk."""

    def __init__(self, root: str | Path | None):
        self.paths = get_paths(root)
        self._cache: dict[Path, tuple[float, object]] = {}

    def _cached(self, path: Path, loader):
        if not path.exists():
            raise HTTPException(status_code=503, detail=NO_DATA)
        mtime = path.stat().st_mtime
        hit = self._cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
        value = loader(path)
        self._cache[path] = (mtime, value)
        return value

    def csv(self, path: Path) -> pd.DataFrame:
        return self._cached(path, pd.read_csv)

    def json(self, path: Path) -> dict:
        return self._cached(path, lambda p: json.loads(p.read_text(encoding="utf-8")))

    @property
    def summary(self) -> pd.DataFrame:
        return self.csv(self.paths.consumers / "dashboard" / "vehicle_summary.csv")

    @property
    def features(self) -> pd.DataFrame:
        return self.csv(self.paths.consumers / "ml_features" / "vehicle_features.csv")

    @property
    def forecast(self) -> pd.DataFrame:
        return self.csv(self.paths.consumers / "forecast" / "soh_forecast.csv")

    def recommendations(self, lang: str) -> pd.DataFrame:
        return self.csv(self.paths.consumers / "recommendations" / f"recommendations_{lang}.csv")

    @property
    def run_summary(self) -> dict:
        return self.json(self.paths.reports / "run_summary.json")

    @property
    def quality(self) -> dict:
        return self.json(self.paths.reports / "quality_report.json")


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of dicts, with NaN turned into null (valid JSON)."""
    return json.loads(df.to_json(orient="records"))


# --------------------------------------------------------------------------- app
def create_app(root: str | Path | None = None) -> FastAPI:
    root = root or os.environ.get("BATTERY_PIPELINE_ROOT")
    store = Store(root)

    def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        expected = os.environ.get("API_KEY")          # read per request: optional, off by default
        if expected and x_api_key != expected:
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")

    app = FastAPI(
        title="EV Battery Data API",
        version=__version__,
        description=(
            "Read-only API over the output of the battery data pipeline. "
            "The default dataset is **synthetic**: it demonstrates the pipeline, not real batteries."
        ),
    )
    auth = [Depends(require_key)]

    def vehicle_row(df: pd.DataFrame, vehicle_id: str) -> dict:
        vid = vehicle_id.upper()
        rows = df[df["vehicle_id"] == vid]
        if rows.empty:
            raise HTTPException(status_code=404, detail=f"Vehicle {vid} not found.")
        return _records(rows.head(1))[0]

    @app.get("/health", response_model=ServiceHealth, tags=["service"])
    def health() -> ServiceHealth:
        """Is the service up, and has the pipeline produced data? (never requires a key)"""
        summary_file = store.paths.consumers / "dashboard" / "vehicle_summary.csv"
        run_file = store.paths.reports / "run_summary.json"
        manifest = store.paths.raw / "_generation_manifest.json"
        last_run = None
        if run_file.exists():
            last_run = json.loads(run_file.read_text(encoding="utf-8")).get("finished_at")
        return ServiceHealth(
            status="ok", data_available=summary_file.exists(),
            data_source="synthetic" if manifest.exists() else "unknown",
            last_run=last_run, version=__version__,
        )

    @app.get("/vehicles", response_model=list[VehicleSummary], tags=["vehicles"], dependencies=auth)
    def list_vehicles(
        status: Optional[Status] = Query(default=None, description="Keep only this status"),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict]:
        """All vehicles, worst degradation first. Optionally filtered by status."""
        df = store.summary
        if status:
            df = df[df["status"] == status]
        return _records(df.sort_values("degradation_rate_pct_30d", ascending=False).head(limit))

    @app.get("/vehicles/{vehicle_id}/health", response_model=VehicleHealth, tags=["vehicles"], dependencies=auth)
    def vehicle_health(
        vehicle_id: str = PathParam(pattern=VEHICLE_ID_PATTERN, description="e.g. A-001 or B-012"),
        lang: Lang = Query(default="es", description="Language of the recommendation"),
    ) -> dict:
        """Current state of health, degradation rate, 6-month outlook and a plain-language recommendation."""
        base = vehicle_row(store.summary, vehicle_id)
        extra = vehicle_row(store.features, vehicle_id)
        rec = vehicle_row(store.recommendations(lang), vehicle_id)
        merged = {**extra, **base}                       # the dashboard values win on overlap
        merged["recommendation"] = rec["recommendation"]
        return merged

    @app.get("/vehicles/{vehicle_id}/forecast", response_model=ForecastResponse, tags=["vehicles"], dependencies=auth)
    def vehicle_forecast(
        vehicle_id: str = PathParam(pattern=VEHICLE_ID_PATTERN),
        days: int = Query(default=90, ge=1, le=FORECAST_HORIZON_DAYS, description="How many days ahead"),
    ) -> dict:
        """State-of-health forecast with its uncertainty band (optimistic: see the README limitations)."""
        vid = vehicle_id.upper()
        df = store.forecast
        rows = df[df["vehicle_id"] == vid].sort_values("date").head(days)
        if rows.empty:
            known = vid in set(store.summary["vehicle_id"])
            detail = f"No forecast for {vid}: not enough history." if known else f"Vehicle {vid} not found."
            raise HTTPException(status_code=404, detail=detail)
        points = _records(rows[["date", "soh_forecast", "soh_lower", "soh_upper"]])
        return {"vehicle_id": vid, "days": len(points), "points": points}

    @app.get("/fleet/summary", response_model=FleetSummary, tags=["fleet"], dependencies=auth)
    def fleet_summary() -> dict:
        """Fleet-level view: status counts, degradation, and how the forecast model was evaluated."""
        feats = store.features.dropna(subset=["degradation_rate_pct_30d"])
        run = store.run_summary
        return {
            "vehicles": int(len(store.features)),
            "status_counts": {k: int(v) for k, v in store.features["status"].value_counts().items()},
            "mean_degradation_rate_pct_30d": round(float(feats["degradation_rate_pct_30d"].mean()), 3) if len(feats) else None,
            "fastest_degrading_vehicle": feats.loc[feats["degradation_rate_pct_30d"].idxmax(), "vehicle_id"] if len(feats) else None,
            "slowest_degrading_vehicle": feats.loc[feats["degradation_rate_pct_30d"].idxmin(), "vehicle_id"] if len(feats) else None,
            "forecast_backtest": run.get("forecast_backtest"),
            "temperature_effect": run.get("temperature_effect"),
            "last_run": run.get("finished_at"),
        }

    @app.get("/quality", tags=["fleet"], dependencies=auth)
    def quality_report() -> dict:
        """The data-quality report of the last pipeline run (rows rejected and why)."""
        return store.quality

    return app


app = create_app()
