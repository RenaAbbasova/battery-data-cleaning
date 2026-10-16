"""The REST API, tested against the output of the real pipeline."""
import json
import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from battery_pipeline.api import create_app
from battery_pipeline.flows import battery_pipeline

START, NOW = "2026-01-01", "2026-07-15"


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    root = tmp_path_factory.mktemp("api_lake")
    battery_pipeline(root=str(root), regenerate=True, seed=11, n_vehicles=8, days=90, start_date=START, now=NOW)
    return root


@pytest.fixture()
def client(lake, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    return TestClient(create_app(lake))


# --- service health --------------------------------------------------------------
def test_health_reports_data_and_that_it_is_synthetic(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["data_available"] is True
    assert body["data_source"] == "synthetic" and body["last_run"]


def test_without_pipeline_output_health_says_so_and_data_endpoints_return_503(tmp_path, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    empty = TestClient(create_app(tmp_path))
    assert empty.get("/health").json()["data_available"] is False
    r = empty.get("/vehicles")
    assert r.status_code == 503 and "battery_pipeline run" in r.json()["detail"]


# --- vehicles --------------------------------------------------------------------
def test_vehicles_are_listed_worst_degradation_first(client):
    rates = [v["degradation_rate_pct_30d"] for v in client.get("/vehicles").json()]
    assert len(rates) == 8 and rates == sorted(rates, reverse=True)


def test_vehicles_can_be_filtered_by_status(client):
    all_vehicles = client.get("/vehicles").json()
    status = all_vehicles[0]["status"]
    filtered = client.get("/vehicles", params={"status": status}).json()
    assert filtered and all(v["status"] == status for v in filtered)
    assert len(filtered) == sum(v["status"] == status for v in all_vehicles)


def test_an_invalid_status_is_rejected(client):
    assert client.get("/vehicles", params={"status": "BROKEN"}).status_code == 422


def test_limit_is_respected_and_bounded(client):
    assert len(client.get("/vehicles", params={"limit": 3}).json()) == 3
    assert client.get("/vehicles", params={"limit": 0}).status_code == 422


def test_vehicle_health_combines_state_trend_and_recommendation(client):
    vid = client.get("/vehicles").json()[0]["vehicle_id"]
    body = client.get(f"/vehicles/{vid}/health").json()
    for key in ("status", "soh_current", "degradation_rate_pct_30d", "soh_in_6_months",
                "temp_max", "days_active", "maker", "recommendation"):
        assert body[key] is not None, key
    assert body["vehicle_id"] == vid


def test_the_recommendation_follows_the_requested_language(client):
    vid = client.get("/vehicles").json()[0]["vehicle_id"]
    es = client.get(f"/vehicles/{vid}/health", params={"lang": "es"}).json()["recommendation"]
    en = client.get(f"/vehicles/{vid}/health", params={"lang": "en"}).json()["recommendation"]
    assert es != en and "batería" in es and "battery" in en
    assert client.get(f"/vehicles/{vid}/health", params={"lang": "fr"}).status_code == 422


def test_vehicle_ids_are_case_insensitive(client):
    vid = client.get("/vehicles").json()[0]["vehicle_id"]
    assert client.get(f"/vehicles/{vid.lower()}/health").json()["vehicle_id"] == vid


def test_an_unknown_vehicle_is_404(client):
    r = client.get("/vehicles/A-999/health")
    assert r.status_code == 404 and "A-999" in r.json()["detail"]


@pytest.mark.parametrize("bad", ["XYZ", "A-1", "C-001", "A-0001"])
def test_a_malformed_vehicle_id_is_422_not_a_server_error(client, bad):
    assert client.get(f"/vehicles/{bad}/health").status_code == 422


# --- forecast --------------------------------------------------------------------
def test_forecast_returns_the_requested_days_in_order_inside_the_band(client):
    vid = client.get("/vehicles").json()[0]["vehicle_id"]
    body = client.get(f"/vehicles/{vid}/forecast", params={"days": 5}).json()
    assert body["days"] == 5 and len(body["points"]) == 5
    dates = [p["date"] for p in body["points"]]
    assert dates == sorted(dates)
    assert all(p["soh_lower"] <= p["soh_forecast"] <= p["soh_upper"] for p in body["points"])


def test_forecast_horizon_cannot_exceed_what_was_computed(client):
    vid = client.get("/vehicles").json()[0]["vehicle_id"]
    assert client.get(f"/vehicles/{vid}/forecast", params={"days": 181}).status_code == 422
    assert client.get(f"/vehicles/{vid}/forecast", params={"days": 0}).status_code == 422


def test_forecast_for_an_unknown_vehicle_is_404(client):
    assert client.get("/vehicles/A-999/forecast").status_code == 404


# --- fleet and quality -----------------------------------------------------------
def test_fleet_summary_is_consistent_with_the_vehicle_list(client):
    summary = client.get("/fleet/summary").json()
    vehicles = client.get("/vehicles").json()
    assert summary["vehicles"] == len(vehicles) == sum(summary["status_counts"].values())
    assert summary["fastest_degrading_vehicle"] == vehicles[0]["vehicle_id"]
    assert summary["forecast_backtest"]["vehicles_evaluated"] == 8


def test_quality_endpoint_returns_the_report_the_pipeline_wrote(client, lake):
    on_disk = json.loads((lake / "reports" / "quality_report.json").read_text(encoding="utf-8"))
    assert client.get("/quality").json() == on_disk


# --- freshness -------------------------------------------------------------------
def test_new_pipeline_output_is_picked_up_without_restarting_the_api(lake, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    c = TestClient(create_app(lake))
    vid = c.get("/vehicles").json()[0]["vehicle_id"]
    path = lake / "data" / "consumers" / "dashboard" / "vehicle_summary.csv"
    original = path.read_text(encoding="utf-8")
    try:
        df = pd.read_csv(path)
        df.loc[df["vehicle_id"] == vid, "status"] = "ATTENTION"
        df.to_csv(path, index=False)
        stat = path.stat()
        os.utime(path, (stat.st_atime + 5, stat.st_mtime + 5))      # make sure the mtime changes
        assert c.get(f"/vehicles/{vid}/health").json()["status"] == "ATTENTION"
    finally:
        path.write_text(original, encoding="utf-8")


# --- authentication --------------------------------------------------------------
def test_api_key_is_required_when_one_is_configured(lake, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    c = TestClient(create_app(lake))
    assert c.get("/vehicles").status_code == 401
    assert c.get("/vehicles", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/vehicles", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert c.get("/health").status_code == 200          # the liveness check stays open


def test_the_api_is_open_when_no_key_is_configured(client):
    assert client.get("/vehicles").status_code == 200


# --- documentation ---------------------------------------------------------------
def test_openapi_documents_every_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    for expected in ("/health", "/vehicles", "/vehicles/{vehicle_id}/health",
                     "/vehicles/{vehicle_id}/forecast", "/fleet/summary", "/quality"):
        assert expected in paths, expected
    assert client.get("/docs").status_code == 200
