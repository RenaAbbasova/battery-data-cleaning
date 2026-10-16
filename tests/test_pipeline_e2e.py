"""The whole flow, orchestrated by Prefect, in a temporary folder."""
import json

import pandas as pd
import pytest

from battery_pipeline import clean as C
from battery_pipeline.flows import battery_pipeline
from battery_pipeline.generate import generate_raw, read_manifest
from battery_pipeline.config import get_paths
from battery_pipeline.quality import DataQualityError

START = "2026-01-01"
NOW = "2026-07-15"          # after the 120 simulated days, before the injected "future" rows


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("lake")
    summary = battery_pipeline(root=str(root), regenerate=True, seed=7, n_vehicles=10, days=120,
                               start_date=START, now=NOW)
    return root, summary


def test_every_expected_output_exists(run):
    root, _ = run
    expected = [
        "data/raw/maker_a.csv", "data/raw/maker_b.csv", "data/processed/battery_clean.csv",
        "data/quarantine/rejected_rows.csv", "data/consumers/dashboard/vehicle_summary.csv",
        "data/consumers/ml_features/vehicle_features.csv", "data/consumers/forecast/soh_forecast.csv",
        "data/consumers/forecast/backtest_per_vehicle.csv", "data/consumers/recommendations/recommendations_es.csv",
        "data/consumers/recommendations/recommendations_en.csv", "data/consumers/anonymized/fleet_anonymized.csv",
        "reports/quality_report.json", "reports/quality_report.md", "reports/run_summary.json",
        "reports/soh_forecast.png", "reports/temperature_vs_degradation.png", "reports/quarantine_reasons.png",
    ]
    missing = [p for p in expected if not (root / p).exists()]
    assert not missing, missing


def test_no_row_is_lost_between_raw_and_the_two_outputs(run):
    root, summary = run
    assert summary["rows_clean"] + summary["rows_quarantined"] == summary["rows_raw"]
    silver = pd.read_csv(root / "data/processed/battery_clean.csv")
    quarantine = pd.read_csv(root / "data/quarantine/rejected_rows.csv")
    assert len(silver) == summary["rows_clean"] and len(quarantine) == summary["rows_quarantined"]


def test_the_injected_defects_are_caught(run):
    root, _ = run
    manifest = read_manifest(get_paths(root).raw)["injected_defects"]
    report = json.loads((root / "reports/quality_report.json").read_text(encoding="utf-8"))
    by_reason = report["quarantined_by_reason"]
    # Every injected defect must be found (some rows have two defects and are counted under the first rule).
    assert by_reason["missing_soh"] >= manifest["missing_soh"] * 0.9
    assert by_reason["soh_out_of_range"] >= manifest["soh_out_of_range"] * 0.9
    assert by_reason["temp_out_of_range"] >= manifest["temp_out_of_range"] * 0.9
    assert by_reason["future_timestamp"] >= manifest["future_timestamp"] * 0.9
    assert by_reason["duplicate"] >= manifest["duplicates"] * 0.9


def test_silver_is_clean_and_uniform(run):
    root, _ = run
    silver = pd.read_csv(root / "data/processed/battery_clean.csv")
    assert silver["vehicle_id"].str.fullmatch(r"[AB]-\d{3}").all()
    assert silver["state_of_health_pct"].between(0, 100).all()
    assert silver["temp_c"].between(-30, 80).all()
    assert not silver.duplicated().any()
    assert silver["vehicle_id"].nunique() == 10


def test_the_temperature_effect_built_into_the_data_is_recovered(run):
    _, summary = run
    assert summary["temperature_effect"]["spearman_temp_vs_degradation"] > 0.6


def test_the_forecast_beats_the_naive_baseline_on_held_out_time(run):
    _, summary = run
    bt = summary["forecast_backtest"]
    assert bt["vehicles_evaluated"] == 10
    assert bt["mean_mae_model_pct_points"] < bt["mean_mae_naive_pct_points"]


def test_the_anonymised_file_contains_no_vehicle_ids(run):
    root, _ = run
    text = (root / "data/consumers/anonymized/fleet_anonymized.csv").read_text(encoding="utf-8")
    assert "A-0" not in text and "B-0" not in text


def test_running_again_on_existing_raw_data_is_repeatable(run, tmp_path):
    root, first = run
    second = battery_pipeline(root=str(root), regenerate=False, now=NOW)
    assert second["rows_raw"] == first["rows_raw"] and second["rows_clean"] == first["rows_clean"]


def test_the_quality_gate_stops_the_pipeline_when_the_raw_data_is_bad(tmp_path):
    paths = get_paths(tmp_path).ensure()
    generate_raw(paths, seed=1, n_vehicles=4, days=60, start_date=START)
    bad = pd.read_csv(paths.raw / "maker_a.csv")
    bad["state_of_health_pct"] = 150.0            # every row impossible
    bad.to_csv(paths.raw / "maker_a.csv", index=False)
    with pytest.raises(DataQualityError):
        battery_pipeline(root=str(tmp_path), regenerate=False, now=NOW)
    assert not (tmp_path / "data/consumers/dashboard/vehicle_summary.csv").exists()   # nothing downstream ran
