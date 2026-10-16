"""Quality gate, recommendations and anonymisation."""
import numpy as np
import pandas as pd
import pytest

from battery_pipeline import anonymize as A
from battery_pipeline import quality as Q
from battery_pipeline import recommend as R


# --- quality report and gate -----------------------------------------------------
def _frames(n_raw=100, n_bad=5):
    raw = pd.DataFrame({"source": ["maker_a"] * n_raw})
    silver = pd.DataFrame({"vehicle_id": ["A-001"] * (n_raw - n_bad), "source": "maker_a",
                           "timestamp": pd.date_range("2026-01-01", periods=n_raw - n_bad, freq="h")})
    quarantine = pd.DataFrame({"source": ["maker_a"] * n_bad, "quarantine_reason": ["duplicate"] * n_bad})
    return raw, silver, quarantine


def test_report_counts_add_up():
    raw, silver, quarantine = _frames()
    r = Q.build_report(raw, silver, quarantine)
    assert r["rows_raw"] == r["rows_clean"] + r["rows_quarantined"] == 100
    assert r["quarantine_rate"] == 0.05
    assert r["quarantined_by_reason"] == {"duplicate": 5}


def test_gate_passes_below_the_threshold():
    raw, silver, quarantine = _frames(n_bad=5)
    Q.assert_quality_gate(Q.build_report(raw, silver, quarantine))


def test_gate_stops_the_pipeline_above_the_threshold():
    raw, silver, quarantine = _frames(n_bad=40)
    with pytest.raises(Q.DataQualityError, match="above"):
        Q.assert_quality_gate(Q.build_report(raw, silver, quarantine))


def test_report_files_are_written(tmp_path):
    raw, silver, quarantine = _frames()
    Q.write_report(Q.build_report(raw, silver, quarantine), tmp_path)
    assert (tmp_path / "quality_report.json").exists()
    assert "Why rows were quarantined" in (tmp_path / "quality_report.md").read_text(encoding="utf-8")


# --- recommendations -------------------------------------------------------------
def _features(**row):
    base = {"vehicle_id": "A-001", "status": "OK", "degradation_rate_pct_30d": 0.5, "temp_max": 30.0, "temp_range": 10.0}
    base.update(row)
    return pd.DataFrame([base])


@pytest.mark.parametrize("status", ["OK", "WATCH", "ATTENTION", "INSUFFICIENT_DATA"])
def test_every_status_has_a_message_in_both_languages(status):
    for lang in ("es", "en"):
        out = R.recommend(_features(status=status), lang)
        assert isinstance(out.loc[0, "recommendation"], str) and out.loc[0, "recommendation"]


def test_heat_adds_a_specific_warning():
    hot = R.recommend(_features(status="WATCH", temp_max=48.0), "en").loc[0, "recommendation"]
    cool = R.recommend(_features(status="WATCH", temp_max=30.0), "en").loc[0, "recommendation"]
    assert "48" in hot and "48" not in cool


def test_large_temperature_swing_adds_a_warning():
    out = R.recommend(_features(temp_range=25.0), "en").loc[0, "recommendation"]
    assert "25" in out


def test_unknown_language_is_rejected():
    with pytest.raises(ValueError):
        R.recommend(_features(), "fr")


# --- anonymisation ---------------------------------------------------------------
def _fleet():
    return pd.DataFrame({
        "vehicle_id": [f"A-{i:03d}" for i in range(1, 9)], "maker": "maker_a",
        "temp_mean": np.linspace(20, 35, 8), "temp_max": np.linspace(30, 48, 8), "temp_range": 15.123,
        "days_active": 180, "degradation_rate_pct_30d": np.linspace(0.4, 1.6, 8), "status": "OK",
    })


def test_no_direct_identifier_survives():
    out = A.anonymize(_fleet(), salt="s")
    assert "vehicle_id" not in out.columns and "maker" not in out.columns
    text = out.to_csv(index=False)
    assert "A-00" not in text and "maker_a" not in text


def test_same_salt_gives_the_same_pseudonym_and_a_new_salt_changes_it():
    assert A.hash_id("A-001", "s1") == A.hash_id("A-001", "s1")
    assert A.hash_id("A-001", "s1") != A.hash_id("A-001", "s2")
    assert A.hash_id("A-001", "s1") != A.hash_id("A-002", "s1")


def test_pseudonyms_are_unique_per_vehicle():
    out = A.anonymize(_fleet(), salt="s")
    assert out["anon_id"].is_unique and len(out) == 8


def test_numbers_are_rounded_and_order_is_shuffled():
    fleet = _fleet()
    out = A.anonymize(fleet, salt="s")
    assert out["temp_range"].iloc[0] == 15.1
    assert out["degradation_rate_pct_30d"].round(2).equals(out["degradation_rate_pct_30d"])
    original_hashes = [A.hash_id(v, "s") for v in fleet["vehicle_id"]]
    assert out["anon_id"].tolist() != original_hashes


def test_default_salt_triggers_a_warning(monkeypatch):
    monkeypatch.delenv("ANON_SALT", raising=False)
    with pytest.warns(UserWarning, match="demo salt"):
        A.anonymize(_fleet())
