"""Silver layer: every cleaning rule has a test that fails if the rule is removed."""
import numpy as np
import pandas as pd
import pytest

from battery_pipeline import clean as C

NOW = "2026-06-01"


def raw_rows(**overrides):
    """One valid raw row from maker A, with fields replaceable per test."""
    row = {"vehicle_id": "A-001", "timestamp": "15/03/2026 10:30", "temp": 30.0, "temp_unit": "C",
           "state_of_health_pct": 95.0, "voltage_v": 350.0, "source": "maker_a"}
    row.update(overrides)
    return row


def frame(*rows):
    return pd.DataFrame(list(rows), columns=C.RAW_COLUMNS)


def reasons(quarantine):
    return quarantine["quarantine_reason"].tolist()


# --- normalisation ---------------------------------------------------------------
@pytest.mark.parametrize("dirty", [" b_001", "B-001 ", "b-001", "B_001", " B-001"])
def test_vehicle_ids_are_normalised(dirty):
    out = C.normalize_vehicle_id(pd.Series([dirty]))
    assert out.iloc[0] == "B-001"


def test_each_manufacturer_timestamp_format_is_parsed():
    df = frame(raw_rows(timestamp="15/03/2026 10:30", source="maker_a"),
               raw_rows(timestamp="2026-03-15T10:30:00", source="maker_b"))
    parsed = C.parse_timestamps(df)
    assert parsed.iloc[0] == parsed.iloc[1] == pd.Timestamp("2026-03-15 10:30")


def test_day_and_month_are_not_swapped():
    df = frame(raw_rows(timestamp="03/04/2026 08:00", source="maker_a"))   # 3 April, not 4 March
    assert C.parse_timestamps(df).iloc[0] == pd.Timestamp("2026-04-03 08:00")


def test_wrong_format_for_the_manufacturer_becomes_bad_timestamp():
    silver, quarantine = C.clean(frame(raw_rows(timestamp="2026-03-15T10:30:00", source="maker_a")), now=NOW)
    assert silver.empty and reasons(quarantine) == ["bad_timestamp"]


def test_fahrenheit_is_converted_to_celsius():
    out = C.to_celsius(pd.Series([212.0, 32.0, 86.0]), pd.Series(["F", "F", "F"]))
    assert out.round(6).tolist() == [100.0, 0.0, 30.0]


def test_celsius_is_left_alone_and_unknown_unit_is_missing():
    out = C.to_celsius(pd.Series([25.0, 25.0]), pd.Series(["C", "K"]))
    assert out.iloc[0] == 25.0 and np.isnan(out.iloc[1])


# --- quarantine rules ------------------------------------------------------------
def test_valid_row_passes_through_unchanged():
    silver, quarantine = C.clean(frame(raw_rows()), now=NOW)
    assert len(silver) == 1 and quarantine.empty
    assert list(silver.columns) == C.CLEAN_COLUMNS
    assert silver.loc[0, "temp_c"] == 30.0


@pytest.mark.parametrize("soh", [100.1, 130.0, -1.0])
def test_impossible_state_of_health_is_quarantined(soh):
    _, quarantine = C.clean(frame(raw_rows(state_of_health_pct=soh)), now=NOW)
    assert reasons(quarantine) == ["soh_out_of_range"]


def test_state_of_health_of_exactly_100_is_valid():
    silver, _ = C.clean(frame(raw_rows(state_of_health_pct=100.0)), now=NOW)
    assert len(silver) == 1


def test_missing_state_of_health_is_quarantined():
    _, quarantine = C.clean(frame(raw_rows(state_of_health_pct=np.nan)), now=NOW)
    assert reasons(quarantine) == ["missing_soh"]


def test_sentinel_temperature_999_is_quarantined():
    _, quarantine = C.clean(frame(raw_rows(temp=999.0)), now=NOW)
    assert reasons(quarantine) == ["temp_out_of_range"]


def test_fahrenheit_reading_is_range_checked_after_conversion():
    # 140 °F = 60 °C is plausible; 999 °F = 537 °C is not.
    silver, quarantine = C.clean(frame(raw_rows(temp=140.0, temp_unit="F"),
                                       raw_rows(temp=999.0, temp_unit="F", timestamp="15/03/2026 11:30")), now=NOW)
    assert len(silver) == 1 and reasons(quarantine) == ["temp_out_of_range"]


def test_timestamp_in_the_future_is_quarantined():
    _, quarantine = C.clean(frame(raw_rows(timestamp="15/03/2027 10:30")), now=NOW)
    assert reasons(quarantine) == ["future_timestamp"]


def test_vehicle_id_that_cannot_be_normalised_is_quarantined():
    _, quarantine = C.clean(frame(raw_rows(vehicle_id="VW-0010")), now=NOW)
    assert reasons(quarantine) == ["bad_vehicle_id"]


def test_exact_duplicate_is_quarantined_and_first_copy_is_kept():
    silver, quarantine = C.clean(frame(raw_rows(), raw_rows()), now=NOW)
    assert len(silver) == 1 and reasons(quarantine) == ["duplicate"]


def test_two_readings_at_different_times_are_not_duplicates():
    silver, quarantine = C.clean(frame(raw_rows(), raw_rows(timestamp="15/03/2026 10:31")), now=NOW)
    assert len(silver) == 2 and quarantine.empty


def test_first_failing_rule_is_the_recorded_reason():
    # Both the timestamp is in the future and the SoH is impossible: the earlier rule wins.
    _, quarantine = C.clean(frame(raw_rows(timestamp="15/03/2027 10:30", state_of_health_pct=150.0)), now=NOW)
    assert reasons(quarantine) == ["future_timestamp"]


# --- guarantees ------------------------------------------------------------------
def test_nothing_is_silently_dropped():
    df = frame(raw_rows(), raw_rows(), raw_rows(state_of_health_pct=140.0, timestamp="16/03/2026 10:30"),
               raw_rows(temp=999.0, timestamp="17/03/2026 10:30"))
    silver, quarantine = C.clean(df, now=NOW)
    assert len(silver) + len(quarantine) == len(df)


def test_quarantine_keeps_the_original_raw_values_for_traceability():
    _, quarantine = C.clean(frame(raw_rows(vehicle_id=" b_007", timestamp="2026-03-15T10:30:00",
                                           source="maker_b", state_of_health_pct=150.0)), now=NOW)
    assert quarantine.loc[0, "vehicle_id"] == " b_007"
    assert quarantine.loc[0, "state_of_health_pct"] == 150.0


def test_silver_is_sorted_by_vehicle_and_time():
    df = frame(raw_rows(vehicle_id="A-002", timestamp="15/03/2026 09:00"),
               raw_rows(vehicle_id="A-001", timestamp="15/03/2026 12:00"),
               raw_rows(vehicle_id="A-001", timestamp="15/03/2026 08:00"))
    silver, _ = C.clean(df, now=NOW)
    assert silver["vehicle_id"].tolist() == ["A-001", "A-001", "A-002"]
    assert silver.groupby("vehicle_id")["timestamp"].apply(lambda s: s.is_monotonic_increasing).all()


def test_missing_raw_columns_are_reported(tmp_path):
    pd.DataFrame({"vehicle_id": ["A-001"]}).to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        C.load_raw(tmp_path)


def test_empty_raw_folder_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        C.load_raw(tmp_path)
