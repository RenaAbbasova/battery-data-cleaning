"""Gold layer: trend, status rule, forecast and the honest backtest."""
import numpy as np
import pandas as pd
import pytest

from battery_pipeline import config as cfg
from battery_pipeline import features as F
from battery_pipeline import forecast as FC


def line_daily(vehicle="A-001", days=90, start_soh=100.0, loss_per_30d=1.0, noise=0.0, seed=0, start="2026-01-01"):
    """A perfectly known degradation line, so the expected answer is a number we can write down."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    soh = start_soh - loss_per_30d / 30 * np.arange(days) + rng.normal(0, noise, days)
    return pd.DataFrame({"vehicle_id": vehicle, "date": dates, "state_of_health_pct": soh,
                         "temp_c": 25.0, "voltage_v": 350.0})


# --- daily series ----------------------------------------------------------------
def test_daily_series_averages_the_readings_of_a_day():
    silver = pd.DataFrame({
        "vehicle_id": ["A-001"] * 3,
        "timestamp": pd.to_datetime(["2026-03-01 08:00", "2026-03-01 20:00", "2026-03-02 09:00"]),
        "temp_c": [20.0, 30.0, 25.0], "state_of_health_pct": [90.0, 92.0, 91.0],
        "voltage_v": [350.0] * 3, "source": "maker_a",
    })
    daily = F.build_daily(silver)
    assert daily["state_of_health_pct"].tolist() == [91.0, 91.0]
    assert daily["temp_c"].tolist() == [25.0, 25.0]


def test_days_without_readings_stay_visible_as_nan():
    silver = pd.DataFrame({
        "vehicle_id": ["A-001"] * 2,
        "timestamp": pd.to_datetime(["2026-03-01 08:00", "2026-03-03 08:00"]),
        "temp_c": [20.0, 20.0], "state_of_health_pct": [90.0, 89.0], "voltage_v": [350.0] * 2, "source": "maker_a",
    })
    daily = F.build_daily(silver)
    assert len(daily) == 3 and daily["state_of_health_pct"].isna().tolist() == [False, True, False]


# --- trend -----------------------------------------------------------------------
def test_trend_recovers_a_known_slope():
    g = line_daily(loss_per_30d=1.0)
    t = F.linear_trend(np.arange(len(g), dtype=float), g["state_of_health_pct"].to_numpy())
    assert t["slope_per_day"] == pytest.approx(-1.0 / 30)
    assert t["r2"] == pytest.approx(1.0)
    assert t["resid_std"] == pytest.approx(0.0, abs=1e-9)


def test_trend_ignores_missing_days():
    g = line_daily(days=90, loss_per_30d=2.0)
    values = g["state_of_health_pct"].to_numpy().copy()
    values[10:20] = np.nan
    t = F.linear_trend(np.arange(90, dtype=float), values)
    assert t["slope_per_day"] == pytest.approx(-2.0 / 30)


def test_too_little_data_gives_no_trend():
    g = line_daily(days=cfg.MIN_DAYS_FOR_TREND - 1)
    t = F.linear_trend(np.arange(len(g), dtype=float), g["state_of_health_pct"].to_numpy())
    assert np.isnan(t["slope_per_day"])


# --- status rule -----------------------------------------------------------------
@pytest.mark.parametrize("rate, temp_max, expected", [
    (cfg.ATTENTION_RATE_PCT_30D, 30, "ATTENTION"),
    (cfg.ATTENTION_RATE_PCT_30D + 1, 30, "ATTENTION"),
    (cfg.WATCH_RATE_PCT_30D, 30, "WATCH"),
    (0.5, cfg.WATCH_TEMP_MAX_C, "WATCH"),
    (0.5, cfg.WATCH_TEMP_MAX_C - 0.1, "OK"),
    (0.0, 20, "OK"),
    (np.nan, 20, "INSUFFICIENT_DATA"),
])
def test_status_thresholds(rate, temp_max, expected):
    assert F.classify(rate, temp_max) == expected


# --- vehicle features ------------------------------------------------------------
def _silver_from_daily(daily, temp):
    return pd.DataFrame({"vehicle_id": daily["vehicle_id"], "timestamp": daily["date"] + pd.Timedelta(hours=12),
                         "temp_c": temp, "state_of_health_pct": daily["state_of_health_pct"],
                         "voltage_v": 350.0, "source": "maker_a"})


def test_a_hot_fast_degrading_vehicle_ranks_above_a_cool_one():
    fast = line_daily("A-001", loss_per_30d=2.0)
    slow = line_daily("A-002", loss_per_30d=0.4)
    silver = pd.concat([_silver_from_daily(fast, 40.0), _silver_from_daily(slow, 22.0)], ignore_index=True)
    feats = F.vehicle_features(silver, F.build_daily(silver))
    assert feats["vehicle_id"].tolist() == ["A-001", "A-002"]          # sorted: worst first
    by_id = feats.set_index("vehicle_id")
    assert by_id.loc["A-001", "degradation_rate_pct_30d"] == pytest.approx(2.0, abs=1e-6)
    assert by_id.loc["A-002", "degradation_rate_pct_30d"] == pytest.approx(0.4, abs=1e-6)
    assert by_id.loc["A-001", "status"] == "ATTENTION" and by_id.loc["A-002", "status"] == "OK"
    assert by_id.loc["A-001", "days_active"] == 90


def test_temperature_effect_is_detected():
    feats = pd.DataFrame({"temp_mean": [20, 24, 28, 32, 36, 40],
                          "degradation_rate_pct_30d": [0.4, 0.6, 0.9, 1.1, 1.4, 1.6]})
    assert F.temperature_effect(feats)["spearman_temp_vs_degradation"] == pytest.approx(1.0)


def test_temperature_effect_handles_ties_like_the_reference_implementation():
    # 0.955882 is the value scipy.stats.spearmanr returns for these numbers (ties included).
    feats = pd.DataFrame({"temp_mean": [20, 24, 24, 32, 36, 40],
                          "degradation_rate_pct_30d": [0.4, 0.6, 0.9, 0.9, 1.4, 1.6]})
    assert F.temperature_effect(feats)["spearman_temp_vs_degradation"] == pytest.approx(0.956, abs=1e-3)


def test_temperature_effect_needs_enough_vehicles():
    feats = pd.DataFrame({"temp_mean": [20, 24], "degradation_rate_pct_30d": [0.4, 0.6]})
    assert F.temperature_effect(feats)["spearman_temp_vs_degradation"] is None


# --- forecast --------------------------------------------------------------------
def test_forecast_continues_the_line_and_starts_after_the_last_reading():
    g = line_daily(days=90, start_soh=100, loss_per_30d=1.0)
    f = FC.forecast_vehicle(g, horizon_days=60)
    assert len(f) == 60
    assert f["date"].min() == g["date"].max() + pd.Timedelta(days=1)
    # 90 days -> 97.0; 60 days later -> 97 - 2 = 95 (a straight line keeps its slope)
    assert f["soh_forecast"].iloc[-1] == pytest.approx(100 - 149 / 30, abs=1e-6)
    assert f["soh_forecast"].is_monotonic_decreasing


def test_forecast_stays_inside_physical_limits():
    g = line_daily(days=90, start_soh=20.0, loss_per_30d=8.0, noise=0.3)
    f = FC.forecast_vehicle(g, horizon_days=300)
    assert f[["soh_forecast", "soh_lower", "soh_upper"]].min().min() >= 0
    assert f[["soh_forecast", "soh_lower", "soh_upper"]].max().max() <= 100


def test_forecast_band_brackets_the_line():
    f = FC.forecast_vehicle(line_daily(noise=0.3), horizon_days=30)
    assert (f["soh_lower"] <= f["soh_forecast"]).all() and (f["soh_forecast"] <= f["soh_upper"]).all()


def test_no_forecast_when_there_is_too_little_history():
    f = FC.forecast_vehicle(line_daily(days=10), horizon_days=30)
    assert f.empty


# --- backtest --------------------------------------------------------------------
def test_backtest_is_near_perfect_on_a_perfect_line_and_beats_naive():
    daily = line_daily(days=120, loss_per_30d=1.2)
    per_vehicle, summary = FC.backtest(daily)
    assert per_vehicle.loc[0, "mae_model"] == pytest.approx(0.0, abs=1e-6)
    assert per_vehicle.loc[0, "mae_naive"] > 0.1
    assert summary["share_vehicles_model_beats_naive"] == 1.0


def test_backtest_does_not_beat_naive_when_the_line_is_flat():
    """If nothing is degrading, 'nothing changes' is already the right answer: no free lunch."""
    daily = line_daily(days=120, loss_per_30d=0.0, noise=0.3, seed=3)
    per_vehicle, _ = FC.backtest(daily)
    assert per_vehicle.loc[0, "mae_model"] == pytest.approx(per_vehicle.loc[0, "mae_naive"], abs=0.15)


def test_backtest_uses_a_time_split_not_a_random_one():
    """After a sudden drop in the last 20% the straight line cannot know about it,
    so it must be wrong there. A random split would have seen the drop and hidden it."""
    daily = line_daily(days=100, loss_per_30d=0.3)
    daily.loc[daily.index[80:], "state_of_health_pct"] -= 5.0
    per_vehicle, _ = FC.backtest(daily)
    assert per_vehicle.loc[0, "mae_model"] > 4.0


def test_backtest_skips_vehicles_with_too_little_data():
    per_vehicle, summary = FC.backtest(line_daily(days=20))
    assert per_vehicle.empty and summary["vehicles_evaluated"] == 0
