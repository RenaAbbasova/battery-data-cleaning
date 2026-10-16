"""Three charts for the report. They answer three questions:

1. What is the health history and forecast of a typical / best / worst vehicle?
2. Do hotter vehicles degrade faster? (the relationship built into the synthetic data)
3. Why was raw data rejected?
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"OK": "#2a9d8f", "WATCH": "#e9a23b", "ATTENTION": "#d1495b", "INSUFFICIENT_DATA": "#8a8f98"}
NOTE = "Synthetic data"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def chart_forecast(daily: pd.DataFrame, forecast: pd.DataFrame, features: pd.DataFrame, out: Path) -> None:
    ranked = features.dropna(subset=["degradation_rate_pct_30d"]).sort_values("degradation_rate_pct_30d")
    if ranked.empty:
        return
    picks = {"slowest degradation": ranked.iloc[0], "median": ranked.iloc[len(ranked) // 2],
             "fastest degradation": ranked.iloc[-1]}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (label, row) in zip(axes, picks.items()):
        vid = row["vehicle_id"]
        h = daily[daily["vehicle_id"] == vid]
        f = forecast[forecast["vehicle_id"] == vid]
        ax.scatter(h["date"], h["state_of_health_pct"], s=6, color="#4a5568", alpha=0.7, label="daily mean")
        if len(f):
            ax.plot(f["date"], f["soh_forecast"], color=COLORS[row["status"]], lw=2, label="forecast")
            ax.fill_between(f["date"], f["soh_lower"], f["soh_upper"], color=COLORS[row["status"]], alpha=0.15)
        _style(ax, f"{vid} · {label}\n{row['degradation_rate_pct_30d']:.2f} % lost / 30 days",
               "", "State of health (%)" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(f"State of health: history and 6-month forecast ({NOTE})", fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "soh_forecast.png", dpi=140)
    plt.close(fig)


def chart_temperature_effect(features: pd.DataFrame, effect: dict, out: Path) -> None:
    f = features.dropna(subset=["degradation_rate_pct_30d", "temp_mean"])
    if f.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for status, g in f.groupby("status"):
        ax.scatter(g["temp_mean"], g["degradation_rate_pct_30d"], s=60, color=COLORS[status], label=status, alpha=0.9)
    if len(f) >= 3:
        k, b = np.polyfit(f["temp_mean"], f["degradation_rate_pct_30d"], 1)
        xs = np.linspace(f["temp_mean"].min(), f["temp_mean"].max(), 50)
        ax.plot(xs, k * xs + b, color="#4a5568", lw=1.5, ls="--")
    rho = effect.get("spearman_temp_vs_degradation")
    txt = f"Spearman ρ = {rho}" if rho is not None else ""
    ax.text(0.03, 0.95, txt, transform=ax.transAxes, va="top", fontsize=11)
    _style(ax, f"Hotter vehicles degrade faster\n({NOTE}: the relationship is built into the generator)",
           "Mean battery temperature (°C)", "Health lost per 30 days (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "temperature_vs_degradation.png", dpi=140)
    plt.close(fig)


def chart_quality(report: dict, out: Path) -> None:
    reasons = report.get("quarantined_by_reason", {})
    if not reasons:
        return
    items = sorted(reasons.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#e9a23b")
    for i, (_, v) in enumerate(items):
        ax.text(v, i, f" {v}", va="center", fontsize=10)
    _style(ax, f"Why raw rows were quarantined ({report['quarantine_rate']:.1%} of {report['rows_raw']})",
           "Rows", "")
    fig.tight_layout()
    fig.savefig(out / "quarantine_reasons.png", dpi=140)
    plt.close(fig)


def make_charts(daily, forecast, features, effect, report, reports_dir: Path) -> list[str]:
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    chart_forecast(daily, forecast, features, out)
    chart_temperature_effect(features, effect, out)
    chart_quality(report, out)
    return sorted(p.name for p in out.glob("*.png"))
