"""Data-quality report and the quality gate that can stop the pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config as cfg


class DataQualityError(RuntimeError):
    """Raised when too much of the raw data had to be rejected."""


def build_report(
    raw: pd.DataFrame,
    silver: pd.DataFrame,
    quarantine: pd.DataFrame,
    manifest: dict | None = None,
) -> dict:
    n_raw, n_silver, n_quar = len(raw), len(silver), len(quarantine)
    by_reason = quarantine["quarantine_reason"].value_counts().to_dict() if n_quar else {}
    by_source = {}
    for source, n in raw["source"].value_counts().items():
        by_source[source] = {
            "raw_rows": int(n),
            "clean_rows": int((silver["source"] == source).sum()),
            "quarantined_rows": int((quarantine["source"] == source).sum()) if n_quar else 0,
        }
    report = {
        "rows_raw": int(n_raw),
        "rows_clean": int(n_silver),
        "rows_quarantined": int(n_quar),
        "quarantine_rate": round(n_quar / n_raw, 4) if n_raw else 0.0,
        "quarantined_by_reason": {k: int(v) for k, v in by_reason.items()},
        "by_source": by_source,
        "vehicles": int(silver["vehicle_id"].nunique()),
        "first_reading": str(silver["timestamp"].min()),
        "last_reading": str(silver["timestamp"].max()),
        "quality_gate_max_quarantine_rate": cfg.MAX_QUARANTINE_RATE,
    }
    if manifest:
        report["synthetic_data_injected_defects"] = manifest.get("injected_defects", {})
    return report


def write_report(report: dict, reports_dir: Path) -> None:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Data quality report",
        "",
        f"- Raw rows: **{report['rows_raw']}**",
        f"- Clean rows (silver): **{report['rows_clean']}**",
        f"- Quarantined rows: **{report['rows_quarantined']}** "
        f"({report['quarantine_rate']:.1%}; gate at {report['quality_gate_max_quarantine_rate']:.0%})",
        f"- Vehicles: **{report['vehicles']}**",
        f"- Period: {report['first_reading']} → {report['last_reading']}",
        "",
        "## Why rows were quarantined",
        "",
        "| Reason | Rows |",
        "|---|---:|",
    ]
    for reason, n in sorted(report["quarantined_by_reason"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {reason} | {n} |")
    lines += ["", "## By manufacturer", "", "| Source | Raw | Clean | Quarantined |", "|---|---:|---:|---:|"]
    for source, d in report["by_source"].items():
        lines.append(f"| {source} | {d['raw_rows']} | {d['clean_rows']} | {d['quarantined_rows']} |")
    if "synthetic_data_injected_defects" in report:
        lines += ["", "## Injected defects (synthetic data only)", "", "| Defect | Rows injected |", "|---|---:|"]
        for k, v in report["synthetic_data_injected_defects"].items():
            lines.append(f"| {k} | {v} |")
    (reports_dir / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_quality_gate(report: dict, max_rate: float = cfg.MAX_QUARANTINE_RATE) -> None:
    """Stop the pipeline if too much of the raw data was rejected."""
    if report["quarantine_rate"] > max_rate:
        raise DataQualityError(
            f"Quarantine rate {report['quarantine_rate']:.1%} is above the {max_rate:.0%} gate; "
            "check the upstream source before trusting any output."
        )
