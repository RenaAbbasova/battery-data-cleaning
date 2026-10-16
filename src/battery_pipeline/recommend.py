"""Rule-based recommendations per vehicle.

Honest scope: these are transparent rules, not AI. The structured output of this
step (status + reasons) is exactly what an LLM would receive as context if one is
added later; starting with rules means every sentence the user sees can be traced
back to a number and a threshold.
"""
from __future__ import annotations

import pandas as pd

from . import config as cfg

BASE = {
    "ATTENTION": {
        "es": "Tu batería pierde salud más rápido de lo normal. Conviene revisarla en un taller.",
        "en": "Your battery is losing health faster than normal. Have it checked at a workshop.",
    },
    "WATCH": {
        "es": "Tu batería está en observación. Cuida las condiciones de carga y temperatura.",
        "en": "Your battery is on watch. Take care with charging and temperature conditions.",
    },
    "OK": {
        "es": "Tu batería mantiene un ritmo de desgaste normal. Sigue así.",
        "en": "Your battery is wearing at a normal pace. Keep it up.",
    },
    "INSUFFICIENT_DATA": {
        "es": "Todavía no hay datos suficientes para valorar tu batería.",
        "en": "Not enough data yet to assess your battery.",
    },
}
EXTRA = {
    "hot": {
        "es": " Ha llegado a {t:.0f} °C: evita cargar o aparcar al sol.",
        "en": " It reached {t:.0f} °C: avoid charging or parking in direct sun.",
    },
    "swing": {
        "es": " La temperatura varía {r:.0f} °C: evita cargas rápidas en los extremos.",
        "en": " Temperature swings by {r:.0f} °C: avoid fast charging at the extremes.",
    },
}
SWING_THRESHOLD_C = 20.0


def recommend(features: pd.DataFrame, lang: str = "es") -> pd.DataFrame:
    if lang not in ("es", "en"):
        raise ValueError("lang must be 'es' or 'en'")
    texts = []
    for _, row in features.iterrows():
        text = BASE[row["status"]][lang]
        if row["temp_max"] >= cfg.WATCH_TEMP_MAX_C:
            text += EXTRA["hot"][lang].format(t=row["temp_max"])
        if row["temp_range"] >= SWING_THRESHOLD_C:
            text += EXTRA["swing"][lang].format(r=row["temp_range"])
        texts.append(text)
    out = features[["vehicle_id", "status", "degradation_rate_pct_30d", "temp_max", "temp_range"]].copy()
    out["recommendation"] = texts
    return out
