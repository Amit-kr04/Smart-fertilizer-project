"""Agronomic decision rules that label the recommendation dataset.

`k` is the field release-rate multiplier from `release_physics.rate_factor`
(k = 1 means lab conditions: 25 °C, moist soil). The proposed system knows k
through Module 1; the existing "static" system implicitly assumes k = 1.
"""
import numpy as np

from config import CROP_CYCLE_DAYS, FERTILIZERS, STAGE_REMAINING
from release_physics import RELEASED_AT_DURATION, rated_duration

PCU_GRADES = ["PCU-60 (thin coat)", "PCU-90 (standard)", "PCU-120 (thick coat)"]
UREA_LOSS_DAYS = 18.0   # days uncoated urea stays available at 25 °C


def remaining_days(crop, stage):
    return CROP_CYCLE_DAYS[crop] * STAGE_REMAINING[stage]


def nutrient_needs(req, soil, stage):
    """kg/ha of N, P, K still to supply: crop removal plus any soil deficit."""
    rem = STAGE_REMAINING[stage]
    return {x: (0.6 * req[x] + 1.2 * max(req[x] - soil[x], 0.0)) * rem for x in "NPK"}


def choose_fertilizer(req, soil, crop, stage, k, jitter=0.0):
    """Pick a product from the relative N/P/K deficit and the field release speed."""
    rel = {x: (req[x] - soil[x]) / req[x] for x in "NPK"}
    ranked = sorted(rel, key=rel.get, reverse=True)
    top, second = ranked[0], ranked[1]

    if rel[top] < 0.05 + jitter:
        return "NPK 19-19-19"                       # nothing clearly limiting: maintenance
    if rel[second] > 0.15 and rel[top] - rel[second] < 0.10 + jitter:
        return "NPK 19-19-19"                       # two nutrients co-limiting
    if top == "P":
        return "DAP (18-46-0)"
    if top == "K":
        return "MOP (0-0-60)"

    days_left = remaining_days(crop, stage)
    if stage in ("flowering", "maturity") or days_left < 25:
        return "Urea (uncoated)"                    # late-season quick N boost
    # Match the coat's *field* release duration to the N demand window.
    target = 0.8 * days_left * (1.0 + jitter)
    durations = [rated_duration(FERTILIZERS[g]["thickness"]) / k for g in PCU_GRADES]
    return PCU_GRADES[int(np.argmin([abs(np.log(d / target)) for d in durations]))]


def field_duration(product, k):
    """Days the product keeps supplying nutrient under field multiplier k."""
    spec = FERTILIZERS[product]
    if spec["thickness"] is not None:
        return rated_duration(spec["thickness"]) / k
    if product == "Urea (uncoated)":
        return UREA_LOSS_DAYS / k
    if product == "NPK 19-19-19":
        return 35.0 / k
    return 60.0 / np.sqrt(k)                        # DAP / MOP: P and K persist longer


def dose_and_timing(product, needs, crop, stage, k):
    """Return (product kg/ha, days until the next application)."""
    spec = FERTILIZERS[product]
    need = max(needs.values()) if spec["nutrient"] == "NPK" else needs[spec["nutrient"]]
    days_left = remaining_days(crop, stage)
    supply_days = field_duration(product, k)

    # Release/demand synchrony: efficiency drops when the product releases much
    # faster or slower than the crop's remaining demand window.
    window = min(days_left, 30.0) if product == "Urea (uncoated)" else 0.8 * days_left
    if spec["thickness"] is not None or product == "Urea (uncoated)":
        sync = np.exp(-0.6 * abs(np.log(supply_days / window)))
    else:
        sync = np.exp(-0.3 * abs(np.log(k)))
    nue = spec["base_nue"] * (0.6 + 0.4 * sync)

    amount = need / (spec["content"] * nue)
    timing = min(0.85 * supply_days, days_left)
    return amount, timing


def module1_features(rate, crop_types, stages):
    """Features Module 2 receives from Module 1: the predicted release rate and a
    synchrony index (predicted release duration / days of crop demand left)."""
    days_left = np.array([remaining_days(c, s) for c, s in zip(crop_types, stages)])
    return {"predicted_release_rate": np.asarray(rate),
            "synchrony_index": (RELEASED_AT_DURATION / np.asarray(rate)) / days_left}
