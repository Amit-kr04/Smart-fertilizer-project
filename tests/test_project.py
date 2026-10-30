"""Tests for the physics, the datasets and the integrated system.

Run after `python main.py`:  .venv/bin/python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import agronomy                      # noqa: E402
import release_physics as phys       # noqa: E402
from config import (FERTILIZER_CLASSES, M1_FEATURES, M2_TARGETS, MODEL_DIR,  # noqa: E402
                    RECOMMEND_DATA, RELEASE_DATA)

LAB = dict(soil_temperature=25, soil_moisture=30, rainfall=100, soil_pH=6.5)


# ---------------------------------------------------------------- physics
def test_lab_conditions_reproduce_rated_duration():
    assert phys.rate_factor(**LAB) == pytest.approx(1.0)
    assert phys.effective_duration(65, **LAB) == pytest.approx(phys.rated_duration(65))
    assert phys.rated_duration(65) == pytest.approx(90, abs=1)


def test_q10_ten_degrees_cuts_duration_by_q10():
    hot = phys.effective_duration(65, **dict(LAB, soil_temperature=35))
    assert phys.rated_duration(65) / hot == pytest.approx(phys.Q10)


def test_thicker_coat_and_drier_soil_release_slower():
    assert phys.rated_duration(90) > phys.rated_duration(40)
    dry = phys.effective_duration(65, **dict(LAB, soil_moisture=10))
    assert dry > phys.effective_duration(65, **LAB)


def test_cumulative_curve_is_monotonic_and_hits_80_at_duration():
    days = np.arange(0, 300)
    curve = phys.cumulative_release(days, 90)
    assert curve[0] == 0 and np.all(np.diff(curve) >= 0) and curve[-1] <= 100
    assert phys.cumulative_release(90, 90) == pytest.approx(80)


# ---------------------------------------------------------------- agronomy
def test_coat_grade_follows_field_release_speed():
    req = {"N": 80, "P": 40, "K": 40}
    soil = {"N": 60, "P": 45, "K": 40}       # nitrogen-limited rice field
    assert agronomy.choose_fertilizer(req, soil, "rice", "sowing", k=2.2).startswith("PCU-120")
    assert agronomy.choose_fertilizer(req, soil, "rice", "sowing", k=0.6).startswith("PCU-60")
    assert agronomy.choose_fertilizer(req, soil, "rice", "flowering", k=1.0) == "Urea (uncoated)"


# ---------------------------------------------------------------- datasets
@pytest.fixture(scope="module")
def release_df():
    return pd.read_csv(RELEASE_DATA)


@pytest.fixture(scope="module")
def recommend_df():
    return pd.read_csv(RECOMMEND_DATA)


def test_release_dataset_schema(release_df):
    needed = set(M1_FEATURES["cumulative_release"]) | {"release_rate", "cumulative_release"}
    assert needed <= set(release_df.columns)
    assert len(release_df) == 11_000 and not release_df.isna().any().any()
    assert release_df["cumulative_release"].between(0, 100).all()
    assert (release_df["release_rate"] > 0).all()


def test_recommendation_dataset_schema(recommend_df):
    assert set(M2_TARGETS) <= set(recommend_df.columns)
    assert len(recommend_df) == 11_000 and not recommend_df.isna().any().any()
    assert set(recommend_df["fertilizer_type"]) == set(FERTILIZER_CLASSES)
    assert (recommend_df["fertilizer_amount"] > 0).all()
    assert (recommend_df["application_timing"] >= 5).all()


def test_dataset_reproduces_release_time_swing(recommend_df):
    dur = phys.rated_duration(65) / recommend_df["field_rate_factor"]
    lo, hi = np.percentile(dur, [5, 95])
    assert lo < 55 and hi > 110          # the 50-120 day swing in the problem statement


# ---------------------------------------------------------------- integrated system
@pytest.fixture(scope="module")
def system():
    if not (MODEL_DIR / "module2_recommender.joblib").exists():
        pytest.skip("models not trained; run main.py first")
    from pipeline import SmartFertilizerSystem
    return SmartFertilizerSystem()


HOT = {"soil_temperature": 36.0, "soil_moisture": 42.0, "rainfall": 240.0, "soil_pH": 6.2}
COOL = {"soil_temperature": 18.0, "soil_moisture": 14.0, "rainfall": 45.0, "soil_pH": 6.8}


def test_module1_hot_field_releases_faster_than_cool(system):
    hot, cool = system.predict_release(HOT), system.predict_release(COOL)
    assert hot["effective_duration"] < 60 < 110 < cool["effective_duration"]


def test_recommendation_output(system):
    rec = system.recommend(HOT, "rice", "sowing", 62, 45, 38, days_since_application=20)
    assert rec["fertilizer_type"] in FERTILIZER_CLASSES
    assert 0 <= rec["confidence"] <= 1
    assert rec["fertilizer_amount_kg_ha"] > 0 and rec["next_application_in_days"] >= 1
    assert 0 <= rec["module1"]["cumulative_release"] <= 100
    assert rec["warnings"] == []


def test_hot_and_cool_fields_get_different_coats(system):
    hot = system.recommend(HOT, "rice", "sowing", 62, 45, 38)["fertilizer_type"]
    cool = system.recommend(COOL, "maize", "sowing", 62, 48, 22)["fertilizer_type"]
    assert hot == "PCU-120 (thick coat)" and cool == "PCU-60 (thin coat)"


def test_out_of_range_inputs_are_flagged(system):
    rec = system.recommend(dict(HOT, soil_temperature=55), "rice", "sowing", 20, 45, 38)
    assert len(rec["warnings"]) == 2


def test_unknown_crop_is_rejected(system):
    with pytest.raises(ValueError):
        system.recommend(HOT, "tomato", "sowing", 62, 45, 38)
