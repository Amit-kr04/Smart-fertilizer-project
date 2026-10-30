"""Build the two project datasets.

Real data
    data/raw/crop_recommendation.csv - 2,200 field records (22 crops) with soil
    N, P, K, pH, air temperature, humidity and rainfall (Kaggle "Crop
    Recommendation Dataset").
    data/raw/crop_nutrient_requirements.csv - crop-wise target N, P, K levels.

No public dataset records coated-fertilizer release under field weather, so
each real field record is expanded into several synthetic scenarios whose
release behaviour is simulated by `release_physics` and whose recommendation
labels come from `agronomy`. Measurement noise, granule-batch variability and
agronomist disagreement are added so the learning problem is not trivial.
"""
import numpy as np
import pandas as pd

import agronomy
import release_physics as phys
from config import (CROP_DATA, CROP_REQ_DATA, FERTILIZER_CLASSES, GROWTH_STAGES,
                    PROCESSED_DIR, RECOMMEND_DATA, REFERENCE_THICKNESS, RELEASE_DATA,
                    SCENARIOS_PER_FIELD, SEED)

LABEL_NOISE = 0.03   # share of recommendations an agronomist would make differently


def load_real():
    fields = pd.read_csv(CROP_DATA).rename(columns={"label": "crop_type", "ph": "soil_pH"})
    req = pd.read_csv(CROP_REQ_DATA).set_index("Crop")[["N", "P", "K"]]
    return fields, req


def field_conditions(fields, rng):
    """Expand real records into soil-level conditions for one season scenario each."""
    df = fields.loc[fields.index.repeat(SCENARIOS_PER_FIELD)].reset_index(drop=True)
    n = len(df)
    # Bare/thinly-covered soil runs a few degrees above air temperature.
    df["soil_temperature"] = np.clip(df["temperature"] + rng.uniform(1, 6, n)
                                     + rng.normal(0, 1.5, n), 8, 48)
    df["rainfall"] = df["rainfall"] * rng.lognormal(0, 0.15, n)
    # Volumetric soil moisture (%) from humidity and rainfall.
    df["soil_moisture"] = np.clip(6 + 0.22 * df["humidity"] + 0.06 * df["rainfall"]
                                  + rng.normal(0, 3, n), 5, 50)
    df["soil_pH"] = np.clip(df["soil_pH"] + rng.normal(0, 0.1, n), 3.5, 9.9)
    return df


def build_release_dataset(fields, rng):
    df = field_conditions(fields, rng)
    n = len(df)
    df["coating_thickness"] = rng.uniform(20, 100, n)
    df["days_since_application"] = rng.integers(0, 151, n)

    batch = rng.lognormal(0, 0.06, n)   # granule-to-granule coating variability
    duration = phys.effective_duration(df["coating_thickness"], df["soil_temperature"],
                                       df["soil_moisture"], df["rainfall"], df["soil_pH"]) * batch
    df["rated_duration"] = phys.rated_duration(df["coating_thickness"])
    df["effective_duration"] = duration
    df["release_rate"] = phys.release_rate(duration) * rng.lognormal(0, 0.03, n)
    df["cumulative_release"] = np.clip(
        phys.cumulative_release(df["days_since_application"], duration)
        + rng.normal(0, 1.5, n), 0, 100)

    cols = ["crop_type", "soil_temperature", "soil_moisture", "rainfall",
            "coating_thickness", "soil_pH", "days_since_application",
            "rated_duration", "effective_duration", "release_rate", "cumulative_release"]
    return df[cols].round(4)


def build_recommendation_dataset(fields, req, rng):
    df = field_conditions(fields, rng)
    n = len(df)
    df["growth_stage"] = rng.choice(GROWTH_STAGES, n)
    k = phys.rate_factor(df["soil_temperature"], df["soil_moisture"],
                         df["rainfall"], df["soil_pH"]) * rng.lognormal(0, 0.06, n)
    # What a sensor-free Module 1 would ideally report: the field release rate
    # of the reference "90-day" coated urea under these conditions.
    df["reference_release_rate"] = phys.release_rate(phys.rated_duration(REFERENCE_THICKNESS) / k)

    types, amounts, timings = [], [], []
    for i, row in enumerate(df.itertuples(index=False)):
        r = req.loc[row.crop_type]
        soil = {"N": row.N, "P": row.P, "K": row.K}
        product = agronomy.choose_fertilizer(r, soil, row.crop_type, row.growth_stage,
                                             k[i], jitter=rng.normal(0, 0.04))
        if rng.random() < LABEL_NOISE:
            product = rng.choice(FERTILIZER_CLASSES)
        needs = agronomy.nutrient_needs(r, soil, row.growth_stage)
        amount, timing = agronomy.dose_and_timing(product, needs, row.crop_type,
                                                  row.growth_stage, k[i])
        types.append(product)
        amounts.append(np.clip(amount * rng.lognormal(0, 0.06), 15, 800))
        timings.append(max(5, round(timing + rng.normal(0, 2))))

    df["fertilizer_type"] = types
    df["fertilizer_amount"] = np.round(amounts, 1)
    df["application_timing"] = timings
    df["field_rate_factor"] = k   # ground truth, kept for analysis only (never a feature)

    cols = ["crop_type", "growth_stage", "N", "P", "K", "soil_pH", "soil_temperature",
            "soil_moisture", "rainfall", "reference_release_rate", "field_rate_factor",
            "fertilizer_type", "fertilizer_amount", "application_timing"]
    return df[cols].round(4)


def main():
    rng = np.random.default_rng(SEED)
    fields, req = load_real()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    release = build_release_dataset(fields, rng)
    release.to_csv(RELEASE_DATA, index=False)
    recommend = build_recommendation_dataset(fields, req, rng)
    recommend.to_csv(RECOMMEND_DATA, index=False)

    print(f"Real field records: {len(fields):,} ({fields['crop_type'].nunique()} crops)")
    print(f"Module 1 dataset -> {RELEASE_DATA.name}: {release.shape}")
    print(f"Module 2 dataset -> {RECOMMEND_DATA.name}: {recommend.shape}")
    print("\nFertilizer class balance:")
    print(recommend["fertilizer_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
