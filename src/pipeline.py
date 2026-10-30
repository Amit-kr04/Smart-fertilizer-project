"""Integrated dual-module system used at inference time."""
import joblib
import pandas as pd

import agronomy
import release_physics as phys
from config import CROP_CYCLE_DAYS, CROP_DATA, FERTILIZERS, GROWTH_STAGES, MODEL_DIR, REFERENCE_THICKNESS

# Field inputs the release model was trained on (soil-level ranges in the dataset).
FIELD_RANGES = {"soil_temperature": (8, 48), "soil_moisture": (5, 50),
                "rainfall": (15, 430), "soil_pH": (3.5, 9.9)}


class SmartFertilizerSystem:
    """Module 1 (release prediction) -> Module 2 (recommendation)."""

    def __init__(self, model_dir=MODEL_DIR):
        self.rate_model = joblib.load(model_dir / "module1_release_rate.joblib")
        self.cumulative_model = joblib.load(model_dir / "module1_cumulative_release.joblib")
        self.recommender = joblib.load(model_dir / "module2_recommender.joblib")
        crops = pd.read_csv(CROP_DATA)
        self.npk_ranges = crops.groupby("label")[["N", "P", "K"]].agg(["min", "max"])

    # ---------------- input checks
    def check_inputs(self, field, crop_type, growth_stage, npk):
        if crop_type not in CROP_CYCLE_DAYS:
            raise ValueError(f"unknown crop '{crop_type}'; choose from {sorted(CROP_CYCLE_DAYS)}")
        if growth_stage not in GROWTH_STAGES:
            raise ValueError(f"unknown growth stage '{growth_stage}'; choose from {GROWTH_STAGES}")
        warnings = []
        for name, (lo, hi) in FIELD_RANGES.items():
            if not lo <= field[name] <= hi:
                warnings.append(f"{name}={field[name]} is outside the training range {lo}-{hi}")
        for nutrient, value in npk.items():
            lo, hi = self.npk_ranges.loc[crop_type, (nutrient, "min")], \
                     self.npk_ranges.loc[crop_type, (nutrient, "max")]
            if not lo <= value <= hi:
                warnings.append(f"soil {nutrient}={value} is outside the range seen for "
                                f"{crop_type} ({lo}-{hi}); the recommendation is an extrapolation")
        return warnings

    # ---------------- Module 1
    def predict_release(self, field, coating_thickness=REFERENCE_THICKNESS, days=None):
        row = dict(field, coating_thickness=coating_thickness)
        rate = float(self.rate_model["model"].predict(
            pd.DataFrame([row])[self.rate_model["features"]])[0])
        rate = max(rate, 0.05)   # guard against extrapolated non-physical rates
        out = {"release_rate": rate,
               "effective_duration": phys.RELEASED_AT_DURATION / rate,
               "rated_duration": float(phys.rated_duration(coating_thickness))}
        if days is not None:
            row["days_since_application"] = days
            out["cumulative_release"] = float(min(max(self.cumulative_model["model"].predict(
                pd.DataFrame([row])[self.cumulative_model["features"]])[0], 0.0), 100.0))
        return out

    # ---------------- Module 1 -> Module 2
    def recommend(self, field, crop_type, growth_stage, N, P, K, days_since_application=None):
        """field: soil_temperature (°C), soil_moisture (%), rainfall (mm), soil_pH."""
        warnings = self.check_inputs(field, crop_type, growth_stage, {"N": N, "P": P, "K": K})
        release = self.predict_release(field, days=days_since_application)
        rec = self.recommender
        m1 = agronomy.module1_features([release["release_rate"]], [crop_type], [growth_stage])
        X = pd.DataFrame([{"crop_type": crop_type, "growth_stage": growth_stage,
                           "N": N, "P": P, "K": K, "soil_pH": field["soil_pH"],
                           **{k: float(v[0]) for k, v in m1.items()}}])

        type_pipe = rec["pipelines"]["fertilizer_type"]
        proba = type_pipe.predict_proba(X[rec["categorical"] + rec["numeric"]])[0]
        classes = rec["label_encoder"].inverse_transform(type_pipe.classes_)
        ranked = sorted(zip(classes, proba), key=lambda t: -t[1])
        product = ranked[0][0]

        X = X.assign(fertilizer_type=product)
        amount = float(rec["pipelines"]["fertilizer_amount"].predict(X)[0])
        timing = float(rec["pipelines"]["application_timing"].predict(X)[0])
        return {
            "module1": release,
            "fertilizer_type": product,
            "confidence": float(ranked[0][1]),
            "alternatives": [(c, float(p)) for c, p in ranked[1:3]],
            "fertilizer_amount_kg_ha": max(amount, 0.0),
            "nutrient_kg_ha": max(amount, 0.0) * FERTILIZERS[product]["content"],
            "next_application_in_days": max(round(timing), 1),
            "warnings": warnings,
        }
