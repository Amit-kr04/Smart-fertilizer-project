"""Shared paths, constants and agronomic lookup tables for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

CROP_DATA = RAW_DIR / "crop_recommendation.csv"            # real (Kaggle, 2,200 rows)
CROP_REQ_DATA = RAW_DIR / "crop_nutrient_requirements.csv"  # real crop-wise N-P-K targets
RELEASE_DATA = PROCESSED_DIR / "nutrient_release_dataset.csv"
RECOMMEND_DATA = PROCESSED_DIR / "fertilizer_recommendation_dataset.csv"

SEED = 42
SCENARIOS_PER_FIELD = 5   # synthetic fertilizer scenarios layered on each real field record
TEST_SIZE = 0.20          # 80/20 split, as in the methodology

# ---------- Module 1 ----------
M1_FIELD_INPUTS = ["soil_temperature", "soil_moisture", "rainfall", "coating_thickness", "soil_pH"]
# Release rate depends only on field conditions; cumulative release also on elapsed time.
M1_FEATURES = {
    "release_rate": M1_FIELD_INPUTS,
    "cumulative_release": M1_FIELD_INPUTS + ["days_since_application"],
}
M1_TARGETS = list(M1_FEATURES)

# ---------- Module 2 ----------
GROWTH_STAGES = ["sowing", "vegetative", "flowering", "maturity"]
# Fraction of the crop cycle (and nutrient demand) still ahead at each stage.
STAGE_REMAINING = {"sowing": 1.00, "vegetative": 0.70, "flowering": 0.40, "maturity": 0.15}

# Days of one crop / fruiting cycle (perennials: one fruiting-season cycle).
CROP_CYCLE_DAYS = {
    "rice": 120, "maize": 110, "chickpea": 100, "kidneybeans": 95, "pigeonpeas": 150,
    "mothbeans": 75, "mungbean": 70, "blackgram": 85, "lentil": 110, "pomegranate": 180,
    "banana": 300, "mango": 150, "grapes": 150, "watermelon": 85, "muskmelon": 85,
    "apple": 160, "orange": 200, "papaya": 270, "coconut": 365, "cotton": 160,
    "jute": 120, "coffee": 240,
}

# Fertilizer products the engine can recommend.
#   nutrient: which nutrient the dose is sized on; content: mass fraction of it
#   thickness: polymer coat (µm) for coated urea; base_nue: nutrient-use efficiency
FERTILIZERS = {
    "Urea (uncoated)":      {"nutrient": "N", "content": 0.46, "thickness": None, "base_nue": 0.45},
    "PCU-60 (thin coat)":   {"nutrient": "N", "content": 0.43, "thickness": 42,   "base_nue": 0.72},
    "PCU-90 (standard)":    {"nutrient": "N", "content": 0.42, "thickness": 65,   "base_nue": 0.72},
    "PCU-120 (thick coat)": {"nutrient": "N", "content": 0.41, "thickness": 88,   "base_nue": 0.72},
    "DAP (18-46-0)":        {"nutrient": "P", "content": 0.46, "thickness": None, "base_nue": 0.50},
    "MOP (0-0-60)":         {"nutrient": "K", "content": 0.60, "thickness": None, "base_nue": 0.60},
    "NPK 19-19-19":         {"nutrient": "NPK", "content": 0.19, "thickness": None, "base_nue": 0.55},
}
FERTILIZER_CLASSES = list(FERTILIZERS)

# Reference product whose field release rate Module 1 reports to Module 2
# (a manufacturer-rated "90-day" polymer-coated urea).
REFERENCE_THICKNESS = 65.0

M2_CATEGORICAL = ["crop_type", "growth_stage"]
# Module 2 inputs as in the system design: Module 1's predicted release rate
# plus crop type, growth stage, initial N-P-K and soil pH.
# synchrony_index = predicted release duration / days of crop demand left.
M2_FROM_MODULE1 = ["predicted_release_rate", "synchrony_index"]
M2_NUMERIC = ["N", "P", "K", "soil_pH"] + M2_FROM_MODULE1
M2_TARGETS = ["fertilizer_type", "fertilizer_amount", "application_timing"]

# ---------- Plot style (validated reference palette, light surface) ----------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_LOW, DIV_MID, DIV_HIGH = "#2a78d6", "#f0efec", "#e34948"
