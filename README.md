# AI-Driven Dual-Module System for Coated-Fertilizer Nutrient Release & Crop Recommendations

 Amit Kumar (24MID0155), Mohammad Afaan Sherief (24MID0178), Bhumi Kumari (24MID0216)

**Module 1** predicts how fast a polymer-coated urea actually releases nitrogen under field weather and soil.
**Module 2** uses that prediction, together with crop, growth stage, soil N-P-K and pH, to recommend the **fertilizer type, amount and timing**.

## Quick start

**One command (recommended):**

```bash
python3 run_project.py
```

It sets up the environment and trains the models if needed (first run only, a few minutes), then starts the
dashboard, opens it in your browser and prints two links: one for this computer and one that other devices on
the same Wi-Fi can open. Press `Ctrl+C` to stop.

**Individual steps:**


```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python main.py        # dataset -> EDA -> Module 1 -> Module 2 -> demo (~2 min)
.venv/bin/streamlit run dashboard.py   # interactive dashboard in the browser
.venv/bin/python predict.py     # interactive recommendation for your own field
.venv/bin/python -m pytest -q   # 13 tests: physics, datasets, integrated system
.venv/bin/jupyter notebook Smart_Fertilizer_Project.ipynb   # walkthrough for the review
.venv/bin/python predict.py --crop rice --stage sowing --N 62 --P 45 --K 38 \
    --temp 36 --moisture 42 --rainfall 240 --ph 6.2 --days 20
```

> **XGBoost on macOS** needs the OpenMP runtime: `brew install libomp`. The results below use XGBoost 2.1.4.
> If XGBoost can't load, the code falls back to scikit-learn's `HistGradientBoosting`.

## Dataset

No public dataset records coated-fertilizer release under real field weather (the literature reports only
lab release curves). The project therefore **combines real field data with a physics-based simulation**:

| File | Source | Rows | Content |
|---|---|---|---|
| `data/raw/crop_recommendation.csv` | **Real** – Kaggle *Crop Recommendation Dataset* | 2,200 | soil N, P, K, pH, temperature, humidity, rainfall for 22 crops |
| `data/raw/crop_nutrient_requirements.csv` | **Real** – crop-wise N-P-K targets | 22 | agronomic requirement per crop |
| `data/processed/nutrient_release_dataset.csv` | Generated (Module 1) | 11,000 | field conditions, coating thickness, days → release rate, cumulative release |
| `data/processed/fertilizer_recommendation_dataset.csv` | Generated (Module 2) | 11,000 | crop, stage, N-P-K, pH, release rate → type, amount, timing |

Each real field record is expanded into 5 season scenarios: soil temperature (air + 1–6 °C), soil moisture
(from humidity and rainfall), coating thickness of 20–100 µm, and days since application.
Release is simulated in [src/release_physics.py](src/release_physics.py) using behaviour reported in the literature:

- **Temperature:** Q10 = 1.8, so +10 °C roughly halves release time (Brown et al. 1966; Kassem et al. 2024).
- **Coating thickness:** rated duration grows linearly with it (≈ 90 days at 65 µm, at 25 °C).
- **Moisture and rainfall:** dry soil slows release; heavy rain speeds it slightly.
- **pH:** only a weak effect.
- **Curve shape:** cumulative release follows a sigmoidal Weibull curve.

Recommendation labels come from the agronomic rules in [src/agronomy.py](src/agronomy.py): nutrient deficit, crop
demand window, and matching the coat's *field* release duration to that window. The data includes 6% batch
variability, measurement noise and 3% agronomist disagreement.

**Result:** across the dataset, a "90-day" coated urea actually lasts **40–126 days** (90% of fields, median 70).
This reproduces the 50–120 day swing that motivates the project.

## Results (80/20 split, 5-fold CV on the training set; model selected by CV score)

### Module 1 – Nutrient release prediction

| Model | Release rate R² | Cumulative release R² |
|---|---|---|
| Linear Regression | 0.811 | 0.809 |
| Polynomial LR (degree 3) | **0.981** | 0.960 |
| Decision Tree (grid-searched) | 0.960 | 0.966 |
| Random Forest (benchmark) | 0.977 | **0.987** |

Plain linear regression underfits because temperature acts exponentially; polynomial terms or trees capture
it. Coating thickness and soil temperature account for 97% of decision-tree importance.

### Module 2 – Recommendation engine (fed by Module 1's *predicted* release rate)

| Target | Decision Tree | Random Forest | **XGBoost** | XGBoost without Module 1 |
|---|---|---|---|---|
| Fertilizer type (accuracy, 7 classes) | 0.804 | 0.830 | **0.857** | 0.836 |
| – coated-urea grade accuracy | 0.753 | 0.785 | **0.804** | 0.710 |
| Amount (R², kg/ha) | 0.987 | 0.992 | **0.992** | 0.991 |
| Timing (R², days) | 0.961 | 0.973 | **0.976** | 0.904 |

XGBoost is selected for all three Module 2 targets by 5-fold CV.

- **Accuracy ceiling.** Because of the deliberate label noise, the exact labelling rule (given the true release
  factor) scores only **0.884**. The 0.857 XGBoost classifier is close to that ceiling.
- **How amount and timing are evaluated.** Module 2 is chained: type first, then amount and timing for that product.
  The R² values above are given the recommended product. End to end, using the *predicted* type, R² is
  0.76 for amount and 0.83 for timing, since a wrong product means a different dose.

### Existing static system vs proposed system (MAE on the test set, same product)

| | Timing, all products | Timing, coated urea | Amount |
|---|---|---|---|
| Existing static system (25 °C lab rating) | 9.2 days | 22.4 days | 12.6 kg/ha |
| ML without Module 1 (soil-test only) | 4.4 days | 8.5 days | 9.0 kg/ha |
| **Proposed dual-module system** | **2.4 days** | **4.0 days** | **8.4 kg/ha** |

Feeding the predicted release rate into Module 2 cuts coated-urea timing error by **82%** compared with
the static manufacturer rating (22.4 → 4.0 days).

### Demo (`main.py`)

| Field | Module 1 | Module 2 |
|---|---|---|
| Hot, wet rice (36 °C, 42% moisture) | "90-day" PCU lasts ~41 days | PCU-120 thick coat, 246 kg/ha, next dose in 46 days |
| Cool, dry maize (18 °C, 14% moisture) | "90-day" PCU lasts ~144 days | PCU-60 thin coat, 223 kg/ha, next dose in 92 days |
| Cotton at flowering | – | Uncoated urea, 204 kg/ha, next dose in 11 days |

## Project structure

```
main.py                 run everything end to end
Smart_Fertilizer_Project.ipynb  executed walkthrough notebook (all outputs included)
tests/test_project.py   pytest suite (physics, datasets, integrated system)
predict.py              CLI recommendation for user inputs (warns on out-of-range inputs)
dashboard.py            Streamlit dashboard: live recommendation, release explorer, model results, dataset
src/config.py           paths, constants, crop cycles, fertilizer products, plot palette
src/release_physics.py  coated-urea release kinetics (simulation ground truth)
src/agronomy.py         recommendation rules, nutrient needs, synchrony index
src/generate_dataset.py builds both datasets from the real field data
src/eda.py              exploratory figures
src/train_module1.py    Module 1 training, evaluation, figures
src/train_module2.py    Module 2 training, pipeline integration, ablation, static baseline
src/pipeline.py         SmartFertilizerSystem: Module 1 -> Module 2 at inference
models/                 trained models (.joblib)
outputs/figures/        all plots
outputs/*.json          full metrics (module1_metrics, module2_metrics, results_summary)
docs/                   course project report, research paper, final review presentation
prototype/              earlier single-script prototype (superseded by this version)
```

## Limitations

- Release behaviour and recommendation labels are **simulated** from literature relationships, so the
  models learn those relationships rather than measured field outcomes. Validating them needs
  field or lysimeter release measurements.
- Soil temperature and moisture are derived from air temperature, humidity and rainfall, not measured with soil sensors.
- Recommendations for N-P-K values outside a crop's range in the real data are extrapolations;
  `predict.py` flags them.

## References

See slide 10 of the zeroth-review deck. Real field data: Kaggle *Crop Recommendation Dataset* (Atharva Ingle),
via the Harvestify repository.
