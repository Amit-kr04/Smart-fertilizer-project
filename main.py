"""Run the full project: dataset -> EDA -> Module 1 -> Module 2 -> demo."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import eda                # noqa: E402
import generate_dataset   # noqa: E402
import train_module1      # noqa: E402
import train_module2      # noqa: E402
from config import OUTPUT_DIR  # noqa: E402
from pipeline import SmartFertilizerSystem  # noqa: E402

DEMO_FIELDS = [
    ("Hot, wet kharif rice field", "rice", "sowing", 62, 45, 38,
     {"soil_temperature": 36.0, "soil_moisture": 42.0, "rainfall": 240.0, "soil_pH": 6.2}),
    ("Cool, dry rabi maize field", "maize", "sowing", 62, 48, 22,
     {"soil_temperature": 18.0, "soil_moisture": 14.0, "rainfall": 45.0, "soil_pH": 6.8}),
    ("Cotton at flowering", "cotton", "flowering", 102, 40, 18,
     {"soil_temperature": 31.0, "soil_moisture": 26.0, "rainfall": 90.0, "soil_pH": 7.1}),
]


def banner(text):
    print("\n" + "=" * 72 + f"\n{text}\n" + "=" * 72)


def main():
    t0 = time.time()
    banner("STEP 1  Dataset (real field data + simulated coated-fertilizer release)")
    generate_dataset.main()
    banner("STEP 1b Exploratory analysis")
    eda_summary = eda.main()
    banner("STEP 2  Train Module 1 - Nutrient Release Prediction")
    m1 = train_module1.main()
    banner("STEP 3/4/5  Train Module 2, integrate pipeline, evaluate")
    m2 = train_module2.main()

    banner("DEMO  Integrated recommendations")
    system = SmartFertilizerSystem()
    demos = []
    for title, crop, stage, n, p, k, field in DEMO_FIELDS:
        rec = system.recommend(field, crop, stage, n, p, k)
        demos.append({"scenario": title, "inputs": {"crop": crop, "stage": stage, "N": n,
                      "P": p, "K": k, **field}, "output": rec})
        m = rec["module1"]
        print(f"\n{title}: {crop} @ {stage}, soil N-P-K {n}-{p}-{k}")
        print(f"  Module 1: '90-day' PCU releases {m['release_rate']:.2f} %/day "
              f"-> lasts ~{m['effective_duration']:.0f} days here")
        print(f"  Module 2: {rec['fertilizer_type']} ({rec['confidence']:.0%}), "
              f"{rec['fertilizer_amount_kg_ha']:.0f} kg/ha, next application in "
              f"{rec['next_application_in_days']} days")
        for w in rec["warnings"]:
            print(f"  WARNING: {w}")

    summary = {"eda": eda_summary, "module1": m1, "module2": m2, "demo": demos,
               "runtime_seconds": round(time.time() - t0, 1)}
    with open(OUTPUT_DIR / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nDone in {summary['runtime_seconds']} s. Figures: outputs/figures/, "
          f"metrics: outputs/results_summary.json")


if __name__ == "__main__":
    main()
