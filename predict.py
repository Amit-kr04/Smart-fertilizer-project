"""Get a fertilizer recommendation for one field.

Examples
  python predict.py --crop rice --stage sowing --N 40 --P 45 --K 38 \\
      --temp 36 --moisture 42 --rainfall 240 --ph 6.2
  python predict.py            # interactive prompts
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import CROP_CYCLE_DAYS, GROWTH_STAGES  # noqa: E402
from pipeline import SmartFertilizerSystem  # noqa: E402

FIELDS = [  # (arg, prompt, type, choices)
    ("crop", f"Crop type {sorted(CROP_CYCLE_DAYS)}", str, sorted(CROP_CYCLE_DAYS)),
    ("stage", f"Growth stage {GROWTH_STAGES}", str, GROWTH_STAGES),
    ("N", "Soil nitrogen N (kg/ha)", float, None),
    ("P", "Soil phosphorus P (kg/ha)", float, None),
    ("K", "Soil potassium K (kg/ha)", float, None),
    ("temp", "Soil temperature (°C)", float, None),
    ("moisture", "Soil moisture (% volumetric)", float, None),
    ("rainfall", "Rainfall over the last month (mm)", float, None),
    ("ph", "Soil pH", float, None),
]


def ask(prompt, cast, choices):
    while True:
        raw = input(f"{prompt}: ").strip().lower()
        try:
            value = cast(raw)
        except ValueError:
            print("  please enter a number")
            continue
        if choices and value not in choices:
            print(f"  choose one of {choices}")
            continue
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, prompt, cast, choices in FIELDS:
        parser.add_argument(f"--{name}", type=cast, choices=choices, help=prompt)
    parser.add_argument("--days", type=float,
                        help="days since a coated fertilizer was applied (optional)")
    args = vars(parser.parse_args())
    for name, prompt, cast, choices in FIELDS:
        if args[name] is None:
            args[name] = ask(prompt, cast, choices)

    field = {"soil_temperature": args["temp"], "soil_moisture": args["moisture"],
             "rainfall": args["rainfall"], "soil_pH": args["ph"]}
    rec = SmartFertilizerSystem().recommend(field, args["crop"], args["stage"],
                                            args["N"], args["P"], args["K"],
                                            days_since_application=args["days"])
    m1 = rec["module1"]
    print("\n--- Module 1: nutrient release (90-day rated polymer-coated urea) ---")
    print(f"Release rate here      : {m1['release_rate']:.2f} % per day")
    print(f"Actual release duration: ~{m1['effective_duration']:.0f} days "
          f"(label says {m1['rated_duration']:.0f})")
    if "cumulative_release" in m1:
        print(f"Released so far        : {m1['cumulative_release']:.0f} % after {args['days']:.0f} days")
    print("\n--- Module 2: recommendation ---")
    print(f"Fertilizer             : {rec['fertilizer_type']}  (confidence {rec['confidence']:.0%})")
    print(f"Amount                 : {rec['fertilizer_amount_kg_ha']:.0f} kg/ha of product "
          f"(~{rec['nutrient_kg_ha']:.0f} kg/ha nutrient)")
    print(f"Next application in    : {rec['next_application_in_days']} days")
    print("Alternatives           : " + ", ".join(f"{c} ({p:.0%})" for c, p in rec["alternatives"]))
    for w in rec["warnings"]:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
