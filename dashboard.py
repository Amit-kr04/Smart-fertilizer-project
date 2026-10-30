"""Interactive dashboard for the coated-fertilizer dual-module system.

Run (after main.py has trained the models):
    .venv/bin/streamlit run dashboard.py
"""
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import release_physics as phys  # noqa: E402
from config import (CROP_CYCLE_DAYS, CROP_DATA, FERTILIZERS, FIG_DIR, GROWTH_STAGES, MODEL_DIR,  # noqa: E402
                    OUTPUT_DIR, RECOMMEND_DATA, REFERENCE_THICKNESS)

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
COAT_GRADES = {"PCU-60 (thin coat)": 42, "PCU-90 (standard)": 65, "PCU-120 (thick coat)": 88}
PRESETS = {
    "Hot, wet rice field": dict(crop="rice", stage="sowing", N=62, P=45, K=38, temp=36.0, moisture=42.0,
                                rainfall=240.0, ph=6.2),
    "Cool, dry maize field": dict(crop="maize", stage="sowing", N=62, P=48, K=22, temp=18.0, moisture=14.0,
                                  rainfall=45.0, ph=6.8),
    "Cotton at flowering": dict(crop="cotton", stage="flowering", N=102, P=40, K=18, temp=31.0, moisture=26.0,
                                rainfall=90.0, ph=7.1),
}

st.set_page_config(page_title="Smart Fertilizer Dashboard", page_icon="🌱", layout="wide")


# ------------------------------------------------------------------ loaders
@st.cache_resource
def load_system():
    from pipeline import SmartFertilizerSystem
    return SmartFertilizerSystem()


@st.cache_data
def load_summary():
    return json.load(open(OUTPUT_DIR / "results_summary.json"))


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


if not (MODEL_DIR / "module2_recommender.joblib").exists() or not (OUTPUT_DIR / "results_summary.json").exists():
    st.error("Models not found. Run `.venv/bin/python main.py` first, then reload this page.")
    st.stop()

system = load_system()
summary = load_summary()


def line_chart(df, x, y, color, domain, palette, x_title, y_title, dash=None, height=320):
    enc = dict(x=alt.X(x, title=x_title), y=alt.Y(y, title=y_title),
               color=alt.Color(color, title=None, scale=alt.Scale(domain=domain, range=palette),
                               legend=alt.Legend(orient="bottom")),
               tooltip=[color, alt.Tooltip(x, format=".0f"), alt.Tooltip(y, format=".1f")])
    if dash:
        enc["strokeDash"] = alt.StrokeDash(dash, legend=None, scale=alt.Scale(domain=domain, range=[[1, 0], [6, 4]]))
    return alt.Chart(df).mark_line(strokeWidth=2.5).encode(**enc).properties(height=height)


def kpi(col, label, value, note=None):
    with col:
        st.metric(label, value, border=True)
        if note:
            st.caption(note)


# ------------------------------------------------------------------ sidebar inputs
st.sidebar.title("🌱 Field inputs")
preset_name = st.sidebar.selectbox("Load an example field", list(PRESETS))
p = PRESETS[preset_name]
k = preset_name  # widget keys change with the preset so its values load

crops = sorted(CROP_CYCLE_DAYS)
crop = st.sidebar.selectbox("Crop type", crops, index=crops.index(p["crop"]), key=f"crop{k}")
stage = st.sidebar.selectbox("Growth stage", GROWTH_STAGES, index=GROWTH_STAGES.index(p["stage"]), key=f"stage{k}")
st.sidebar.markdown("**Initial soil nutrients (kg/ha)**")
c1, c2, c3 = st.sidebar.columns(3)
N = c1.number_input("N", 0, 200, p["N"], key=f"N{k}")
P = c2.number_input("P", 0, 200, p["P"], key=f"P{k}")
K = c3.number_input("K", 0, 250, p["K"], key=f"K{k}")
st.sidebar.markdown("**Field conditions**")
temp = st.sidebar.slider("Soil temperature (°C)", 8.0, 48.0, p["temp"], 0.5, key=f"t{k}")
moisture = st.sidebar.slider("Soil moisture (% volumetric)", 5.0, 50.0, p["moisture"], 0.5, key=f"m{k}")
rainfall = st.sidebar.slider("Rainfall, last month (mm)", 15.0, 430.0, p["rainfall"], 5.0, key=f"r{k}")
ph = st.sidebar.slider("Soil pH", 3.5, 9.9, p["ph"], 0.1, key=f"ph{k}")
use_days = st.sidebar.checkbox("Coated fertilizer already applied?", key=f"ud{k}")
days = st.sidebar.slider("Days since application", 0, 150, 30, key=f"d{k}") if use_days else None

field = {"soil_temperature": temp, "soil_moisture": moisture, "rainfall": rainfall, "soil_pH": ph}
rec = system.recommend(field, crop, stage, N, P, K, days_since_application=days)
m1 = rec["module1"]

# ------------------------------------------------------------------ header
st.title("Smart Fertilizer Dashboard")
st.caption("AI-driven dual-module system: **Module 1** predicts how fast a coated fertilizer releases nitrogen in "
           "your field; **Module 2** uses that prediction to recommend the fertilizer type, amount and timing.")

tab_rec, tab_rel, tab_perf, tab_data = st.tabs(["Recommendation", "Release explorer", "Model performance", "Dataset"])

# ================================================================== Recommendation
with tab_rec:
    for w in rec["warnings"]:
        st.warning(f"⚠️ {w}")

    a, b, c, d = st.columns(4)
    name, _, detail = rec["fertilizer_type"].partition(" (")
    kpi(a, "Recommended fertilizer", name, f"{detail.rstrip(')')} · {rec['confidence']:.0%} model confidence")
    kpi(b, "Application amount", f"{rec['fertilizer_amount_kg_ha']:.0f} kg/ha",
        f"≈ {rec['nutrient_kg_ha']:.0f} kg/ha of nutrient")
    kpi(c, "Next application in", f"{rec['next_application_in_days']} days", f"{stage} stage of {crop}")
    kpi(d, "\"90-day\" coated urea lasts", f"{m1['effective_duration']:.0f} days",
        f"{m1['effective_duration'] - m1['rated_duration']:+.0f} days vs the label in this field")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Module 1: release in your field")
        t = np.arange(0, 181)
        curves = pd.concat([
            pd.DataFrame({"Day": t, "Released (%)": phys.cumulative_release(t, m1["effective_duration"]),
                          "Curve": "Your field (predicted)"}),
            pd.DataFrame({"Day": t, "Released (%)": phys.cumulative_release(t, m1["rated_duration"]),
                          "Curve": "Lab rating (25 °C)"}),
        ])
        chart = line_chart(curves, "Day:Q", "Released (%):Q", "Curve:N",
                           ["Your field (predicted)", "Lab rating (25 °C)"], [BLUE, GREY],
                           "Days since application", "Cumulative N released (%)", dash="Curve:N")
        rule = alt.Chart(pd.DataFrame({"y": [80]})).mark_rule(color=GREY, strokeDash=[2, 3]).encode(y="y:Q")
        layers = [chart, rule]
        if days is not None:
            pt = pd.DataFrame({"Day": [days], "Released (%)": [m1["cumulative_release"]]})
            layers.append(alt.Chart(pt).mark_point(size=110, filled=True, color=ORANGE, stroke="white",
                                                   strokeWidth=1.5).encode(
                x="Day:Q", y="Released (%):Q",
                tooltip=[alt.Tooltip("Day:Q", title="Day"), alt.Tooltip("Released (%):Q", format=".0f")]))
        st.altair_chart(alt.layer(*layers), use_container_width=True)
        msg = (f"Release rate here: **{m1['release_rate']:.2f}% per day**. A product rated for "
               f"{m1['rated_duration']:.0f} days reaches 80% release after about **{m1['effective_duration']:.0f} days** "
               "in these conditions.")
        if days is not None:
            msg += f" After {days} days, about **{m1['cumulative_release']:.0f}%** has been released."
        st.markdown(msg)

    with right:
        st.subheader("Module 2: product choice")
        probs = pd.DataFrame([(rec["fertilizer_type"], rec["confidence"])] + rec["alternatives"],
                             columns=["Fertilizer", "Probability"])
        bars = alt.Chart(probs).mark_bar(color=BLUE, cornerRadiusEnd=4, height=22).encode(
            x=alt.X("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title="Model probability"),
            y=alt.Y("Fertilizer:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            tooltip=["Fertilizer", alt.Tooltip("Probability:Q", format=".0%")])
        labels = bars.mark_text(align="left", dx=4, color="#52514e").encode(text=alt.Text("Probability:Q", format=".0%"))
        st.altair_chart((bars + labels).properties(height=150), use_container_width=True)
        spec = FERTILIZERS[rec["fertilizer_type"]]
        st.markdown(
            f"- **Why this product:** recommended for *{crop}* at the *{stage}* stage with soil N-P-K "
            f"{N}-{P}-{K}.\n"
            f"- **Nutrient content:** {spec['content']:.0%} {spec['nutrient']}"
            + (f", polymer coat {spec['thickness']} µm" if spec["thickness"] else "") + ".\n"
            f"- **Crop cycle:** {CROP_CYCLE_DAYS[crop]} days; the next dose is due in "
            f"{rec['next_application_in_days']} days.")

# ================================================================== Release explorer
with tab_rel:
    st.subheader("How long does each coat grade last as soil warms?")
    st.caption(f"Module 1 predictions at your soil moisture ({moisture:.0f}%), rainfall ({rainfall:.0f} mm) "
               f"and pH ({ph:.1f}), over the 15–44 °C range covered by the training data. Hover for values.")
    temps = np.arange(15, 44.5, 0.5)  # where 98% of the training data lies
    rows = []
    for name, thick in COAT_GRADES.items():
        X = pd.DataFrame({"soil_temperature": temps, "soil_moisture": moisture, "rainfall": rainfall,
                          "soil_pH": ph, "coating_thickness": thick})
        rate = np.clip(system.rate_model["model"].predict(X[system.rate_model["features"]]), 0.05, None)
        rows.append(pd.DataFrame({"Soil temperature (°C)": temps, "Days to 80% release": 80 / rate, "Product": name}))
    curves = pd.concat(rows)
    domain = list(COAT_GRADES)
    chart = line_chart(curves, "Soil temperature (°C):Q", "Days to 80% release:Q", "Product:N", domain,
                       [BLUE, ORANGE, AQUA], "Soil temperature (°C)", "Days to 80% release", height=380)
    chart = chart.encode(x=alt.X("Soil temperature (°C):Q", scale=alt.Scale(domain=[15, 44], nice=False)))
    rule = alt.Chart(pd.DataFrame({"Soil temperature (°C)": [temp]})).mark_rule(color=GREY, strokeDash=[4, 3]).encode(
        x="Soil temperature (°C):Q")
    st.altair_chart(chart + rule, use_container_width=True)
    cols = st.columns(3)
    for col, (name, thick) in zip(cols, COAT_GRADES.items()):
        dur = system.predict_release(field, coating_thickness=thick)["effective_duration"]
        rated = phys.rated_duration(thick)
        kpi(col, f"{name} at {temp:.0f} °C", f"{dur:.0f} days", f"label says {rated:.0f} days")

# ================================================================== Model performance
with tab_perf:
    m1s, m2s = summary["module1"], summary["module2"]
    cmp = m2s["system_comparison"]
    coated = cmp["application_timing_coated_urea"]
    red = 1 - coated["Proposed dual-module system"]["MAE"] / coated["Existing static system"]["MAE"]
    t = m2s["fertilizer_type"]

    a, b, c, d = st.columns(4)
    kpi(a, "Module 1 release-rate R²", f"{m1s['release_rate']['models'][m1s['release_rate']['best']]['R2']:.3f}",
        m1s["release_rate"]["best"])
    kpi(b, "Fertilizer-type accuracy", f"{t['models'][t['best']]['accuracy']:.1%}",
        f"{t['best']} · maximum possible {t['rule_ceiling_accuracy']:.1%}")
    kpi(c, "Timing R²", f"{m2s['application_timing']['models'][m2s['application_timing']['best']]['R2']:.3f}",
        m2s["application_timing"]["best"])
    kpi(d, "Coated-urea timing error vs static", f"−{red:.0%}",
        f"{coated['Existing static system']['MAE']:.1f} → {coated['Proposed dual-module system']['MAE']:.1f} days")

    left, right = st.columns(2)
    with left:
        st.subheader("Module 1: test R² by model")
        rows = [{"Model": n.replace(" (benchmark)", ""), "Target": lbl, "R²": v["R2"]}
                for tgt, lbl in [("release_rate", "Release rate"), ("cumulative_release", "Cumulative release")]
                for n, v in m1s[tgt]["models"].items()]
        df = pd.DataFrame(rows)
        bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("Model:N", title=None, sort=None, axis=alt.Axis(labelAngle=0, labelLimit=140)),
            xOffset=alt.XOffset("Target:N", sort=["Release rate", "Cumulative release"]),
            y=alt.Y("R²:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Target:N", scale=alt.Scale(domain=["Release rate", "Cumulative release"], range=[BLUE, ORANGE]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Model", "Target", alt.Tooltip("R²:Q", format=".3f")])
        st.altair_chart(bars.properties(height=320), use_container_width=True)
    with right:
        st.subheader("Module 2: test scores by model")
        rows = []
        for n in t["models"]:
            name = n.replace(" (baseline)", "")
            rows += [{"Model": name, "Metric": "Type accuracy", "Score": t["models"][n]["accuracy"]},
                     {"Model": name, "Metric": "Coated-grade accuracy", "Score": t["models"][n]["coated_urea_accuracy"]},
                     {"Model": name, "Metric": "Amount R²", "Score": m2s["fertilizer_amount"]["models"][n]["R2"]},
                     {"Model": name, "Metric": "Timing R²", "Score": m2s["application_timing"]["models"][n]["R2"]}]
        df = pd.DataFrame(rows)
        bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("Metric:N", title=None, sort=None, axis=alt.Axis(labelAngle=0, labelLimit=140)),
            xOffset=alt.XOffset("Model:N", sort=None), y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", sort=None, scale=alt.Scale(range=[AQUA, ORANGE, BLUE]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".3f")])
        st.altair_chart(bars.properties(height=320), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Static system vs proposed")
        rows = [{"System": s, "Products": lbl, "Timing MAE (days)": cmp[key][s]["MAE"]}
                for key, lbl in [("application_timing", "All products"), ("application_timing_coated_urea", "Coated urea")]
                for s in cmp[key]]
        df = pd.DataFrame(rows)
        systems = list(cmp["application_timing"])
        bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("Products:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("System:N", sort=systems),
            y=alt.Y("Timing MAE (days):Q"),
            color=alt.Color("System:N", sort=systems, scale=alt.Scale(domain=systems, range=[ORANGE, AQUA, BLUE]),
                            legend=alt.Legend(orient="bottom", title=None, columns=1)),
            tooltip=["System", "Products", alt.Tooltip("Timing MAE (days):Q", format=".1f")])
        st.altair_chart(bars.properties(height=340), use_container_width=True)
        st.caption("Mean absolute error in the date of the next application, on the 20% test set "
                   "(lower is better). All systems are given the same recommended product.")
    with right:
        st.subheader("Fertilizer-type confusion matrix")
        st.image(str(FIG_DIR / "m2_confusion_matrix.png"), width="stretch")

    with st.expander("Full metrics tables"):
        st.markdown("**Module 1**")
        st.dataframe(pd.DataFrame([{"Target": tgt, "Model": n, **{k: v[k] for k in ["R2", "RMSE", "MAE", "MSE", "CV_R2_mean"]}}
                                   for tgt in m1s for n, v in m1s[tgt]["models"].items()]).round(4),
                     hide_index=True, width="stretch")
        st.markdown("**Module 2: fertilizer type**")
        st.dataframe(pd.DataFrame([{"Model": n, **{k: v[k] for k in ["accuracy", "macro_F1", "coated_urea_accuracy",
                                                                     "CV_accuracy_mean"]}}
                                   for n, v in t["models"].items()]).round(4), hide_index=True, width="stretch")
        st.markdown("**Module 2: amount and timing**")
        st.dataframe(pd.DataFrame([{"Target": tgt, "Model": n, "R2": v["R2"], "MAE": v["MAE"], "RMSE": v["RMSE"],
                                    "End-to-end R2": v["end_to_end"]["R2"]}
                                   for tgt in ["fertilizer_amount", "application_timing"]
                                   for n, v in m2s[tgt]["models"].items()]).round(4),
                     hide_index=True, width="stretch")

# ================================================================== Dataset
with tab_data:
    real = load_csv(CROP_DATA)
    recd = load_csv(RECOMMEND_DATA)
    a, b, c, d = st.columns(4)
    kpi(a, "Real field records", f"{len(real):,}", "Kaggle Crop Recommendation dataset")
    kpi(b, "Crops", real["label"].nunique(), "100 field records each")
    kpi(c, "Generated rows per module", f"{len(recd):,}", "5 season scenarios per real record")
    lo, hi = np.percentile(phys.rated_duration(REFERENCE_THICKNESS) / recd["field_rate_factor"], [5, 95])
    kpi(d, "\"90-day\" product real duration", f"{lo:.0f}–{hi:.0f} days", "90% of field scenarios")

    left, right = st.columns(2)
    with left:
        st.subheader("Simulated release time of a \"90-day\" coated urea")
        dur = pd.DataFrame({"Days": (phys.rated_duration(REFERENCE_THICKNESS) / recd["field_rate_factor"]).clip(upper=200)})
        hist = alt.Chart(dur).mark_bar(color=BLUE).encode(
            x=alt.X("Days:Q", bin=alt.Bin(step=5), title="Days to 80% release in the field"),
            y=alt.Y("count():Q", title="Field scenarios"), tooltip=[alt.Tooltip("count():Q", title="Scenarios")])
        label = alt.Chart(pd.DataFrame({"x": [90]})).mark_rule(color=ORANGE, strokeWidth=2).encode(x="x:Q")
        st.altair_chart((hist + label).properties(height=300), use_container_width=True)
        st.caption("Orange line: the 90-day label. The same product lasts 40–126 days depending on field conditions.")
    with right:
        st.subheader("Recommended fertilizer classes")
        counts = recd["fertilizer_type"].value_counts().rename_axis("Fertilizer").reset_index(name="Rows")
        bars = alt.Chart(counts).mark_bar(color=BLUE, cornerRadiusEnd=3).encode(
            x=alt.X("Rows:Q"), y=alt.Y("Fertilizer:N", sort="-x", title=None), tooltip=["Fertilizer", "Rows"])
        st.altair_chart(bars.properties(height=300), use_container_width=True)
        st.caption("Class balance of the 11,000-row recommendation dataset.")

    st.subheader("Real field data (first 200 rows)")
    st.dataframe(real.head(200), width="stretch", hide_index=True)
