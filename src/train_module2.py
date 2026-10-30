"""Module 2 - Fertilizer Recommendation Engine (+ pipeline integration).

Inputs: Module 1's predicted release rate + crop type, growth stage, initial
N-P-K and soil pH. Outputs: fertilizer type, amount (kg/ha), timing (days
until the next application).

Also evaluates
  * an ablation without Module 1 (a soil-test-only ML recommender), and
  * the existing static system (manufacturer rating at 25 °C, i.e. k = 1).
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import agronomy
import plots
from config import (CROP_REQ_DATA, FERTILIZER_CLASSES, INK, INK_2, M2_CATEGORICAL,
                    M2_FROM_MODULE1, M2_NUMERIC,
                    MODEL_DIR, MUTED, OUTPUT_DIR, RECOMMEND_DATA, REFERENCE_THICKNESS, SEED,
                    SERIES, TEST_SIZE)

warnings.filterwarnings("ignore", category=UserWarning)

# Amount and timing are sized for the product chosen in step 1.
REG_CATEGORICAL = M2_CATEGORICAL + ["fertilizer_type"]

try:
    from xgboost import XGBClassifier, XGBRegressor
    BOOST_NAME = "XGBoost"
except Exception:  # noqa: BLE001 - xgboost raises XGBoostError if libomp is missing
    XGBClassifier = XGBRegressor = None
    BOOST_NAME = "Gradient Boosting (HistGB)"


def boosted_classifier():
    if XGBClassifier is not None:
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                             subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                             n_jobs=2, eval_metric="mlogloss")
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                          max_leaf_nodes=48, random_state=SEED)


def boosted_regressor():
    if XGBRegressor is not None:
        return XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.06,
                            subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=2)
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06,
                                         max_leaf_nodes=48, random_state=SEED)


def classifiers():
    return {
        "Decision Tree (baseline)": DecisionTreeClassifier(max_depth=14, min_samples_leaf=3,
                                                           random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=1,
                                                random_state=SEED, n_jobs=-1),
        BOOST_NAME: boosted_classifier(),
    }


def regressors():
    return {
        "Decision Tree (baseline)": DecisionTreeRegressor(max_depth=14, min_samples_leaf=3,
                                                          random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                               random_state=SEED, n_jobs=-1),
        BOOST_NAME: boosted_regressor(),
    }


def make_pipeline(model, numeric, categorical=M2_CATEGORICAL):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", "passthrough", numeric),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def add_module1_predictions(df):
    """Pipeline integration: Module 1 -> predicted release rate of the reference coat."""
    m1 = joblib.load(MODEL_DIR / "module1_release_rate.joblib")
    rate = m1["model"].predict(df.assign(coating_thickness=REFERENCE_THICKNESS)[m1["features"]])
    return df.assign(**agronomy.module1_features(rate, df["crop_type"], df["growth_stage"]))


def rule_ceiling(df, req):
    """Accuracy of the exact labelling rule (true release factor, no jitter):
    the best any model can do given label noise."""
    pred = np.array([agronomy.choose_fertilizer(req.loc[r.crop_type], {"N": r.N, "P": r.P, "K": r.K},
                                                r.crop_type, r.growth_stage, r.field_rate_factor)
                     for r in df.itertuples(index=False)])
    return accuracy_score(df["fertilizer_type"], pred)


def reg_metrics(y, p):
    return {"MSE": mean_squared_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": mean_absolute_error(y, p), "R2": r2_score(y, p)}


def fit_classifier(name, model, numeric, X_tr, y_tr, X_te, y_te, enc):
    pipe = make_pipeline(model, numeric)
    pipe.fit(X_tr[M2_CATEGORICAL + numeric], enc.transform(y_tr))
    pred = enc.inverse_transform(pipe.predict(X_te[M2_CATEGORICAL + numeric]))
    return pipe, pred, {"accuracy": accuracy_score(y_te, pred),
                        "macro_F1": f1_score(y_te, pred, average="macro"),
                        "weighted_F1": f1_score(y_te, pred, average="weighted")}


def static_baseline(test, req):
    """What today's static system recommends: same product, lab rating (k = 1)."""
    amounts, timings = [], []
    for row in test.itertuples(index=False):
        r = req.loc[row.crop_type]
        needs = agronomy.nutrient_needs(r, {"N": row.N, "P": row.P, "K": row.K}, row.growth_stage)
        a, t = agronomy.dose_and_timing(row.fertilizer_type, needs, row.crop_type,
                                        row.growth_stage, k=1.0)
        amounts.append(a)
        timings.append(t)
    return np.array(amounts), np.array(timings)


# ---------------------------------------------------------------- figures
def plot_confusion(y_te, pred, name):
    labels = FERTILIZER_CLASSES
    cm = confusion_matrix(y_te, pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plots.plt.subplots(figsize=(8.6, 7))
    ax.imshow(cm_norm, cmap=plots.SEQ_CMAP, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9,
                        color="white" if cm_norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted fertilizer")
    ax.set_ylabel("Recommended (ground truth)")
    ax.set_title(f"Fertilizer-type confusion matrix, test set\n{name}")
    for s in ax.spines.values():
        s.set_visible(False)
    return plots.save(fig, "m2_confusion_matrix.png")


def plot_model_comparison(results):
    panels = [("fertilizer_type", "accuracy", "Fertilizer type: accuracy"),
              ("fertilizer_amount", "R2", "Amount: test R²"),
              ("application_timing", "R2", "Timing: test R²")]
    fig, axes = plots.plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (target, metric, title) in zip(axes, panels):
        res = results[target]["models"]
        names = list(res)
        vals = [res[n][metric] for n in names]
        best = results[target]["best"]
        bars = ax.barh(names, vals, height=0.6,
                       color=[SERIES[0] if n == best else "#b7d3f6" for n in names])
        for b, v in zip(bars, vals):
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center",
                    color=INK_2, fontsize=9)
        ax.set_xlim(0, 1.15)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle("Module 2 model comparison (best model in dark blue)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    return plots.save(fig, "m2_model_comparison.png")


def plot_amount_timing(y_amt, p_amt, y_tim, p_tim):
    fig, axes = plots.plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, y, p, label in [(axes[0], y_amt, p_amt, "Fertilizer amount (kg/ha)"),
                            (axes[1], y_tim, p_tim, "Application timing (days)")]:
        ax.scatter(y, p, s=9, alpha=0.35, color=SERIES[0], edgecolors="none")
        lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
        ax.text(0.03, 0.95, f"R² = {r2_score(y, p):.3f}\nMAE = {mean_absolute_error(y, p):.1f}",
                transform=ax.transAxes, va="top", color=INK_2, fontsize=9)
        ax.set_xlabel(f"Actual {label.lower()}")
        ax.set_ylabel("Predicted")
        ax.set_title(label)
    fig.suptitle("Module 2: actual vs predicted on the 20% test set", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    return plots.save(fig, "m2_actual_vs_predicted.png")


def plot_system_comparison(comparison):
    """Timing and amount error: static system vs ML without Module 1 vs full system."""
    systems = list(comparison["application_timing"])
    colors = [SERIES[1], "#b7d3f6", SERIES[0]]
    panels = [("application_timing", "d", "Timing error, all products (MAE)"),
              ("application_timing_coated_urea", "d", "Timing error, coated urea (MAE)"),
              ("fertilizer_amount", "kg/ha", "Amount error (MAE)")]
    fig, axes = plots.plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, (target, unit, title) in zip(axes, panels):
        vals = [comparison[target][s]["MAE"] for s in systems]
        bars = ax.barh(systems, vals, color=colors, height=0.6)
        for b, v in zip(bars, vals):
            ax.text(v + max(vals) * 0.015, b.get_y() + b.get_height() / 2, f"{v:.1f} {unit}",
                    va="center", color=INK_2, fontsize=9)
        ax.set_xlim(0, max(vals) * 1.3)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle("Existing static system vs proposed dual-module system (lower is better)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    return plots.save(fig, "system_comparison.png")


# ---------------------------------------------------------------- main
def main():
    plots.apply_style()
    req = pd.read_csv(CROP_REQ_DATA).set_index("Crop")[["N", "P", "K"]]
    df = add_module1_predictions(pd.read_csv(RECOMMEND_DATA))
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED,
                                   stratify=df["fertilizer_type"])
    enc = LabelEncoder().fit(FERTILIZER_CLASSES)
    no_m1 = [c for c in M2_NUMERIC if c not in M2_FROM_MODULE1]
    print(f"\n[Module 2] boosting backend: {BOOST_NAME}")
    print(f"  Module 1 -> Module 2 release-rate R2 vs truth: "
          f"{r2_score(df['reference_release_rate'], df['predicted_release_rate']):.4f}")

    results, best_pipes, preds = {}, {}, {}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pcu = test["fertilizer_type"].str.startswith("PCU").to_numpy()

    # ---- step 1: fertilizer type
    print("\n[Module 2] target = fertilizer_type")
    res, pipes, cls_preds = {}, {}, {}
    for name, model in classifiers().items():
        pipe, pred, m = fit_classifier(name, model, M2_NUMERIC, train, train["fertilizer_type"],
                                       test, test["fertilizer_type"], enc)
        cv_acc = cross_val_score(make_pipeline(model, M2_NUMERIC),
                                 train[M2_CATEGORICAL + M2_NUMERIC],
                                 enc.transform(train["fertilizer_type"]), cv=cv, n_jobs=-1)
        m["CV_accuracy_mean"], m["CV_accuracy_std"] = float(cv_acc.mean()), float(cv_acc.std())
        m["coated_urea_accuracy"] = accuracy_score(test["fertilizer_type"][pcu], pred[pcu])
        res[name], pipes[name], cls_preds[name] = m, pipe, pred
        print(f"  {name:<28} acc={m['accuracy']:.4f}  macroF1={m['macro_F1']:.4f}  "
              f"CV acc={m['CV_accuracy_mean']:.4f}±{m['CV_accuracy_std']:.4f}  "
              f"PCU-grade acc={m['coated_urea_accuracy']:.4f}")
    best = max(res, key=lambda n: res[n]["CV_accuracy_mean"])
    results["fertilizer_type"] = {"models": res, "best": best,
                                  "rule_ceiling_accuracy": rule_ceiling(test, req)}
    best_pipes["fertilizer_type"], preds["fertilizer_type"] = pipes[best], cls_preds[best]
    _, abl_type, abl = fit_classifier(best, classifiers()[best], no_m1, train,
                                      train["fertilizer_type"], test, test["fertilizer_type"], enc)
    abl["coated_urea_accuracy"] = accuracy_score(test["fertilizer_type"][pcu], abl_type[pcu])
    results["fertilizer_type"]["without_module1"] = abl
    print(f"  noise ceiling (exact rule on noisy labels): "
          f"{results['fertilizer_type']['rule_ceiling_accuracy']:.4f}")
    print(f"  -> best: {best};  without Module 1: acc={abl['accuracy']:.4f}  "
          f"PCU-grade acc={abl['coated_urea_accuracy']:.4f}")

    # ---- step 2: amount & timing for the recommended product (chained on type)
    test_e2e = test.assign(fertilizer_type=preds["fertilizer_type"])
    test_e2e_abl = test.assign(fertilizer_type=abl_type)
    ablation_reg = {}
    for target in ["fertilizer_amount", "application_timing"]:
        print(f"\n[Module 2] target = {target}")
        res, pipes, reg_preds = {}, {}, {}
        for name, model in regressors().items():
            pipe = make_pipeline(model, M2_NUMERIC, REG_CATEGORICAL).fit(train, train[target])
            p = pipe.predict(test)
            m = reg_metrics(test[target], p)
            cv_r2 = cross_val_score(make_pipeline(model, M2_NUMERIC, REG_CATEGORICAL), train,
                                    train[target], cv=KFold(5, shuffle=True, random_state=SEED),
                                    scoring="r2", n_jobs=-1)
            m["CV_R2_mean"], m["CV_R2_std"] = float(cv_r2.mean()), float(cv_r2.std())
            m["end_to_end"] = reg_metrics(test[target], pipe.predict(test_e2e))
            res[name], pipes[name], reg_preds[name] = m, pipe, p
            print(f"  {name:<28} R2={m['R2']:.4f}  MAE={m['MAE']:.3f}  "
                  f"CV R2={m['CV_R2_mean']:.4f}±{m['CV_R2_std']:.4f}  "
                  f"end-to-end R2={m['end_to_end']['R2']:.4f}")
        best = max(res, key=lambda n: res[n]["CV_R2_mean"])
        abl_pipe = make_pipeline(regressors()[best], no_m1, REG_CATEGORICAL).fit(train, train[target])
        ablation_reg[target] = abl_pipe.predict(test)
        results[target] = {
            "models": res, "best": best,
            "without_module1": reg_metrics(test[target], ablation_reg[target]),
            "without_module1_end_to_end": reg_metrics(test[target], abl_pipe.predict(test_e2e_abl)),
        }
        best_pipes[target], preds[target] = pipes[best], reg_preds[best]
        print(f"  -> best: {best};  without Module 1: "
              f"R2={results[target]['without_module1']['R2']:.4f}")

    # ---- existing static system vs proposed (all given the same recommended product)
    static_amt, static_tim = static_baseline(test, req)
    comparison = {}
    for target, static in [("application_timing", static_tim), ("fertilizer_amount", static_amt)]:
        comparison[target] = {
            "Existing static system": reg_metrics(test[target], static),
            "ML without Module 1": reg_metrics(test[target], ablation_reg[target]),
            "Proposed dual-module system": reg_metrics(test[target], preds[target]),
        }
    # Coated-urea rows: where the static lab rating matters most.
    comparison["application_timing_coated_urea"] = {
        k: reg_metrics(test["application_timing"][pcu], v[pcu]) for k, v in [
            ("Existing static system", static_tim),
            ("ML without Module 1", ablation_reg["application_timing"]),
            ("Proposed dual-module system", preds["application_timing"])]}
    results["system_comparison"] = comparison
    print("\n[System comparison] MAE  (static | no-Module-1 | proposed)")
    for target, c in comparison.items():
        print(f"  {target:<31} " + " | ".join(f"{v['MAE']:.2f}" for v in c.values()))

    # ---- save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipelines": best_pipes, "label_encoder": enc,
                 "categorical": M2_CATEGORICAL, "regression_categorical": REG_CATEGORICAL,
                 "numeric": M2_NUMERIC,
                 "names": {t: results[t]["best"] for t in best_pipes}},
                MODEL_DIR / "module2_recommender.joblib", compress=3)
    with open(OUTPUT_DIR / "module2_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    plot_confusion(test["fertilizer_type"], preds["fertilizer_type"],
                   results["fertilizer_type"]["best"])
    plot_model_comparison(results)
    plot_amount_timing(test["fertilizer_amount"].to_numpy(), preds["fertilizer_amount"],
                       test["application_timing"].to_numpy(), preds["application_timing"])
    plot_system_comparison(comparison)
    return results


if __name__ == "__main__":
    main()
