"""Module 1 - Nutrient Release Prediction.

Predicts, from field conditions and coating thickness:
  * release_rate        - % of the nutrient released per day
  * cumulative_release  - % released so far, given days since application
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

import plots
from config import (INK_2, M1_FEATURES, M1_TARGETS, MODEL_DIR, MUTED, OUTPUT_DIR,
                    RELEASE_DATA, SEED, SERIES, TEST_SIZE)

TARGET_LABELS = {"release_rate": "Release rate (% per day)",
                 "cumulative_release": "Cumulative release (%)"}


def candidate_models():
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Polynomial LR (deg 3)": make_pipeline(StandardScaler(), PolynomialFeatures(3),
                                               LinearRegression()),
        "Decision Tree": GridSearchCV(
            DecisionTreeRegressor(random_state=SEED),
            {"max_depth": [6, 10, 14, None], "min_samples_leaf": [1, 5, 10]},
            cv=5, scoring="neg_mean_squared_error", n_jobs=-1),
        "Random Forest (benchmark)": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    }


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MSE": mse, "RMSE": float(np.sqrt(mse)),
            "MAE": mean_absolute_error(y_true, y_pred), "R2": r2_score(y_true, y_pred)}


def train_target(df, target):
    X, y = df[M1_FEATURES[target]], df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    cv = KFold(5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, model in candidate_models().items():
        model.fit(X_tr, y_tr)
        est = model.best_estimator_ if isinstance(model, GridSearchCV) else model
        cv_r2 = cross_val_score(est, X_tr, y_tr, cv=cv, scoring="r2", n_jobs=-1)
        m = regression_metrics(y_te, model.predict(X_te))
        m["CV_R2_mean"], m["CV_R2_std"] = float(cv_r2.mean()), float(cv_r2.std())
        if isinstance(model, GridSearchCV):
            m["best_params"] = model.best_params_
        results[name], fitted[name] = m, est
        print(f"  {name:<26} test R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  "
              f"CV R2={m['CV_R2_mean']:.4f}±{m['CV_R2_std']:.4f}")

    best = max(results, key=lambda n: results[n]["CV_R2_mean"])  # select on CV, not test
    print(f"  -> best: {best}")
    return results, fitted, best, (X_te, y_te)


def plot_model_comparison(all_results):
    fig, axes = plots.plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, target in zip(axes, M1_TARGETS):
        res = all_results[target]["models"]
        names = list(res)
        r2 = [res[n]["R2"] for n in names]
        best = all_results[target]["best"]
        colors = [SERIES[0] if n == best else "#b7d3f6" for n in names]
        bars = ax.barh(names, r2, color=colors, height=0.6)
        for b, v in zip(bars, r2):
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                    va="center", color=INK_2, fontsize=9)
        ax.set_xlim(0, 1.12)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Test R² (higher is better)")
        ax.set_title(TARGET_LABELS[target])
    fig.suptitle("Module 1 model comparison (best model in dark blue)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    return plots.save(fig, "m1_model_comparison.png")


def plot_actual_vs_pred(all_results, fitted_best, test_sets):
    fig, axes = plots.plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, target in zip(axes, M1_TARGETS):
        X_te, y_te = test_sets[target]
        pred = fitted_best[target].predict(X_te)
        ax.scatter(y_te, pred, s=9, alpha=0.35, color=SERIES[0], edgecolors="none")
        lo, hi = float(min(y_te.min(), pred.min())), float(max(y_te.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
        ax.text(0.03, 0.95, f"{all_results[target]['best']}\n"
                f"R² = {r2_score(y_te, pred):.3f}", transform=ax.transAxes,
                va="top", color=INK_2, fontsize=9)
        ax.set_xlabel(f"Actual {TARGET_LABELS[target].lower()}")
        ax.set_ylabel("Predicted")
        ax.set_title(TARGET_LABELS[target])
    fig.suptitle("Module 1: actual vs predicted on the 20% test set", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    return plots.save(fig, "m1_actual_vs_predicted.png")


def plot_feature_importance(fitted):
    fig, axes = plots.plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, target in zip(axes, M1_TARGETS):
        tree = fitted[target]["Decision Tree"]
        imp = pd.Series(tree.feature_importances_, index=M1_FEATURES[target]).sort_values()
        ax.barh(imp.index.str.replace("_", " "), imp.values, color=SERIES[0], height=0.6)
        for i, v in enumerate(imp.values):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", color=INK_2, fontsize=9)
        ax.set_xlim(0, max(imp.values) * 1.18)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Decision-tree feature importance")
        ax.set_title(TARGET_LABELS[target])
    fig.suptitle("Module 1: which field conditions drive release", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    return plots.save(fig, "m1_feature_importance.png")


def main():
    plots.apply_style()
    df = pd.read_csv(RELEASE_DATA)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    all_results, fitted, fitted_best, test_sets = {}, {}, {}, {}
    for target in M1_TARGETS:
        print(f"\n[Module 1] target = {target}")
        res, models, best, test = train_target(df, target)
        all_results[target] = {"models": res, "best": best}
        fitted[target], fitted_best[target], test_sets[target] = models, models[best], test
        joblib.dump({"model": models[best], "features": M1_FEATURES[target], "name": best},
                    MODEL_DIR / f"module1_{target}.joblib", compress=3)

    plot_model_comparison(all_results)
    plot_actual_vs_pred(all_results, fitted_best, test_sets)
    plot_feature_importance(fitted)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "module1_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    return all_results


if __name__ == "__main__":
    main()
