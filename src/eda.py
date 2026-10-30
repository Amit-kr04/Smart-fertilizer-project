"""Exploratory figures: the problem (release-time swing) and the datasets."""
import numpy as np
import pandas as pd

import plots
import release_physics as phys
from config import (CROP_DATA, INK_2, MUTED, RECOMMEND_DATA, REFERENCE_THICKNESS, RELEASE_DATA,
                    SERIES)


def release_curves():
    """Same "90-day" coated urea, three field climates."""
    days = np.arange(0, 191)
    # (label, soil °C, moisture %, rainfall mm, color, label position)
    scenarios = [("Cool & dry: 18 °C, 12% moisture", 18, 12, 40, SERIES[0], (118, 38)),
                 ("Lab rating: 25 °C, 30% moisture", 25, 30, 100, MUTED, (97, 70)),
                 ("Hot & wet: 35 °C, 40% moisture", 35, 40, 220, SERIES[1], (4, 90))]
    fig, ax = plots.plt.subplots(figsize=(9, 4.6))
    for label, t, m, r, color, pos in scenarios:
        dur = float(phys.effective_duration(REFERENCE_THICKNESS, t, m, r, 6.5))
        ax.plot(days, phys.cumulative_release(days, dur), color=color,
                ls="--" if color == MUTED else "-")
        ax.annotate(f"{label}\n80% released by day {dur:.0f}", xy=(dur, 80), xytext=pos,
                    fontsize=9, color=INK_2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=5))
        ax.plot([dur], [80], "o", color=color, ms=7, mec="white", mew=1.5)
    ax.axhline(80, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(0, 190)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Days since application")
    ax.set_ylabel("Cumulative N released (%)")
    ax.set_title('One "90-day" coated urea, three field climates')
    return plots.save(fig, "release_curves.png")


def duration_histogram():
    """Real release duration of a 90-day product across the 11,000 field scenarios."""
    df = pd.read_csv(RECOMMEND_DATA)
    dur = phys.rated_duration(REFERENCE_THICKNESS) / df["field_rate_factor"]
    lo, hi = np.percentile(dur, [5, 95])
    fig, ax = plots.plt.subplots(figsize=(9, 4.2))
    ax.hist(dur.clip(upper=200), bins=60, color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.6)
    ax.axvline(90, color=SERIES[1], lw=2)
    ax.text(92, ax.get_ylim()[1] * 0.92, "Label says 90 days", color=INK_2, fontsize=9)
    ax.axvspan(lo, hi, color=SERIES[0], alpha=0.07, lw=0)
    ax.text(hi + 2, ax.get_ylim()[1] * 0.80, f"Shaded: 90% of fields\nlast {lo:.0f}–{hi:.0f} days",
            color=INK_2, fontsize=9)
    ax.set_xlabel("Actual days to 80% release in the field")
    ax.set_ylabel("Field scenarios")
    ax.set_title('How long a "90-day" coated urea really lasts across the dataset')
    plots.save(fig, "duration_histogram.png")
    return float(lo), float(hi), float(np.median(dur))


def correlation_heatmap():
    df = pd.read_csv(RELEASE_DATA)
    cols = ["soil_temperature", "soil_moisture", "rainfall", "coating_thickness", "soil_pH",
            "days_since_application", "release_rate", "cumulative_release"]
    corr = df[cols].corr()
    labels = [c.replace("_", " ") for c in cols]
    fig, ax = plots.plt.subplots(figsize=(8, 6.6))
    im = ax.imshow(corr, cmap=plots.DIV_CMAP, vmin=-1, vmax=1)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(v) > 0.6 else "#0b0b0b")
    ax.set_xticks(range(len(cols)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(cols)), labels)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    ax.set_title("Module 1 dataset: Pearson correlations")
    return plots.save(fig, "correlation_heatmap.png")


def real_data_overview():
    """Soil N-P-K by crop in the real Kaggle field data."""
    df = pd.read_csv(CROP_DATA)
    med = df.groupby("label")[["N", "P", "K"]].median().sort_values("N")
    fig, ax = plots.plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(med))
    w = 0.27
    for i, (col, color) in enumerate(zip(["N", "P", "K"], SERIES[:3])):
        ax.bar(x + (i - 1) * w, med[col], width=w - 0.03, color=color, label=col)
    ax.set_xticks(x, med.index, rotation=45, ha="right")
    ax.set_ylabel("Median soil level (kg/ha)")
    ax.legend(ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    ax.set_title("Real field data: initial soil N-P-K by crop (2,200 records)")
    return plots.save(fig, "real_data_npk_by_crop.png")


def main():
    plots.apply_style()
    release_curves()
    lo, hi, med = duration_histogram()
    correlation_heatmap()
    real_data_overview()
    print(f"[EDA] '90-day' coated urea real duration: median {med:.0f} d, "
          f"90% of fields {lo:.0f}–{hi:.0f} d")
    return {"duration_p5": lo, "duration_p95": hi, "duration_median": med}


if __name__ == "__main__":
    main()
