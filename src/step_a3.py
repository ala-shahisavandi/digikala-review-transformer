"""
Step A3: Target distributions.

- Recommendation classes (recommended / no_idea / not_recommended / missing), RAW and CLEAN.
- rating_band from raw rate: low [0,2.5), medium [2.5,4.0), high [4.0,5.0]; everything else invalid.
- Raw rate statistics (min, max, mean, median, std, quantiles, n_unique, fractional share).
- rate vs recommendation_status and rating_band vs recommendation_status.
- Diagnostics for suspicious rate=0 and fractional rates (by year and is_buyer).

Run from repo root:  python src/step_a3.py
Outputs: artifacts/eda/step_a3_targets.json, reports/figures/A3_target_distributions.png
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

RAW_PARQUET = Path("data/interim/digikala_comments.parquet")
CLEAN_PARQUET = Path("data/interim/digikala_clean.parquet")
OUT_JSON = Path("artifacts/eda/step_a3_targets.json")
OUT_FIG = Path("reports/figures/A3_target_distributions.png")

REC_ORDER = ["recommended", "no_idea", "not_recommended", "missing"]
BAND_ORDER = ["low", "medium", "high", "invalid"]
RATE_ORDER = ["0", "1", "2", "3", "4", "5", "fractional", "invalid"]
REC_COLORS = {"recommended": "#388e3c", "no_idea": "#9e9e9e",
              "not_recommended": "#d32f2f", "missing": "#1976d2"}


def build_frame(path: Path) -> pl.DataFrame:
    s = pl.col("recommendation_status").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    rate = pl.col("rate")
    invalid = rate.is_null() | (rate < 0) | (rate > 5)
    return pl.scan_parquet(path).select([
        rate,
        pl.col("is_buyer"),
        pl.when(pl.col("recommendation_status").is_null() | s.is_in(["", "nan", "none", "null"]))
        .then(pl.lit("missing")).otherwise(s).alias("rec_class"),
        pl.when(invalid).then(pl.lit("invalid"))
        .when(rate < 2.5).then(pl.lit("low"))
        .when(rate < 4.0).then(pl.lit("medium"))
        .otherwise(pl.lit("high")).alias("rating_band"),
        pl.when(invalid).then(pl.lit("invalid"))
        .when(rate % 1 == 0).then(rate.cast(pl.Int64).cast(pl.Utf8))
        .otherwise(pl.lit("fractional")).alias("rate_group"),
        pl.col("created_at").cast(pl.Utf8).str.extract(r"(\d{4})\s*$", 1)
        .fill_null("unparsed").alias("year"),
    ]).collect(engine="streaming")


def dist(df: pl.DataFrame, col: str, order: list) -> list:
    total = len(df)
    counts = {r[col]: r["len"] for r in df.group_by(col).len().to_dicts()}
    return [{"class": k, "count": int(counts.get(k, 0)),
             "pct": round(counts.get(k, 0) / total * 100, 3)} for k in order]


def rate_stats(series: pl.Series) -> dict:
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    return {
        "n": len(series),
        "min": float(series.min()), "max": float(series.max()),
        "mean": round(float(series.mean()), 4), "median": float(series.median()),
        "std": round(float(series.std()), 4),
        "quantiles": {f"p{int(q*100)}": float(series.quantile(q)) for q in qs},
        "n_unique": int(series.n_unique()),
        "fractional_share_pct": round(float(((series % 1) != 0).mean()) * 100, 3),
    }


def crosstab(df: pl.DataFrame, row: str, order: list) -> list:
    out = []
    g = df.group_by([row, "rec_class"]).len().to_dicts()
    table = {}
    for r in g:
        table.setdefault(r[row], {})[r["rec_class"]] = r["len"]
    for k in order:
        cells = table.get(k, {})
        n = sum(cells.values())
        labeled = sum(cells.get(c, 0) for c in REC_ORDER[:3])
        out.append({
            row: k, "n": n,
            **{f"{c}": cells.get(c, 0) for c in REC_ORDER},
            **{f"{c}_pct_of_row": round(cells.get(c, 0) / n * 100, 2) if n else None for c in REC_ORDER},
            **{f"{c}_pct_of_labeled": round(cells.get(c, 0) / labeled * 100, 2) if labeled else None
               for c in REC_ORDER[:3]},
        })
    return out


def print_dist(title: str, rows: list) -> None:
    print(f"\n{title}")
    for r in rows:
        print(f"  {r['class']:>16s}: {r['count']:>10,}  ({r['pct']:.3f}%)")


def print_crosstab(rows: list, key: str) -> None:
    print(f"{key:>11s}{'n':>11s}{'recomm%lab':>12s}{'no_idea%lab':>13s}{'not_rec%lab':>13s}{'missing%row':>13s}")
    for r in rows:
        if r["n"]:
            print(f"{r[key]:>11s}{r['n']:>11,}{r['recommended_pct_of_labeled']!s:>12}"
                  f"{r['no_idea_pct_of_labeled']!s:>13}{r['not_recommended_pct_of_labeled']!s:>13}"
                  f"{r['missing_pct_of_row']!s:>13}")


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    df_raw = build_frame(RAW_PARQUET)
    df = build_frame(CLEAN_PARQUET)

    print("=" * 70)
    print("A3-1: CLASS DISTRIBUTIONS")
    print("=" * 70)
    rec_raw, rec_clean = dist(df_raw, "rec_class", REC_ORDER), dist(df, "rec_class", REC_ORDER)
    band_raw, band_clean = dist(df_raw, "rating_band", BAND_ORDER), dist(df, "rating_band", BAND_ORDER)
    print_dist("recommendation (RAW)", rec_raw)
    print_dist("recommendation (CLEAN)", rec_clean)
    print_dist("rating_band (RAW)", band_raw)
    print_dist("rating_band (CLEAN)", band_clean)

    def imbalance(rows: list, classes: list) -> dict:
        c = {r["class"]: r["count"] for r in rows if r["class"] in classes}
        mx, mn = max(c, key=c.get), min(c, key=c.get)
        return {"majority": mx, "minority": mn, "ratio": round(c[mx] / c[mn], 2)}

    imb_rec = imbalance(rec_clean, REC_ORDER[:3])
    imb_band = imbalance(band_clean, BAND_ORDER[:3])
    print(f"\nImbalance (clean, labeled classes): recommendation {imb_rec} | band {imb_band}")

    print("\n" + "=" * 70)
    print("A3-2: RAW rate STATISTICS (clean dataset)")
    print("=" * 70)
    stats_all = rate_stats(df["rate"])
    stats_valid = rate_stats(df.filter(pl.col("rating_band") != "invalid")["rate"])
    print("All rows:", json.dumps(stats_all))
    print("Valid [0,5] only:", json.dumps(stats_valid))

    print("\n" + "=" * 70)
    print("A3-3: rate vs recommendation_status (% of labeled rows; missing = % of all rows)")
    print("=" * 70)
    ct_rate = crosstab(df, "rate_group", RATE_ORDER)
    print_crosstab(ct_rate, "rate_group")
    print("\nrating_band vs recommendation_status")
    ct_band = crosstab(df, "rating_band", BAND_ORDER)
    print_crosstab(ct_band, "rating_band")

    print("\n" + "=" * 70)
    print("A3-4: DIAGNOSTICS for suspicious rate=0 and fractional rates")
    print("=" * 70)

    def share_table(key: str) -> list:
        return (
            df.group_by(key)
            .agg(pl.len().alias("n"),
                 (pl.col("rate_group") == "0").sum().alias("rate0"),
                 (pl.col("rate_group") == "fractional").sum().alias("fractional"))
            .with_columns((pl.col("rate0") / pl.col("n") * 100).round(2).alias("rate0_pct"),
                          (pl.col("fractional") / pl.col("n") * 100).round(2).alias("fractional_pct"))
            .sort(key).to_dicts()
        )

    by_year = share_table("year")
    print(f"{'year':>9s}{'n':>11s}{'rate0':>10s}{'rate0%':>9s}{'fract':>10s}{'fract%':>9s}")
    for r in by_year:
        print(f"{r['year']:>9s}{r['n']:>11,}{r['rate0']:>10,}{r['rate0_pct']:>9.2f}"
              f"{r['fractional']:>10,}{r['fractional_pct']:>9.2f}")
    by_buyer = share_table("is_buyer")
    print("\nBy is_buyer:")
    for r in by_buyer:
        print(f"  is_buyer={r['is_buyer']}: n={r['n']:,} rate0={r['rate0_pct']}% fractional={r['fractional_pct']}%")

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)

    ax = axes[0, 0]
    vals = [r["count"] for r in rec_clean]
    bars = ax.bar(REC_ORDER, vals, color=[REC_COLORS[c] for c in REC_ORDER], edgecolor="white")
    for b, r in zip(bars, rec_clean):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{r['pct']:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("A3-a: recommendation_status classes (clean)", fontweight="bold")
    ax.set_ylabel("Rows")

    ax = axes[0, 1]
    vals = [r["count"] for r in band_clean]
    bars = ax.bar(BAND_ORDER, vals, color=["#d32f2f", "#f9a825", "#388e3c", "#9e9e9e"], edgecolor="white")
    for b, r in zip(bars, band_clean):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{r['pct']:.2f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("A3-b: rating_band classes (clean)", fontweight="bold")
    ax.set_ylabel("Rows")

    ax = axes[1, 0]
    groups = [r for r in ct_rate if r["n"] and r["rate_group"] != "invalid"]
    labels = [r["rate_group"] for r in groups]
    bottom = np.zeros(len(groups))
    for c in REC_ORDER:
        v = np.array([r[f"{c}_pct_of_row"] for r in groups], dtype=float)
        ax.bar(labels, v, bottom=bottom, color=REC_COLORS[c], label=c, edgecolor="white")
        bottom += v
    ax.set_title("A3-c: recommendation mix by rate value (row %)", fontweight="bold")
    ax.set_ylabel("% of rows")
    ax.set_xlabel("rate value")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 1]
    yrs = [r for r in by_year if r["year"] != "unparsed" and r["n"] >= 1000]
    ax.plot([r["year"] for r in yrs], [r["rate0_pct"] for r in yrs], marker="o", label="rate = 0 (%)")
    ax.plot([r["year"] for r in yrs], [r["fractional_pct"] for r in yrs], marker="s", label="fractional rate (%)")
    ax.set_title("A3-d: share of rate=0 and fractional rate by year", fontweight="bold")
    ax.set_ylabel("% of rows in year")
    ax.set_xlabel("Persian year")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)

    result = {
        "band_definition": {"low": "0 <= rate < 2.5", "medium": "2.5 <= rate < 4.0",
                            "high": "4.0 <= rate <= 5.0", "invalid": "null, <0 or >5"},
        "recommendation_raw": rec_raw, "recommendation_clean": rec_clean,
        "rating_band_raw": band_raw, "rating_band_clean": band_clean,
        "imbalance_recommendation": imb_rec, "imbalance_band": imb_band,
        "rate_stats_all_rows": stats_all, "rate_stats_valid_0_5": stats_valid,
        "rate_vs_recommendation": ct_rate, "band_vs_recommendation": ct_band,
        "by_year": by_year, "by_is_buyer": by_buyer,
        "figure_path": str(OUT_FIG),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_FIG}")


if __name__ == "__main__":
    main()
