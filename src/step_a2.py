"""
Step A2: Missing-value patterns.

1) Per-column missing count/percentage (null OR '', 'nan', 'none', 'null') on RAW and CLEAN.
2) Is a missing recommendation_status related to is_buyer, rate, product frequency,
   review year, text length?  Reports missing-share per group and Cramer's V (effect size).

Run from repo root:  python src/step_a2.py
Outputs: artifacts/eda/step_a2_missing.json, reports/figures/A2_missing_patterns.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

RAW_PARQUET = Path("data/interim/digikala_comments.parquet")
CLEAN_PARQUET = Path("data/interim/digikala_clean.parquet")
OUT_JSON = Path("artifacts/eda/step_a2_missing.json")
OUT_FIG = Path("reports/figures/A2_missing_patterns.png")

ORIGINAL_COLUMNS = [
    "id", "title", "body", "created_at", "rate", "recommendation_status", "is_buyer",
    "product_id", "advantages", "disadvantages", "likes", "dislikes",
    "seller_title", "seller_code", "true_to_size_rate",
]
MISSING_TOKENS = ["", "nan", "none", "null"]
PERSIAN_DIGITS = {"۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
                  "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
                  "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
                  "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9"}

PROD_ORDER = ["1", "2-5", "6-20", "21-99", "100-199", "200+"]
BODY_ORDER = ["no_text", "1-20", "21-50", "51-100", "101-300", ">300"]
RATE_ORDER = ["0", "1", "2", "3", "4", "5", "invalid", "missing"]


def norm_str(c: str) -> pl.Expr:
    return pl.col(c).cast(pl.Utf8).str.strip_chars().str.to_lowercase()


def is_missing(c: str) -> pl.Expr:
    return (pl.col(c).is_null() | norm_str(c).is_in(MISSING_TOKENS)).fill_null(False)


def column_missing(path: Path) -> list:
    lf = pl.scan_parquet(path)
    cols = [c for c in ORIGINAL_COLUMNS if c in lf.collect_schema().names()]
    exprs = [pl.len().alias("__n")]
    for c in cols:
        exprs.append(pl.col(c).is_null().sum().alias(f"{c}__null"))
        exprs.append(is_missing(c).sum().alias(f"{c}__miss"))
    row = lf.select(exprs).collect(engine="streaming").row(0, named=True)
    n = row["__n"]
    schema = lf.collect_schema()
    out = []
    for c in cols:
        miss = int(row[f"{c}__miss"])
        out.append({
            "column": c,
            "dtype": str(schema[c]),
            "null_count": int(row[f"{c}__null"]),
            "literal_nan_or_empty": miss - int(row[f"{c}__null"]),
            "missing_total": miss,
            "missing_pct": round(miss / n * 100, 3),
        })
    return out


def ascii_digits(e: pl.Expr) -> pl.Expr:
    for k, v in PERSIAN_DIGITS.items():
        e = e.str.replace_all(k, v, literal=True)
    return e


def cramers_v(groups: list) -> float:
    n = np.array([g["n"] for g in groups], dtype=float)
    m = np.array([g["missing"] for g in groups], dtype=float)
    keep = n > 0
    n, m = n[keep], m[keep]
    if len(n) < 2:
        return 0.0
    obs = np.vstack([m, n - m])
    total = obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((obs - exp) ** 2 / exp)
    return float(np.sqrt(chi2 / total))  # 2 x k table -> min(r,c)-1 = 1


def group_stats(df: pl.DataFrame, col: str, order: list | None = None) -> dict:
    g = (
        df.group_by(col)
        .agg(pl.len().alias("n"), pl.col("rec_missing").sum().alias("missing"))
        .with_columns((pl.col("missing") / pl.col("n") * 100).round(2).alias("missing_pct"))
    )
    rows = g.to_dicts()
    if order:
        rows.sort(key=lambda r: order.index(r[col]) if r[col] in order else 999)
    else:
        rows.sort(key=lambda r: str(r[col]))
    pcts = [r["missing_pct"] for r in rows if r["n"] >= 1000]
    return {
        "factor": col,
        "groups": rows,
        "cramers_v": round(cramers_v(rows), 4),
        "missing_pct_min_group_n_ge_1000": min(pcts) if pcts else None,
        "missing_pct_max_group_n_ge_1000": max(pcts) if pcts else None,
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("A2-1: MISSING VALUES PER COLUMN (null OR '', nan, none, null)")
    print("=" * 70)
    col_raw = column_missing(RAW_PARQUET)
    col_clean = column_missing(CLEAN_PARQUET)
    print(f"{'column':24s}{'dtype':10s}{'raw_missing':>14s}{'raw_%':>9s}{'clean_missing':>16s}{'clean_%':>9s}")
    for r, c in zip(col_raw, col_clean):
        print(f"{r['column']:24s}{r['dtype']:10s}{r['missing_total']:>14,}{r['missing_pct']:>9.3f}"
              f"{c['missing_total']:>16,}{c['missing_pct']:>9.3f}")

    lf = pl.scan_parquet(CLEAN_PARQUET)
    cnt = pl.len().over("product_id")
    body_len = pl.col("body").str.len_chars()
    year = (
        ascii_digits(pl.col("created_at").cast(pl.Utf8))
        .str.extract(r"(\d{4})\s*$", 1)
        .fill_null("unparsed")
    )
    df = lf.select([
        is_missing("recommendation_status").alias("rec_missing"),
        pl.col("is_buyer").cast(pl.Utf8).alias("is_buyer"),
        pl.when(pl.col("rate").is_null()).then(pl.lit("missing"))
        .when((pl.col("rate") < 0) | (pl.col("rate") > 5)).then(pl.lit("invalid"))
        .otherwise(pl.col("rate").floor().cast(pl.Int64).cast(pl.Utf8)).alias("rate_bucket"),
        pl.when(cnt == 1).then(pl.lit("1"))
        .when(cnt <= 5).then(pl.lit("2-5"))
        .when(cnt <= 20).then(pl.lit("6-20"))
        .when(cnt <= 99).then(pl.lit("21-99"))
        .when(cnt <= 199).then(pl.lit("100-199"))
        .otherwise(pl.lit("200+")).alias("prod_bucket"),
        year.alias("year"),
        pl.when(is_missing("body")).then(pl.lit("no_text"))
        .when(body_len <= 20).then(pl.lit("1-20"))
        .when(body_len <= 50).then(pl.lit("21-50"))
        .when(body_len <= 100).then(pl.lit("51-100"))
        .when(body_len <= 300).then(pl.lit("101-300"))
        .otherwise(pl.lit(">300")).alias("body_len_bucket"),
    ]).collect(engine="streaming")

    total = len(df)
    total_missing = int(df["rec_missing"].sum())
    overall_pct = round(total_missing / total * 100, 3)
    print("\n" + "=" * 70)
    print(f"A2-2: recommendation_status MISSING vs METADATA (clean). "
          f"Overall: {total_missing:,} / {total:,} = {overall_pct}%")
    print("=" * 70)

    factors = [
        group_stats(df, "is_buyer"),
        group_stats(df, "rate_bucket", RATE_ORDER),
        group_stats(df, "prod_bucket", PROD_ORDER),
        group_stats(df, "year"),
        group_stats(df, "body_len_bucket", BODY_ORDER),
    ]
    for f in factors:
        print(f"\n--- factor: {f['factor']} | Cramer's V = {f['cramers_v']} "
              f"| missing% range (groups n>=1000): {f['missing_pct_min_group_n_ge_1000']} .. "
              f"{f['missing_pct_max_group_n_ge_1000']}")
        for r in f["groups"]:
            print(f"   {str(r[f['factor']]):>10s}: n={r['n']:>9,}  missing={r['missing']:>8,}  ({r['missing_pct']:.2f}%)")

    # Figure: 6 panels
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150)
    for ax, f in zip(axes.flat[:5], factors):
        labels = [str(r[f["factor"]]) for r in f["groups"]]
        vals = [r["missing_pct"] for r in f["groups"]]
        ax.bar(labels, vals, color="#2b5c8f", edgecolor="white")
        ax.axhline(overall_pct, color="#d95f02", linestyle="--", linewidth=1.5,
                   label=f"overall {overall_pct:.1f}%")
        ax.set_title(f"Missing recommendation_status by {f['factor']}\n(Cramer's V = {f['cramers_v']})",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("% missing label")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
    ax = axes.flat[5]
    names = [c["column"] for c in col_clean][::-1]
    pcts = [c["missing_pct"] for c in col_clean][::-1]
    ax.barh(names, pcts, color="#388e3c", edgecolor="white")
    ax.set_title("Missing % per column (clean)", fontsize=10, fontweight="bold")
    ax.set_xlabel("% missing (null / '' / nan)")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)

    result = {
        "definition_of_missing": "null OR string in ['', 'nan', 'none', 'null'] after strip/lowercase",
        "per_column_raw": col_raw,
        "per_column_clean": col_clean,
        "recommendation_missing_overall": {"missing": total_missing, "total": total, "pct": overall_pct},
        "factors": factors,
        "figure_path": str(OUT_FIG),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_FIG}")


if __name__ == "__main__":
    main()
