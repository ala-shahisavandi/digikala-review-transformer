"""
Step A1 (reference artifact): dataset size, schema, dtypes, sellers and
product-concentration, computed from the CURRENT parquet files, raw vs clean.

Replaces the stale artifacts/eda/A1_schema_and_dtypes.* (they were produced by an old
script that cast seller_code to Int64 and therefore reported 0 sellers).

Run from repo root:  python src/step_a1_schema.py
"""
import json
from pathlib import Path
import polars as pl

from step0_ingest import find_raw_csv

RAW_PARQUET = Path("data/interim/digikala_comments.parquet")
CLEAN_PARQUET = Path("data/interim/digikala_clean.parquet")
OUT_JSON = Path("artifacts/eda/A1_schema_and_dtypes.json")
OUT_MD = Path("artifacts/eda/A1_schema_and_dtypes.md")

ORIGINAL_COLUMNS = [
    "id", "title", "body", "created_at", "rate", "recommendation_status", "is_buyer",
    "product_id", "advantages", "disadvantages", "likes", "dislikes",
    "seller_title", "seller_code", "true_to_size_rate",
]


def not_nan(col: str) -> pl.Expr:
    s = pl.col(col).cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    return pl.col(col).is_not_null() & (s != "nan") & (s != "")


def describe(path: Path) -> dict:
    lf = pl.scan_parquet(path)
    base = lf.select([
        pl.len().alias("total_records"),
        pl.col("product_id").drop_nulls().n_unique().alias("unique_products"),
        pl.col("seller_code").filter(not_nan("seller_code")).n_unique().alias("unique_seller_codes_excl_nan"),
        pl.col("seller_title").filter(not_nan("seller_title")).n_unique().alias("unique_seller_titles_excl_nan"),
        (~not_nan("seller_code")).sum().alias("rows_without_seller"),
        pl.col("id").n_unique().alias("unique_ids"),
    ]).collect(engine="streaming").to_dicts()[0]

    prod_counts = (
        lf.drop_nulls("product_id").group_by("product_id").len()
        .rename({"len": "review_count"}).collect(engine="streaming")
    )
    dist = prod_counts.select([
        pl.col("review_count").min().alias("min"),
        pl.col("review_count").quantile(0.25).alias("p25"),
        pl.col("review_count").median().alias("median"),
        pl.col("review_count").mean().round(2).alias("mean"),
        pl.col("review_count").std().round(2).alias("std"),
        pl.col("review_count").quantile(0.75).alias("p75"),
        pl.col("review_count").quantile(0.90).alias("p90"),
        pl.col("review_count").quantile(0.99).alias("p99"),
        pl.col("review_count").max().alias("max"),
        (pl.col("review_count") == 1).sum().alias("products_with_1_review"),
        (pl.col("review_count") <= 5).sum().alias("products_with_le_5_reviews"),
        (pl.col("review_count") >= 100).sum().alias("products_with_100plus_reviews"),
        (pl.col("review_count") >= 200).sum().alias("products_with_200plus_reviews"),
        (pl.col("review_count") >= 1000).sum().alias("products_with_1000plus_reviews"),
    ]).to_dicts()[0]
    base["product_review_distribution"] = dist
    base["parquet_file_size_mb"] = round(path.stat().st_size / (1024 * 1024), 2)
    return base


def main() -> None:
    raw_csv = find_raw_csv()
    csv_mb = round(raw_csv.stat().st_size / (1024 * 1024), 2)

    schema = pl.scan_parquet(RAW_PARQUET).collect_schema()
    original_schema = [{"column": c, "dtype": str(schema[c])} for c in ORIGINAL_COLUMNS]
    extra_parquet_cols = [c for c in schema.names() if c not in ORIGINAL_COLUMNS]

    result = {
        "title": "A1: Dataset Size, Schema and Concentration (raw vs clean)",
        "csv_file_size_mb": csv_mb,
        "num_original_columns": len(ORIGINAL_COLUMNS),
        "extra_columns_added_by_pipeline": extra_parquet_cols,
        "schema": original_schema,
        "raw": describe(RAW_PARQUET),
        "clean": describe(CLEAN_PARQUET),
    }
    result["parquet_size_reduction_pct"] = round(
        (1 - result["raw"]["parquet_file_size_mb"] / csv_mb) * 100, 1
    )
    result["csv_to_parquet_ratio"] = round(csv_mb / result["raw"]["parquet_file_size_mb"], 2)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    lines = ["# A1: Dataset Size, Schema and Concentration (raw vs clean)", "",
             "| column | dtype |", "| --- | --- |"]
    lines += [f"| {r['column']} | {r['dtype']} |" for r in original_schema]
    lines += ["", f"- CSV size: {csv_mb} MB; raw Parquet: {result['raw']['parquet_file_size_mb']} MB "
              f"({result['parquet_size_reduction_pct']}% smaller, {result['csv_to_parquet_ratio']}x)",
              f"- Extra pipeline columns in Parquet: {extra_parquet_cols}", ""]
    for label in ("raw", "clean"):
        lines.append(f"## {label.upper()}")
        for k, v in result[label].items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("=== A1 REFERENCE ARTIFACT REGENERATED ===")
    print(f"CSV {csv_mb} MB | raw Parquet {result['raw']['parquet_file_size_mb']} MB "
          f"| reduction {result['parquet_size_reduction_pct']}% ({result['csv_to_parquet_ratio']}x)")
    for label in ("raw", "clean"):
        d = result[label]
        print(f"[{label}] rows={d['total_records']:,} ids={d['unique_ids']:,} products={d['unique_products']:,} "
              f"seller_codes={d['unique_seller_codes_excl_nan']:,} seller_titles={d['unique_seller_titles_excl_nan']:,} "
              f"no_seller_rows={d['rows_without_seller']:,}")
        print(f"        dist={d['product_review_distribution']}")
    print(f"Saved: {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
