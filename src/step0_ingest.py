"""
Step 0: Raw CSV Ingestion, Strict String-to-Numeric Casting Audit,
Text Hygiene Counts, Rate Out-of-Range Flagging, and Raw 400-Review Cap Check.
Automatically locates the raw CSV file inside data/raw/ (or project root/data).
"""
import json
import time
from pathlib import Path
import polars as pl

INTERIM_PARQUET_PATH = Path("data/interim/digikala_comments.parquet")
REPORT_JSON_PATH = Path("artifacts/eda/step0_summary.json")


def find_raw_csv() -> Path:
    candidates = [
        Path("data/raw/digikala_comments.csv"),
        Path("data/digikala_comments.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    for folder in [Path("data/raw"), Path("data"), Path(".")]:
        if folder.exists():
            csvs = sorted(folder.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
            if csvs:
                return csvs[0]
    raise FileNotFoundError("Could not find any raw CSV file in data/raw/ or data/.")


def run_step0() -> dict:
    INTERIM_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_csv_path = find_raw_csv()
    print(f"Found raw CSV at: {raw_csv_path}")

    # 1. Read CSV as raw strings for strict parse auditing and attach row_idx
    t0_csv = time.perf_counter()
    df_raw = pl.read_csv(
        raw_csv_path,
        encoding="utf8-lossy",
        infer_schema_length=10000,
        schema_overrides={
            "id": pl.Utf8,
            "product_id": pl.Utf8,
            "rate": pl.Utf8,
            "recommendation_status": pl.Utf8,
            "body": pl.Utf8,
            "title": pl.Utf8,
            "likes": pl.Utf8,
            "dislikes": pl.Utf8,
            "seller_code": pl.Utf8,
        },
        null_values=[""],
    ).with_row_index(name="row_idx", offset=0)
    csv_read_seconds = round(time.perf_counter() - t0_csv, 2)

    total_rows = len(df_raw)

    # 2. Audit unparseable values in rate and product_id before replacing
    rate_clean_str = pl.col("rate").str.strip_chars()
    prod_clean_str = pl.col("product_id").str.strip_chars()

    cast_audit = df_raw.select([
        pl.col("rate").is_null().sum().alias("rate_raw_null"),
        (
            pl.col("rate").is_not_null() &
            (rate_clean_str.str.to_lowercase() == "nan")
        ).sum().alias("rate_literal_nan"),
        (
            pl.col("rate").is_not_null() &
            (rate_clean_str != "") &
            (rate_clean_str.str.to_lowercase() != "nan") &
            rate_clean_str.cast(pl.Float64, strict=False).is_null()
        ).sum().alias("rate_unparseable_non_null"),
        pl.col("product_id").is_null().sum().alias("product_id_raw_null"),
        (
            pl.col("product_id").is_not_null() &
            (prod_clean_str.str.to_lowercase() == "nan")
        ).sum().alias("product_id_literal_nan"),
        (
            pl.col("product_id").is_not_null() &
            (prod_clean_str != "") &
            (prod_clean_str.str.to_lowercase() != "nan") &
            prod_clean_str.cast(pl.Int64, strict=False).is_null()
        ).sum().alias("product_id_unparseable_non_null"),
    ]).to_dicts()[0]

    # 3. Audit body, title, and recommendation_status text hygiene (null, whitespace-only, literal 'nan')
    text_hygiene = df_raw.select([
        pl.col("body").is_null().sum().alias("body_null"),
        (pl.col("body").is_not_null() & (pl.col("body").str.strip_chars() == "")).sum().alias("body_whitespace_only"),
        (pl.col("body").is_not_null() & (pl.col("body").str.strip_chars().str.to_lowercase() == "nan")).sum().alias("body_literal_nan"),
        pl.col("title").is_null().sum().alias("title_null"),
        (pl.col("title").is_not_null() & (pl.col("title").str.strip_chars() == "")).sum().alias("title_whitespace_only"),
        (pl.col("title").is_not_null() & (pl.col("title").str.strip_chars().str.to_lowercase() == "nan")).sum().alias("title_literal_nan"),
        pl.col("recommendation_status").is_null().sum().alias("rec_raw_null"),
        (
            pl.col("recommendation_status").is_not_null() &
            (pl.col("recommendation_status").str.strip_chars().str.to_lowercase() == "nan")
        ).sum().alias("rec_literal_nan_converted_to_null"),
        pl.col("seller_code").is_null().sum().alias("seller_code_null"),
    ]).to_dicts()[0]

    # 4. Transform and cast columns, convert literal 'nan' in recommendation_status to null,
    #    and add rate_out_of_range without altering rate values.
    df_typed = df_raw.with_columns([
        pl.col("id").str.strip_chars().cast(pl.Int64, strict=True),
        pl.when(prod_clean_str.str.to_lowercase().is_in(["nan", "null", "none", ""]))
        .then(None)
        .otherwise(prod_clean_str)
        .cast(pl.Int64, strict=False)
        .alias("product_id"),
        pl.when(rate_clean_str.str.to_lowercase().is_in(["nan", "null", "none", ""]))
        .then(None)
        .otherwise(rate_clean_str)
        .cast(pl.Float64, strict=False)
        .alias("rate"),
        pl.when(
            pl.col("recommendation_status").is_null() |
            pl.col("recommendation_status").str.strip_chars().str.to_lowercase().is_in(["nan", "null", "none", ""])
        )
        .then(None)
        .otherwise(pl.col("recommendation_status").str.strip_chars())
        .alias("recommendation_status"),
        pl.col("likes").str.strip_chars().cast(pl.Int64, strict=False),
        pl.col("dislikes").str.strip_chars().cast(pl.Int64, strict=False),
    ]).with_columns([
        (
            pl.col("rate").is_not_null() &
            ((pl.col("rate") < 0.0) | (pl.col("rate") > 100.0))
        ).alias("rate_out_of_range")
    ])

    # 5. Write Parquet and benchmark Parquet read speed
    t0_pq_write = time.perf_counter()
    df_typed.write_parquet(INTERIM_PARQUET_PATH, compression="zstd")
    parquet_write_seconds = round(time.perf_counter() - t0_pq_write, 2)

    t0_pq_read = time.perf_counter()
    df_check = pl.read_parquet(INTERIM_PARQUET_PATH)
    parquet_read_seconds = round(time.perf_counter() - t0_pq_read, 2)

    # 6. Rate out-of-range details
    out_of_range_rows = (
        df_check.filter(pl.col("rate_out_of_range"))
        .select(["row_idx", "id", "product_id", "rate"])
        .to_dicts()
    )

    # 7. Raw product review distribution (exact 400 cap and >300 distribution)
    prod_counts = (
        df_check.drop_nulls("product_id")
        .group_by("product_id")
        .len()
        .rename({"len": "review_count"})
    )
    prod_dist_raw = prod_counts.select([
        pl.len().alias("unique_products"),
        (pl.col("review_count") == 400).sum().alias("products_exact_400_reviews"),
        (pl.col("review_count") > 400).sum().alias("products_over_400_reviews"),
        (pl.col("review_count") > 300).sum().alias("products_over_300_reviews"),
        (pl.col("review_count").is_between(301, 350)).sum().alias("products_301_to_350"),
        (pl.col("review_count").is_between(351, 399)).sum().alias("products_351_to_399"),
        pl.col("review_count").max().alias("max_reviews_per_product"),
        pl.col("review_count").quantile(0.50).alias("p50_reviews"),
        pl.col("review_count").quantile(0.90).alias("p90_reviews"),
        pl.col("review_count").quantile(0.99).alias("p99_reviews"),
    ]).to_dicts()[0]

    top_counts_over_300 = (
        prod_counts.filter(pl.col("review_count") >= 300)
        .group_by("review_count")
        .len()
        .rename({"len": "num_products"})
        .sort("num_products", descending=True)
        .head(10)
        .to_dicts()
    )

    summary = {
        "raw_csv_path": str(raw_csv_path),
        "total_rows": total_rows,
        "unique_ids": int(df_check["id"].n_unique()),
        "benchmark_seconds": {
            "csv_read_seconds": csv_read_seconds,
            "parquet_write_seconds": parquet_write_seconds,
            "parquet_read_seconds": parquet_read_seconds,
        },
        "strict_cast_audit": cast_audit,
        "text_hygiene_audit": text_hygiene,
        "rate_out_of_range_count": len(out_of_range_rows),
        "rate_out_of_range_records": out_of_range_rows,
        "raw_product_review_distribution": prod_dist_raw,
        "raw_top_frequencies_ge_300": top_counts_over_300,
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== STEP 0: INGESTION & AUDIT COMPLETE ===")
    print(f"Total Rows: {total_rows:,} | Unique IDs: {summary['unique_ids']:,}")
    print(f"CSV Read: {csv_read_seconds}s | Parquet Write: {parquet_write_seconds}s | Parquet Read: {parquet_read_seconds}s")
    print(f"Cast Audit -> rate unparseable: {cast_audit['rate_unparseable_non_null']}, product_id unparseable: {cast_audit['product_id_unparseable_non_null']}")
    print(f"Text Hygiene -> body null: {text_hygiene['body_null']:,}, body ws-only: {text_hygiene['body_whitespace_only']:,}, body 'nan': {text_hygiene['body_literal_nan']:,}")
    print(f"Text Hygiene -> title null: {text_hygiene['title_null']:,}, title ws-only: {text_hygiene['title_whitespace_only']:,}, title 'nan': {text_hygiene['title_literal_nan']:,}")
    print(f"Rec 'nan' converted to null: {text_hygiene['rec_literal_nan_converted_to_null']:,}")
    print(f"Rate out of range [0, 100] flagged: {len(out_of_range_rows)} -> {out_of_range_rows}")
    print(f"Raw 400-Cap Check -> Exact 400: {prod_dist_raw['products_exact_400_reviews']:,} | >400: {prod_dist_raw['products_over_400_reviews']:,} | >300: {prod_dist_raw['products_over_300_reviews']:,}")
    print(f"Top counts >= 300: {top_counts_over_300[:5]}")
    return summary


if __name__ == "__main__":
    run_step0()
