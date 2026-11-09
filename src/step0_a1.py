"""
Step 0 (Schema & Quality Gate + Parquet Conversion) & Step A1 (Dataset Size & Structure).
"""
import time
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import src.eda_utils as u

def run_step0_and_a1():
    u.ensure_directories()
    u.setup_plot_style()
    csv_path = u.locate_raw_csv()
    parquet_path = u.DATA_INTERIM_DIR / "digikala_comments.parquet"

    print("=" * 70)
    print("STEP 0: SCHEMA VALIDATION, QUALITY GATE & PARQUET CONVERSION")
    print("=" * 70)

    t0 = time.time()
    lf_raw = pl.scan_csv(
        csv_path,
        infer_schema_length=0,
        ignore_errors=False,
        encoding="utf8",
    )
    actual_cols = lf_raw.collect_schema().names()
    missing_cols = [c for c in u.EXPECTED_COLUMNS if c not in actual_cols]
    extra_cols = [c for c in actual_cols if c not in u.EXPECTED_COLUMNS]

    print(f"Actual columns ({len(actual_cols)}): {actual_cols}")
    print(f"Missing expected columns: {missing_cols}")
    print(f"Extra columns: {extra_cols}")

    print("\nAuditing raw records via CSV parser (no silent coercion)...")
    audit_exprs = [
        pl.len().alias("total_rows"),
        pl.col("id").is_null().sum().alias("null_id_count"),
        (pl.col("id").is_not_null() & pl.col("id").cast(pl.Int64, strict=False).is_null())
        .sum()
        .alias("non_integer_id_count"),
        pl.col("id").n_unique().alias("unique_id_count"),
        pl.col("rate").is_null().sum().alias("null_rate_raw_count"),
        (pl.col("rate").is_not_null() & pl.col("rate").cast(pl.Float64, strict=False).is_null())
        .sum()
        .alias("non_numeric_rate_count"),
        (
            (pl.col("rate").cast(pl.Float64, strict=False) < 0.0)
            | (pl.col("rate").cast(pl.Float64, strict=False) > 5.0)
        )
        .sum()
        .alias("out_of_range_rate_count"),
        pl.col("body").is_null().sum().alias("null_body_count"),
        (
            pl.col("body").is_not_null()
            & (
                (pl.col("body").str.strip_chars() == "")
                | (pl.col("body").str.to_lowercase() == "nan")
            )
        )
        .sum()
        .alias("empty_or_nan_body_count"),
    ]
    audit_df = lf_raw.select(audit_exprs).collect(engine="streaming")
    audit_res = audit_df.to_dicts()[0]
    audit_res["duplicate_id_count"] = (
        audit_res["total_rows"] - audit_res["unique_id_count"]
    )

    # Inspect out-of-range rate record(s)
    out_of_range_df = (
        lf_raw.filter(
            (pl.col("rate").cast(pl.Float64, strict=False) < 0.0)
            | (pl.col("rate").cast(pl.Float64, strict=False) > 5.0)
        )
        .select(["id", "rate", "recommendation_status", "product_id"])
        .collect(engine="streaming")
    )
    print("\n--- Out-of-Range `rate` Record(s) ---")
    print(out_of_range_df)

    # Inspect raw vocabulary of recommendation_status and is_buyer
    rec_vocab_df = (
        lf_raw.group_by("recommendation_status")
        .len()
        .sort("len", descending=True)
        .collect(engine="streaming")
    )
    buyer_vocab_df = (
        lf_raw.group_by("is_buyer")
        .len()
        .sort("len", descending=True)
        .collect(engine="streaming")
    )

    print("\n--- Observed `recommendation_status` Vocabulary ---")
    print(rec_vocab_df)
    print("\n--- Observed `is_buyer` Vocabulary ---")
    print(buyer_vocab_df)

    # Helper expression to normalize literal "nan" / "None" / "" strings to proper Null
    def clean_str_col(col_name: str) -> pl.Expr:
        return (
            pl.when(
                pl.col(col_name).is_null()
                | (pl.col(col_name).str.strip_chars() == "")
                | (pl.col(col_name).str.to_lowercase() == "nan")
                | (pl.col(col_name).str.to_lowercase() == "none")
                | (pl.col(col_name).str.to_lowercase() == "null")
            )
            .then(pl.lit(None, dtype=pl.Utf8))
            .otherwise(pl.col(col_name))
            .alias(col_name)
        )

    # Write typed Parquet (normalizing literal "nan" strings to proper nulls & explicit boolean mapping)
    print(f"\nWriting typed Parquet to {parquet_path} ...")
    lf_typed = lf_raw.with_columns([
        pl.col("id").cast(pl.Int64, strict=False),
        clean_str_col("title"),
        clean_str_col("body"),
        clean_str_col("created_at"),
        pl.col("rate").cast(pl.Float64, strict=False),
        clean_str_col("recommendation_status"),
        pl.when(pl.col("is_buyer").str.to_lowercase().is_in(["true", "1", "1.0", "yes"]))
        .then(pl.lit(True))
        .when(pl.col("is_buyer").str.to_lowercase().is_in(["false", "0", "0.0", "no"]))
        .then(pl.lit(False))
        .otherwise(pl.lit(None, dtype=pl.Boolean))
        .alias("is_buyer"),
        pl.col("product_id").cast(pl.Int64, strict=False),
        clean_str_col("advantages"),
        clean_str_col("disadvantages"),
        pl.col("likes").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False),
        pl.col("dislikes").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False),
        clean_str_col("seller_title"),
        pl.col("seller_code").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False),
        pl.col("true_to_size_rate").cast(pl.Float64, strict=False),
    ])
    lf_typed.sink_parquet(parquet_path, compression="zstd")
    print(f"Parquet conversion completed in {time.time() - t0:.1f}s.")

    # Save Quality Gate Artifact
    quality_gate_payload = {
        "schema_validation": {
            "expected_columns": u.EXPECTED_COLUMNS,
            "actual_columns": actual_cols,
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "schema_passed": len(missing_cols) == 0,
        },
        "quality_counts": audit_res,
        "out_of_range_rate_records": out_of_range_df.to_dicts(),
        "recommendation_status_raw_vocabulary": rec_vocab_df.to_dicts(),
        "is_buyer_raw_vocabulary": buyer_vocab_df.to_dicts(),
    }
    u.save_json_artifact(quality_gate_payload, "00_quality_gate_summary.json")

    print("\n" + "=" * 70)
    print("STEP A1: DATASET SIZE, STRUCTURE & PRODUCT/SELLER CONCENTRATION")
    print("=" * 70)

    lf = pl.scan_parquet(parquet_path)
    csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
    pq_size_mb = parquet_path.stat().st_size / (1024 * 1024)

    a1_stats = lf.select([
        pl.len().alias("total_records"),
        pl.col("product_id").drop_nulls().n_unique().alias("unique_products"),
        pl.col("seller_code").drop_nulls().n_unique().alias("unique_seller_codes"),
        pl.col("seller_title").drop_nulls().n_unique().alias("unique_seller_titles"),
    ]).collect()

    a1_dict = a1_stats.to_dicts()[0]
    a1_dict["num_columns"] = len(actual_cols)
    a1_dict["csv_file_size_mb"] = round(csv_size_mb, 2)
    a1_dict["parquet_file_size_mb"] = round(pq_size_mb, 2)

    # Product review-count distribution
    prod_counts = (
        lf.filter(pl.col("product_id").is_not_null())
        .group_by("product_id")
        .len()
        .rename({"len": "review_count"})
        .collect()
    )

    prod_dist = prod_counts.select([
        pl.col("review_count").min().alias("min_reviews_per_product"),
        pl.col("review_count").quantile(0.25).alias("p25_reviews"),
        pl.col("review_count").median().alias("median_reviews"),
        pl.col("review_count").mean().alias("mean_reviews"),
        pl.col("review_count").std().alias("std_reviews"),
        pl.col("review_count").quantile(0.75).alias("p75_reviews"),
        pl.col("review_count").quantile(0.90).alias("p90_reviews"),
        pl.col("review_count").quantile(0.99).alias("p99_reviews"),
        pl.col("review_count").max().alias("max_reviews_per_product"),
        (pl.col("review_count") == 1).sum().alias("products_with_only_1_review"),
        (pl.col("review_count") <= 5).sum().alias("products_with_le_5_reviews"),
        (pl.col("review_count") >= 100).sum().alias("products_with_100plus_reviews"),
        (pl.col("review_count") >= 1000).sum().alias("products_with_1000plus_reviews"),
    ]).to_dicts()[0]

    a1_dict["product_review_distribution"] = prod_dist

    schema_rows = [
        {"column": col, "dtype": str(dtype)}
        for col, dtype in lf.collect_schema().items()
    ]
    u.save_table_artifact(
        pl.DataFrame(schema_rows),
        "A1_schema_and_dtypes",
        title="A1: Dataset Schema and Data Types",
        extra_metadata=a1_dict,
    )

    print("\n--- A1 Summary Metrics ---")
    for k, v in a1_dict.items():
        if k != "product_review_distribution":
            print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    print("\n--- Product Review Count Distribution ---")
    for k, v in prod_dist.items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v:,}")

    # Plot A1: Distribution of Reviews per Product (Log-Log Scale)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    counts_np = prod_counts["review_count"].to_numpy()
    bins = np.logspace(0, np.log10(counts_np.max() + 1), 50)
    ax.hist(counts_np, bins=bins, color="#2b5c8f", edgecolor="white", alpha=0.88)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("A1: Distribution of Review Counts per Product (Log-Log Scale, Full Data)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Reviews per Product (Log Scale)", fontsize=11)
    ax.set_ylabel("Number of Products (Log Scale)", fontsize=11)
    ax.axvline(prod_dist["median_reviews"], color="#d95f02", linestyle="--", linewidth=1.8, label=f"Median = {prod_dist['median_reviews']:.0f}")
    ax.axvline(prod_dist["mean_reviews"], color="#1b9e77", linestyle="-.", linewidth=1.8, label=f"Mean = {prod_dist['mean_reviews']:.1f}")
    ax.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    fig_path = u.FIGURES_DIR / "A1_product_review_distribution.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"\n[Figure Saved] -> {fig_path.relative_to(u.PROJECT_ROOT)}")

if __name__ == "__main__":
    run_step0_and_a1()
