"""
Step A1: Product review distribution and 200-review cap analysis on the clean dataset.

Finding from audit_verification.py: the raw 400-review products were products scraped
twice (200 unique ids, each duplicated). After deduplication the real per-product cap
is 200 reviews.

Reads data/interim/digikala_clean.parquet, writes artifacts/eda/step_a1_product_dist.json
and reports/figures/A1_product_review_distribution.png.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

CLEAN_PARQUET = Path("data/interim/digikala_clean.parquet")
OUTPUT_JSON = Path("artifacts/eda/step_a1_product_dist.json")
OUTPUT_FIG = Path("reports/figures/A1_product_review_distribution.png")
CAP = 200


def run_step_a1() -> dict:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(CLEAN_PARQUET, columns=["id", "product_id"])
    total_clean_rows = len(df)
    null_product_rows = int(df["product_id"].is_null().sum())

    prod_counts = (
        df.drop_nulls("product_id")
        .group_by("product_id")
        .len()
        .rename({"len": "review_count"})
    )

    stats = prod_counts.select([
        pl.len().alias("unique_products"),
        pl.col("review_count").min().alias("min"),
        pl.col("review_count").mean().round(2).alias("mean"),
        pl.col("review_count").std().round(2).alias("std"),
        pl.col("review_count").quantile(0.25).alias("p25"),
        pl.col("review_count").quantile(0.50).alias("p50"),
        pl.col("review_count").quantile(0.75).alias("p75"),
        pl.col("review_count").quantile(0.90).alias("p90"),
        pl.col("review_count").quantile(0.95).alias("p95"),
        pl.col("review_count").quantile(0.99).alias("p99"),
        pl.col("review_count").max().alias("max"),
        (pl.col("review_count") == 1).sum().alias("single_review_products"),
        (pl.col("review_count") < CAP).sum().alias("below_200_products"),
        (pl.col("review_count") == CAP).sum().alias("exact_200_products"),
        (pl.col("review_count") > CAP).sum().alias("over_200_products"),
        (pl.col("review_count") >= CAP).sum().alias("at_least_200_products"),
    ]).to_dicts()[0]

    reviews_in_ge_200 = int(
        prod_counts.filter(pl.col("review_count") >= CAP)["review_count"].sum()
    )
    stats["reviews_in_products_ge_200"] = reviews_in_ge_200
    stats["share_of_reviews_in_products_ge_200_pct"] = round(
        reviews_in_ge_200 / total_clean_rows * 100, 2
    )
    stats["share_of_products_ge_200_pct"] = round(
        stats["at_least_200_products"] / stats["unique_products"] * 100, 2
    )

    spikes_ge_190 = (
        prod_counts.filter(pl.col("review_count") >= 190)
        .group_by("review_count")
        .len()
        .rename({"len": "num_products"})
        .sort("review_count")
        .to_dicts()
    )

    counts_np = prod_counts["review_count"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Panel 1: log-log histogram (log-spaced bins) of reviews per product
    bins = np.logspace(0, np.log10(counts_np.max() + 1), 60)
    axes[0].hist(counts_np, bins=bins, color="#2b5c8f", edgecolor="white", alpha=0.9)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].axvline(
        CAP, color="#d95f02", linestyle="--", linewidth=1.8,
        label=f"200-review cap ({stats['exact_200_products']:,} products)",
    )
    axes[0].set_title("A1-a: Reviews per Product (Clean, Log-Log)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Reviews per Product (log scale)")
    axes[0].set_ylabel("Number of Products (log scale)")
    axes[0].legend()
    axes[0].grid(True, which="both", linestyle=":", alpha=0.5)

    # Panel 2: zoom 190..max, one integer bin per count, showing the spike at 200
    tail_np = counts_np[counts_np >= 190]
    axes[1].hist(
        tail_np, bins=np.arange(189.5, stats["max"] + 1.5, 1),
        color="#388e3c", edgecolor="white", alpha=0.9,
    )
    axes[1].set_yscale("log")
    axes[1].axvline(
        CAP, color="#d95f02", linestyle="--", linewidth=1.8,
        label=f"Spike at 200: {stats['exact_200_products']:,} products",
    )
    axes[1].set_title("A1-b: Zoom on Reviews >= 190 (Clean)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Reviews per Product")
    axes[1].set_ylabel("Number of Products (log scale)")
    axes[1].legend()
    axes[1].grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close(fig)

    result = {
        "source_parquet": str(CLEAN_PARQUET),
        "total_clean_rows": total_clean_rows,
        "null_product_id_rows": null_product_rows,
        "cap_analyzed": CAP,
        "note": "Raw 400-review products were scraped twice (200 unique ids each, all Category A); real cap after dedup is 200.",
        "clean_product_distribution": stats,
        "counts_ge_190": spikes_ge_190,
        "figure_path": str(OUTPUT_FIG),
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("=== STEP A1 (CLEAN DATASET) COMPLETE ===")
    print(f"Clean Rows: {total_clean_rows:,} | Unique Products: {stats['unique_products']:,}")
    print(f"Quantiles -> Min: {stats['min']}, P50: {stats['p50']}, P90: {stats['p90']}, P99: {stats['p99']}, Max: {stats['max']}")
    print(f"Cap 200 -> exact 200: {stats['exact_200_products']:,} | >200: {stats['over_200_products']:,} | >=200: {stats['at_least_200_products']:,} ({stats['share_of_products_ge_200_pct']}% of products)")
    print(f"Reviews in products >=200: {reviews_in_ge_200:,} ({stats['share_of_reviews_in_products_ge_200_pct']}% of all reviews)")
    print(f"Saved plot to: {OUTPUT_FIG}")
    return result


if __name__ == "__main__":
    run_step_a1()
