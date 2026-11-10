"""
Step A1: Product Review Distribution and 400-Review Cap Analysis on Clean Dataset.
Reads data/interim/digikala_clean.parquet and saves plot to reports/figures/.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

CLEAN_PARQUET = Path("data/interim/digikala_clean.parquet")
OUTPUT_JSON = Path("artifacts/eda/step_a1_product_dist.json")
OUTPUT_FIG = Path("reports/figures/A1_product_review_distribution.png")


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
        (pl.col("review_count") > 300).sum().alias("over_300_products"),
        (pl.col("review_count").is_between(301, 350)).sum().alias("between_301_350_products"),
        (pl.col("review_count").is_between(351, 399)).sum().alias("between_351_399_products"),
        (pl.col("review_count") == 400).sum().alias("exact_400_products"),
        (pl.col("review_count") > 400).sum().alias("over_400_products"),
    ]).to_dicts()[0]

    top_spikes_ge_300 = (
        prod_counts.filter(pl.col("review_count") >= 300)
        .group_by("review_count")
        .len()
        .rename({"len": "num_products"})
        .sort("num_products", descending=True)
        .head(15)
        .to_dicts()
    )

    # Generate publication-quality two-panel figure
    counts_np = prod_counts["review_count"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Panel 1: Log-scale histogram of reviews per product
    axes[0].hist(counts_np, bins=80, color="#2b5c8f", edgecolor="black", alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].axvline(400, color="#d95f02", linestyle="--", linewidth=1.8, label=f"400-Review Cap (n={stats['exact_400_products']:,})")
    axes[0].set_title("A1-a: Product Review Count Distribution (Log Scale)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Reviews per Product")
    axes[0].set_ylabel("Number of Products (Log10 Scale)")
    axes[0].legend()
    axes[0].grid(True, which="both", linestyle=":", alpha=0.5)

    # Panel 2: Zoomed tail (300 to max) showing the spike at 400
    tail_np = counts_np[counts_np >= 300]
    axes[1].hist(tail_np, bins=range(300, int(stats["max"]) + 5, 2), color="#388e3c", edgecolor="black", alpha=0.85)
    axes[1].axvline(400, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Exact 400 spike: {stats['exact_400_products']:,} products")
    axes[1].set_title("A1-b: Tail Distribution (Reviews >= 300)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Reviews per Product (>= 300)")
    axes[1].set_ylabel("Number of Products")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close(fig)

    result = {
        "source_parquet": str(CLEAN_PARQUET),
        "total_clean_rows": total_clean_rows,
        "null_product_id_rows": null_product_rows,
        "clean_product_distribution": stats,
        "top_count_spikes_ge_300": top_spikes_ge_300,
        "figure_path": str(OUTPUT_FIG),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("=== STEP A1 (CLEAN DATASET) COMPLETE ===")
    print(f"Clean Rows: {total_clean_rows:,} | Unique Products: {stats['unique_products']:,}")
    print(f"Quantiles -> Min: {stats['min']}, P50: {stats['p50']}, P90: {stats['p90']}, P99: {stats['p99']}, Max: {stats['max']}")
    print(f"Tail (>300) -> 301..350: {stats['between_301_350_products']:,}, 351..399: {stats['between_351_399_products']:,}, Exact 400: {stats['exact_400_products']:,}, >400: {stats['over_400_products']:,}")
    print(f"Saved plot to: {OUTPUT_FIG}")
    return result


if __name__ == "__main__":
    run_step_a1()
