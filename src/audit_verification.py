# -*- coding: utf-8 -*-
"""
Independent verification of the six audit points (rate, 400-review products,
vanished product, Unicode normalization, I/O benchmark, body == 'nan').

Run from repo root:  python src/audit_verification.py
"""
import json
import time
from pathlib import Path
import polars as pl

from step0_ingest import find_raw_csv
from eda_utils import normalize_text_minimal
from dedup import classify_duplicates

RAW_PARQUET = Path("data/interim/digikala_comments.parquet")
CLEAN_PARQUET = Path("data/interim/digikala_clean.parquet")
DEDUP_POLICY_PATH = Path("artifacts/eda/dedup_policy.json")
OUT_JSON = Path("artifacts/eda/verification_6points.json")


def main() -> None:
    raw_csv = find_raw_csv()
    print(f"Using Raw CSV: {raw_csv}")
    report = {}

    lf_clean = pl.scan_parquet(CLEAN_PARQUET)
    lf_raw = pl.scan_parquet(RAW_PARQUET)

    # =========================================================================
    # 1. RATE DISTRIBUTION & RAW CSV INSPECTION FOR id=38631814
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. RATE COLUMN AUDIT & RAW CSV ROW FOR id=38631814")
    print("=" * 70)

    rate_stats = lf_raw.select([
        pl.col("rate").min().alias("min"),
        pl.col("rate").max().alias("max"),
        pl.col("rate").n_unique().alias("n_unique"),
        pl.col("rate").null_count().alias("null_count"),
        (pl.col("rate") < 0).sum().alias("lt_0"),
        (pl.col("rate") > 5).sum().alias("gt_5"),
        (pl.col("rate") > 100).sum().alias("gt_100"),
    ]).collect(engine="streaming").row(0, named=True)

    top20_rates = (
        lf_raw.group_by("rate").len().sort("len", descending=True).head(20)
        .collect(engine="streaming").to_dicts()
    )
    all_rates_gt_5 = (
        lf_raw.filter(pl.col("rate") > 5).group_by("rate").len().sort("rate")
        .collect(engine="streaming").to_dicts()
    )

    raw_csv_schema = {
        "id": pl.Utf8, "title": pl.Utf8, "body": pl.Utf8, "created_at": pl.Utf8,
        "rate": pl.Utf8, "recommendation_status": pl.Utf8, "is_buyer": pl.Utf8,
        "product_id": pl.Utf8, "advantages": pl.Utf8, "disadvantages": pl.Utf8,
        "likes": pl.Utf8, "dislikes": pl.Utf8,
    }

    def scan_raw_csv() -> pl.LazyFrame:
        return pl.scan_csv(raw_csv, schema_overrides=raw_csv_schema, encoding="utf8-lossy")

    csv_row_38631814 = (
        scan_raw_csv().with_row_index("csv_row_idx")
        .filter(pl.col("id") == "38631814")
        .collect(engine="streaming").to_dicts()
    )
    csv_gt_100 = (
        scan_raw_csv().with_row_index("csv_row_idx")
        .filter(pl.col("rate").cast(pl.Float64, strict=False) > 100)
        .collect(engine="streaming").to_dicts()
    )

    print(f"Rate Stats (Raw): {rate_stats}")
    print("Top 20 Rate values (Raw):")
    for r in top20_rates:
        print(f"  rate={r['rate']}: {r['len']:,}")
    print(f"All Rate values > 5 (distinct buckets count={len(all_rates_gt_5)}):")
    for r in all_rates_gt_5:
        print(f"  rate={r['rate']}: {r['len']:,}")
    print("\nDirect RAW CSV row(s) for id='38631814':")
    for r in csv_row_38631814:
        print(" ", json.dumps(r, ensure_ascii=False))
    print("Direct RAW CSV row(s) with float(rate) > 100:")
    for r in csv_gt_100:
        print(" ", json.dumps(r, ensure_ascii=False))

    report["q1_rate"] = {
        "stats_raw": rate_stats,
        "top20_rates": top20_rates,
        "all_rates_gt_5": all_rates_gt_5,
        "csv_row_38631814": csv_row_38631814,
        "csv_rows_gt_100": csv_gt_100,
    }

    # =========================================================================
    # Recompute duplicate categories with the SAME logic used by dedup.py
    # (the old duplicate_ids_report.json does not store id lists)
    # =========================================================================
    print("\nRecomputing duplicate categories (shared classify_duplicates)...")
    dup_ids = (
        lf_raw.group_by("id").len().filter(pl.col("len") > 1).select("id")
        .collect(engine="streaming")["id"].to_list()
    )
    dup_df = (
        lf_raw.filter(pl.col("id").is_in(dup_ids))
        .select(["row_idx", "id", "body", "rate", "recommendation_status"])
        .collect(engine="streaming")
    )
    cls = classify_duplicates(dup_df)
    cat_a_set = set(cls["cat_a_ids"])
    cat_b_set = set(cls["cat_b_ids"])
    cat_c_set = set(cls["cat_c_ids"])

    with open(DEDUP_POLICY_PATH, "r", encoding="utf-8") as f:
        policy = json.load(f)
    categories_match_policy = (
        len(dup_ids) == policy["total_duplicate_ids"]
        and len(cat_a_set) == policy["cat_A_ids_count"]
        and len(cat_b_set) == policy["cat_B_ids_count"]
        and len(cat_c_set) == policy["cat_C_ids_count"]
    )
    print(
        f"Recomputed: dup_ids={len(dup_ids)}, A={len(cat_a_set)}, B={len(cat_b_set)}, C={len(cat_c_set)} "
        f"| matches dedup_policy.json: {categories_match_policy}"
    )
    report["categories_recomputed"] = {
        "total_duplicate_ids": len(dup_ids),
        "cat_A": len(cat_a_set),
        "cat_B": len(cat_b_set),
        "cat_C": len(cat_c_set),
        "matches_dedup_policy_json": categories_match_policy,
    }

    # =========================================================================
    # 2. 400-REVIEW PRODUCTS (BEFORE/AFTER DEDUP) & 190..233 HISTOGRAM
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. 400-REVIEW PRODUCTS AUDIT & 190..233 HISTOGRAM")
    print("=" * 70)

    raw_prod_counts = lf_raw.group_by("product_id").len().rename({"len": "raw_count"})
    clean_prod_counts = lf_clean.group_by("product_id").len().rename({"len": "clean_count"})

    prod_400_df = (
        raw_prod_counts.filter(pl.col("raw_count") == 400)
        .join(clean_prod_counts, on="product_id", how="left")
        .collect(engine="streaming")
    )

    prod_400_details = []
    for row in prod_400_df.to_dicts():
        pid = row["product_id"]
        rows_for_pid = (
            lf_raw.filter(pl.col("product_id") == pid)
            .select(["row_idx", "id"]).collect(engine="streaming")
        )
        unique_ids = set(rows_for_pid["id"].to_list())
        id_counts = rows_for_pid.group_by("id").len().filter(pl.col("len") > 1)
        dup_ids_in_pid = set(id_counts["id"].to_list())

        detail = {
            "product_id": int(pid),
            "raw_rows": int(row["raw_count"]),
            "unique_ids_in_product": len(unique_ids),
            "clean_rows": int(row["clean_count"]) if row["clean_count"] is not None else 0,
            "duplicate_ids_count": len(dup_ids_in_pid),
            "dup_ids_in_Cat_A": len(dup_ids_in_pid & cat_a_set),
            "dup_ids_in_Cat_B": len(dup_ids_in_pid & cat_b_set),
            "dup_ids_in_Cat_C": len(dup_ids_in_pid & cat_c_set),
        }
        prod_400_details.append(detail)
        print(f"Product {pid}: {detail}")

    hist_190_233 = (
        clean_prod_counts.filter(
            (pl.col("clean_count") >= 190) & (pl.col("clean_count") <= 233)
        )
        .group_by("clean_count").len()
        .rename({"clean_count": "review_count", "len": "product_count"})
        .sort("review_count")
        .collect(engine="streaming").to_dicts()
    )
    exact_200_clean = next(
        (x["product_count"] for x in hist_190_233 if x["review_count"] == 200), 0
    )
    print(f"\nProducts with EXACTLY 200 reviews (Clean): {exact_200_clean:,}")
    print("Histogram of Clean Products (190..233 reviews):")
    for h in hist_190_233:
        print(f"  reviews={h['review_count']}: {h['product_count']:,} products")

    report["q2_products_400_and_hist"] = {
        "products_with_400_raw": prod_400_details,
        "exact_200_clean_products": exact_200_clean,
        "histogram_190_to_233": hist_190_233,
    }

    # =========================================================================
    # 3. VANISHED PRODUCT BETWEEN RAW AND CLEAN
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. VANISHED PRODUCT AUDIT (RAW: 331,599 vs CLEAN: 331,598)")
    print("=" * 70)

    vanished_pids = (
        raw_prod_counts.join(clean_prod_counts, on="product_id", how="anti")
        .collect(engine="streaming")["product_id"].to_list()
    )
    vanished_details = []
    for vpid in vanished_pids:
        v_rows = (
            lf_raw.filter(pl.col("product_id") == vpid)
            .collect(engine="streaming").to_dicts()
        )
        v_ids = [r["id"] for r in v_rows]
        all_in_cat_c = all(i in cat_c_set for i in v_ids)
        vanished_details.append({
            "product_id": int(vpid),
            "raw_row_count": len(v_rows),
            "ids": v_ids,
            "all_ids_in_cat_c": all_in_cat_c,
            "rows": v_rows,
        })
        print(f"Vanished Product ID: {vpid} | all_ids_in_cat_c={all_in_cat_c}")
        for r in v_rows:
            print("  Row:", json.dumps(r, ensure_ascii=False, default=str))

    report["q3_vanished_products"] = vanished_details

    # =========================================================================
    # 4. UNICODE NORMALIZATION TEST ON U+064A AND U+0643
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. UNICODE NORMALIZATION TEST (U+064A & U+0643)")
    print("=" * 70)

    test_inputs = {
        "arabic_yeh_only": "\u064a",
        "arabic_kaf_only": "\u0643",
        "combined_with_zwnj_ws": " \u0643\u062a\u0627\u0628\u200c\u0647\u0627\u064a   \u062e\u0648\u0628 ",
    }
    unicode_results = []
    for name, in_str in test_inputs.items():
        out_str = normalize_text_minimal(in_str)
        item = {
            "test_case": name,
            "input_escaped": in_str.encode("unicode_escape").decode("ascii"),
            "output_escaped": out_str.encode("unicode_escape").decode("ascii"),
            "input_codepoints": [f"U+{ord(c):04X}" for c in in_str],
            "output_codepoints": [f"U+{ord(c):04X}" for c in out_str],
        }
        unicode_results.append(item)
        print(f"Case '{name}':")
        print(f"  Input  : {item['input_escaped']} -> {item['input_codepoints']}")
        print(f"  Output : {item['output_escaped']} -> {item['output_codepoints']}")

    report["q4_unicode_test"] = unicode_results

    # =========================================================================
    # 5. 3-RUN I/O BENCHMARK (IDENTICAL 12 COLUMNS: CSV vs PARQUET)
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. 3-RUN I/O BENCHMARK (CSV vs PARQUET, IDENTICAL 12 COLUMNS)")
    print("=" * 70)

    cols_12 = list(raw_csv_schema.keys())
    csv_times, pq_times = [], []
    for i in range(3):
        t0 = time.perf_counter()
        _ = scan_raw_csv().select(cols_12).collect(engine="streaming")
        dt_csv = time.perf_counter() - t0
        csv_times.append(round(dt_csv, 4))

        t1 = time.perf_counter()
        _ = pl.scan_parquet(RAW_PARQUET).select(cols_12).collect(engine="streaming")
        dt_pq = time.perf_counter() - t1
        pq_times.append(round(dt_pq, 4))
        print(f"  Run {i+1}: CSV = {dt_csv:.4f}s | Parquet = {dt_pq:.4f}s")

    avg_csv = round(sum(csv_times) / 3.0, 4)
    avg_pq = round(sum(pq_times) / 3.0, 4)
    print(f"Average (3 runs): CSV = {avg_csv:.4f}s | Parquet = {avg_pq:.4f}s")

    report["q5_io_benchmark"] = {
        "columns_tested": cols_12,
        "csv_runs_sec": csv_times,
        "parquet_runs_sec": pq_times,
        "csv_avg_sec": avg_csv,
        "parquet_avg_sec": avg_pq,
    }

    # =========================================================================
    # 6. BODY == "nan" CROSS-TAB BY RECOMMENDATION_STATUS AND RATE
    # =========================================================================
    print("\n" + "=" * 70)
    print("6. BODY == 'nan' BREAKDOWN BY RECOMMENDATION_STATUS & RATE")
    print("=" * 70)

    for label, lf_target in [("raw", lf_raw), ("clean", lf_clean)]:
        body_nan_lf = lf_target.filter(
            pl.col("body").str.strip_chars().str.to_lowercase() == "nan"
        )
        total_body_nan = body_nan_lf.select(pl.len()).collect(engine="streaming").item()
        by_rec = (
            body_nan_lf.group_by("recommendation_status").len().sort("len", descending=True)
            .collect(engine="streaming").to_dicts()
        )
        by_rate = (
            body_nan_lf.group_by("rate").len().sort("rate")
            .collect(engine="streaming").to_dicts()
        )
        by_both = (
            body_nan_lf.group_by(["recommendation_status", "rate"]).len()
            .sort(["recommendation_status", "rate"])
            .collect(engine="streaming").to_dicts()
        )
        print(f"\n[{label.upper()} DATASET] Total body == 'nan': {total_body_nan}")
        print(f"  By recommendation_status: {by_rec}")
        print(f"  By rate: {by_rate}")
        report[f"q6_body_nan_{label}"] = {
            "total": total_body_nan,
            "by_recommendation_status": by_rec,
            "by_rate": by_rate,
            "by_rec_and_rate": by_both,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved full verification report to: {OUT_JSON}")


if __name__ == "__main__":
    main()
