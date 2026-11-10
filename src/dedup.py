"""
Step 2: Self-Contained Deduplication Policy Execution.
- Category A: duplicates identical in body/rate/recommendation_status.
- Category B: duplicates differing only by null/'nan' or whitespace/ZWNJ/Yeh/Kaf variants.
- Category C: real conflicts (different body text, rate, or recommendation).
- A & B: keep one row per id (most complete, then lowest row_idx).
- C: drop ALL occurrences of the id.
- Verifies clean_total_rows == raw_unique_ids - len(cat_C_ids).

classify_duplicates() is reusable (audit_verification.py imports it so both
scripts use exactly the same category logic).
"""
import json
from pathlib import Path
import polars as pl

from eda_utils import normalize_text_minimal

INPUT_PARQUET = Path("data/interim/digikala_comments.parquet")
REPORT_JSON = Path("artifacts/eda/duplicate_ids_report.json")
OUTPUT_PARQUET = Path("data/interim/digikala_clean.parquet")
POLICY_JSON = Path("artifacts/eda/dedup_policy.json")


def is_effectively_empty(val: object) -> bool:
    if val is None:
        return True
    s = str(val).strip()
    return s == "" or s.lower() == "nan"


def classify_duplicates(dup_df: pl.DataFrame) -> dict:
    """dup_df: only rows whose id appears more than once.
    Required columns: row_idx, id, body, rate, recommendation_status."""
    dup_df = dup_df.sort(["id", "row_idx"])
    id_groups: dict = {}
    for row in dup_df.select(["row_idx", "id", "body", "rate", "recommendation_status"]).iter_rows(named=True):
        id_groups.setdefault(row["id"], []).append(row)

    cat_a_ids, cat_b_ids, cat_c_ids = [], [], []
    body_diff_only_whitespace = 0
    body_actual_text_diff = 0

    for gid, rows in id_groups.items():
        rates = [r["rate"] for r in rows]
        rates_valid = [r for r in rates if r is not None]
        has_rate_conflict = len(set(rates_valid)) > 1
        has_rate_null_diff = (len(set(rates_valid)) == 1) and (len(rates_valid) < len(rates))

        recs = [r["recommendation_status"] for r in rows]
        recs_valid = [str(r).strip() for r in recs if not is_effectively_empty(r)]
        has_rec_conflict = len(set(recs_valid)) > 1
        has_rec_null_diff = (len(set(recs_valid)) == 1) and (len(recs_valid) < len(recs))

        bodies = [r["body"] for r in rows]
        bodies_valid = [str(b) for b in bodies if not is_effectively_empty(b)]
        bodies_unique_valid = set(bodies_valid)
        norm_bodies_valid = {normalize_text_minimal(b) for b in bodies_valid}

        has_body_raw_diff = len(set(bodies)) > 1
        has_body_conflict = len(norm_bodies_valid) > 1
        has_body_ws_diff = (len(bodies_unique_valid) > 1) and (len(norm_bodies_valid) == 1)
        has_body_null_diff = (len(bodies_unique_valid) == 1) and (len(bodies_valid) < len(bodies))

        if has_body_raw_diff:
            if has_body_conflict:
                body_actual_text_diff += 1
            elif has_body_ws_diff:
                body_diff_only_whitespace += 1

        if has_body_conflict or has_rate_conflict or has_rec_conflict:
            cat_c_ids.append(gid)
        elif has_body_null_diff or has_body_ws_diff or has_rate_null_diff or has_rec_null_diff:
            cat_b_ids.append(gid)
        else:
            cat_a_ids.append(gid)

    return {
        "cat_a_ids": cat_a_ids,
        "cat_b_ids": cat_b_ids,
        "cat_c_ids": cat_c_ids,
        "body_diff_only_whitespace": body_diff_only_whitespace,
        "body_actual_text_diff": body_actual_text_diff,
    }


def run_dedup() -> dict:
    df = pl.read_parquet(INPUT_PARQUET)
    if "row_idx" not in df.columns:
        df = df.with_row_index(name="row_idx", offset=0)

    raw_rows = len(df)
    raw_unique_ids = int(df["id"].n_unique())

    id_counts = df.group_by("id").len().filter(pl.col("len") > 1)
    dup_ids_list = id_counts["id"].to_list()
    total_dup_ids = len(dup_ids_list)

    dup_df = df.filter(pl.col("id").is_in(dup_ids_list))
    cls = classify_duplicates(dup_df)
    cat_a_ids, cat_b_ids, cat_c_ids = cls["cat_a_ids"], cls["cat_b_ids"], cls["cat_c_ids"]
    body_diff_only_whitespace = cls["body_diff_only_whitespace"]
    body_actual_text_diff = cls["body_actual_text_diff"]

    # 1. Drop all rows belonging to Category C IDs
    df_no_c = df.filter(~pl.col("id").is_in(cat_c_ids))
    rows_removed_cat_c = raw_rows - len(df_no_c)

    # 2. Keep one row per id for Category A & B: most complete, then lowest row_idx
    is_body_valid = (
        pl.col("body").is_not_null()
        & (pl.col("body").str.strip_chars() != "")
        & (pl.col("body").str.strip_chars().str.to_lowercase() != "nan")
    ).cast(pl.Int8)
    is_rate_valid = pl.col("rate").is_not_null().cast(pl.Int8)
    is_rec_valid = (
        pl.col("recommendation_status").is_not_null()
        & (pl.col("recommendation_status").str.strip_chars() != "")
        & (pl.col("recommendation_status").str.strip_chars().str.to_lowercase() != "nan")
    ).cast(pl.Int8)

    df_clean = (
        df_no_c.with_columns(
            (is_body_valid + is_rate_valid + is_rec_valid).alias("_completeness_score")
        )
        .sort(["id", "_completeness_score", "row_idx"], descending=[False, True, False])
        .unique(subset=["id"], keep="first", maintain_order=False)
        .drop("_completeness_score")
        .sort("row_idx")
    )

    clean_rows = len(df_clean)
    clean_unique_ids = int(df_clean["id"].n_unique())
    rows_removed_cat_ab = len(df_no_c) - clean_rows
    total_removed = raw_rows - clean_rows

    expected_clean_rows = raw_unique_ids - len(cat_c_ids)
    formula_matches = clean_rows == expected_clean_rows

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_clean.write_parquet(OUTPUT_PARQUET, compression="zstd")

    policy_summary = {
        "raw_total_rows": raw_rows,
        "raw_unique_ids": raw_unique_ids,
        "total_duplicate_ids": total_dup_ids,
        "cat_A_ids_count": len(cat_a_ids),
        "cat_B_ids_count": len(cat_b_ids),
        "cat_C_ids_count": len(cat_c_ids),
        "body_diff_whitespace_zwnj_yeh_kaf_only": body_diff_only_whitespace,
        "body_actual_text_diff": body_actual_text_diff,
        "rows_removed_from_cat_C_drop_all": rows_removed_cat_c,
        "rows_removed_from_cat_AB_keep_first_valid": rows_removed_cat_ab,
        "total_rows_removed": total_removed,
        "expected_clean_rows_formula": expected_clean_rows,
        "clean_total_rows": clean_rows,
        "clean_unique_ids": clean_unique_ids,
        "formula_verified": formula_matches,
        "output_parquet": str(OUTPUT_PARQUET),
    }

    POLICY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(POLICY_JSON, "w", encoding="utf-8") as f:
        json.dump(policy_summary, f, indent=2, ensure_ascii=False)

    print("=== STEP 2: DEDUPLICATION POLICY COMPLETE ===")
    print(f"Raw Total Rows:               {raw_rows:,}")
    print(f"Raw Unique IDs:               {raw_unique_ids:,}")
    print(f"Categories:                   A={len(cat_a_ids):,}, B={len(cat_b_ids):,}, C={len(cat_c_ids):,}")
    print(f"Body Diff (WS/ZWNJ/Yeh-Kaf):  {body_diff_only_whitespace:,} | Body Real Conflict: {body_actual_text_diff:,}")
    print(f"Category C IDs (drop all):    {len(cat_c_ids):,} IDs ({rows_removed_cat_c:,} rows removed)")
    print(f"Category A+B extra rows:      {rows_removed_cat_ab:,} rows removed")
    print(f"Total Rows Removed:           {total_removed:,}")
    print(f"Expected Clean Rows (U - C):  {expected_clean_rows:,}")
    print(f"Actual Clean Rows:            {clean_rows:,}")
    print(f"Formula Match Verified:       {formula_matches}")

    if not formula_matches:
        raise ValueError(f"Clean row count mismatch! Expected {expected_clean_rows}, got {clean_rows}")

    return policy_summary


if __name__ == "__main__":
    run_dedup()
