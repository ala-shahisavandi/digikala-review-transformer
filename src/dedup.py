"""
Step 2: Self-Contained Deduplication Policy Execution.
- Audits duplicate IDs with fixed Unicode escapes (U+200C, U+064A->U+06CC, U+0643->U+06A9).
- Category A & B: Keep first occurrence based on row_idx, prioritizing valid non-null/non-empty values.
- Category C: Drop ALL occurrences of conflicting IDs.
- Verifies clean_total_rows == raw_unique_ids - len(cat_C_ids).
"""
import json
import re
from pathlib import Path
import polars as pl

INPUT_PARQUET = Path("data/interim/digikala_comments.parquet")
REPORT_JSON = Path("artifacts/eda/duplicate_ids_report.json")
OUTPUT_PARQUET = Path("data/interim/digikala_clean.parquet")
POLICY_JSON = Path("artifacts/eda/dedup_policy.json")


def normalize_text_minimal(text: str | None) -> str:
    if text is None:
        return ""
    s = (
        text.replace("\u200c", " ")
        .replace("\u064a", "\u06cc")
        .replace("\u0643", "\u06a9")
    )
    return re.sub(r"\s+", " ", s).strip()


def is_effectively_empty(val: object) -> bool:
    if val is None:
        return True
    s = str(val).strip()
    return s == "" or s.lower() == "nan"


def run_dedup() -> dict:
    df = pl.read_parquet(INPUT_PARQUET)
    if "row_idx" not in df.columns:
        df = df.with_row_index(name="row_idx", offset=0)

    raw_rows = len(df)
    raw_unique_ids = int(df["id"].n_unique())

    id_counts = df.group_by("id").len().filter(pl.col("len") > 1)
    dup_ids_list = id_counts["id"].to_list()
    total_dup_ids = len(dup_ids_list)

    dup_df = df.filter(pl.col("id").is_in(dup_ids_list)).sort(["id", "row_idx"])

    id_groups = {}
    for row in dup_df.select(["row_idx", "id", "body", "rate", "recommendation_status"]).iter_rows(named=True):
        gid = row["id"]
        if gid not in id_groups:
            id_groups[gid] = []
        id_groups[gid].append(row)

    cat_a_ids = []
    cat_b_ids = []
    cat_c_ids = []
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

    # 1. Drop all rows belonging to Category C IDs
    df_no_c = df.filter(~pl.col("id").is_in(cat_c_ids))
    rows_removed_cat_c = raw_rows - len(df_no_c)

    # 2. Keep first occurrence by row_idx for Category A & B, prioritizing valid non-null values
    is_body_valid = (
        pl.col("body").is_not_null() &
        (pl.col("body").str.strip_chars() != "") &
        (pl.col("body").str.strip_chars().str.to_lowercase() != "nan")
    ).cast(pl.Int8)

    is_rate_valid = pl.col("rate").is_not_null().cast(pl.Int8)

    is_rec_valid = (
        pl.col("recommendation_status").is_not_null() &
        (pl.col("recommendation_status").str.strip_chars() != "") &
        (pl.col("recommendation_status").str.strip_chars().str.to_lowercase() != "nan")
    ).cast(pl.Int8)

    df_clean = (
        df_no_c.with_columns([
            (is_body_valid + is_rate_valid + is_rec_valid).alias("_completeness_score")
        ])
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
    formula_matches = (clean_rows == expected_clean_rows)

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
    print(f"Categories (Fixed Escapes):   A={len(cat_a_ids):,}, B={len(cat_b_ids):,}, C={len(cat_c_ids):,}")
    print(f"Body Diff (WS/ZWNJ/Yeh-Kaf):  {body_diff_only_whitespace:,} | Body Real Conflict: {body_actual_text_diff:,}")
    print(f"Category C IDs (drop all):    {len(cat_c_ids):,} IDs ({rows_removed_cat_c:,} rows removed)")
    print(f"Category A+B extra rows:      {rows_removed_cat_ab:,} rows removed (kept first valid by row_idx)")
    print(f"Total Rows Removed:           {total_removed:,}")
    print(f"Expected Clean Rows (U - C):  {expected_clean_rows:,}")
    print(f"Actual Clean Rows:            {clean_rows:,}")
    print(f"Formula Match Verified:       {formula_matches}")

    if not formula_matches:
        raise ValueError(f"Clean row count mismatch! Expected {expected_clean_rows}, got {clean_rows}")

    return policy_summary


if __name__ == "__main__":
    run_dedup()
