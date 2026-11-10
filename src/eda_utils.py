"""
EDA Utilities for Digikala Review Dataset
Includes rigorous audit for duplicate IDs and content conflicts.
"""
import json
import re
from pathlib import Path
import polars as pl


def normalize_text_minimal(text: str | None) -> str:
    """Single canonical minimal normalizer (used by dedup, audit, and later stages).

    - ZWNJ (U+200C)          -> space
    - Arabic Yeh (U+064A)    -> Persian Yeh (U+06CC)
    - Arabic Kaf (U+0643)    -> Persian Keheh (U+06A9)
    - collapse whitespace, strip
    """
    if text is None:
        return ""
    s = (
        text.replace("\u200c", " ")
        .replace("\u064a", "\u06cc")
        .replace("\u0643", "\u06a9")
    )
    return re.sub(r"\s+", " ", s).strip()


def audit_duplicate_ids(
    parquet_path: str = "data/interim/digikala_comments.parquet",
    output_json_path: str = "artifacts/eda/duplicate_ids_report.json",
) -> dict:
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(parquet_path)

    total_rows = len(df)
    unique_ids = df["id"].n_unique()
    extra_rows = total_rows - unique_ids

    # 1. Identify IDs with duplicates
    id_counts = df.group_by("id").len().filter(pl.col("len") > 1)
    dup_ids_list = id_counts["id"].to_list()
    total_dup_ids = len(dup_ids_list)

    # IDs appearing more than 2 times
    ids_over_two = id_counts.filter(pl.col("len") > 2).sort("len", descending=True)
    top_recurring_ids = ids_over_two.to_dicts()

    dup_df = df.filter(pl.col("id").is_in(dup_ids_list)).sort(["id", "created_at"])

    id_groups = {}
    cols_of_interest = ["id", "body", "rate", "recommendation_status", "likes", "dislikes", "title", "product_id"]
    for row in dup_df.select(cols_of_interest).iter_rows(named=True):
        gid = row["id"]
        if gid not in id_groups:
            id_groups[gid] = []
        id_groups[gid].append(row)

    identical_target_ids = 0
    null_vs_value_ids = 0
    actual_conflict_ids = 0

    body_exact_same = 0
    body_diff_only_whitespace = 0
    body_actual_text_diff = 0

    ids_with_body_diff = set()
    ids_with_rate_diff = set()
    ids_with_rec_diff = set()

    for gid, rows in id_groups.items():
        # Rate
        rates = [r["rate"] for r in rows]
        rates_valid = [r for r in rates if r is not None]
        rates_unique_valid = set(rates_valid)
        has_rate_conflict = len(rates_unique_valid) > 1
        has_rate_null_diff = (len(rates_unique_valid) == 1) and (len(rates_valid) < len(rates))
        if has_rate_conflict or has_rate_null_diff:
            ids_with_rate_diff.add(gid)

        # Recommendation status
        recs = [r["recommendation_status"] for r in rows]
        recs_valid = [
            r for r in recs
            if r is not None and str(r).strip() != "" and str(r).lower() != "nan"
        ]
        recs_unique_valid = set(recs_valid)
        has_rec_conflict = len(recs_unique_valid) > 1
        has_rec_null_diff = (len(recs_unique_valid) == 1) and (len(recs_valid) < len(recs))
        if has_rec_conflict or has_rec_null_diff:
            ids_with_rec_diff.add(gid)

        # Body
        bodies = [r["body"] for r in rows]
        bodies_valid = [
            b for b in bodies
            if b is not None and str(b).strip() != "" and str(b).lower() != "nan"
        ]
        bodies_unique_valid = set(bodies_valid)

        has_body_diff = len(set(bodies)) > 1
        if has_body_diff:
            ids_with_body_diff.add(gid)
            if len(bodies_unique_valid) <= 1:
                pass  # only null/nan vs string
            else:
                normalized_bodies = {normalize_text_minimal(b) for b in bodies_valid}
                if len(normalized_bodies) == 1:
                    body_diff_only_whitespace += 1
                else:
                    body_actual_text_diff += 1
        else:
            body_exact_same += 1

        has_any_actual_conflict = (
            has_rate_conflict
            or has_rec_conflict
            or (len(bodies_unique_valid) > 1 and len({normalize_text_minimal(b) for b in bodies_valid}) > 1)
        )
        has_any_null_vs_val = (
            (has_rate_null_diff and not has_rate_conflict)
            or (has_rec_null_diff and not has_rec_conflict)
            or (len(bodies_unique_valid) == 1 and len(bodies_valid) < len(bodies))
            or (len(bodies_unique_valid) > 1 and len({normalize_text_minimal(b) for b in bodies_valid}) == 1)
        )

        if has_any_actual_conflict:
            actual_conflict_ids += 1
        elif has_any_null_vs_val:
            null_vs_value_ids += 1
        else:
            identical_target_ids += 1

    union_diff_ids = ids_with_body_diff.union(ids_with_rate_diff).union(ids_with_rec_diff)

    report = {
        "summary": {
            "total_rows": total_rows,
            "unique_ids": unique_ids,
            "extra_rows": extra_rows,
            "total_duplicate_ids": total_dup_ids,
        },
        "question_1_breakdown": {
            "identical_targets_count": identical_target_ids,
            "identical_targets_pct": round(identical_target_ids / total_dup_ids * 100, 2),
            "null_or_format_variant_count": null_vs_value_ids,
            "null_or_format_variant_pct": round(null_vs_value_ids / total_dup_ids * 100, 2),
            "actual_conflict_count": actual_conflict_ids,
            "actual_conflict_pct": round(actual_conflict_ids / total_dup_ids * 100, 2),
        },
        "question_2_body_analysis": {
            "body_exact_same_count": body_exact_same,
            "body_diff_whitespace_or_zwnj_only": body_diff_only_whitespace,
            "body_actual_text_diff": body_actual_text_diff,
        },
        "question_3_union_diff": {
            "ids_differing_in_body": len(ids_with_body_diff),
            "ids_differing_in_rate": len(ids_with_rate_diff),
            "ids_differing_in_recommendation_status": len(ids_with_rec_diff),
            "union_ids_differing_in_any_of_three": len(union_diff_ids),
        },
        "question_4_recurring_ids": top_recurring_ids,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== AUDIT SUMMARY ===")
    print(f"Total Unique Duplicate IDs: {total_dup_ids:,}")
    print(f"(A) Identical in targets (body, rate, rec): {identical_target_ids:,} ({report['question_1_breakdown']['identical_targets_pct']}%)")
    print(f"(B) Null/Whitespace variation only: {null_vs_value_ids:,} ({report['question_1_breakdown']['null_or_format_variant_pct']}%)")
    print(f"(C) Actual semantic/label conflict: {actual_conflict_ids:,} ({report['question_1_breakdown']['actual_conflict_pct']}%)")
    print(f"\nBody Details: Whitespace/ZWNJ diff = {body_diff_only_whitespace:,} | Real text diff = {body_actual_text_diff:,}")
    print(f"Union of IDs differing in body, rate, or rec: {len(union_diff_ids):,}")
    print(f"IDs appearing > 2 times: {len(top_recurring_ids)}")
    for item in top_recurring_ids:
        print(f"  - id={item['id']}: count={item['len']}")
    print(f"\nArtifact saved to: {output_json_path}")
    return report


if __name__ == "__main__":
    audit_duplicate_ids()
