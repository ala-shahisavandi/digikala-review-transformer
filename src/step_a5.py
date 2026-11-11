"""
Step A5: Repetition, label conflicts, and product/seller concentration (clean dataset).

1) Duplicate review bodies, on RAW text and on NORMALIZED text (normalize_text_minimal).
   The conclusions use the normalized text, because that is what the model sees.
2) Identical (normalized) texts with conflicting labels, reported separately for short
   texts (<= 3 words, e.g. "good", "thanks") and longer texts (> 3 words).
   Band conflicts are also reported with rate=0 rows excluded (rate=0 is suspicious, see A3).
3) Concentration of reviews among products and sellers (top-share, Gini, Lorenz curve).
4) Leakage simulation: row-level random split vs product-grouped split
   (share of test rows whose product / normalized text also appears in train).
   This is an ILLUSTRATION with a hash split, not the official Part B split.

Run from repo root:  python src/step_a5.py
Outputs: artifacts/eda/step_a5_repetition_concentration.json,
         reports/figures/A5_repetition_concentration.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from eda_utils import normalize_text_minimal

CLEAN_PARQUET = Path("data/interim/digikala_clean.parquet")
OUT_JSON = Path("artifacts/eda/step_a5_repetition_concentration.json")
OUT_FIG = Path("reports/figures/A5_repetition_concentration.png")
SEED = 2026
SHORT_MAX_WORDS = 3


def pct(a: int, b: int) -> float:
    return round(a / b * 100, 3) if b else 0.0


def gini(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(np.float64))
    n = len(x)
    i = np.arange(1, n + 1)
    return float((2 * np.sum(i * x)) / (n * x.sum()) - (n + 1) / n)


def concentration(counts: np.ndarray, label: str) -> dict:
    desc = np.sort(counts)[::-1]
    total = int(desc.sum())
    n = len(desc)
    out = {"entity": label, "n_entities": n, "total_reviews": total, "gini": round(gini(counts), 4)}
    for frac in (0.01, 0.05, 0.10, 0.20):
        k = max(1, int(round(n * frac)))
        out[f"top_{int(frac*100)}pct_share_of_reviews"] = pct(int(desc[:k].sum()), total)
    for k in (10, 100):
        out[f"top_{k}_share_of_reviews"] = pct(int(desc[:k].sum()), total)
    return out


def lorenz(counts: np.ndarray, points: int = 2000):
    x = np.sort(counts.astype(np.float64))
    cum = np.cumsum(x) / x.sum()
    idx = np.unique(np.linspace(0, len(x) - 1, points).astype(int))
    return (idx + 1) / len(x), cum[idx]


def conflict_summary(grp: pl.DataFrame, n_labels: str, n_labeled: str) -> dict:
    sub = grp.filter(pl.col(n_labeled) >= 2)
    conf = sub.filter(pl.col(n_labels) >= 2)
    return {
        "groups_with_2plus_labeled_rows": sub.height,
        "conflicting_groups": conf.height,
        "conflict_pct_of_groups": pct(conf.height, sub.height),
        "rows_in_conflicting_groups": int(conf["n"].sum()) if conf.height else 0,
    }


def leakage(df: pl.DataFrame, split_expr: pl.Expr) -> dict:
    d = df.with_columns(split_expr.alias("split"))
    tr = d.filter(pl.col("split") == "train")
    te = d.filter(pl.col("split") == "test")
    tr_texts = tr.select("norm").unique()
    tr_prods = tr.select("product_id").unique()
    te_long = te.filter(pl.col("words") > SHORT_MAX_WORDS)
    prod_in = te.join(tr_prods, on="product_id", how="semi").height
    text_in = te.join(tr_texts, on="norm", how="semi").height
    text_in_long = te_long.join(tr_texts, on="norm", how="semi").height
    return {
        "train_rows": tr.height, "test_rows": te.height,
        "test_rows_product_in_train": prod_in,
        "test_rows_product_in_train_pct": pct(prod_in, te.height),
        "test_rows_text_in_train": text_in,
        "test_rows_text_in_train_pct": pct(text_in, te.height),
        "test_long_rows": te_long.height,
        "test_long_rows_text_in_train_pct": pct(text_in_long, te_long.height),
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    rec = pl.col("recommendation_status").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    rate = pl.col("rate")
    seller_ok = (
        pl.col("seller_code").is_not_null()
        & (pl.col("seller_code").cast(pl.Utf8).str.strip_chars().str.to_lowercase() != "nan")
        & (pl.col("seller_code").cast(pl.Utf8).str.strip_chars() != "")
    )
    df = pl.scan_parquet(CLEAN_PARQUET).select([
        pl.col("id"), pl.col("product_id"), pl.col("body"),
        pl.when(seller_ok).then(pl.col("seller_code").cast(pl.Utf8)).otherwise(None).alias("seller_code"),
        pl.when(rec.is_in(["recommended", "no_idea", "not_recommended"]))
        .then(rec).otherwise(pl.lit("missing")).alias("rec_class"),
        pl.when(rate.is_null() | (rate < 0) | (rate > 5)).then(pl.lit("invalid"))
        .when(rate < 2.5).then(pl.lit("low"))
        .when(rate < 4.0).then(pl.lit("medium"))
        .otherwise(pl.lit("high")).alias("band"),
        pl.when(rate.is_null() | (rate <= 0) | (rate > 5)).then(None)
        .when(rate < 2.5).then(pl.lit("low"))
        .when(rate < 4.0).then(pl.lit("medium"))
        .otherwise(pl.lit("high")).alias("band_nz"),
    ]).collect(engine="streaming")

    # Keep only rows with a real body (same rule as A4)
    has_text = (
        pl.col("body").is_not_null()
        & (pl.col("body").str.strip_chars() != "")
        & (pl.col("body").str.strip_chars().str.to_lowercase() != "nan")
    )
    n_all = df.height
    df = df.filter(has_text)
    n = df.height
    print(f"Rows: {n_all:,} | with real body: {n:,}")

    # ---------------------------------------------------------------- A5-1
    print("\n" + "=" * 70)
    print("A5-1: DUPLICATE BODIES (raw vs normalized)")
    print("=" * 70)
    raw_groups = df.group_by("body").len()
    raw_unique = raw_groups.height
    raw_dup_groups = raw_groups.filter(pl.col("len") > 1)
    raw_dup_rows = int(raw_dup_groups["len"].sum())
    del raw_groups

    df = df.with_columns(
        pl.Series("norm", [normalize_text_minimal(b) for b in df["body"].to_list()])
    ).drop("body")
    df = df.with_columns(pl.col("norm").str.count_matches(r"\S+").alias("words"))

    grp_all = df.group_by("norm").agg(
        pl.len().alias("n"),
        pl.col("words").first().alias("words"),
        pl.col("product_id").n_unique().alias("n_products"),
        pl.col("rec_class").filter(pl.col("rec_class") != "missing").n_unique().alias("n_rec_labels"),
        pl.col("rec_class").filter(pl.col("rec_class") != "missing").len().alias("n_rec_labeled"),
        pl.col("band").filter(pl.col("band") != "invalid").n_unique().alias("n_band_labels"),
        pl.col("band").filter(pl.col("band") != "invalid").len().alias("n_band_labeled"),
        pl.col("band_nz").drop_nulls().n_unique().alias("n_bandnz_labels"),
        pl.col("band_nz").drop_nulls().len().alias("n_bandnz_labeled"),
    )
    norm_unique = grp_all.height
    grp = grp_all.filter(pl.col("n") > 1)
    del grp_all
    norm_dup_rows = int(grp["n"].sum())
    norm_extra_rows = norm_dup_rows - grp.height

    dup_summary = {
        "raw": {
            "unique_bodies": raw_unique,
            "duplicated_bodies": raw_dup_groups.height,
            "rows_in_duplicated_bodies": raw_dup_rows,
            "rows_in_duplicated_bodies_pct": pct(raw_dup_rows, n),
        },
        "normalized": {
            "unique_bodies": norm_unique,
            "duplicated_bodies": grp.height,
            "rows_in_duplicated_bodies": norm_dup_rows,
            "rows_in_duplicated_bodies_pct": pct(norm_dup_rows, n),
            "extra_rows_beyond_first": norm_extra_rows,
        },
        "unique_bodies_merged_by_normalization": raw_unique - norm_unique,
    }
    for k in ("raw", "normalized"):
        d = dup_summary[k]
        print(f"[{k}] unique={d['unique_bodies']:,} | duplicated bodies={d['duplicated_bodies']:,} | "
              f"rows in duplicated bodies={d['rows_in_duplicated_bodies']:,} ({d['rows_in_duplicated_bodies_pct']}%)")
    print(f"Unique bodies merged by normalization: {dup_summary['unique_bodies_merged_by_normalization']:,}")

    top_texts = (
        grp.sort("n", descending=True).head(20)
        .select(["norm", "n", "words", "n_products", "n_rec_labels", "n_band_labels"]).to_dicts()
    )
    print("\nTop 10 most repeated normalized texts (the text itself is in the JSON; open it in VS Code):")
    for r in top_texts[:10]:
        print(f"  count={r['n']:>8,} words={r['words']} products={r['n_products']:,} "
              f"rec_labels={r['n_rec_labels']} band_labels={r['n_band_labels']}")

    # duplicate share by length bucket (row level)
    df = df.with_columns((pl.len().over("norm") > 1).alias("is_dup"))
    bucket = (
        pl.when(pl.col("words") <= 1).then(pl.lit("1"))
        .when(pl.col("words") == 2).then(pl.lit("2"))
        .when(pl.col("words") == 3).then(pl.lit("3"))
        .when(pl.col("words") <= 5).then(pl.lit("4-5"))
        .when(pl.col("words") <= 10).then(pl.lit("6-10"))
        .when(pl.col("words") <= 20).then(pl.lit("11-20"))
        .otherwise(pl.lit("21+"))
    )
    b_order = ["1", "2", "3", "4-5", "6-10", "11-20", "21+"]
    bt = (
        df.with_columns(bucket.alias("wb")).group_by("wb")
        .agg(pl.len().alias("n"), pl.col("is_dup").sum().alias("dup_rows"))
        .to_dicts()
    )
    bt = sorted(bt, key=lambda r: b_order.index(r["wb"]))
    for r in bt:
        r["dup_pct"] = pct(int(r["dup_rows"]), int(r["n"]))
    print("\nShare of rows whose normalized text appears more than once, by word count:")
    for r in bt:
        print(f"  words={r['wb']:>6s}: n={r['n']:>9,}  duplicated={r['dup_rows']:>9,}  ({r['dup_pct']:.2f}%)")

    # ---------------------------------------------------------------- A5-2
    print("\n" + "=" * 70)
    print(f"A5-2: IDENTICAL TEXT, CONFLICTING LABELS (short = <= {SHORT_MAX_WORDS} words)")
    print("=" * 70)
    segments = {
        "all": grp,
        "short": grp.filter(pl.col("words") <= SHORT_MAX_WORDS),
        "long": grp.filter(pl.col("words") > SHORT_MAX_WORDS),
    }
    conflicts = {}
    for seg, g in segments.items():
        conflicts[seg] = {
            "duplicated_groups": g.height,
            "recommendation": conflict_summary(g, "n_rec_labels", "n_rec_labeled"),
            "rating_band": conflict_summary(g, "n_band_labels", "n_band_labeled"),
            "rating_band_excluding_rate0": conflict_summary(g, "n_bandnz_labels", "n_bandnz_labeled"),
        }
        print(f"\n[{seg}] duplicated groups: {g.height:,}")
        for tname in ("recommendation", "rating_band", "rating_band_excluding_rate0"):
            c = conflicts[seg][tname]
            print(f"  {tname:>28s}: groups with >=2 labeled rows={c['groups_with_2plus_labeled_rows']:>8,} | "
                  f"conflicting={c['conflicting_groups']:>7,} ({c['conflict_pct_of_groups']:.2f}%) | "
                  f"rows in conflicting groups={c['rows_in_conflicting_groups']:,}")

    multi_product = grp.filter(pl.col("n_products") > 1)
    cross_product = {
        "duplicated_groups_spanning_multiple_products": multi_product.height,
        "pct_of_duplicated_groups": pct(multi_product.height, grp.height),
        "rows_in_those_groups": int(multi_product["n"].sum()) if multi_product.height else 0,
    }
    print(f"\nDuplicated texts that appear in more than one product: {cross_product}")

    # ---------------------------------------------------------------- A5-3
    print("\n" + "=" * 70)
    print("A5-3: CONCENTRATION AMONG PRODUCTS AND SELLERS (clean data)")
    print("=" * 70)
    prod_counts = df.group_by("product_id").len()["len"].to_numpy()
    conc_prod = concentration(prod_counts, "product")
    sel = df.filter(pl.col("seller_code").is_not_null())
    sel_counts = sel.group_by("seller_code").len()
    conc_sell = concentration(sel_counts["len"].to_numpy(), "seller")
    conc_sell["rows_without_seller"] = n - sel.height
    top_sellers = sel_counts.sort("len", descending=True).head(10).to_dicts()
    for c in (conc_prod, conc_sell):
        print(json.dumps(c))
    print("Top 10 sellers (code: reviews):", {r["seller_code"]: r["len"] for r in top_sellers})
    print("Note: the scraper capped reviews per product (about 200), so product concentration is "
          "understated relative to real review volume.")

    # ---------------------------------------------------------------- A5-4
    print("\n" + "=" * 70)
    print("A5-4: LEAKAGE SIMULATION (row-level random vs product-grouped; hash split 70/15/15)")
    print("=" * 70)

    def split_from(h: pl.Expr) -> pl.Expr:
        return pl.when(h < 70).then(pl.lit("train")).when(h < 85).then(pl.lit("val")).otherwise(pl.lit("test"))

    leak_row = leakage(df, split_from(pl.col("id").hash(seed=SEED) % 100))
    leak_grp = leakage(df, split_from(pl.col("product_id").hash(seed=SEED) % 100))
    for name, lk in (("row-level random", leak_row), ("product-grouped", leak_grp)):
        print(f"[{name}] test rows={lk['test_rows']:,} | product also in train: {lk['test_rows_product_in_train_pct']}% | "
              f"text also in train: {lk['test_rows_text_in_train_pct']}% | "
              f"text also in train (>{SHORT_MAX_WORDS} words only): {lk['test_long_rows_text_in_train_pct']}%")

    # ---------------------------------------------------------------- Figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)

    ax = axes[0, 0]
    freqs = np.sort(grp["n"].to_numpy())[::-1]
    ax.loglog(np.arange(1, len(freqs) + 1), freqs, color="#2b5c8f")
    ax.set_title("A5-a: Rank-frequency of repeated normalized texts", fontweight="bold")
    ax.set_xlabel("Rank (log)")
    ax.set_ylabel("Occurrences (log)")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    ax = axes[0, 1]
    ax.bar([r["wb"] for r in bt], [r["dup_pct"] for r in bt], color="#388e3c", edgecolor="white")
    ax.set_title("A5-b: % of rows with a repeated text, by word count", fontweight="bold")
    ax.set_xlabel("Words in review")
    ax.set_ylabel("% of rows")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    ax = axes[1, 0]
    for counts, label, color in ((prod_counts, f"products (Gini {conc_prod['gini']})", "#2b5c8f"),
                                 (sel_counts["len"].to_numpy(), f"sellers (Gini {conc_sell['gini']})", "#d95f02")):
        x, y = lorenz(counts)
        ax.plot(x, y, label=label, color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9e9e9e", label="equality")
    ax.set_title("A5-c: Lorenz curve of review concentration", fontweight="bold")
    ax.set_xlabel("Cumulative share of entities (smallest first)")
    ax.set_ylabel("Cumulative share of reviews")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    ax = axes[1, 1]
    names = ["recommendation", "rating_band", "rating_band_excluding_rate0"]
    short_v = [conflicts["short"][t]["conflict_pct_of_groups"] for t in names]
    long_v = [conflicts["long"][t]["conflict_pct_of_groups"] for t in names]
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, short_v, 0.4, label=f"short (<= {SHORT_MAX_WORDS} words)", color="#d32f2f")
    ax.bar(xs + 0.2, long_v, 0.4, label=f"long (> {SHORT_MAX_WORDS} words)", color="#1976d2")
    ax.set_xticks(xs)
    ax.set_xticklabels(["recommendation", "band", "band (no rate=0)"])
    ax.set_title("A5-d: % of repeated texts with conflicting labels", fontweight="bold")
    ax.set_ylabel("% of groups with >= 2 labeled rows")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)

    result = {
        "rows_with_text": n,
        "comparison_basis": "normalized text via eda_utils.normalize_text_minimal (raw reported for reference)",
        "short_text_definition": f"<= {SHORT_MAX_WORDS} words",
        "duplicate_bodies": dup_summary,
        "duplicate_share_by_word_count": bt,
        "top20_repeated_normalized_texts": top_texts,
        "label_conflicts": conflicts,
        "duplicated_texts_across_products": cross_product,
        "concentration": {"products": conc_prod, "sellers": conc_sell, "top10_sellers": top_sellers,
                          "note": "reviews per product are capped by the scraper (about 200)"},
        "leakage_simulation": {"seed": SEED, "split": "hash 70/15/15, illustration only",
                               "row_level_random": leak_row, "product_grouped": leak_grp},
        "figure_path": str(OUT_FIG),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_FIG}")


if __name__ == "__main__":
    main()
