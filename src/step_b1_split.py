"""
Part B / step B1: normalization, product-grouped 70/15/15 split, public masking protocol,
standard data artifacts and split_manifest.json.

Input : data/interim/digikala_clean.parquet   (after the A/B/C dedup policy)
Output: data/processed/{train_supervised, train_pseudo_unlabeled, natural_unlabeled,
                        validation, test}.csv, pseudo_reference_labels.csv,
        data/processed/split_manifest.json  (+ copy in artifacts/split/)

Protocol (from the project definition):
  * Only `body` is a model input. `product_id` is used ONLY to build the grouped split.
  * One shared, deterministic product-grouped split (a product lives in exactly one split).
    Products are shuffled with numpy default_rng(SPLIT_SEED) and cut at 70% / 85% of the
    cumulative ROW count, so the row proportions are almost exactly 70/15/15.
  * Rows with a missing target stay in their product's split; a head simply ignores them.
  * Masking: inside the TRAIN split, for each recommendation class, rows are sorted by
    stable_hash(id) and the first floor(0.20 * class_count) rows move to the artificial
    unlabeled pool. Validation and test rows are never masked.
  * train_pseudo_unlabeled.csv contains only id and body. The true labels of those rows are
    saved separately in pseudo_reference_labels.csv and may be used ONLY after pseudo-label
    selection, to audit pseudo-label quality.
  * natural_unlabeled.csv = TRAIN rows whose recommendation label is naturally missing
    (id, body). These rows also stay in train_supervised.csv for the rating-band head.
  * validation.csv / test.csv carry a `text_in_train` flag (the normalized text also occurs in
    train). It is a data property, not a label, and enables the "text not in train" metric
    subset suggested by A5.

`rate` is kept as a column in the labeled files ONLY for the rate=0 ablation and sensitivity
analysis. It must never be passed to a model as a feature.

Run from repo root:  python src/step_b1_split.py
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from preprocess import normalize_text, NORMALIZER_VERSION, MAX_REPEAT

CLEAN_PARQUET = Path("data/interim/digikala_clean.parquet")
OUT_DIR = Path("data/processed")
MANIFEST_COPY = Path("artifacts/split/split_manifest.json")

SPLIT_SEED = 2026
MASK_SEED = "digikala-semi-supervised-v1"
MASK_RATE = 0.20
MASK_RATE_PCT = 20            # integer arithmetic: floor(n * 20 / 100) == floor(0.20 * n)
SPLIT_FRACTIONS = (0.70, 0.15)  # train, validation; test gets the rest (0.15)
SPLITS = ["train", "validation", "test"]
REC_CLASSES = ["recommended", "no_idea", "not_recommended"]
BANDS = ["low", "medium", "high"]
BAND_ENCODING = {"low": 0, "medium": 1, "high": 2}
SHORT_MAX_WORDS = 3
LABEL_COLS = ["id", "product_id", "body", "recommendation_status", "rating_band", "rate"]


def stable_hash(row_id: str) -> int:
    key = f"{MASK_SEED}|{row_id}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:16], 16)


def stable_hash_hex(row_id: str) -> str:
    """Same value as stable_hash(), kept as the 16-char hex string (sorts identically)."""
    return hashlib.sha256(f"{MASK_SEED}|{row_id}".encode("utf-8")).hexdigest()[:16]


def pct(a: int, b: int) -> float:
    return round(a / b * 100, 3) if b else 0.0


def dist(d: pl.DataFrame, col: str) -> dict:
    out = {}
    for row in d.group_by(col).len().iter_rows():
        out["missing" if row[0] is None else row[0]] = int(row[1])
    return out


def leak_stats(d: pl.DataFrame) -> dict:
    words = pl.col("body").str.count_matches(r"\S+")
    r = d.select([
        pl.len().alias("n"),
        pl.col("text_in_train").sum().alias("in_train"),
        (words > SHORT_MAX_WORDS).sum().alias("long_n"),
        ((words > SHORT_MAX_WORDS) & pl.col("text_in_train")).sum().alias("long_in_train"),
    ]).row(0, named=True)
    return {
        "rows": int(r["n"]),
        "rows_text_in_train": int(r["in_train"]),
        "rows_text_in_train_pct": pct(int(r["in_train"]), int(r["n"])),
        "rows_text_not_in_train": int(r["n"] - r["in_train"]),
        "long_rows": int(r["long_n"]),
        "long_rows_text_in_train_pct": pct(int(r["long_in_train"]), int(r["long_n"])),
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_COPY.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load + clean text
    rec = pl.col("recommendation_status").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    rate = pl.col("rate")
    has_text = (
        pl.col("body").is_not_null()
        & (pl.col("body").str.strip_chars() != "")
        & (pl.col("body").str.strip_chars().str.to_lowercase() != "nan")
    )
    raw = (pl.scan_parquet(CLEAN_PARQUET)
           .select(["id", "product_id", "body", "rate", "recommendation_status"])
           .collect(engine="streaming"))
    n_clean = raw.height
    raw = raw.filter(has_text)
    n_with_text = raw.height
    print(f"Clean rows: {n_clean:,} | with real body: {n_with_text:,}")

    print("Normalizing text (this takes a few minutes)...")
    norm = [normalize_text(b) for b in raw["body"].to_list()]
    raw = raw.with_columns(pl.Series("body", norm))
    df = raw.select([
        pl.col("id"), pl.col("product_id"), pl.col("body"), pl.col("rate"),
        pl.when(rec.is_in(REC_CLASSES)).then(rec).otherwise(None).alias("recommendation_status"),
        pl.when(rate.is_null() | (rate < 0) | (rate > 5)).then(None)
        .when(rate < 2.5).then(pl.lit("low"))
        .when(rate < 4.0).then(pl.lit("medium"))
        .otherwise(pl.lit("high")).alias("rating_band"),
    ]).filter(pl.col("body") != "")
    del raw, norm
    n_model = df.height
    print(f"Rows kept for modeling: {n_model:,} (excluded: {n_clean - n_model:,} without usable text)")

    # ------------------------------------------------------------------ product-grouped split
    counts = df.group_by("product_id").len().sort("product_id")
    pids = counts["product_id"].to_numpy()
    sizes = counts["len"].to_numpy().astype(np.int64)
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(pids))
    cum = np.cumsum(sizes[perm])
    total = int(cum[-1])
    c_train = SPLIT_FRACTIONS[0] * total
    c_val = (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]) * total
    split_idx = np.where(cum <= c_train, 0, np.where(cum <= c_val, 1, 2)).astype(np.int8)
    prod_map = pl.DataFrame({"product_id": pids[perm], "split_idx": split_idx})
    df = df.join(prod_map, on="product_id", how="left")
    assert df["split_idx"].null_count() == 0
    df = df.with_columns(
        pl.when(pl.col("split_idx") == 0).then(pl.lit("train"))
        .when(pl.col("split_idx") == 1).then(pl.lit("validation"))
        .otherwise(pl.lit("test")).alias("split")
    ).drop("split_idx")

    # ------------------------------------------------------------------ masking (train only)
    train_lab = (df.filter((pl.col("split") == "train") & pl.col("recommendation_status").is_not_null())
                   .select(["id", "recommendation_status"]))
    print(f"Hashing {train_lab.height:,} labeled train ids for masking...")
    hexes = [stable_hash_hex(str(i)) for i in train_lab["id"].to_list()]
    train_lab = train_lab.with_columns(pl.Series("h", hexes))
    masked_ids, mask_per_class = [], {}
    for c in REC_CLASSES:
        sub = train_lab.filter(pl.col("recommendation_status") == c).sort(["h", "id"])
        k = (sub.height * MASK_RATE_PCT) // 100
        masked_ids.extend(sub["id"].head(k).to_list())
        mask_per_class[c] = {"labeled_train_rows": int(sub.height), "masked": int(k),
                             "kept_labeled": int(sub.height - k)}
    masked_df = pl.DataFrame({"id": masked_ids, "is_masked": [True] * len(masked_ids)})
    df = df.join(masked_df, on="id", how="left").with_columns(pl.col("is_masked").fill_null(False))
    assert int(df["is_masked"].sum()) == len(masked_ids)

    # ------------------------------------------------------------------ build artifacts
    train = df.filter(pl.col("split") == "train")
    pseudo = train.filter(pl.col("is_masked"))
    train_sup = train.filter(~pl.col("is_masked"))
    natural = train.filter(pl.col("recommendation_status").is_null())
    train_text = train.select("body").unique().with_columns(pl.lit(True).alias("text_in_train"))

    def with_flag(d: pl.DataFrame) -> pl.DataFrame:
        return (d.join(train_text, on="body", how="left")
                 .with_columns(pl.col("text_in_train").fill_null(False)))

    val = with_flag(df.filter(pl.col("split") == "validation"))
    test = with_flag(df.filter(pl.col("split") == "test"))

    # ------------------------------------------------------------------ integrity checks
    checks = {}
    multi = df.group_by("product_id").agg(pl.col("split").n_unique().alias("k")).filter(pl.col("k") > 1)
    checks["no_product_in_two_splits"] = multi.height == 0
    checks["masked_only_in_train"] = int(df.filter(pl.col("is_masked") & (pl.col("split") != "train")).height) == 0
    checks["masked_rows_all_labeled"] = int(pseudo["recommendation_status"].null_count()) == 0
    checks["train_equals_supervised_plus_pseudo"] = train.height == train_sup.height + pseudo.height
    checks["masked_counts_match_floor_rule"] = all(
        v["masked"] == (v["labeled_train_rows"] * MASK_RATE_PCT) // 100 for v in mask_per_class.values())
    checks["ids_unique_across_files"] = (
        df["id"].n_unique() == df.height)
    for k, v in checks.items():
        print(f"  check {k}: {v}")
    assert all(checks.values()), f"integrity check failed: {checks}"

    # ------------------------------------------------------------------ write CSVs
    files = {}

    def write(name: str, d: pl.DataFrame, cols: list) -> None:
        path = OUT_DIR / name
        d.select(cols).sort("id").write_csv(path)
        files[name] = {"rows": int(d.height), "columns": cols, "size_mb": round(path.stat().st_size / 1048576, 2)}
        print(f"  wrote {path} ({d.height:,} rows)")

    write("train_supervised.csv", train_sup, LABEL_COLS)
    write("train_pseudo_unlabeled.csv", pseudo, ["id", "body"])
    write("pseudo_reference_labels.csv", pseudo, ["id", "recommendation_status"])
    write("natural_unlabeled.csv", natural, ["id", "body"])
    write("validation.csv", val, LABEL_COLS + ["text_in_train"])
    write("test.csv", test, LABEL_COLS + ["text_in_train"])

    # ------------------------------------------------------------------ statistics for the manifest
    per_split = {}
    for name, d in (("train", train), ("validation", val), ("test", test)):
        per_split[name] = {
            "rows": int(d.height),
            "rows_pct": pct(int(d.height), n_model),
            "products": int(d["product_id"].n_unique()),
            "recommendation": dist(d, "recommendation_status"),
            "rating_band": {("invalid" if k == "missing" else k): v for k, v in dist(d, "rating_band").items()},
        }
    overall = {"recommendation": dist(df, "recommendation_status"),
               "rating_band": {("invalid" if k == "missing" else k): v
                               for k, v in dist(df, "rating_band").items()}}

    def share(counts: dict, keys: list) -> dict:
        tot = sum(counts.get(k, 0) for k in keys)
        return {k: counts.get(k, 0) / tot * 100 if tot else 0.0 for k in keys}

    rep = {"max_abs_pp_deviation_recommendation": 0.0, "max_abs_pp_deviation_rating_band": 0.0,
           "missing_recommendation_share_pct": {}}
    ov_rec = share(overall["recommendation"], REC_CLASSES)
    ov_band = share(overall["rating_band"], BANDS)
    for name, s in per_split.items():
        sr = share(s["recommendation"], REC_CLASSES)
        sb = share(s["rating_band"], BANDS)
        rep["max_abs_pp_deviation_recommendation"] = round(max(
            rep["max_abs_pp_deviation_recommendation"], max(abs(sr[k] - ov_rec[k]) for k in REC_CLASSES)), 3)
        rep["max_abs_pp_deviation_rating_band"] = round(max(
            rep["max_abs_pp_deviation_rating_band"], max(abs(sb[k] - ov_band[k]) for k in BANDS)), 3)
        rep["missing_recommendation_share_pct"][name] = pct(s["recommendation"].get("missing", 0), s["rows"])
    print(f"Representativeness (max |share diff| vs overall, percentage points): "
          f"recommendation {rep['max_abs_pp_deviation_recommendation']} | "
          f"band {rep['max_abs_pp_deviation_rating_band']}")

    leak = {"validation": leak_stats(val), "test": leak_stats(test)}
    for name, lk in leak.items():
        print(f"  official split {name}: text also in train {lk['rows_text_in_train_pct']}% "
              f"| >{SHORT_MAX_WORDS}-word rows: {lk['long_rows_text_in_train_pct']}%")

    manifest = {
        "script": "src/step_b1_split.py",
        "seeds": {"SPLIT_SEED": SPLIT_SEED, "MASK_SEED": MASK_SEED, "MASK_RATE": MASK_RATE},
        "normalizer": {"module": "src/preprocess.py", "version": NORMALIZER_VERSION,
                       "max_repeat": MAX_REPEAT,
                       "keeps": ["emoji", "latin letters", "punctuation", "negation words", "stop words"],
                       "digits": "Persian/Arabic-Indic digits mapped to ASCII (not deleted)",
                       "zwnj": "mapped to space", "urls": "replaced by the token URL"},
        "rating_band": {
            "source_column": "rate (used only to derive the target; never a model input)",
            "thresholds": {"low": "0 <= rate < 2.5", "medium": "2.5 <= rate < 4.0", "high": "4.0 <= rate <= 5.0"},
            "ordinal_encoding": BAND_ENCODING,
            "invalid": "null, < 0 or > 5 -> rating_band left empty (1 row: id 38631814, rate 2500)",
        },
        "recommendation_classes": REC_CLASSES,
        "split_strategy": (
            "product-grouped; products shuffled with numpy.random.default_rng(SPLIT_SEED) and cut at 70% and 85% "
            "of the cumulative row count (train / validation / test)"),
        "masking": {
            "rule": "within each recommendation class of TRAIN, sort by stable_hash(id) ascending and mask "
                    "floor(MASK_RATE * class_count) rows",
            "hash": "int(sha256(f'{MASK_SEED}|{id}').hexdigest()[:16], 16)",
            "rounding": "floor (integer arithmetic n*20//100)",
            "per_class": mask_per_class,
            "total_masked": int(len(masked_ids)),
            "never_masked": ["validation", "test"],
        },
        "input": {"clean_rows": int(n_clean), "rows_with_text": int(n_with_text),
                  "rows_used_for_modeling": int(n_model),
                  "excluded_no_usable_text": int(n_clean - n_model),
                  "products": int(df["product_id"].n_unique())},
        "splits": per_split,
        "overall": overall,
        "representativeness": rep,
        "text_leakage_official_split": leak,
        "checks": checks,
        "files": files,
        "notes": [
            "Rows with a missing target stay in their product's split; each head uses only rows whose target is valid.",
            "natural_unlabeled.csv rows are also present in train_supervised.csv (they have a valid rating band).",
            "pseudo_reference_labels.csv may be used only to audit pseudo-label quality after selection.",
            "`rate` is kept only for the rate=0 ablation/sensitivity analysis, never as a feature.",
        ],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    for path in (OUT_DIR / "split_manifest.json", MANIFEST_COPY):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nSaved manifest: {OUT_DIR / 'split_manifest.json'} and {MANIFEST_COPY}")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
