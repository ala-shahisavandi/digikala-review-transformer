"""
Step A4: Text-length analysis of `body` (clean dataset).

1) Character and word length on the FULL clean dataset: mean, median, percentiles,
   very short / very long reviews.
2) Length by recommendation class and by rating_band (full data).
3) xlm-roberta-base token length on a fixed random sample (seed=2026, N=200,000):
   percentiles and share of reviews truncated at max_length 64/128/256/512.
4) Data-driven max_length suggestion.

Run from repo root:  python src/step_a4.py
Outputs: artifacts/eda/step_a4_text_length.json, reports/figures/A4_text_length.png
"""
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from eda_utils import normalize_text_minimal

CLEAN_PARQUET = Path("data/interim/digikala_clean.parquet")
OUT_JSON = Path("artifacts/eda/step_a4_text_length.json")
OUT_FIG = Path("reports/figures/A4_text_length.png")

TOKENIZER_NAME = "xlm-roberta-base"
SAMPLE_SEED = 2026
SAMPLE_N = 200_000
MAX_LENGTHS = [64, 128, 256, 512]
QS = [0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
REC_ORDER = ["recommended", "no_idea", "not_recommended"]
BAND_ORDER = ["low", "medium", "high"]


def q_dict(s: pl.Series) -> dict:
    return {f"p{round(q*100, 1):g}": float(s.quantile(q)) for q in QS}


def len_summary(s: pl.Series) -> dict:
    return {
        "n": len(s), "min": int(s.min()), "max": int(s.max()),
        "mean": round(float(s.mean()), 2), "median": float(s.median()),
        "std": round(float(s.std()), 2), **q_dict(s),
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    rec = pl.col("recommendation_status").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    rate = pl.col("rate")
    df = pl.scan_parquet(CLEAN_PARQUET).select([
        pl.col("id"),
        pl.col("body"),
        pl.when(rec.is_in(["recommended", "no_idea", "not_recommended"]))
        .then(rec).otherwise(pl.lit("missing")).alias("rec_class"),
        pl.when(rate.is_null() | (rate < 0) | (rate > 5)).then(pl.lit("invalid"))
        .when(rate < 2.5).then(pl.lit("low"))
        .when(rate < 4.0).then(pl.lit("medium"))
        .otherwise(pl.lit("high")).alias("rating_band"),
    ]).collect(engine="streaming")

    # Rows with a real body (literal 'nan' bodies are treated as no text)
    has_text = (
        pl.col("body").is_not_null()
        & (pl.col("body").str.strip_chars() != "")
        & (pl.col("body").str.strip_chars().str.to_lowercase() != "nan")
    )
    n_all = len(df)
    df = df.filter(has_text)
    n_text = len(df)
    print(f"Rows: {n_all:,} | with real body text: {n_text:,} | no text: {n_all - n_text:,}")

    # Lengths on lightly normalized text (ZWNJ -> space etc.), same normalizer as the pipeline
    bodies = df["body"].to_list()
    norm = [normalize_text_minimal(b) for b in bodies]
    char_len = pl.Series("char_len", [len(t) for t in norm])
    word_len = pl.Series("word_len", [len(t.split()) for t in norm])
    df = df.with_columns([char_len, word_len])

    print("\n" + "=" * 70)
    print("A4-1: CHARACTER AND WORD LENGTH (full clean data)")
    print("=" * 70)
    char_stats = len_summary(df["char_len"])
    word_stats = len_summary(df["word_len"])
    print("chars:", json.dumps(char_stats))
    print("words:", json.dumps(word_stats))

    thresholds = {
        "chars_le_3": int((df["char_len"] <= 3).sum()),
        "chars_le_10": int((df["char_len"] <= 10).sum()),
        "words_eq_1": int((df["word_len"] == 1).sum()),
        "words_le_3": int((df["word_len"] <= 3).sum()),
        "chars_gt_300": int((df["char_len"] > 300).sum()),
        "chars_gt_1000": int((df["char_len"] > 1000).sum()),
        "words_gt_100": int((df["word_len"] > 100).sum()),
        "words_gt_200": int((df["word_len"] > 200).sum()),
    }
    thresholds_pct = {k: round(v / n_text * 100, 3) for k, v in thresholds.items()}
    print("\nShort/long counts:")
    for k, v in thresholds.items():
        print(f"  {k:>14s}: {v:>10,}  ({thresholds_pct[k]:.3f}%)")

    print("\n" + "=" * 70)
    print("A4-2: LENGTH BY CLASS (full clean data, words)")
    print("=" * 70)

    def by_class(col: str, order: list) -> list:
        out = []
        for k in order:
            sub = df.filter(pl.col(col) == k)
            if len(sub) == 0:
                continue
            out.append({
                "class": k, "n": len(sub),
                "words_mean": round(float(sub["word_len"].mean()), 2),
                "words_median": float(sub["word_len"].median()),
                "words_p90": float(sub["word_len"].quantile(0.90)),
                "words_p99": float(sub["word_len"].quantile(0.99)),
                "chars_median": float(sub["char_len"].median()),
                "chars_p99": float(sub["char_len"].quantile(0.99)),
            })
        return out

    by_rec = by_class("rec_class", REC_ORDER + ["missing"])
    by_band = by_class("rating_band", BAND_ORDER)
    for title, rows in [("recommendation", by_rec), ("rating_band", by_band)]:
        print(f"\n{title}")
        print(f"{'class':>16s}{'n':>11s}{'mean':>8s}{'median':>8s}{'p90':>7s}{'p99':>7s}")
        for r in rows:
            print(f"{r['class']:>16s}{r['n']:>11,}{r['words_mean']:>8.1f}"
                  f"{r['words_median']:>8.0f}{r['words_p90']:>7.0f}{r['words_p99']:>7.0f}")

    print("\n" + "=" * 70)
    print(f"A4-3: {TOKENIZER_NAME} TOKEN LENGTH (sample seed={SAMPLE_SEED}, N={SAMPLE_N:,})")
    print("=" * 70)
    token_result = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        sample = df.sample(n=min(SAMPLE_N, len(df)), seed=SAMPLE_SEED)
        texts = [normalize_text_minimal(b) for b in sample["body"].to_list()]
        enc = tok(texts, add_special_tokens=True, truncation=False)["input_ids"]
        tok_len = pl.Series("tok_len", [len(x) for x in enc])
        sample = sample.with_columns(tok_len)
        tok_stats = len_summary(tok_len)
        trunc = {
            str(m): {
                "truncated_count": int((tok_len > m).sum()),
                "truncated_pct": round(float((tok_len > m).mean()) * 100, 3),
            } for m in MAX_LENGTHS
        }
        ratio = round(float(tok_len.sum()) / float(sample["word_len"].sum()), 3)
        print("tokens:", json.dumps(tok_stats))
        print(f"tokens per word (sample): {ratio}")
        for m in MAX_LENGTHS:
            print(f"  max_length={m:>3d}: truncated {trunc[str(m)]['truncated_pct']:.3f}% "
                  f"({trunc[str(m)]['truncated_count']:,})")

        # truncation by class at each max_length (does truncation hit one class more?)
        trunc_by_class = {}
        for m in (128, 256):
            rows = []
            for col, order in [("rec_class", REC_ORDER), ("rating_band", BAND_ORDER)]:
                for k in order:
                    sub = sample.filter(pl.col(col) == k)
                    if len(sub):
                        rows.append({"target": col, "class": k, "n": len(sub),
                                     "truncated_pct": round(float((sub["tok_len"] > m).mean()) * 100, 3)})
            trunc_by_class[str(m)] = rows
        token_result = {
            "tokenizer": TOKENIZER_NAME, "seed": SAMPLE_SEED, "sample_n": len(sample),
            "token_stats": tok_stats, "tokens_per_word": ratio,
            "truncation": trunc, "truncation_by_class": trunc_by_class,
        }
    except Exception as e:  # tokenizer unavailable (offline / no VPN)
        print(f"Tokenizer step skipped: {type(e).__name__}: {e}")
        sample = None

    # Suggestion: smallest max_length in the list that truncates <= 2% of reviews
    suggestion = None
    if token_result:
        for m in MAX_LENGTHS:
            if token_result["truncation"][str(m)]["truncated_pct"] <= 2.0:
                suggestion = m
                break
        print(f"\nSuggested max_length (smallest candidate truncating <= 2%): {suggestion}")

    # ---------------- Figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)

    ax = axes[0, 0]
    w = df["word_len"].to_numpy()
    ax.hist(np.clip(w, 0, 150), bins=np.arange(0, 152, 2), color="#2b5c8f", edgecolor="white")
    ax.set_yscale("log")
    ax.axvline(word_stats["median"], color="#d95f02", linestyle="--", label=f"median = {word_stats['median']:.0f}")
    ax.axvline(word_stats["p99"], color="#d32f2f", linestyle="-.", label=f"p99 = {word_stats['p99']:.0f}")
    ax.set_title("A4-a: Words per review (full data, clipped at 150)", fontweight="bold")
    ax.set_xlabel("Words")
    ax.set_ylabel("Reviews (log)")
    ax.legend()

    ax = axes[0, 1]
    if token_result:
        t = sample["tok_len"].to_numpy()
        ax.hist(np.clip(t, 0, 300), bins=np.arange(0, 302, 4), color="#388e3c", edgecolor="white")
        ax.set_yscale("log")
        for m, c in [(64, "#9e9e9e"), (128, "#d95f02"), (256, "#d32f2f")]:
            ax.axvline(m, color=c, linestyle="--",
                       label=f"max_length {m}: {token_result['truncation'][str(m)]['truncated_pct']:.2f}% cut")
        ax.legend()
        ax.set_title(f"A4-b: {TOKENIZER_NAME} tokens (sample N={len(sample):,}, seed {SAMPLE_SEED})",
                     fontweight="bold")
        ax.set_xlabel("Tokens (clipped at 300)")
        ax.set_ylabel("Reviews (log)")
    else:
        ax.text(0.5, 0.5, "tokenizer unavailable", ha="center")
        ax.axis("off")

    ax = axes[1, 0]
    data, labels = [], []
    for r in by_rec[:3]:
        data.append(df.filter(pl.col("rec_class") == r["class"])["word_len"].to_numpy())
        labels.append(r["class"])
    ax.boxplot([np.clip(d, 0, 100) for d in data], tick_labels=labels, showfliers=False)
    ax.set_title("A4-c: Words by recommendation class (full data)", fontweight="bold")
    ax.set_ylabel("Words")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    ax = axes[1, 1]
    data, labels = [], []
    for r in by_band:
        data.append(df.filter(pl.col("rating_band") == r["class"])["word_len"].to_numpy())
        labels.append(r["class"])
    ax.boxplot([np.clip(d, 0, 100) for d in data], tick_labels=labels, showfliers=False)
    ax.set_title("A4-d: Words by rating_band (full data)", fontweight="bold")
    ax.set_ylabel("Words")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)

    result = {
        "rows_total": n_all, "rows_with_text": n_text, "rows_without_text": n_all - n_text,
        "normalizer": "eda_utils.normalize_text_minimal",
        "char_length": char_stats, "word_length": word_stats,
        "short_long_counts": thresholds, "short_long_pct": thresholds_pct,
        "by_recommendation": by_rec, "by_rating_band": by_band,
        "tokens": token_result, "suggested_max_length": suggestion,
        "figure_path": str(OUT_FIG),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_FIG}")


if __name__ == "__main__":
    main()
