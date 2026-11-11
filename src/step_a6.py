"""
Step A6: Vocabulary-level analysis (bounded sample, seed=2026).

Sample: SAMPLE_N reviews drawn at random (seed 2026) from clean rows with real body text.
Text is normalized with eda_utils.normalize_text_minimal, punctuation replaced by spaces,
lowercased, split on whitespace. All counts are DOCUMENT frequencies (a word counts once
per review) so that one long or spammy review cannot dominate.

1) Frequent words and bigrams (overall).
2) Class-specific terms (smoothed one-vs-rest log-odds) for recommendation and rating_band.
3) Labeled vs naturally unlabeled (recommendation missing): distinctive terms, vocabulary overlap, OOV.
4) Negation/contrast words per class + "would a frequency stoplist remove them?" (stopword
   removal is reported as EVIDENCE, not applied as a rule).
5) Orthographic noise indicators (elongation, Latin letters, digits, emoji).

Run from repo root:  python src/step_a6.py
Outputs: artifacts/eda/step_a6_vocabulary.json, artifacts/eda/step_a6_vocab_tables.md,
         reports/figures/A6_vocabulary.png
Persian text prints badly in PowerShell: open the .md / .json in VS Code.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from eda_utils import normalize_text_minimal

CLEAN_PARQUET = Path("data/interim/digikala_clean.parquet")
OUT_JSON = Path("artifacts/eda/step_a6_vocabulary.json")
OUT_MD = Path("artifacts/eda/step_a6_vocab_tables.md")
OUT_FIG = Path("reports/figures/A6_vocabulary.png")

SEED = 2026
SAMPLE_N = 1_000_000
MIN_DF = 200          # minimum overall document frequency for class-specific terms
ALPHA = 0.5           # smoothing for log-odds
TOP_K = 15
REC = ["recommended", "no_idea", "not_recommended", "missing"]
BANDS = ["low", "medium", "high"]
NEG_WORDS = ["نه", "نیست", "نبود", "نداره", "ندارد", "نمی", "نشد", "بد", "ولی", "اما",
             "متاسفانه", "نخرید", "نخرین", "هیچ", "بدون"]

PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)
ELONG = re.compile(r"(.)\1{2,}")
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"\d")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
NEG_SET = set(NEG_WORDS)


def logodds_top(df_c: Counter, n_c: int, df_all: Counter, n_all: int, vocab: list, k: int = TOP_K) -> list:
    n_r = n_all - n_c
    rows = []
    for w in vocab:
        a = df_c.get(w, 0)
        b = df_all[w] - a
        lo = math.log((a + ALPHA) / (n_c - a + ALPHA)) - math.log((b + ALPHA) / (n_r - b + ALPHA))
        rows.append((lo, w, a, b))
    rows.sort(reverse=True)
    return [{"word": w, "log_odds": round(lo, 3),
             "pct_in_group": round(a / n_c * 100, 3), "pct_in_rest": round(b / n_r * 100, 3)}
            for lo, w, a, b in rows[:k]]


def md_table(title: str, rows: list, cols: list) -> str:
    out = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    rec = pl.col("recommendation_status").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    rate = pl.col("rate")
    has_text = (
        pl.col("body").is_not_null()
        & (pl.col("body").str.strip_chars() != "")
        & (pl.col("body").str.strip_chars().str.to_lowercase() != "nan")
    )
    df = pl.scan_parquet(CLEAN_PARQUET).select([
        pl.col("body"),
        pl.when(rec.is_in(["recommended", "no_idea", "not_recommended"]))
        .then(rec).otherwise(pl.lit("missing")).alias("rec_class"),
        pl.when(rate.is_null() | (rate < 0) | (rate > 5)).then(pl.lit("invalid"))
        .when(rate < 2.5).then(pl.lit("low"))
        .when(rate < 4.0).then(pl.lit("medium"))
        .otherwise(pl.lit("high")).alias("band"),
    ]).filter(has_text).collect(engine="streaming")
    population = df.height
    sample = df.sample(n=min(SAMPLE_N, population), seed=SEED)
    N = sample.height
    print(f"Population (rows with text): {population:,} | sample N={N:,} (seed={SEED})")

    texts = [normalize_text_minimal(b) for b in sample["body"].to_list()]
    rec_list = sample["rec_class"].to_list()
    band_list = sample["band"].to_list()

    df_rec = {c: Counter() for c in REC}
    n_rec = Counter()
    df_band = {b: Counter() for b in BANDS}
    n_band = Counter()
    tf_all = Counter()
    bi_df = Counter()
    noise = {c: Counter() for c in REC}

    for text, rc, bd in zip(texts, rec_list, band_list):
        toks = PUNCT.sub(" ", text).lower().split()
        s = set(toks)
        df_rec[rc].update(s)
        n_rec[rc] += 1
        if bd in df_band:
            df_band[bd].update(s)
            n_band[bd] += 1
        tf_all.update(toks)
        if len(toks) > 1:
            bi_df.update({f"{a} {b}" for a, b in zip(toks, toks[1:])})
        nz = noise[rc]
        if ELONG.search(text):
            nz["elongation"] += 1
        if LATIN.search(text):
            nz["latin_letters"] += 1
        if DIGIT.search(text):
            nz["digits"] += 1
        if EMOJI.search(text):
            nz["emoji"] += 1
        if s & NEG_SET:
            nz["any_negation_word"] += 1

    df_all = Counter()
    for c in REC:
        df_all.update(df_rec[c])
    df_lab = Counter()
    for c in REC[:3]:
        df_lab.update(df_rec[c])
    n_lab = sum(n_rec[c] for c in REC[:3])
    n_nat = n_rec["missing"]
    vocab = [w for w, c in df_all.items() if c >= MIN_DF]

    # ------------------------------------------------------------------ A6-1
    print("\n" + "=" * 70)
    print("A6-1: VOCABULARY SIZE AND FREQUENT WORDS/BIGRAMS")
    print("=" * 70)
    total_tokens = sum(tf_all.values())
    sorted_tf = np.array(sorted(tf_all.values(), reverse=True), dtype=np.float64)
    cum = np.cumsum(sorted_tf) / sorted_tf.sum()
    coverage = {f"top_{k}": round(float(cum[min(k, len(cum)) - 1]) * 100, 2) for k in (50, 100, 1000, 5000, 20000)}
    vocab_stats = {
        "sample_reviews": N, "total_tokens": int(total_tokens),
        "unique_words": len(tf_all), "words_df_ge_5": sum(1 for c in df_all.values() if c >= 5),
        "words_df_ge_MIN_DF": len(vocab), "hapax_words": sum(1 for c in tf_all.values() if c == 1),
        "coverage_of_token_occurrences_pct": coverage,
    }
    vocab_stats["hapax_share_of_vocab_pct"] = round(vocab_stats["hapax_words"] / vocab_stats["unique_words"] * 100, 2)
    print(json.dumps(vocab_stats))
    top_words = [{"rank": i + 1, "word": w, "df": c, "pct_reviews": round(c / N * 100, 3)}
                 for i, (w, c) in enumerate(df_all.most_common(30))]
    top_bigrams = [{"rank": i + 1, "bigram": w, "df": c, "pct_reviews": round(c / N * 100, 3)}
                   for i, (w, c) in enumerate(bi_df.most_common(30))]

    # ------------------------------------------------------------------ A6-2
    print("\n" + "=" * 70)
    print("A6-2: CLASS-SPECIFIC TERMS (smoothed one-vs-rest log-odds; tables in the .md file)")
    print("=" * 70)
    class_terms = {"recommendation": {}, "rating_band": {}}
    for c in REC[:3]:
        class_terms["recommendation"][c] = logodds_top(df_rec[c], n_rec[c], df_all, N, vocab)
    for b in BANDS:
        class_terms["rating_band"][b] = logodds_top(df_band[b], n_band[b], df_all, sum(n_band.values()) or N,
                                                    [w for w in vocab])
    print("done (counts: " + ", ".join(f"{c}={n_rec[c]:,}" for c in REC) + ")")
    print("bands: " + ", ".join(f"{b}={n_band[b]:,}" for b in BANDS))

    # ------------------------------------------------------------------ A6-3
    print("\n" + "=" * 70)
    print("A6-3: LABELED vs NATURALLY UNLABELED (recommendation missing)")
    print("=" * 70)
    nat_terms = logodds_top(df_rec["missing"], n_nat, df_all, N, vocab, k=20)
    lab_terms = logodds_top(Counter({w: df_lab[w] for w in vocab}), n_lab, df_all, N, vocab, k=20)
    top_lab = {w for w, _ in df_lab.most_common(5000)}
    top_nat = {w for w, _ in df_rec["missing"].most_common(5000)}
    jaccard = round(len(top_lab & top_nat) / len(top_lab | top_nat), 4)
    vocab_lab = {w for w, c in df_lab.items() if c >= 5}
    nat_total = sum(df_rec["missing"].values())
    nat_oov = sum(c for w, c in df_rec["missing"].items() if w not in vocab_lab)
    lab_total = sum(df_lab.values())
    vocab_nat = {w for w, c in df_rec["missing"].items() if c >= 5}
    lab_oov = sum(c for w, c in df_lab.items() if w not in vocab_nat)
    compare = {
        "labeled_reviews": n_lab, "natural_unlabeled_reviews": n_nat,
        "jaccard_top5000_words": jaccard,
        "oov_rate_natural_vs_labeled_vocab_pct": round(nat_oov / nat_total * 100, 3),
        "oov_rate_labeled_vs_natural_vocab_pct": round(lab_oov / lab_total * 100, 3),
        "note": "OOV is weighted by document frequency; vocab = words with df >= 5 in the other group",
    }
    print(json.dumps(compare))

    # ------------------------------------------------------------------ A6-4
    print("\n" + "=" * 70)
    print("A6-4: NEGATION/CONTRAST WORDS AND STOPLIST RISK")
    print("=" * 70)
    neg_table = []
    for w in NEG_WORDS:
        row = {"word": w, "pct_overall": round(df_all[w] / N * 100, 3)}
        for c in REC:
            row[f"pct_{c}"] = round(df_rec[c][w] / n_rec[c] * 100, 3) if n_rec[c] else None
        neg_table.append(row)
    top50 = [w for w, _ in df_all.most_common(50)]
    top100 = [w for w, _ in df_all.most_common(100)]
    stoplist_risk = {
        "top50_contains_negation_words": [w for w in top50 if w in NEG_SET],
        "top100_contains_negation_words": [w for w in top100 if w in NEG_SET],
        "top50_token_coverage_pct": coverage["top_50"],
        "note": "A frequency-based stoplist would delete these words; stopword removal must stay an experiment.",
    }
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in stoplist_risk.items()}))

    # ------------------------------------------------------------------ A6-5
    print("\n" + "=" * 70)
    print("A6-5: ORTHOGRAPHIC NOISE INDICATORS (% of reviews)")
    print("=" * 70)
    keys = ["elongation", "latin_letters", "digits", "emoji"]
    noise_pct = {}
    for c in REC:
        noise_pct[c] = {k: round(noise[c][k] / n_rec[c] * 100, 3) for k in keys}
    lab_noise = {k: round(sum(noise[c][k] for c in REC[:3]) / n_lab * 100, 3) for k in keys}
    nat_noise = {k: round(noise["missing"][k] / n_nat * 100, 3) for k in keys}
    overall_noise = {k: round(sum(noise[c][k] for c in REC) / N * 100, 3) for k in keys}
    for c in REC:
        print(f"  {c:>16s}: {noise_pct[c]}")
    print(f"  {'labeled':>16s}: {lab_noise}\n  {'natural_unlab':>16s}: {nat_noise}")
    neg_any = {c: round(noise[c]["any_negation_word"] / n_rec[c] * 100, 3) for c in REC}
    print("  any negation word (%):", neg_any)

    # ------------------------------------------------------------------ Figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)

    ax = axes[0, 0]
    dfs = np.sort(np.fromiter(df_all.values(), dtype=np.int64))[::-1]
    ax.loglog(np.arange(1, len(dfs) + 1), dfs, color="#2b5c8f")
    ax.axhline(MIN_DF, color="#d95f02", linestyle="--", label=f"df = {MIN_DF}")
    ax.set_title("A6-a: Zipf plot of word document-frequency", fontweight="bold")
    ax.set_xlabel("Word rank (log)")
    ax.set_ylabel("Reviews containing word (log)")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    ax = axes[0, 1]
    ks = np.arange(1, min(30000, len(cum)) + 1)
    ax.semilogx(ks, cum[: len(ks)] * 100, color="#388e3c")
    for k in (1000, 5000, 20000):
        if k <= len(cum):
            ax.plot(k, cum[k - 1] * 100, "o", color="#d32f2f")
            ax.annotate(f"top {k:,}: {cum[k - 1] * 100:.1f}%", (k, cum[k - 1] * 100),
                        textcoords="offset points", xytext=(8, -12), fontsize=9)
    ax.set_title("A6-b: Token coverage by top-k words", fontweight="bold")
    ax.set_xlabel("k (log)")
    ax.set_ylabel("% of token occurrences covered")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    ax = axes[1, 0]
    xs = np.arange(len(keys))
    ax.bar(xs - 0.2, [lab_noise[k] for k in keys], 0.4, label="labeled", color="#1976d2")
    ax.bar(xs + 0.2, [nat_noise[k] for k in keys], 0.4, label="natural unlabeled", color="#f9a825")
    ax.set_xticks(xs)
    ax.set_xticklabels(keys)
    ax.set_title("A6-c: Orthographic noise indicators (% of reviews)", fontweight="bold")
    ax.set_ylabel("% of reviews")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    ax = axes[1, 1]
    vals = [neg_any[c] for c in REC]
    ax.bar(REC, vals, color=["#388e3c", "#9e9e9e", "#d32f2f", "#1976d2"], edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("A6-d: Reviews containing a negation/contrast word", fontweight="bold")
    ax.set_ylabel("% of reviews in class")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------ Save
    result = {
        "sample": {"population_rows_with_text": population, "n": N, "seed": SEED,
                   "rule": "random sample of clean rows with real body text", "min_df_class_terms": MIN_DF},
        "vocabulary": vocab_stats,
        "top_words": top_words, "top_bigrams": top_bigrams,
        "class_terms": class_terms,
        "labeled_vs_natural": {**compare, "terms_more_in_natural_unlabeled": nat_terms,
                               "terms_more_in_labeled": lab_terms},
        "negation_words": neg_table, "stoplist_risk": stoplist_risk,
        "noise_indicators_pct": {"by_recommendation_class": noise_pct, "labeled": lab_noise,
                                 "natural_unlabeled": nat_noise, "overall": overall_noise,
                                 "any_negation_word_by_class": neg_any},
        "figure_path": str(OUT_FIG),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    md = [f"# A6 vocabulary tables (sample N={N:,}, seed {SEED}, document frequency)\n",
          md_table("Top 30 words", top_words, ["rank", "word", "df", "pct_reviews"]),
          md_table("Top 30 bigrams", top_bigrams, ["rank", "bigram", "df", "pct_reviews"])]
    for c, rows in class_terms["recommendation"].items():
        md.append(md_table(f"Recommendation: terms distinctive of {c}", rows,
                           ["word", "log_odds", "pct_in_group", "pct_in_rest"]))
    for b, rows in class_terms["rating_band"].items():
        md.append(md_table(f"Rating band: terms distinctive of {b}", rows,
                           ["word", "log_odds", "pct_in_group", "pct_in_rest"]))
    md.append(md_table("Terms more frequent in NATURALLY UNLABELED reviews", nat_terms,
                       ["word", "log_odds", "pct_in_group", "pct_in_rest"]))
    md.append(md_table("Terms more frequent in LABELED reviews", lab_terms,
                       ["word", "log_odds", "pct_in_group", "pct_in_rest"]))
    md.append(md_table("Negation / contrast words (% of reviews)", neg_table,
                       ["word", "pct_overall"] + [f"pct_{c}" for c in REC]))
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_MD}\nSaved: {OUT_FIG}")


if __name__ == "__main__":
    main()
