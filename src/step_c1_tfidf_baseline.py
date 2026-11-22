"""
Part C / step C1: TF-IDF + Logistic Regression baselines for BOTH targets.

Protocol
  * Input is `body` only (already normalized by preprocess.py in step B1).
  * TF-IDF vocabulary, IDF weights and class weights are fitted on the TRAIN SUBSET ONLY
    (data/subset/train_supervised_sub.csv, the same rows the Transformer is trained on).
  * Recommendation head uses train rows with a valid recommendation label;
    rating_band head uses train rows with a valid band (rows with a missing target are
    used only by the head for which the target is valid).
  * Features: word 1-2-grams + character 2-5-grams (char_wb), because A6 showed heavy
    spelling noise (elongation, glued words). Set USE_CHAR = False for a faster word-only run.
  * Evaluation: validation (FULL validation.csv) and test (FULL test.csv), with extra subsets:
      - text_not_in_train (A5: duplicate-text leakage removed),
      - long_text (more than 3 words),
      - rating_band excluding rate == 0 (A3: rate=0 is suspicious).

Run from repo root:  python src/step_c1_tfidf_baseline.py
Outputs: artifacts/baselines/tfidf_metrics.json,
         artifacts/models/tfidf_{vectorizers,rec_lr,band_lr}.joblib (not in git),
         data/processed/preds_tfidf_{validation,test}.csv (not in git)
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from metrics_utils import evaluate, round_floats

SUB = Path("data/subset")
PROC = Path("data/processed")
OUT_JSON = Path("artifacts/baselines/tfidf_metrics.json")
MODEL_DIR = Path("artifacts/models")

REC = ["recommended", "no_idea", "not_recommended"]
BANDS = ["low", "medium", "high"]
USE_CHAR = True
C = 2.0
MAX_ITER = 300
SHORT_MAX_WORDS = 3


def read(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=20000)


def prep(d: pl.DataFrame) -> pl.DataFrame:
    exprs = [
        pl.col("body").fill_null("").alias("body"),
        pl.col("recommendation_status").replace_strict({c: i for i, c in enumerate(REC)}, default=None,
                                                       return_dtype=pl.Int64).alias("y_rec"),
        pl.col("rating_band").replace_strict({c: i for i, c in enumerate(BANDS)}, default=None,
                                             return_dtype=pl.Int64).alias("y_band"),
        (pl.col("body").fill_null("").str.count_matches(r"\S+") > SHORT_MAX_WORDS).alias("is_long"),
    ]
    if "text_in_train" in d.columns:
        exprs.append((pl.col("text_in_train").cast(pl.Utf8).str.to_lowercase() == "true").alias("text_in_train"))
    return d.with_columns(exprs)


def build_features(train_texts: list):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=400_000, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
    mats = [word.fit_transform(train_texts)]
    vecs = {"word": word}
    if USE_CHAR:
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=300_000,
                               sublinear_tf=True, dtype=np.float32)
        mats.append(char.fit_transform(train_texts))
        vecs["char"] = char
    return hstack(mats).tocsr(), vecs


def transform(vecs: dict, texts: list):
    mats = [vecs["word"].transform(texts)]
    if "char" in vecs:
        mats.append(vecs["char"].transform(texts))
    return hstack(mats).tocsr()


def main() -> None:
    t0 = time.time()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train = prep(read(SUB / "train_supervised_sub.csv"))
    val = prep(read(PROC / "validation.csv"))
    test = prep(read(PROC / "test.csv"))
    print(f"train_sub={train.height:,} | validation={val.height:,} | test={test.height:,}")

    print("Fitting TF-IDF on the train subset ...")
    X_train, vecs = build_features(train["body"].to_list())
    print(f"  features: {X_train.shape[1]:,} | fit time {time.time() - t0:.0f}s")

    models = {}
    for head, ycol, names in (("recommendation", "y_rec", REC), ("rating_band", "y_band", BANDS)):
        mask = train[ycol].is_not_null().to_numpy()
        y = train[ycol].to_numpy()[mask].astype(np.int64)
        print(f"Training LR for {head}: {int(mask.sum()):,} rows ...")
        t1 = time.time()
        clf = LogisticRegression(C=C, max_iter=MAX_ITER, class_weight="balanced")
        clf.fit(X_train[mask], y)
        models[head] = clf
        print(f"  done in {time.time() - t1:.0f}s")

    results = {"config": {"use_char_ngrams": USE_CHAR, "C": C, "max_iter": MAX_ITER, "class_weight": "balanced",
                          "word_ngrams": [1, 2], "char_ngrams": [2, 5] if USE_CHAR else None,
                          "n_features": int(X_train.shape[1]), "train_rows": int(train.height),
                          "fit_on": "data/subset/train_supervised_sub.csv"}}
    for split_name, d in (("validation", val), ("test", test)):
        X = transform(vecs, d["body"].to_list())
        pred = {}
        for head, names in (("recommendation", REC), ("rating_band", BANDS)):
            proba = models[head].predict_proba(X)
            pred[head] = (proba, proba.argmax(axis=1))
        res = {"recommendation": {}, "rating_band": {}}

        y = d["y_rec"].to_numpy()
        valid = d["y_rec"].is_not_null().to_numpy()
        subsets = {"all": valid,
                   "long_text": valid & d["is_long"].to_numpy()}
        if "text_in_train" in d.columns:
            subsets["text_not_in_train"] = valid & ~d["text_in_train"].to_numpy()
        for name, m in subsets.items():
            res["recommendation"][name] = evaluate(y[m].astype(np.int64), pred["recommendation"][1][m], REC)

        yb = d["y_band"].to_numpy()
        validb = d["y_band"].is_not_null().to_numpy()
        rate0 = (d["rate"] == 0).fill_null(False).to_numpy()
        subsets_b = {"all": validb, "excluding_rate0": validb & ~rate0,
                     "long_text": validb & d["is_long"].to_numpy()}
        if "text_in_train" in d.columns:
            subsets_b["text_not_in_train"] = validb & ~d["text_in_train"].to_numpy()
        for name, m in subsets_b.items():
            res["rating_band"][name] = evaluate(yb[m].astype(np.int64), pred["rating_band"][1][m], BANDS, ordinal=True)
        results[split_name] = res

        out = pl.DataFrame({
            "id": d["id"],
            "rec_true": d["recommendation_status"],
            "rec_pred": [REC[i] for i in pred["recommendation"][1]],
            **{f"rec_p_{c}": pred["recommendation"][0][:, i].astype(np.float32) for i, c in enumerate(REC)},
            "band_true": d["rating_band"],
            "band_pred": [BANDS[i] for i in pred["rating_band"][1]],
            **{f"band_p_{c}": pred["rating_band"][0][:, i].astype(np.float32) for i, c in enumerate(BANDS)},
        })
        out.write_csv(PROC / f"preds_tfidf_{split_name}.csv")

    results["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(round_floats(results), f, indent=2, ensure_ascii=False)
    joblib.dump(vecs, MODEL_DIR / "tfidf_vectorizers.joblib")
    joblib.dump(models["recommendation"], MODEL_DIR / "tfidf_rec_lr.joblib")
    joblib.dump(models["rating_band"], MODEL_DIR / "tfidf_band_lr.joblib")

    print("\n" + "=" * 70)
    print("TF-IDF + LR RESULTS (Macro-F1 / Balanced Acc / Accuracy)")
    print("=" * 70)
    for split_name in ("validation", "test"):
        for head in ("recommendation", "rating_band"):
            for sub, m in results[split_name][head].items():
                extra = f" | ordinal MAE {m['ordinal_mean_abs_error']:.3f}" if head == "rating_band" else ""
                print(f"{split_name:>10s} {head:>14s} {sub:>18s}: n={m['n']:>8,} "
                      f"macroF1={m['macro_f1']:.4f} balAcc={m['balanced_accuracy']:.4f} acc={m['accuracy']:.4f}{extra}")
    print(f"\nSaved: {OUT_JSON}\nElapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
