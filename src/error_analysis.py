"""
Error analysis (no GPU, no model inference): uses SAVED predictions only.

Inputs:
  data/processed/test.csv                          (labels, body, rate)
  data/processed/preds_tfidf_test.csv              (TF-IDF predictions)
  artifacts/models/supervised/test_preds.npz       (xlm-roberta probabilities)

Outputs:
  reports/error_analysis_samples.csv   (sampled errors by category, with an empty `your_note` column)
  artifacts/results/error_analysis_summary.json  (confusion matrices, error rate by text feature)

Run from repo root:  python src/error_analysis.py
Sampling: seed 2026, 3 rows per category (fixed, reproducible).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
N_PER = 3
REC = ["recommended", "no_idea", "not_recommended"]
BANDS = ["low", "medium", "high"]

OUT_CSV = Path("reports/error_analysis_samples.csv")
OUT_JSON = Path("artifacts/results/error_analysis_summary.json")


def main() -> None:
    test = pd.read_csv("data/processed/test.csv", encoding="utf-8")
    test["body"] = test["body"].fillna("").astype(str)
    tf = pd.read_csv("data/processed/preds_tfidf_test.csv",
                     usecols=["id", "rec_pred", "band_pred"], encoding="utf-8")
    z = np.load("artifacts/models/supervised/test_preds.npz")
    xl = pd.DataFrame({
        "id": z["id"],
        "xl_rec": np.array(REC)[z["pr"].argmax(1)],
        "xl_rec_conf": z["pr"].max(1),
        "xl_band": np.array(BANDS)[z["pb"].argmax(1)],
        "xl_band_conf": z["pb"].max(1),
    })
    df = test.merge(xl, on="id", how="inner").merge(tf, on="id", how="left")
    assert len(df) == len(test) == 922953, (len(df), len(test))
    print("rows:", len(df))

    df["words"] = df["body"].str.split().str.len().fillna(0).astype(int)
    has_rec = df["recommendation_status"].isin(REC)
    has_band = df["rating_band"].isin(BANDS)
    rec_err = has_rec & (df["xl_rec"] != df["recommendation_status"])
    band_err = has_band & (df["xl_band"] != df["rating_band"])
    tf_rec_ok = has_rec & (df["rec_pred"] == df["recommendation_status"])
    tf_band_ok = has_band & (df["band_pred"] == df["rating_band"])
    xl_rec_ok = has_rec & ~rec_err
    xl_band_ok = has_band & ~band_err

    neg = df["body"].str.contains("نیست|نبود|نمی|ندارد|نداره|هیچ|بدون", regex=True)
    mixed = df["body"].str.contains("اما|ولی|ولي", regex=True)
    arabic_chars = df["body"].str.contains("\u064a|\u0643", regex=True)
    short = df["words"] <= 3
    rate = df["rate"]
    near_boundary = ((rate >= 2.3) & (rate <= 2.7)) | ((rate >= 3.8) & (rate <= 4.2))

    # (category, head, mask of rows to sample from)
    cats = [
        ("negation_words", "rec", rec_err & neg),
        ("short_text_le3_words", "rec", rec_err & short),
        ("mixed_opinion_but", "rec", rec_err & mixed),
        ("spelling_arabic_yeh_kaf", "rec", rec_err & arabic_chars),
        ("repeated_text_in_train", "rec", rec_err & df["text_in_train"].astype(bool)),
        ("no_idea_predicted_recommended", "rec",
         rec_err & (df["recommendation_status"] == "no_idea") & (df["xl_rec"] == "recommended")),
        ("band_boundary_cases", "band", band_err & near_boundary),
        ("rate_zero_noise", "band", band_err & (rate == 0)),
        ("tfidf_right_xlmr_wrong_rec", "rec", rec_err & tf_rec_ok),
        ("xlmr_right_tfidf_wrong_rec", "rec", xl_rec_ok & ~tf_rec_ok),
        ("tfidf_right_xlmr_wrong_band", "band", band_err & tf_band_ok),
        ("xlmr_right_tfidf_wrong_band", "band", xl_band_ok & ~tf_band_ok),
    ]

    rows = []
    counts = {}
    for name, head, mask in cats:
        sub = df[mask]
        counts[name] = int(len(sub))
        take = sub.sample(n=min(N_PER, len(sub)), random_state=SEED)
        for _, r in take.iterrows():
            if head == "rec":
                true, pred, conf, other = (r["recommendation_status"], r["xl_rec"],
                                           r["xl_rec_conf"], r["rec_pred"])
            else:
                true, pred, conf, other = (r["rating_band"], r["xl_band"],
                                           r["xl_band_conf"], r["band_pred"])
            rows.append({
                "category": name, "head": head, "id": r["id"],
                "body": r["body"][:400], "true": true, "xlmr_pred": pred,
                "xlmr_conf": round(float(conf), 3), "tfidf_pred": other,
                "rate": r["rate"], "words": r["words"], "your_note": "",
            })
    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("sampled errors:", len(out), "->", OUT_CSV)

    # error rate by text feature (recommendation head, xlm-roberta vs TF-IDF)
    feats = {"all": pd.Series(True, index=df.index), "negation_words": neg,
             "mixed_opinion_but": mixed, "short_text_le3_words": short,
             "arabic_yeh_kaf": arabic_chars}
    by_feat = {}
    for fname, fm in feats.items():
        m = has_rec & fm
        n = int(m.sum())
        by_feat[fname] = {
            "n_labeled": n,
            "xlmr_error_pct": round(float(rec_err[m].mean()) * 100, 2) if n else None,
            "tfidf_error_pct": round(float((~tf_rec_ok)[m].mean()) * 100, 2) if n else None,
        }
    print("error rate by feature (rec head):")
    for k, v in by_feat.items():
        print(" ", k, v)

    summary = {
        "seed": SEED, "n_per_category": N_PER,
        "pool_size_per_category": counts,
        "rec_error_rate_by_feature": by_feat,
        "confusion_rec_xlmr": pd.crosstab(
            df.loc[has_rec, "recommendation_status"], df.loc[has_rec, "xl_rec"]).to_dict(),
        "confusion_band_xlmr": pd.crosstab(
            df.loc[has_band, "rating_band"], df.loc[has_band, "xl_band"]).to_dict(),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print("saved:", OUT_JSON)
    print("pool sizes:", counts)


if __name__ == "__main__":
    main()
