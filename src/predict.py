"""
Offline prediction (CPU, no API): one review body -> normalize -> tokenize ->
TF-IDF + fine-tuned xlm-roberta -> recommendation and rating_band probabilities + a short explanation.

Run from repo root:
    python src/predict.py --text "کیفیت محصول خیلی خوب بود اما بسته‌بندی ضعیف بود"
    python src/predict.py --demo
Needs: artifacts/models/tfidf_*.joblib, artifacts/models/supervised/{best.pt,tokenizer/}.
The first run downloads xlm-roberta-base (about 1.1 GB) once; the frozen word embeddings
are not stored in best.pt and come from that pretrained checkpoint.
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from transformers import AutoModel, AutoTokenizer

from preprocess import normalize_text  # same normalizer as training

MODELS = Path("artifacts/models")
CKPT = MODELS / "supervised"
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 128
REC = ["recommended", "no_idea", "not_recommended"]
BANDS = ["low", "medium", "high"]

NEG_CUES = ("نیست", "نبود", "نمی", "ندارد", "نداره", "هیچ", "بدون")
MIX_CUES = ("اما", "ولی", "ولي", "هرچند", "اگرچه")

DEMO = [
    ("short", "عالی بود"),
    ("mixed (spec example)", "کیفیت محصول خیلی خوب بود اما بسته‌بندی ضعیف بود"),
    ("negative", "اصلا توصیه نمیکنم، جنسش خیلی بد بود و زود خراب شد"),
    ("ambiguous", "قبلا خریده بودم"),
    ("mixed + negation", "خوبه جنسش ، ولی عالی نیست"),
]


class TwoHead(nn.Module):
    """Same architecture as digikala_lib.TwoHead (copied: that module is Colab-specific)."""

    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_NAME, add_pooling_layer=False)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.rec = nn.Linear(h, 3)
        self.band = nn.Linear(h, 3)

    def forward(self, x, m):
        o = self.enc(input_ids=x, attention_mask=m).last_hidden_state
        mf = m.unsqueeze(-1).to(o.dtype)
        p = (o * mf).sum(1) / mf.sum(1).clamp(min=1)
        return self.rec(self.drop(p)), self.band(self.drop(p))


def load_transformer():
    tok = AutoTokenizer.from_pretrained(str(CKPT / "tokenizer"))
    model = TwoHead()
    ck = torch.load(CKPT / "best.pt", map_location="cpu", weights_only=False)
    res = model.load_state_dict(ck["model"], strict=False)
    bad = [k for k in res.missing_keys if "word_embeddings" not in k]
    if bad or res.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing={bad} unexpected={res.unexpected_keys}")
    model.eval()
    return tok, model


def load_tfidf():
    vecs = joblib.load(MODELS / "tfidf_vectorizers.joblib")
    if isinstance(vecs, dict):
        order = [k for k in ("word", "char") if k in vecs] or list(vecs)
        vl = [vecs[k] for k in order]
    else:
        vl = list(vecs)
    return vl, joblib.load(MODELS / "tfidf_rec_lr.joblib"), joblib.load(MODELS / "tfidf_band_lr.joblib")


def tfidf_predict(text, vl, rec_m, band_m):
    X = sparse.hstack([v.transform([text]) for v in vl]).tocsr()
    for m in (rec_m, band_m):
        if hasattr(m, "coef_") and m.coef_.shape[1] != X.shape[1]:
            raise RuntimeError(f"TF-IDF feature mismatch: X has {X.shape[1]}, model expects {m.coef_.shape[1]}. "
                               "Check the order of vectorizers in tfidf_vectorizers.joblib.")
    out = {}
    for name, m in (("rec", rec_m), ("band", band_m)):
        p = m.predict_proba(X)[0]
        names = REC if name == "rec" else BANDS
        out[name] = {names[int(c)]: float(v) for c, v in zip(m.classes_, p)}
    top_words = []
    try:  # words pushing the predicted recommendation class (word features only)
        ci = int(np.argmax(rec_m.predict_proba(X)[0]))
        contrib = X.multiply(rec_m.coef_[ci]).toarray()[0]
        n_word = len(vl[0].get_feature_names_out())
        names = vl[0].get_feature_names_out()
        idx = [i for i in np.argsort(-contrib) if i < n_word and contrib[i] > 0][:5]
        top_words = [str(names[i]) for i in idx]
    except Exception:
        pass
    return out, top_words


@torch.no_grad()
def transformer_predict(text, tok, model):
    n_full = len(tok(text, truncation=False)["input_ids"])
    enc = tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    a, b = model(enc["input_ids"], enc["attention_mask"])
    pr = torch.softmax(a.float(), -1)[0].numpy()
    pb = torch.softmax(b.float(), -1)[0].numpy()
    return ({"rec": dict(zip(REC, map(float, pr))), "band": dict(zip(BANDS, map(float, pb)))},
            n_full, n_full > MAX_LEN)


def fmt(d):
    return "  ".join(f"{k}={v:.3f}" for k, v in d.items())


def run_one(raw, tok, model, vl, rec_m, band_m):
    t0 = time.time()
    text = normalize_text(raw)
    tr, n_tok, trunc = transformer_predict(text, tok, model)
    tf, words = tfidf_predict(text, vl, rec_m, band_m)
    tr_rec = max(tr["rec"], key=tr["rec"].get)
    tr_band = max(tr["band"], key=tr["band"].get)
    tf_rec = max(tf["rec"], key=tf["rec"].get)
    tf_band = max(tf["band"], key=tf["band"].get)

    print("-" * 70)
    print("normalized:", text)
    print(f"tokens: {n_tok} (max_length {MAX_LEN}, truncated: {trunc})")
    print("[xlm-roberta] recommendation:", tr_rec, "|", fmt(tr["rec"]))
    print("[xlm-roberta] rating_band   :", tr_band, "|", fmt(tr["band"]))
    print("[TF-IDF]      recommendation:", tf_rec, "|", fmt(tf["rec"]))
    print("[TF-IDF]      rating_band   :", tf_band, "|", fmt(tf["band"]))
    if words:
        print("TF-IDF words supporting its recommendation:", ", ".join(words))

    notes = []
    if len(text.split()) <= 3:
        notes.append("very short text: little evidence")
    if any(c in text for c in NEG_CUES):
        notes.append("contains negation cue (hard case, see error analysis)")
    if any(c in text for c in MIX_CUES):
        notes.append("contains contrast cue 'but' (mixed opinion, models often predict no_idea/medium)")
    if max(tr["rec"].values()) < 0.6:
        notes.append("low recommendation confidence (< 0.60)")
    if tr_rec != tf_rec or tr_band != tf_band:
        notes.append("the two models disagree")
    print("explanation:", "; ".join(notes) if notes else "clear case, no risk cue found")
    print(f"latency: {time.time() - t0:.2f}s")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="review body")
    ap.add_argument("--demo", action="store_true", help="run 5 demo reviews")
    ap.add_argument("--out", help="write output to this UTF-8 file")
    args = ap.parse_args()
    if args.out:
      sys.stdout = open(args.out, "w", encoding="utf-8")
    if not args.text and not args.demo:
        ap.error("give --text or --demo")

    tok, model = load_transformer()
    vl, rec_m, band_m = load_tfidf()
    if args.demo:
        for label, t in DEMO:
            print("\n### case:", label)
            run_one(t, tok, model, vl, rec_m, band_m)
    if args.text:
        run_one(args.text, tok, model, vl, rec_m, band_m)


if __name__ == "__main__":
    main()
