import os, sys, shutil, zipfile, time, json, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

DRIVE = "/content/drive/MyDrive/digikala"
WORK = "/content/work"
SUB = WORK + "/subset"
CACHE = DRIVE + "/cache"
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 128
REC = ["recommended", "no_idea", "not_recommended"]
BANDS = ["low", "medium", "high"]
REC_ID = {c: i for i, c in enumerate(REC)}
BAND_ID = {c: i for i, c in enumerate(BANDS)}
DEV = "cuda"


def setup_data():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    for f in ("preprocess.py", "metrics_utils.py", "digikala_subset.zip"):
        if not os.path.exists(WORK + "/" + f):
            shutil.copy(DRIVE + "/" + f, WORK)
    if not os.path.isdir(SUB):
        with zipfile.ZipFile(WORK + "/digikala_subset.zip") as z:
            z.extractall(SUB)
    if WORK not in sys.path:
        sys.path.insert(0, WORK)


setup_data()
from metrics_utils import evaluate

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
PAD = tok.pad_token_id

CFG = dict(
    name="exp1_full", smoke=False,
    label_smoothing=0.1, mask_rate0_band=True,
    ordinal_soft=0.2, class_weights=True,
    lr_enc=2e-5, lr_head=5e-4, wd=0.01,
    batch=64, epochs=3, warmup=0.06,
    freeze_word_emb=True, ckpt_every=1000,
    patience=2, seed=2026,
)


def load_csv(name):
    d = pd.read_csv(SUB + "/" + name)
    d["body"] = d["body"].fillna("").astype(str)
    if "recommendation_status" in d.columns:
        yr = d["recommendation_status"].map(REC_ID)
        yb = d["rating_band"].map(BAND_ID)
        d["y_rec"] = yr.fillna(-100).astype(np.int64)
        d["y_band"] = yb.fillna(-100).astype(np.int64)
    return d


def encode_and_cache(d, tag):
    path = CACHE + "/" + tag + ".npz"
    if os.path.exists(path):
        print(tag, "cache exists, skipping")
        return
    t0 = time.time()
    out = tok(d["body"].tolist(), truncation=True, max_length=MAX_LEN)
    ids = out["input_ids"]
    lens = np.array([len(x) for x in ids], dtype=np.int32)
    flat = np.fromiter((t for x in ids for t in x),
                       dtype=np.int32, count=int(lens.sum()))
    extra = {}
    if "y_rec" in d.columns:
        extra["y_rec"] = d["y_rec"].to_numpy()
        extra["y_band"] = d["y_band"].to_numpy()
    if "text_in_train" in d.columns:
        extra["text_in_train"] = d["text_in_train"].astype(bool).to_numpy()
    if "rate" in d.columns:
        extra["rate_is_zero"] = (d["rate"] == 0).fillna(False).to_numpy()
    tmp = path + ".tmp.npz"
    np.savez(tmp, flat=flat, lens=lens, id=d["id"].to_numpy(), **extra)
    os.replace(tmp, path)
    print(tag, len(d), "rows | mean len", round(float(lens.mean()), 1),
          "|", round(time.time() - t0), "s")


def prepare_cache():
    encode_and_cache(load_csv("train_supervised_sub.csv"), "train_sup")
    encode_and_cache(load_csv("validation_sub.csv"), "val_sub")


def load_npz(tag):
    z = np.load(CACHE + "/" + tag + ".npz")
    lens = z["lens"].astype(np.int64)
    off = np.concatenate([[0], np.cumsum(lens)])
    D = dict(flat=z["flat"], lens=lens, off=off, id=z["id"])
    for k in ("y_rec", "y_band", "rate_is_zero", "text_in_train"):
        if k in z.files:
            D[k] = z[k]
    return D


def head_rows(D, n):
    out = dict(D)
    keys = ("lens", "y_rec", "y_band", "id", "rate_is_zero", "text_in_train")
    for k in keys:
        if k in D:
            out[k] = D[k][:n]
    out["off"] = D["off"][:n + 1]
    return out


def make_batches(D, bs, seed, shuffle):
    n = len(D["lens"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    batches = []
    for s in range(0, n, bs * 50):
        c = idx[s:s + bs * 50]
        c = c[np.argsort(D["lens"][c], kind="stable")]
        batches += [c[i:i + bs] for i in range(0, len(c), bs)]
    if shuffle:
        order = rng.permutation(len(batches))
        batches = [batches[i] for i in order]
    return batches


def collate(D, b):
    L = int(D["lens"][b].max())
    x = np.full((len(b), L), PAD, dtype=np.int64)
    for j, i in enumerate(b):
        a, z = D["off"][i], D["off"][i + 1]
        x[j, :D["lens"][i]] = D["flat"][a:z]
    x = torch.from_numpy(x)
    return x, (x != PAD)


class TwoHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(
            MODEL_NAME,
            add_pooling_layer=False,
        )
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.rec = nn.Linear(h, 3)
        self.band = nn.Linear(h, 3)

    def forward(self, x, m):
        o = self.enc(input_ids=x, attention_mask=m).last_hidden_state
        mf = m.unsqueeze(-1).to(o.dtype)
        p = (o * mf).sum(1) / mf.sum(1).clamp(min=1)
        p = self.drop(p)
        return self.rec(p), self.band(p)


def soft_ce(logits, y, valid, M, w):
    t = M[y]
    lp = F.log_softmax(logits.float(), -1)
    wt = w[y] * valid.float()
    loss = -(t * lp).sum(-1) * wt
    return loss.sum() / wt.sum().clamp(min=1e-6)


def build_targets(cfg):
    ls = cfg["label_smoothing"]
    e = cfg["ordinal_soft"]
    I = torch.eye(3)
    ordm = torch.tensor([[1 - e, e, 0],
                         [e / 2, 1 - e, e / 2],
                         [0, e, 1 - e]])
    M_rec = ((1 - ls) * I + ls / 3).to(DEV)
    M_band = ((1 - ls) * ordm + ls / 3).to(DEV)
    return M_rec, M_band


def class_w(y, enabled):
    if not enabled:
        return torch.ones(3, device=DEV)
    c = np.bincount(y[y >= 0], minlength=3).astype(np.float64)
    w = c.sum() / (3 * c)
    return torch.tensor(w, dtype=torch.float32, device=DEV)


@torch.no_grad()
def predict(model, D, bs=512):
    model.eval()
    n = len(D["lens"])
    pr = np.zeros((n, 3), np.float32)
    pb = np.zeros((n, 3), np.float32)
    for b in make_batches(D, bs, 0, False):
        x, m = collate(D, b)
        with torch.autocast("cuda", dtype=torch.float16):
            a, c = model(x.to(DEV), m.to(DEV))
        pr[b] = F.softmax(a.float(), -1).cpu().numpy()
        pb[b] = F.softmax(c.float(), -1).cpu().numpy()
    model.train()
    return pr, pb


def eval_val(model, V):
    pr, pb = predict(model, V)
    mr = V["y_rec"] >= 0
    mb = V["y_band"] >= 0
    r = evaluate(V["y_rec"][mr], pr[mr].argmax(1), REC)
    b = evaluate(V["y_band"][mb], pb[mb].argmax(1), BANDS, ordinal=True)
    return r, b


def save_atomic(obj, exp_dir, name):
    tmp = "/content/" + name + ".tmp"
    torch.save(obj, tmp)
    dst = exp_dir + "/" + name
    shutil.copy(tmp, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    os.remove(tmp)


def trainable_state(model):
    return {n: p.detach().cpu()
            for n, p in model.named_parameters() if p.requires_grad}


def load_best(name="exp1_full"):
    model = TwoHead().to(DEV)
    model.enc.embeddings.word_embeddings.weight.requires_grad = False
    path = DRIVE + "/runs/" + name + "/best.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    print("loaded best.pt | val record:", ck["val"])
    return model


def train(cfg):
    exp = DRIVE + "/runs/" + cfg["name"]
    os.makedirs(exp, exist_ok=True)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    T, V = load_npz("train_sup"), load_npz("val_sub")
    if cfg["smoke"]:
        V = head_rows(V, 4000)
    y_rec = T["y_rec"]
    y_band = T["y_band"].copy()
    if cfg["mask_rate0_band"]:
        y_band[T["rate_is_zero"]] = -100
    M_rec, M_band = build_targets(cfg)
    w_rec = class_w(y_rec, cfg["class_weights"])
    w_band = class_w(y_band, cfg["class_weights"])
    print("class weights", w_rec.tolist(), w_band.tolist())

    model = TwoHead().to(DEV)
    if cfg["freeze_word_emb"]:
        model.enc.embeddings.word_embeddings.weight.requires_grad = False
    params = list(model.named_parameters())
    tot = sum(p.numel() for _, p in params)
    trn = sum(p.numel() for _, p in params if p.requires_grad)
    print("params total", tot, "| trainable", trn)

    enc_p = [p for n, p in params if p.requires_grad and n.startswith("enc.")]
    head_p = [p for n, p in params
              if p.requires_grad and not n.startswith("enc.")]
    opt = torch.optim.AdamW(
        [{"params": enc_p, "lr": cfg["lr_enc"]},
         {"params": head_p, "lr": cfg["lr_head"]}],
        weight_decay=cfg["wd"])
    spe = len(make_batches(T, cfg["batch"], 0, True))
    if cfg["smoke"]:
        spe = 150
    total = spe * cfg["epochs"]
    wu = int(total * cfg["warmup"])

    def lr_lambda(s):
        if s < wu:
            return (s + 1) / wu
        return max(0.0, (total - s) / max(1, total - wu))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    st = dict(step=0, epoch=0, bi=0, best=-1.0, bad=0, hist=[])
    last = exp + "/last.pt"
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        st = ck["st"]
        print("RESUMED from", {k: st[k] for k in ("step", "epoch", "bi", "best")})

    def save_last():
        obj = dict(model=trainable_state(model), opt=opt.state_dict(),
                   sched=sched.state_dict(), scaler=scaler.state_dict(),
                   st=st, cfg=cfg)
        save_atomic(obj, exp, "last.pt")

    model.train()
    t0 = time.time()
    run = 0.0
    nrun = 0
    while st["epoch"] < cfg["epochs"] and st["bad"] < cfg["patience"]:
        seed = cfg["seed"] + st["epoch"]
        batches = make_batches(T, cfg["batch"], seed, True)[:spe]
        for bi in range(st["bi"], len(batches)):
            b = batches[bi]
            x, m = collate(T, b)
            yr = torch.from_numpy(y_rec[b]).to(DEV)
            yb = torch.from_numpy(y_band[b]).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                a, c = model(x.to(DEV), m.to(DEV))
            l1 = soft_ce(a, yr.clamp(min=0), yr >= 0, M_rec, w_rec)
            l2 = soft_ce(c, yb.clamp(min=0), yb >= 0, M_band, w_band)
            loss = l1 + l2
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(enc_p + head_p, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            st["step"] += 1
            st["bi"] = bi + 1
            run += loss.item()
            nrun += 1
            if st["step"] % 200 == 0:
                speed = nrun / (time.time() - t0)
                print("ep", st["epoch"], "step", st["step"],
                      "loss", round(run / nrun, 4),
                      "|", round(speed, 1), "it/s")
                run = 0.0
                nrun = 0
                t0 = time.time()
            if st["step"] % cfg["ckpt_every"] == 0:
                save_last()
        r, b_ = eval_val(model, V)
        score = (r["macro_f1"] + b_["macro_f1"]) / 2
        rec = dict(epoch=st["epoch"], step=st["step"],
                   rec_f1=r["macro_f1"], band_f1=b_["macro_f1"],
                   band_mae=b_["ordinal_mean_abs_error"], score=score)
        st["hist"].append(rec)
        print("VAL", rec)
        if score > st["best"]:
            st["best"] = score
            st["bad"] = 0
            obj = dict(model=trainable_state(model), cfg=cfg, val=rec)
            save_atomic(obj, exp, "best.pt")
        else:
            st["bad"] += 1
        st["epoch"] += 1
        st["bi"] = 0
        save_last()
        with open(exp + "/history.json", "w") as f:
            json.dump(st["hist"], f, indent=1)
    print("DONE. best val score:", st["best"])
    return model


TFIDF_TEST_F1 = {
    "rec": {"all": 0.6601, "long_text": 0.6506,
            "text_not_in_train": 0.6440},
    "band": {"all": 0.5427, "excluding_rate0": 0.5648,
             "long_text": 0.5338, "text_not_in_train": 0.5312},
}


def eval_test(model, name="exp1_full"):
    exp = DRIVE + "/runs/" + name
    test = load_csv("test.csv")
    encode_and_cache(test, "test")
    Te = load_npz("test")
    assert len(Te["lens"]) == len(test) == 922953
    is_long = (test["body"].str.split().str.len() > 3).to_numpy()

    pr, pb = predict(model, Te)
    np.savez(exp + "/test_preds.npz", id=Te["id"], pr=pr, pb=pb)
    tok.save_pretrained(exp + "/tokenizer")

    vr = Te["y_rec"] >= 0
    vb = Te["y_band"] >= 0
    nit = ~Te["text_in_train"]
    r0 = Te["rate_is_zero"]
    subs_r = {"all": vr, "long_text": vr & is_long,
              "text_not_in_train": vr & nit}
    subs_b = {"all": vb, "excluding_rate0": vb & ~r0,
              "long_text": vb & is_long, "text_not_in_train": vb & nit}

    res = {"recommendation": {}, "rating_band": {}}
    jobs = (("recommendation", Te["y_rec"], pr, REC, subs_r, "rec"),
            ("rating_band", Te["y_band"], pb, BANDS, subs_b, "band"))
    print("head | subset | n | F1 xlmr | F1 tfidf | balAcc | acc | MAE")
    for head, y, p, names, subs, key in jobs:
        for s, m in subs.items():
            ordinal = head == "rating_band"
            r = evaluate(y[m], p[m].argmax(1), names, ordinal=ordinal)
            res[head][s] = r
            mae = round(r["ordinal_mean_abs_error"], 3) if ordinal else "-"
            print(head, "|", s, "|", r["n"], "|",
                  round(r["macro_f1"], 4), "|",
                  TFIDF_TEST_F1[key][s], "|",
                  round(r["balanced_accuracy"], 4), "|",
                  round(r["accuracy"], 4), "|", mae)
    for head in res:
        print("\nper-class (all):", head)
        for k, v in res[head]["all"]["per_class"].items():
            print(" ", k, "P", round(v["precision"], 3),
                  "R", round(v["recall"], 3),
                  "F1", round(v["f1"], 3), "support", v["support"])
    with open(exp + "/test_metrics.json", "w") as f:
        json.dump(res, f, indent=1)
    return res
