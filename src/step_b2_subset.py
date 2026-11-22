"""
Part B / step B2: deterministic modeling subset (for Kaggle and for the baselines).

The full split (step_b1_split.py) is untouched. Because GPU time is limited, Transformer
training uses a documented, deterministic subset of TRAIN rows; validation/test evaluation
is done on a fixed validation subset (model selection) and the FULL test set (final numbers).

Sampling rule (same for every Karvand):
  * a train row is kept when  int(sha256(f"{SUBSET_SEED}|{id}")[:8], 16) % 10000 < SUBSET_PCT * 100
    (applied to train_supervised, train_pseudo_unlabeled and pseudo_reference_labels with the
     same id rule, so the 80/20 labeled/masked proportion of the masking protocol is preserved;
     the subset hash uses a DIFFERENT seed than the masking hash, so the two are independent);
  * validation_sub = the VAL_N validation rows with the smallest sha256(f"{SUBSET_SEED}-val|{id}");
  * test = the complete test.csv.

Run from repo root:  python src/step_b2_subset.py
Output folder: data/subset/   (upload this folder to Kaggle as one Dataset)
"""
import hashlib
import json
import shutil
from pathlib import Path

import polars as pl

PROC = Path("data/processed")
OUT = Path("data/subset")
MANIFEST_COPY = Path("artifacts/split/subset_manifest.json")

SUBSET_SEED = "digikala-subset-v1"
SUBSET_PCT = 15          # percent of TRAIN rows kept
VAL_N = 150_000          # validation rows used for model selection
REC = ["recommended", "no_idea", "not_recommended"]
BANDS = ["low", "medium", "high"]


def bucket(row_id) -> int:
    return int(hashlib.sha256(f"{SUBSET_SEED}|{row_id}".encode("utf-8")).hexdigest()[:8], 16) % 10000


def read(name: str) -> pl.DataFrame:
    return pl.read_csv(PROC / name, infer_schema_length=20000)


def counts(d: pl.DataFrame, col: str) -> dict:
    return {("missing" if r[0] is None else r[0]): int(r[1]) for r in d.group_by(col).len().iter_rows()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST_COPY.parent.mkdir(parents=True, exist_ok=True)
    thr = SUBSET_PCT * 100
    manifest = {"SUBSET_SEED": SUBSET_SEED, "SUBSET_PCT": SUBSET_PCT, "VAL_N": VAL_N, "files": {}}

    for name, out_name in (("train_supervised.csv", "train_supervised_sub.csv"),
                           ("train_pseudo_unlabeled.csv", "train_pseudo_unlabeled_sub.csv"),
                           ("pseudo_reference_labels.csv", "pseudo_reference_labels_sub.csv")):
        d = read(name)
        keep = pl.Series([bucket(i) < thr for i in d["id"].to_list()])
        sub = d.filter(keep)
        sub.write_csv(OUT / out_name)
        manifest["files"][out_name] = {"rows": int(sub.height), "full_rows": int(d.height)}
        if name == "train_supervised.csv":
            manifest["train_supervised_sub"] = {
                "recommendation": counts(sub, "recommendation_status"),
                "rating_band": counts(sub, "rating_band"),
                "rate_zero_rows": int((sub["rate"] == 0).sum()),
            }
        if name == "pseudo_reference_labels.csv":
            manifest["pseudo_reference_sub_recommendation"] = counts(sub, "recommendation_status")
        print(f"{out_name}: {sub.height:,} / {d.height:,} rows")

    val = read("validation.csv")
    hexes = [hashlib.sha256(f"{SUBSET_SEED}-val|{i}".encode("utf-8")).hexdigest() for i in val["id"].to_list()]
    val_sub = (val.with_columns(pl.Series("h", hexes)).sort("h").head(VAL_N).drop("h").sort("id"))
    val_sub.write_csv(OUT / "validation_sub.csv")
    manifest["files"]["validation_sub.csv"] = {"rows": int(val_sub.height), "full_rows": int(val.height)}
    manifest["validation_sub"] = {"recommendation": counts(val_sub, "recommendation_status"),
                                  "rating_band": counts(val_sub, "rating_band")}
    print(f"validation_sub.csv: {val_sub.height:,} / {val.height:,} rows")

    shutil.copy(PROC / "test.csv", OUT / "test.csv")
    shutil.copy(PROC / "split_manifest.json", OUT / "split_manifest.json")
    manifest["files"]["test.csv"] = {"rows": int(pl.scan_csv(PROC / "test.csv").select(pl.len()).collect().item())}

    total_mb = 0.0
    for p in sorted(OUT.glob("*.csv")):
        mb = p.stat().st_size / 1048576
        total_mb += mb
        manifest["files"].setdefault(p.name, {})["size_mb"] = round(mb, 1)
        print(f"  {p.name}: {mb:.1f} MB")
    manifest["total_csv_mb"] = round(total_mb, 1)
    print(f"Total CSV size: {total_mb:.1f} MB  -> upload the folder {OUT} to Kaggle as one Dataset")

    sup = manifest["train_supervised_sub"]["recommendation"]
    lab = sum(sup.get(c, 0) for c in REC)
    pseudo = manifest["files"]["train_pseudo_unlabeled_sub.csv"]["rows"]
    print(f"Masked share inside the subset: {pseudo / (lab + pseudo) * 100:.2f}% "
          f"(target: about 20% of labeled rows = {0.2 * (lab + pseudo):,.0f})")

    for path in (OUT / "subset_manifest.json", MANIFEST_COPY):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved: {OUT / 'subset_manifest.json'} and {MANIFEST_COPY}")


if __name__ == "__main__":
    main()
