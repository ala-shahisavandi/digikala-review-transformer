"""
Shared evaluation helper (numpy only, so it runs locally and on Kaggle without sklearn).

evaluate(y_true, y_pred, class_names, ordinal=False) returns:
  n, accuracy, balanced_accuracy, macro_precision/recall/f1,
  per_class {precision, recall, f1, support}, confusion_matrix (rows = true, cols = predicted),
  and for ordinal targets (rating_band): ordinal_mean_abs_error, share_distance_1, share_distance_2.

Labels must be integer ids 0..K-1 in the order of class_names.
"""
import numpy as np


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float64)
    np.divide(a, b, out=out, where=b > 0)
    return out


def evaluate(y_true, y_pred, class_names: list, ordinal: bool = False) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    k = len(class_names)
    n = int(len(y_true))
    cm = np.zeros((k, k), dtype=np.int64)
    if n:
        np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    prec = _safe_div(tp, predicted)
    rec = _safe_div(tp, support)
    f1 = _safe_div(2 * prec * rec, prec + rec)
    present = support > 0
    out = {
        "n": n,
        "accuracy": float(tp.sum() / n) if n else 0.0,
        "balanced_accuracy": float(rec[present].mean()) if present.any() else 0.0,
        "macro_precision": float(prec[present].mean()) if present.any() else 0.0,
        "macro_recall": float(rec[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {
            name: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                   "support": int(support[i])}
            for i, name in enumerate(class_names)
        },
        "labels": list(class_names),
        "confusion_matrix": cm.tolist(),
    }
    if ordinal and n:
        d = np.abs(y_true - y_pred)
        out["ordinal_mean_abs_error"] = float(d.mean())
        out["share_distance_1"] = float((d == 1).mean())
        out["share_distance_2"] = float((d >= 2).mean())
    return out


def round_floats(obj, digits: int = 5):
    """Recursively round floats for compact JSON."""
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, digits) for v in obj]
    return obj


if __name__ == "__main__":
    yt = [0, 0, 1, 1, 2, 2, 2, 2]
    yp = [0, 1, 1, 1, 2, 2, 0, 2]
    r = evaluate(yt, yp, ["low", "medium", "high"], ordinal=True)
    assert r["n"] == 8 and abs(r["accuracy"] - 6 / 8) < 1e-9
    assert r["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 3]]
    assert abs(r["per_class"]["high"]["recall"] - 0.75) < 1e-9
    assert abs(r["per_class"]["low"]["precision"] - 0.5) < 1e-9
    assert abs(r["balanced_accuracy"] - (0.5 + 1.0 + 0.75) / 3) < 1e-9
    assert abs(r["ordinal_mean_abs_error"] - (1 + 2) / 8) < 1e-9
    assert abs(r["share_distance_2"] - 1 / 8) < 1e-9
    print("metrics_utils self-test OK")
