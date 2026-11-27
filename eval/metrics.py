"""Field-level metrics for receipt extraction."""
from src.data_mcocr import FIELDS
from src.normalize import norm_text, norm_total, norm_timestamp


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def text_sim(p, g):
    """1 - normalized edit distance (ANLS style), 0 if below 0.5."""
    if p is None or g is None:
        return float(p is None and g is None)
    d = levenshtein(p, g) / max(len(p), len(g))
    s = 1 - d
    return s if s >= 0.5 else 0.0


def score_field(field, pred, gold):
    """Returns (exact_match: 0/1, soft_score: 0..1)."""
    if field == "total_cost":
        p, g = norm_total(pred), norm_total(gold)
        em = float(p == g)
        return em, em
    if field == "timestamp":
        p, g = norm_timestamp(pred), norm_timestamp(gold)
        if p is None or g is None:
            em = float(p is None and g is None)
            return em, em
        date_ok = p[0] == g[0]
        time_ok = g[1] is None or p[1] == g[1]
        em = float(date_ok and time_ok)
        return em, (date_ok + time_ok) / 2
    p, g = norm_text(pred), norm_text(gold)
    return float(p == g), text_sim(p, g)


def evaluate(preds, golds):
    """preds, golds: dict img_id -> record. Missing prediction counts as all None."""
    out = {f: {"em": 0.0, "soft": 0.0} for f in FIELDS}
    n = len(golds)
    for img_id, g in golds.items():
        p = preds.get(img_id, {})
        for f in FIELDS:
            em, soft = score_field(f, p.get(f), g.get(f))
            out[f]["em"] += em
            out[f]["soft"] += soft
    for f in FIELDS:
        out[f] = {k: round(v / n, 4) for k, v in out[f].items()}
    out["mean_em"] = round(sum(out[f]["em"] for f in FIELDS) / len(FIELDS), 4)
    out["mean_soft"] = round(sum(out[f]["soft"] for f in FIELDS) / len(FIELDS), 4)
    out["n"] = n
    return out
