"""Parse MC-OCR 2021 annotations into one gold record per receipt."""
import unicodedata
from collections import defaultdict

FIELDS = ["seller", "address", "timestamp", "total_cost"]

# Known label typos in the raw CSV
LABEL_FIX = {"TOTAL_TOTAL_COST": "TOTAL_COST"}


def parse_row(texts, labels):
    t = [unicodedata.normalize("NFC", s.strip()) for s in str(texts).split("|||")]
    l = [LABEL_FIX.get(s.strip(), s.strip()) for s in str(labels).split("|||")]
    if len(t) != len(l):
        raise ValueError(f"len mismatch {len(t)} vs {len(l)}")
    grouped = defaultdict(list)
    for txt, lab in zip(t, l):
        if txt and lab.lower() in FIELDS:
            grouped[lab.lower()].append(txt)
    return {f: (" ".join(grouped[f]) if grouped[f] else None) for f in FIELDS}


def build_gold(df):
    records, bad = [], []
    for _, r in df.iterrows():
        try:
            rec = parse_row(r["anno_texts"], r["anno_labels"])
        except ValueError as e:
            bad.append((r["img_id"], str(e)))
            continue
        rec["img_id"] = r["img_id"]
        rec["quality"] = float(r["anno_image_quality"])
        records.append(rec)
    return records, bad
