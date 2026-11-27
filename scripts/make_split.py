"""Build gold labels from MC-OCR CSV and a fixed train/val/test split.

Usage: python scripts/make_split.py <mcocr_train_df.csv> <out_dir>
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_mcocr import build_gold  # noqa: E402

SEED = 42
RATIOS = (0.8, 0.1, 0.1)


def main(csv_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records, bad = build_gold(pd.read_csv(csv_path))
    if bad:
        print("skipped rows:", bad)

    ids = sorted(r["img_id"] for r in records)
    random.Random(SEED).shuffle(ids)
    n = len(ids)
    n_train, n_val = int(n * RATIOS[0]), int(n * RATIOS[1])
    split = {
        "seed": SEED,
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train:n_train + n_val]),
        "test": sorted(ids[n_train + n_val:]),
    }
    (out / "split.json").write_text(json.dumps(split, indent=1))
    with open(out / "gold.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print({k: len(v) for k, v in split.items() if k != "seed"})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
