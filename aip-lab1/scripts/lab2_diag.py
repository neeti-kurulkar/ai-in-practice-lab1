#!/usr/bin/env python3
"""Part A diagnostic: WHY did few-shot lose?

Runs zero_shot and few_shot on the held-out dev cases (few-shot examples
excluded) and shows, per model-decided field:

  - how many outputs were malformed / missing the field   (implementation bug?)
  - the category confusion matrix for each variant          (label bias?)
  - the urgency gold-vs-predicted histogram                 (pulled toward the
                                                             example patterns?)

All calls are served from structured()'s cache if you have already run
`grid.py --variants zero_shot few_shot --split dev`, so this is nearly free.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labs.lab2.variants import FEW_SHOT_IDS, few_shot, zero_shot  # noqa: E402

FIELDS = ("category", "urgency", "sentiment", "product")
CATS = ["billing", "claims", "policy_change", "technical", "complaint", "information"]


def load_holdout() -> list[dict]:
    rows = [json.loads(l) for l in
            (ROOT / "data/eval/extraction_dev.jsonl").open(encoding="utf-8")]
    return [r for r in rows if r["id"] not in set(FEW_SHOT_IDS)]


def run(variant, rows) -> list[dict]:
    return [variant(r["input"]) for r in rows]


def report(name: str, rows: list[dict], preds: list[dict]) -> None:
    print(f"\n================  {name}  (n={len(rows)})  ================")

    # 1. malformed / missing fields
    missing = collections.Counter()
    for p in preds:
        for f in FIELDS:
            if not isinstance(p, dict) or f not in p or p[f] is None:
                missing[f] += 1
    print("missing/None per field:", dict(missing) or "none")

    # 2. per-field accuracy
    for f in FIELDS:
        ok = sum(1 for r, p in zip(rows, preds)
                 if str(p.get(f)).strip().lower() == str(r["expected"][f]).strip().lower())
        print(f"  {f:<10} acc {ok / len(rows):.3f}")

    # 3. category confusion (rows = gold, cols = predicted)
    conf = collections.Counter()
    for r, p in zip(rows, preds):
        conf[(r["expected"]["category"], str(p.get("category")))] += 1
    print("\ncategory confusion (gold \\ pred):")
    print("            " + "".join(f"{c[:9]:>11}" for c in CATS))
    for g in CATS:
        print(f"  {g:<10}" + "".join(f"{conf.get((g, c), 0):>11}" for c in CATS))

    # 4. urgency histogram
    gold_h = collections.Counter(r["expected"]["urgency"] for r in rows)
    pred_h = collections.Counter(p.get("urgency") for p in preds)
    print("\nurgency   1   2   3   4   5")
    print("  gold  " + "".join(f"{gold_h.get(u, 0):>4}" for u in range(1, 6)))
    print("  pred  " + "".join(f"{pred_h.get(u, 0):>4}" for u in range(1, 6)))


def main() -> None:
    rows = load_holdout()
    print(f"held-out dev cases: {len(rows)}  (excluded {sorted(FEW_SHOT_IDS)})")
    report("zero_shot", rows, run(zero_shot, rows))
    report("few_shot", rows, run(few_shot, rows))


if __name__ == "__main__":
    main()
