#!/usr/bin/env python3
"""Part E — error analysis. Failure dump + a confusion matrix.

Offline, from a grid JSON saved by `grid.py --save` joined back to the dev
golden set for the gold labels.

  E1  every case the variant got wrong: the wrong fields, gold vs predicted,
      and the ticket text — read these and cluster them by hand.
  E2  the gold-vs-predicted matrix for one field across ALL cases — the
      systematic confusion the aggregate hides.

    python scripts/lab2_errors.py [reports/lab2_grid.json] --variant zero_shot
    python scripts/lab2_errors.py --variant zero_shot --field urgency --limit 20
    python scripts/lab2_errors.py --variant zero_shot --field category
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRADED = ["category", "urgency", "sentiment", "product", "language",
          "policy_number", "contains_pii", "escalate"]


def load_gold() -> dict[str, dict]:
    rows = [json.loads(l) for l in
            (ROOT / "data/eval/extraction_dev.jsonl").open(encoding="utf-8")]
    return {r["id"]: r for r in rows}


def norm(v: object) -> str:
    return str(v).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", nargs="?", default="reports/lab2_grid.json")
    ap.add_argument("--variant", default="zero_shot")
    ap.add_argument("--field", default="urgency")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    data = json.loads((ROOT / args.grid).read_text(encoding="utf-8"))
    if args.variant not in data:
        ap.error(f"{args.variant!r} not in {args.grid}. Have: {', '.join(data)}")
    results = data[args.variant]["results"]
    gold = load_gold()

    # ----- E1: failures -----
    fails = [r for r in results
             if r["error"] or r["metrics"].get("record_accuracy", 0.0) < 1.0]
    by_field = collections.Counter()
    print(f"=== {args.variant}: {len(fails)} imperfect records of {len(results)} "
          f"(showing {min(args.limit, len(fails))}) ===\n")
    for r in fails[:args.limit]:
        g = gold.get(r["id"], {}).get("expected", {}) or {}
        p = r["output"] or {}
        wrong = [f for f in GRADED if norm(g.get(f)) != norm(p.get(f))]
        by_field.update(wrong)
        text = " ".join(gold.get(r["id"], {}).get("input", "").split())
        tag = "  (ERROR)" if r["error"] else ""
        print(f"{r['id']}  wrong={wrong}{tag}")
        for f in wrong:
            print(f"    {f:<14} gold={g.get(f)!r:<16} pred={p.get(f)!r}")
        print(f"    {textwrap.shorten(text, 150)}\n")

    if by_field:
        print("wrong-field tally (across the shown failures):")
        for f, n in by_field.most_common():
            print(f"  {f:<14} {n}")
        print()

    # ----- E2: confusion matrix for one field, all cases -----
    def gv(r):
        return str(gold.get(r["id"], {}).get("expected", {}).get(args.field))

    def pv(r):
        return str((r["output"] or {}).get(args.field))

    labels = sorted({gv(r) for r in results} | {pv(r) for r in results})
    conf = collections.Counter((gv(r), pv(r)) for r in results)
    w = max([len(x) for x in labels] + [4]) + 1
    hit = sum(conf[(x, x)] for x in labels)
    print(f"=== confusion for '{args.field}'  (rows = gold, cols = pred)   "
          f"acc {hit}/{len(results)} = {hit / len(results):.3f} ===")
    print(" " * (w + 2) + "".join(f"{x:>{w}}" for x in labels))
    for g in labels:
        print(f"  {g:<{w}}" + "".join(f"{conf.get((g, p), 0):>{w}}" for p in labels))


if __name__ == "__main__":
    main()
