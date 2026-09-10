#!/usr/bin/env python3
"""Part C — interrogate the cascade.  Offline, from a grid JSON with a `cascade`
row saved by `grid.py --save`.

Reports:
  - escalation rate and the reason breakdown
  - whether the trigger carries signal: sample agreement when SMALL is
    record-correct vs record-wrong, and how many errors that actually catches
  - SMALL accuracy on escalated vs accepted tickets
  - blended cost/accuracy note

Relies on the fields cascade() writes onto each record: `_path`
('small'|'large'), `_small_agree` (bool|None), `_escalate_reason`.

    python scripts/lab2_cascade.py [reports/lab2_grid.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def correct(r: dict) -> bool:
    return r["metrics"].get("record_accuracy", 0.0) == 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", nargs="?", default="reports/lab2_grid.json")
    ap.add_argument("--variant", default="cascade")
    args = ap.parse_args()

    data = json.loads((ROOT / args.grid).read_text(encoding="utf-8"))
    if args.variant not in data:
        ap.error(f"{args.variant!r} not in {args.grid}. Have: {', '.join(data)}")
    rows = data[args.variant]["results"]
    n = len(rows)
    out = [r["output"] or {} for r in rows]

    esc = [r for r in rows if (r["output"] or {}).get("_path") == "large"]
    acc = [r for r in rows if (r["output"] or {}).get("_path") == "small"]

    print(f"=== {args.variant}: {n} tickets ===\n")
    print(f"escalation rate       {len(esc)}/{n} = {len(esc) / n:.1%}")
    print("escalate reasons      "
          + str(dict(collections.Counter(o.get("_escalate_reason") for o in out))))
    print()

    def rate(group, label):
        k = sum(correct(r) for r in group)
        m = len(group)
        print(f"  {label:<26} {k}/{m} = {k / m:.2f}" if m else f"  {label:<26} —")

    print("SMALL record-accuracy by path:")
    rate(esc, "escalated (disagree)")
    rate(acc, "accepted (agree)")
    print()

    right = [r for r in rows if correct(r)]
    wrong = [r for r in rows if not correct(r)]

    def agree(group, label):
        k = sum(1 for r in group if (r["output"] or {}).get("_small_agree") is True)
        m = len(group)
        print(f"  {label:<34} {k}/{m} = {k / m:.2f}" if m else f"  {label:<34} —")

    print("two-sample agreement (does the trigger carry signal?):")
    agree(right, "when SMALL is record-correct")
    agree(wrong, "when SMALL is record-wrong")
    caught = sum(1 for r in wrong if (r["output"] or {}).get("_path") == "large")
    print(f"\n  errors flagged by the trigger: {caught}/{len(wrong)} "
          f"= {caught / len(wrong):.0%}")
    false_esc = sum(1 for r in esc if correct(r))
    print(f"  escalations of already-correct answers: {false_esc}/{len(esc)}")

    b = data[args.variant]["budget"]
    print(f"\nbudget: cost ${b.get('cost_usd', 0):.4f}  "
          f"p95 {b.get('latency_p95_ms', 0):.0f} ms  "
          f"calls {b.get('calls', 0)} ({b.get('cached_calls', 0)} cached)")
    print("note: blended accuracy == pure SMALL wherever a MAIN escalation "
          "failed (check for 'cascade MAIN call failed' warnings in the run).")


if __name__ == "__main__":
    main()
