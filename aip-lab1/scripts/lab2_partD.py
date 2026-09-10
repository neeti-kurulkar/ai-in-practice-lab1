#!/usr/bin/env python3
"""Part D — is the difference real?  Wilson CIs + paired McNemar tests.

Runs entirely offline on a grid JSON saved by `grid.py --save`.

  D1  record-accuracy point estimate with its 95% Wilson interval
  D2  paired McNemar exact test on the same items (b, c, p)
  D3  the verdict — "no significant difference" is a real answer (choose on cost)

A4 hold-out (ON by default): a `few_shot*` variant was shown 6 dev tickets in
its own prompt, so scoring it on those tickets is unfair. Any CI or comparison
that involves a `few_shot*` variant therefore drops FEW_SHOT_IDS; `zero_shot`
and `cascade`, which never saw them, keep all items. `grid.py` is left stock and
its JSON still holds every case — the hold-out happens only here. Pass
--keep-fewshot for the raw, un-held-out numbers.

    python scripts/lab2_partD.py [reports/lab2_grid.json]
    python scripts/lab2_partD.py reports/lab2_grid.json --baseline zero_shot
    python scripts/lab2_partD.py --keep-fewshot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labs.lab2.stats import paired_test, wilson_interval  # noqa: E402

try:
    from labs.lab2.variants import FEW_SHOT_IDS  # noqa: E402
except Exception:  # keep the analysis usable even if variants.py won't import
    FEW_SHOT_IDS = ["T0097", "T0056", "T0200", "T0021", "T0081", "T0222"]


def is_fewshot(name: str) -> bool:
    return name.startswith("few_shot")


def per_item_correct(variant: dict, metric: str) -> dict[str, bool]:
    """id -> did this variant get this item exactly right (metric == 1.0)."""
    return {r["id"]: r["metrics"].get(metric, 0.0) == 1.0
            for r in variant["results"] if not r["error"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", nargs="?", default="reports/lab2_grid.json")
    ap.add_argument("--baseline", default="",
                    help="only compare every variant against this one")
    ap.add_argument("--metric", default="record_accuracy")
    ap.add_argument("--keep-fewshot", action="store_true",
                    help="disable the A4 hold-out (score few-shot on all items)")
    args = ap.parse_args()

    data = json.loads((ROOT / args.grid).read_text(encoding="utf-8"))
    names = list(data)
    corr = {n: per_item_correct(data[n], args.metric) for n in names}
    heldout = set() if args.keep_fewshot else set(FEW_SHOT_IDS)

    print(f"grid   : {args.grid}")
    print(f"metric : {args.metric}")
    if heldout:
        print(f"A4     : {len(heldout)} few-shot example(s) held out of any CI / "
              f"comparison involving a few_shot* variant")
        print(f"         ({', '.join(sorted(heldout))})")
    print()

    print("D1 — point estimate and 95% Wilson interval")
    print(f"  {'variant':<26}{'n':>5}{'acc':>9}   95% CI")
    for n in names:
        drop = heldout if is_fewshot(n) else set()
        c = {i: v for i, v in corr[n].items() if i not in drop}
        k, tot = sum(c.values()), len(c)
        if tot == 0:
            print(f"  {n:<26}{0:>5}      —")
            continue
        lo, hi = wilson_interval(k, tot)
        tag = f"  (held out {len(drop)})" if drop else ""
        print(f"  {n:<26}{tot:>5}{k / tot:>9.3f}   [{lo:.3f}, {hi:.3f}]{tag}")

    print("\nD2 / D3 — paired McNemar (same items scored by both systems)")
    print("  b = left right & right wrong,  c = right right & left wrong\n")
    if args.baseline:
        if args.baseline not in names:
            ap.error(f"--baseline {args.baseline!r} not in grid ({', '.join(names)})")
        pairs = [(args.baseline, o) for o in names if o != args.baseline]
    else:
        pairs = [(names[i], names[j])
                 for i in range(len(names)) for j in range(i + 1, len(names))]

    for a, b in pairs:
        drop = heldout if (is_fewshot(a) or is_fewshot(b)) else set()
        ids = [i for i in corr[a] if i in corr[b] and i not in drop]
        if not ids:
            print(f"  {a}  vs  {b}: no shared items")
            continue
        res = paired_test([corr[a][i] for i in ids], [corr[b][i] for i in ids])
        note = f", {len(drop)} held out" if drop else ""
        print(f"  {a}  vs  {b}   (n={len(ids)}{note})")
        print(f"      b={res['b']}  c={res['c']}  p={res['p_value']:.4f}"
              f"  ->  {res['verdict']}")


if __name__ == "__main__":
    main()
