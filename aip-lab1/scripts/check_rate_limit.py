#!/usr/bin/env python3
"""Measure your API key's rate-limit headroom before Lab 1.

    python scripts/check_rate_limit.py

Providers no longer publish fixed free-tier RPM figures -- they are per key,
not per model, and linking a billing account silently moves you to a higher
tier. One person's key tells you nothing about another's, so guessing is not
an option. This fires a burst of small, live calls at increasing concurrency
and reports the highest concurrency that ran clean before the first 429.

Lab 1 needs about 500 calls over three hours; this probe uses about 40 of
them. On the free tiers this repo recommends (gemini, nvidia) that is
effectively free; on a paid profile it is a few cents at most.
"""
from __future__ import annotations

import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aip.config import resolve_model, settings  # noqa: E402
from aip.llm import raw_call  # noqa: E402

LEVELS = (1, 2, 4, 8)   # concurrency levels to try, in order
CALLS_PER_LEVEL = 10    # len(LEVELS) * CALLS_PER_LEVEL = ~40 calls, worst case

# Same markers aip.llm._is_retryable uses, split into "this IS the rate limit"
# vs "the provider had a hiccup". A 503 mid-burst is not evidence about your
# concurrency headroom -- providers serve those regardless of how politely you
# ask -- so it must not be reported as if it were a 429.
_RATE_LIMIT_MARKERS = ("ratelimit", "429")
_TRANSIENT_MARKERS = ("timeout", "overloaded", "apiconnection", "internalserver",
                      "serviceunavailable", "529", "503", "502", "500")


def _classify(detail: str) -> str:
    d = detail.lower()
    if any(m in d for m in _RATE_LIMIT_MARKERS):
        return "rate_limited"
    if any(m in d for m in _TRANSIENT_MARKERS):
        return "transient"
    return "hard"


def _one_call(model: str, nonce: int) -> tuple[bool, float, str]:
    """Fire one minimal, live, uncached call. Returns (ok, latency_s, detail)."""
    t0 = time.perf_counter()
    try:
        raw_call(
            [{"role": "user", "content": f"Reply with exactly: {nonce}"}],
            model=model, temperature=0, max_tokens=5,
        )
        return True, time.perf_counter() - t0, ""
    except Exception as exc:                                      # noqa: BLE001
        return False, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"


def probe(model: str) -> tuple[int | None, int | None, str]:
    """Returns (level that first hit a 429, highest level that ran clean, hard error detail)."""
    hit_429 = None
    best_clean = None
    hard_detail = ""
    nonce = 0

    for workers in LEVELS:
        print(f"-- {workers} concurrent worker{'s' if workers != 1 else ''} "
              f"({CALLS_PER_LEVEL} calls) --")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one_call, model, nonce + i) for i in range(CALLS_PER_LEVEL)]
            results = [f.result() for f in as_completed(futures)]
        nonce += CALLS_PER_LEVEL

        ok = [r for r in results if r[0]]
        errors = [(ok_flag, lat, detail, _classify(detail))
                  for ok_flag, lat, detail in results if not ok_flag]
        rate_limited = [e for e in errors if e[3] == "rate_limited"]
        transient = [e for e in errors if e[3] == "transient"]
        hard = [e for e in errors if e[3] == "hard"]

        if ok:
            lat = statistics.median(r[1] for r in ok)
            print(f"   {len(ok)}/{len(results)} ok, median latency {lat * 1000:.0f} ms")
        if rate_limited:
            print(f"   {len(rate_limited)} rate-limited (429):")
            print(f"     {rate_limited[0][2][:160]}")
            hit_429 = workers
            break
        if hard:
            print(f"   {len(hard)} unexpected error(s):")
            print(f"     {hard[0][2][:160]}")
            hard_detail = hard[0][2]
            break
        if transient:
            print(f"   {len(transient)} transient provider error(s) (not a rate limit) "
                  f"-- continuing")
            print(f"     {transient[0][2][:160]}")

        best_clean = workers

    return hit_429, best_clean, hard_detail


def main() -> None:
    if settings.offline:
        print("AIP_OFFLINE=1 -- rate-limit probing needs live calls. Unset it and re-run.")
        sys.exit(1)

    model = resolve_model("SMALL")
    print("AI in Practice I - Module 1: rate-limit probe\n")
    print(f"profile={settings.profile}  model={model}")
    print(f"Up to {len(LEVELS) * CALLS_PER_LEVEL} live calls, working up through "
          f"{', '.join(str(w) for w in LEVELS)} concurrent workers.\n")

    # The client's own retry sleeps up to 30s with backoff -- exactly what
    # would hide a 429 from this probe. Go single-shot so the real error
    # surfaces immediately, then restore the normal setting for the labs.
    original_retries = settings.max_retries
    settings.max_retries = 1
    try:
        hit_429, best_clean, hard_detail = probe(model)
    finally:
        settings.max_retries = original_retries

    print()
    if hit_429 is not None:
        recommended = max(1, hit_429 // 2)
        print(f"Hit 429 at {hit_429} concurrent workers.")
        print(f"Recommended: --workers {recommended} for Lab 1. The client's own "
              f"retry (backoff + jitter) will absorb the occasional 429 at that level.")
    elif hard_detail:
        print("Stopped on an unexpected error -- this is not a rate-limit signal, "
              "something else is wrong:")
        print(f"  {hard_detail[:200]}")
        print("Check your API key and AIP_PROFILE before starting Lab 1.")
        sys.exit(1)
    elif best_clean:
        print(f"Clean through {best_clean} concurrent workers, no 429s.")
        print(f"Recommended: --workers {best_clean} for Lab 1 "
              f"(about 500 calls over three hours is comfortably inside this).")
    else:
        print("Every call failed with a transient provider error before a clean "
              "batch completed. The provider may be having a bad moment -- retry "
              "the probe in a minute before assuming anything about your limits.")
        sys.exit(1)

    from aip.cost import global_budget
    print("\n" + global_budget().report())


if __name__ == "__main__":
    main()
