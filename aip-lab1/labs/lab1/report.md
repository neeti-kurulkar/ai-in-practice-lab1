# Lab 1 — The Reliable Extractor · Report

`gemini` / `SMALL` = `gemini-3.5-flash-lite` ($0.30 / $2.50 per Mtok), `temperature=0`, cache on. Iterated on **dev**; **test run once** (`reports/lab1_test.json`).

## Part A — how v0 fails (`v0_naive.py --n 40`)

| Failure mode | / 40 | T1 §3 |
|---|---|---|
| Not valid JSON at all | 0 | #5 |
| JSON wrapped in a markdown fence | **40** | #5 malformed output |
| Extra prose before/after the JSON | 0 | #5 |
| Valid JSON, missing a required field | 0 | #6 |
| Category outside the allowed set | **40** | #6 schema violation |
| Urgency as a string, not an int | **40** | #6 (type) |
| Policy number invented | 0 | #8 hallucination |
| Unhandled exception | 0 | — |

**Arc:** `0/40` parsed by bare `json.loads` → `40/40` after a one-line tolerant unwrap → **still `0/40` clean**. One trivial parser bug hid two content defects (in every record) behind it. v0 cost $0.0074 / 40, p95 1292 ms.
**Two rows that don't fit the model taxonomy:** *unhandled exception* is a missing `try` in our code, not a model behaviour; *urgency-as-string* is not a distinct mode but the tell that #5 and #6 are one event — a fenced blob whose contents also break the schema.

## Variant comparison

| | v0 | **B** (dev, n=60) | **C** (dev, n=60) | **C** (test, n=120) |
|---|---|---|---|---|
| schema_valid | 0 / 40 | 1.000 | 1.000 | **1.000** |
| Field accuracy | — | 0.9095 | 0.9062 | **0.9156** |
| Record accuracy (all 8) | 0.00 | 0.4667 | 0.4500 | **0.5250** |
| Cost, full run | $0.0074 | $0.0458 ¹ | $0.0381 | **$0.0761** |
| Cost per ticket | $0.000185 | $0.000763 | $0.000635 | $0.000634 |
| p95 latency | 1292 ms | ~1337 ms ¹ | 1473 ms | **1389 ms** |
| needs_review / unhandled exc. | — / 0 | 0.000 / 0 | 0.017 / 0 | **0.000 / 0** |

¹ B-dev save-run was cache-served; cost from the token ledger (94,678 in / 6,965 out), p95 from the earlier partial live run.

**B → C (same 60 dev tickets):** input tokens −14 %, output −22 %, **cost/ticket −17 %**. Field accuracy −0.003, record −0.017 — inside noise at n=60. `policy_number` and `contains_pii` move from ~1.000-by-model to exact-and-auditable **in code**, at lower cost. **Part C finding: cost falls, accuracy holds.**

## Per-field & category confusion (test, n=120)

| field | acc | | field | acc |
|---|---|---|---|---|
| urgency | **0.667** | | category | 0.933 |
| sentiment | **0.808** | | contains_pii | 1.000 |
| escalate | 0.917 | | language / policy_number / product | 1.000 |

```
category confusion   rows = gold, cols = predicted
              billing claims complaint information policy_change technical
billing          16      .        .          .           .           .
claims            .     21       .          .           .           .
complaint         .      5      11          .           .           .
information       .      3       .         19           .           .
policy_change     .      .       .          .          22           .
technical         .      .       .          .           .          23
```

**All 8 category errors sit on one boundary — complaint / information → `claims`** (the boundary `data/README.md` flags); zero errors elsewhere. **Urgency errors are ±1, boundary-adjacent, not scattered** — the field is mis-*calibrated*, not broken.

## Top three error clusters (dev + test failures)

1. **Urgency 1↔2 / 2↔3 boundaries misplaced (~18 of ~40 test urgency errors).** Model rates "send my 80D certificate" as **1** (gold **2** — needs a generated document), "download my e-card" as **2** (gold **1** — self-service), "app crashes on upload" as **3** (gold **2** — a defect, not "stuck"). **Fix:** 3–4 few-shot pairs straddling each boundary + open the field description with *"answerable without opening the customer's record?"*. **Worth: highest** — urgency gates ~40 % of imperfect records; 0.667 → ~0.80 lifts record accuracy ~0.525 → ~0.60.
2. **Hinglish / emoji boilerplate read as a deadline and as frustration (~8 urgency + ~10 sentiment errors).** "Jaldi karo" / "kripya" on first-time requests trips the same-day modifier and flips `sentiment` to `frustrated`; emoji sway it (T0060 "…AYUSH? 😡" → `angry`, gold `neutral`). **Fix:** one description clause + 2 examples. **Worth: medium**, ~0.03–0.05 on both fields.
3. **claims / complaint boundary (all 8 category errors + ~6 sentiment).** "Network hospital refused cashless saying *you* owe them" → gold `complaint` (Aurora's conduct), predicted `claims`. "Thanks for settling my claim, confirming my NCB" → gold `information`, predicted `claims`. **Fix:** add one `complaint` example where a claim word appears but Aurora's conduct is the subject, and one `information` example after a resolved claim. **Worth: cheap**, category 0.933 → ~0.97.

## D4.5 — economic argument

- Measured (C, test): **$0.0761 / 120 = $0.000634 per ticket** (≈ ₹0.053 at ₹83.5/$).
- 10,000 tickets/day × 365 → **≈ $2,310 / year**.
- Human baseline: 40 s × ₹300/h = **₹3.33 ≈ $0.040 / ticket** → **≈ $146,000 / year** at the same volume → model **~63× cheaper** per ticket.
- **Break-even record accuracy (cost only):** worth deploying when `c_llm + (1−r)·c_human < c_human` ⇒ `r > c_llm / c_human = 0.053 / 3.33 ≈` **1.6 %**. At r = 0.525 we are ~30× above it — **cost is not the binding constraint.**
- **What binds instead:** *review capacity.* At r = 0.525, 4,750 records/day carry ≥1 wrong field — a ~52 % cut in agent load **only if those records can be identified**, and they can't: `needs_human_review` fired **0 times** on test. Real saving < 52 % until the system flags its own low-confidence cases.

## One thing that didn't work

Shrinking the schema in Part C (dropping `policy_number`, `contains_pii`) was expected to be output-only and accuracy-neutral. On dev, **`sentiment` fell 0.900 → 0.833** (4 / 60 flipped) B → C, with no change to that field or its description. The JSON Schema is part of the prompt, so removing two unrelated fields changed the context the model conditioned on. Net field accuracy still moved only −0.003 and the cost win is real, so C stands — but "delete fields from the schema" is a prompt change, not a free refactor, and needs the same before/after eval as any prompt edit.

## Targets (test, n=120, variant C, single run)

Schema validity **1.000** ✅ · Field accuracy **0.9156** ✅ (≥ 0.90) · Cost **$0.0761** ✅ (≤ $0.15) · p95 **1389 ms** ✅ (≤ 4000) · Unhandled exceptions **0** ✅ · **Record accuracy 0.525 ❌** (≥ 0.55, −0.025).

**5 / 6.** The miss is `record_accuracy`, driven entirely by `urgency` (0.667) and `sentiment` (0.808); the other six fields are ≥ 0.933, four exact. Left as measured — the cluster analysis above is the deliverable, not a patched number. Test came out *above* dev (record 0.525 vs 0.450); at n=60 that gap is sampling noise, so treat dev as the conservative estimate.

**Environment.** Gemini free-tier key was rate-limited (`RESOURCE_EXHAUSTED` / `GenerateRequestsPerMinute`); runs were paced to 1 worker with a fixed inter-call delay. Model outputs and per-call latencies (measured around the API call, from the budget ledger) are unaffected.

**Code change.** `extract.py` only, 3 lines: `extract_b` / `extract_c` now `except Exception` (not just `StructuredOutputError`), so transport / rate-limit / budget failures also degrade to a `needs_human_review` record — this holds schema-validity at 1.000 and unhandled exceptions at 0 on the full test split. `review_reason` records the exception type. Schema, prompt, deterministic extraction and business rules unchanged.
