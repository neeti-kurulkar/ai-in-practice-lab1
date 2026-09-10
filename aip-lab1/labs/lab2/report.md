# Lab 2 — The Prompt Lab

**Provider:** OpenAI profile — `SMALL` = `gpt-4.1-mini`, `MAIN` = `gpt-4.1`.
Everything on `dev` (n=60) unless noted; calibration on `test` (n=120).
Gemini free tier was used first but hit its per-day quota on three separate
days across two keys before the grid could complete; the run was moved to a
paid OpenAI key. `grid.py`, `stats.py` left stock; analysis helpers added
under `scripts/` (`lab2_partD.py`, `lab2_errors.py`, `lab2_cascade.py`,
`lab2_diag.py`).

---

## 1 · The grid (dev, n=60)

| variant | record_acc | field_acc | schema_valid | repair_rate | cost / 1k | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| zero_shot | 0.467 | 0.910 | 1.00 | ~0 | **$0.73** | ~2000 | 3341 |
| zero_shot_main | 0.483 | 0.910 | 1.00 | ~0 | ~$3.7 (est) | ~1500 | 1572 |
| few_shot | 0.483 | 0.898 | 1.00 | ~0 | $1.16 | ~1900 | 3775 |
| **few_shot_main** | **0.683** | **0.944** | 1.00 | ~0 | **$5.83** | 1356 | 11352† |
| few_shot_reasoned | 0.467 | 0.892 | 1.00 | ~0 | $1.48 | 2897 | 3960 |
| few_shot_reasoned_main | 0.683 | 0.942 | 1.00 | ~0 | $7.48 | 1973 | 14493† |
| cascade | 0.467 | 0.915 | 1.00 | ~0 | ~$2.0 (est) | ~0 | 2338 |

† p95 is inflated by rate-limit backoff on the shared key; p50 (~1.4 s) is
representative of steady state. Cost marked "est" was measured on a partly
cached run and is a lower bound; the ratio (`MAIN` ≈ 5× `SMALL` per token)
holds.

**Calibration (test, n=120):** `zero_shot` record **0.450** (95% CI
[0.364, 0.539]); `few_shot_main` record **0.600** (95% CI [0.511, 0.683]).
Field accuracy 0.905 vs 0.924.

### Part B answers

1. **Which axis moved the numbers — prompt or model tier?** *Neither, alone.*
   `zero_shot → zero_shot_main` (tier only): 0.467 → 0.483, **p = 1.00**.
   `zero_shot → few_shot` (prompt only): 0.467 → 0.483, **p = 1.00**.
   Only the **interaction** moved anything: `few_shot_main` (both) → 0.683,
   **p = 0.031** (A4 held out). The strong model can use six worked examples;
   the small one cannot, and the strong model given no examples has nothing to
   work with. The headline number is an interaction effect, not a main effect.

2. **What did the reasoning field cost, and buy?** `few_shot_reasoned_main`
   scored **identically** to `few_shot_main` (0.683, same discordant pairs
   b=4 c=14) while costing **28 % more** ($7.48 vs $5.83 / 1k) and running
   slower. Accuracy points per rupee: **zero** — the delta is noise. On the
   small tier the reasoning field made record accuracy slightly *worse*
   (0.467 vs 0.483).

3. **Dominated configurations.** `few_shot_reasoned_main` is dominated by
   `few_shot_main`: equal record and field accuracy, higher cost, higher
   latency. (Ignore `grid.py`'s dominance line on a cache-warm run — cached
   variants report $0.00 and break the comparison.)

---

## 2 · Few-shot selection (Part A)

Six dev tickets, chosen as edges that target the fields the zero-shot baseline
fails (`urgency`, `category`, `sentiment` — `policy_number`/`contains_pii` are
code-extracted and already at ~1.00, so a few-shot slot there is wasted):

| ID | teaches |
|---|---|
| T0097 | "third time / double debit / ombudsman" is **billing**, not complaint; the ombudsman line drives `escalate`, not the category |
| T0056 | a mishandled grievance is **complaint** but urgency 3 / `escalate` false — breaks the "complaint ⇒ urgent" reflex |
| T0200 | "what is X and why does it apply to me?" about a settled claim is **claims**, not information; also Hinglish |
| T0021 | portal down *at the hospital desk* = urgency 5; Hinglish; no policy number; product read from the plan name |
| T0081 | an app you are not blocked on = urgency 2, not 4; "reinstalled twice" (a prior failure) = frustrated, not neutral |
| T0222 | praise + a question = sentiment **satisfied**, urgency 1 — tone is not urgency |

Two archetypes from the brief were not used and are documented in
`variants.py`: *"policy number only in a quoted reply"* (no such ticket exists
in dev, and the field is code-extracted) and *"satisfied-but-urgent"* (no dev
ticket is both `satisfied` and urgency ≥ 4). Their slots went to T0056 and
T0081/T0222.

### A4 — the dev-set contamination problem, and the fix

The six examples are drawn from `dev` and the comparison is scored on `dev`,
so the model has effectively seen six answers. **Fix:** any confidence
interval or paired comparison involving a `few_shot*` variant excludes those
six IDs (n = 54); `zero_shot`, `zero_shot_main` and `cascade`, which never
saw them, keep all 60. Implemented in `scripts/lab2_partD.py`; `grid.py` left
stock, and its saved JSON still holds every case so the choice is reversible.
On the `test` split the six dev IDs match nothing, so the calibration is
uncontaminated by construction.

---

## 3 · The cascade (Part C)

`SMALL` first; escalate to `MAIN` when the `SMALL` call fails validation, has
an empty `evidence` span, **or** a second `SMALL` sample drawn at
`temperature = 0.7` disagrees on category/urgency/sentiment/product. The
second sample is drawn at T > 0 deliberately — at T = 0 it is the same request,
the cache serves it, the answers are byte-identical, and the escalation rate
is a silent 0.00.

| metric | value |
|---|---|
| **escalation rate** | **11.7 %** (7/60), all triggered by disagreement |
| **blended cost** | ~$2 / 1k — roughly 2× `zero_shot` (two SMALL calls per ticket) plus the escalations; still far below `few_shot_main`'s $5.83 |
| **blended accuracy** | **0.467 — identical to `zero_shot`** (b=1, c=1, p = 1.00) |

**The cascade cannot help here, structurally.** It escalates to *zero-shot*
`MAIN`, and zero-shot `MAIN` is not better than zero-shot `SMALL` (p = 1.00) —
the tier only pays off *with* few-shot. To work, the cascade would have to
escalate to `few_shot_main`.

**Does the trigger carry signal?** Two samples agree 93 % of the time when
`SMALL` is record-correct vs **84 %** when it is wrong — a 9-point gap that
flags only **5 of 32 errors (16 %)** and escalates 2 already-correct answers.
Escalated tickets do have lower `SMALL` accuracy (0.29 vs 0.49), so
disagreement is not pure noise — but it detects *variance*, and the model's
errors are mostly *bias*: it is consistently wrong, not uncertain.

---

## 4 · Is the difference real? (Part D)

95 % Wilson intervals and McNemar's exact paired test, `zero_shot` as baseline.

**Dev (few-shot variants n = 54, A4 held out):**

| comparison | b | c | p | verdict |
|---|---|---|---|---|
| zero_shot vs zero_shot_main | 7 | 8 | 1.00 | no difference — choose on cost |
| zero_shot vs few_shot | 8 | 6 | 0.79 | no difference — choose on cost |
| zero_shot vs few_shot_reasoned | 8 | 5 | 0.58 | no difference — choose on cost |
| zero_shot vs cascade | 1 | 1 | 1.00 | no difference — choose on cost |
| **zero_shot vs few_shot_main** | 4 | 14 | **0.031** | **few_shot_main better** |
| zero_shot vs few_shot_reasoned_main | 4 | 14 | 0.031 | better, but = few_shot_main at higher cost |

**Test (n = 120):** zero_shot 0.450 [0.36, 0.54] vs few_shot_main 0.600
[0.51, 0.68] — **b = 10, c = 28, p = 0.0051.** The dev result replicates out
of sample.

---

## 5 · Error analysis (Part E) — `few_shot_main`, the recommended config

19 imperfect records of 60 (record acc 0.683). Wrong-field tally: **urgency
14**, sentiment 6, escalate 4, category 3. Escalate errors are downstream of
urgency (`escalate = urgency ≥ 4 or "ombudsman"`).

**Top three clusters:**

1. **Urgency 4 → 3 at the escalation boundary (4 cases: T0054, T0230, T0086,
   T0214).** Money/access at risk with no loud deadline — mis-selling refund
   demands and a claim pending 7 days — read as "merely stuck" (3) instead of
   "at risk now" (4). Every one silently flips `escalate` to False. Three of
   the four are the *same template*: *"your agent mis-sold me … I want a full
   refund."* **Highest-impact cluster — it is a routing miss on the tickets
   that most need routing.**
2. **Procedural questions rated as trivial (4 cases: T0167, T0045, T0169,
   T0199).** "What documents do I need for reimbursement", "how do I submit
   post-hospitalisation bills" → urgency 1, gold 2–3. The model treats
   "requires Aurora to act on this account" as a general-knowledge FAQ.
3. **Sentiment `frustrated` ↔ `neutral` when the cue is a prior-failure
   reference, not emotive words (T0128, T0192 frustrated→neutral; T0137
   neutral→frustrated).** "Debited twice … I have the bank statement if you
   need it" is calm in tone but references a repeat failure — the rubric's
   definition of frustrated, which the model misses without emotional
   language.

**E2 — confusion matrix for the worst field, `urgency` (acc 46/60 = 0.767):**

```
        pred 1   2   3   4   5
gold 1:   11   0   1   0   0
gold 2:    2  14   0   0   0
gold 3:    2   1   8   0   0
gold 4:    0   0   4   8   2
gold 5:    0   0   0   2   5
```

The aggregate 0.767 hides a **directional bias at one boundary**: gold-4 is
predicted 3 four times (and never the reverse — gold-3 → pred-4 is 0). The
error is asymmetric and it maps one-to-one onto dropped escalations. Errors
elsewhere are ±1 and symmetric (gold-5↔4). Category, by contrast, is 0.95 —
the only pattern is `complaint → claims` ×2 (Aurora's unpaid dues misread as
a claim because of the hospitalisation context).

---

## 6 · Recommendation

**Ship `few_shot_main`** — the six-example few-shot prompt on `gpt-4.1`.
On held-out `test` it scores **record accuracy 0.600 (95 % CI [0.51, 0.68]),
field accuracy 0.924**, against the `zero_shot` baseline's 0.450 — a paired,
significant **+0.15** (b = 10, c = 28, **p = 0.0051**). It costs **~$5.83 per
1 000 tickets** (≈ **$21,000 / year** at 10 000 tickets/day) versus ~$0.73 for
`zero_shot`, at a steady-state p50 latency of ~1.4 s. The 7.7× cost premium is
justified: at 10 000/day the 15-point record-accuracy gain removes ~550 000
tickets/year from the human-review queue, which pays for the extra ~$18 k of
API at any review cost above ~$0.04 per ticket.

**I would change this recommendation if:** (a) a re-run puts `few_shot_main`
record accuracy below ~0.55, collapsing the margin over baseline; (b)
human-review cost per ticket falls below ~$0.04, making the cheap baseline the
better trade; or (c) the deployment is latency-sensitive (synchronous chat
rather than async triage), where ~1.4 s median — and a rate-limited p95 —
is too slow.

---

## 7 · Negative results

- **Few-shot alone does nothing** (p = 0.79 dev, p = 1.00 test-style) and on
  the small model actively *hurts* category accuracy (0.944 → 0.870, `claims`
  bleeding into `information`).
- **A bigger model alone does nothing** — `zero_shot_main` vs `zero_shot`,
  p = 1.00. Five times the token price for no measurable gain.
- **The reasoning field does nothing** on top of few-shot (identical accuracy,
  +28 % cost) and slightly hurts on the small tier.
- **The cascade does nothing** (p = 1.00): it escalates to zero-shot `MAIN`,
  which is not better than zero-shot `SMALL`; and its disagreement trigger
  catches only 16 % of errors because the model's failures are bias, not
  variance.
- **Urgency did not improve** even in the winning config — 0.767, still the
  worst field, still biased downward at the 3↔4 boundary. Whatever
  `few_shot_main` fixed, it was not the systematic urgency error.
