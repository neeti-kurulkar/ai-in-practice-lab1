#!/usr/bin/env python3
"""Lab 2 — the configurations under test.

Each variant is a callable `str -> dict`. `grid.py` runs them all through the
same harness, so the only thing that differs between rows of your table is the
thing you intended to differ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pydantic import Field  # noqa: E402

from aip.llm import structured  # noqa: E402
from labs.lab1.extract import (  # noqa: E402
    SYSTEM_PROMPT, TicketRecord, apply_business_rules, extract_deterministic,
)

# ---------------------------------------------------------------------------
# A1 — your six chosen examples.
# ---------------------------------------------------------------------------
# TODO A1: choose 6 dev-set tickets. For EACH, write one line saying what it
#          teaches that prose cannot. Pick edges, not averages (T2 §2.2):
#            - the billing/complaint boundary
#            - a ticket with no policy number (teaches null)
#            - a Hinglish ticket
#            - a satisfied-but-urgent ticket (the sentiment/urgency trap)
#            - a ticket whose policy number is only in a quoted reply
#            - one you got wrong in Lab 1
#
# How these 6 picks map to the archetype list above:
#   - billing/complaint boundary ....... T0097 (billing side) + T0056 (complaint side)
#   - no policy number (null) ........... T0021
#   - Hinglish .......................... T0200, T0021
#   - one you got wrong in Lab 1 ....... T0097, T0021, T0081
#
# Two archetypes were NOT used, and what replaced them:
#
#   (a) "a ticket whose policy number is only in a quoted reply"
#       NOT FOUND: no dev ticket has an AUR-####### that appears only inside a
#       '>' quoted reply (the quotes carry SR-1000xx reference strings, not
#       policy numbers). Also moot here — policy_number is produced by
#       extract_deterministic() in code, not by the model, so a few-shot
#       example cannot move it.
#       REPLACED WITH: T0056 (complaint vs billing/claims boundary — a
#       mishandled grievance, urgency 3 / escalate false).
#
#   (b) "a satisfied-but-urgent ticket (the sentiment/urgency trap)"
#       NOT FOUND: no dev ticket is both satisfied in tone and urgency >= 4;
#       both satisfied tickets (T0222, T0029) are urgency 1.
#       REPLACED WITH: T0222 (satisfied + a question -> sentiment satisfied,
#       urgency 1 — the "tone is not urgency" lesson from the low-urgency
#       side) and T0081 (frustrated + not-blocked defect -> urgency 2).
#
# Closest literal-brief substitutes if you want them: T0238 (reference string
# only in the quote -> null) and T0201 (neutral tone + "Jaldi karo" + deadline
# -> urgency 4).
FEW_SHOT_IDS: list[str] = [
    "T0097",  # teaches: "third time... double debit... ombudsman" is BILLING (the subject is money), not complaint; the ombudsman line drives escalate, not the category
    "T0056",  # teaches: a mishandled grievance is COMPLAINT, but urgency 3 / escalate false — breaks the "complaint ⇒ urgent ⇒ escalate" reflex
    "T0200",  # teaches: "what is X and why does it apply to me?" about a settled claim is CLAIMS, not information — the info-sink error; also hi-en ("Koi solution batayiye")
    "T0021",  # teaches: portal down WHILE at the hospital desk = urgency 5 (emergency in progress); hi-en ("Jaldi karo"); no policy number; product read from "Aurora Gold policy"
    "T0081",  # teaches: an app you are not blocked on = urgency 2, not 4; "reinstalled twice" (a prior failure) = frustrated, not neutral
    "T0222",  # teaches: praise + a question = sentiment satisfied and urgency 1 — tone is not urgency
]

# A hand-written `evidence` span for each example: the quoted phrase that
# decides `category`. Kept under the schema's 200-char cap and shown in the
# same shape the model is asked to produce (evidence first, T2 §3.3).
FEW_SHOT_EVIDENCE: dict[str, str] = {
    "T0097": "double debit on AUR-7548999. Rs 8750 taken twice, nobody has called back in 45 days",
    "T0056": "The grievance I raised 24 days ago on AUR-8916484 was closed without anyone contacting me",
    "T0200": "My claim on AUR-9674338 was settled at Rs 41800 but the hospital bill was much higher",
    "T0021": "Your portal has been down all morning and I am standing at the hospital insurance desk",
    "T0081": "The Aurora app crashes every time I try to upload a document for AUR-5985112",
    "T0222": "Can you confirm the restoration benefit is still available this year on AUR-4940770",
}

# The fields an example record shows, in TicketRecord declaration order.
# `evidence` is prepended from FEW_SHOT_EVIDENCE; needs_human_review and
# review_reason are code-set and never shown.
_FEW_SHOT_FIELDS = ("category", "urgency", "sentiment", "product", "language",
                    "policy_number", "contains_pii")


def load_examples(ids: list[str]) -> list[dict]:
    rows = [json.loads(l) for l in
            (ROOT / "data/eval/extraction_dev.jsonl").open(encoding="utf-8")]
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown example ids: {missing}")
    return [by_id[i] for i in ids]


def _example_reasoning(g: dict, evidence: str) -> str:
    """A short reasoning string for a worked example (Part B).

    Models the discipline we want the reasoned variant to copy: cite the
    deciding span, name the category it implies, then check urgency/sentiment
    against the situation rather than the loudness of the wording.
    """
    return (
        f'The deciding span is "{evidence}", which makes category '
        f'{g["category"]}. Urgency is {g["urgency"]} and sentiment '
        f'{g["sentiment"]}, judged from the situation and tone, not from how '
        f'loudly the message is written.'
    )


def few_shot_block(ids: list[str], with_reasoning: bool = False) -> str:
    """TODO A2: render the examples into the prompt.

    The example output format must be byte-identical to the format you are
    asking the model to produce. A mismatch here is a classic own goal.

    ---
    Implementation: worked `Ticket: ... / JSON: {...}` pairs. The JSON shown
    matches what structured() asks the model to return — `evidence` first
    (T2 §3.3), then the judgement fields, then the two code-checked fields —
    joined with `---`. When `with_reasoning=True` (Part B, few_shot_reasoned)
    a leading `reasoning` key is added so the examples match the
    TicketRecordReasoned output shape too.
    """
    parts = ["Here are worked examples. Produce output in exactly this JSON "
             "format — same keys, same order, nothing extra.\n"]
    for row in load_examples(ids):
        g = row["expected"]
        ev = FEW_SHOT_EVIDENCE.get(row["id"], "")
        record: dict = {}
        if with_reasoning:
            record["reasoning"] = _example_reasoning(g, ev)
        record["evidence"] = ev
        record.update({k: g[k] for k in _FEW_SHOT_FIELDS})
        parts.append(
            f"Ticket:\n{row['input']}\n\n"
            f"JSON:\n{json.dumps(record, indent=2, ensure_ascii=False)}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# The variants
# ---------------------------------------------------------------------------
def _extract(prompt: str, ticket: str, tier: str,
             schema: type = TicketRecord) -> dict:
    """One model call, then the deterministic fields and the business rules.

    Shared by every prompt-only variant so they differ *only* in `prompt`.
    `ticket` is passed separately because the code-side extraction runs on the
    raw ticket, not on a prompt that may also contain few-shot examples.
    Never raises: any model/transport failure degrades to a review record,
    exactly as Lab 1's extract_c() does.
    """
    try:
        fields = structured(prompt, schema=schema, system=SYSTEM_PROMPT,
                            tier=tier).model_dump()
    except Exception as exc:  # noqa: BLE001 - never raise: degrade any failure to a review record
        # Still never raise (matches Lab 1 extract_c), but do NOT fail
        # silently: a rate-limited run that quietly returns stubs looks like a
        # real measurement and is not. Surface it so the grid row is not
        # mistaken for signal.
        print(f"  warn: call failed, using review-record stub -> "
              f"{type(exc).__name__}: {str(exc)[:140]}", file=sys.stderr)
        stub = dict(evidence="", category="information", urgency=1,
                    sentiment="neutral", product="unknown", language="en",
                    needs_human_review=True,
                    review_reason=f"{type(exc).__name__}: {exc}"[:300])
        if "reasoning" in schema.model_fields:
            stub["reasoning"] = ""
        fields = schema(**stub).model_dump()

    fields.update(extract_deterministic(ticket))
    return apply_business_rules(fields, ticket)


def zero_shot(ticket: str, tier: str = "SMALL") -> dict:
    """TODO B: Lab 1 Part C, no examples. This is your baseline.

    `tier` is the only knob — the grid runs this on SMALL and MAIN.
    """
    return _extract(f"Ticket:\n{ticket}", ticket, tier)


def _few_shot_prompt(ticket: str, with_reasoning: bool = False) -> str:
    """The user turn shared by few_shot and few_shot_reasoned: the worked
    examples, then this ticket. Only `with_reasoning` differs between them."""
    return (
        f"{few_shot_block(FEW_SHOT_IDS, with_reasoning=with_reasoning)}\n\n"
        f"---\n\n"
        f"Now output the JSON object for this ticket, in the same format as "
        f"the examples above:\nTicket:\n{ticket}"
    )


def few_shot(ticket: str, tier: str = "SMALL") -> dict:
    """TODO B: zero_shot + the few-shot block.

    Same SYSTEM_PROMPT as zero_shot; the examples go in the user turn, so the
    only difference from the baseline is their presence.
    """
    return _extract(_few_shot_prompt(ticket), ticket, tier)


class TicketRecordReasoned(TicketRecord):
    """TODO B: add a `reasoning: str` field FIRST (T2 §3.3).

    Pydantic keeps declaration order, and field order in the JSON Schema
    influences generation order. Putting reasoning first makes it condition the
    answer; putting it last makes it a post-hoc rationalisation. You want the
    first. Measure the difference in output tokens.

    ---
    Implementation: Pydantic v2 appends subclass fields *after* inherited
    ones, so declaring `reasoning` here alone would put it last. We therefore
    move it to the front of `model_fields` and rebuild, so it leads the JSON
    Schema `properties` and is generated before the answer fields.
    """

    reasoning: str = Field(
        description="Think before you answer. In 2-4 sentences: quote the span "
                    "that fixes `category`, state the category it implies, then "
                    "check `urgency` and `sentiment` against the situation and "
                    "tone. Write this FIELD FIRST, before any answer field.",
    )


# Move `reasoning` to the front: generation order becomes reasoning -> answer.
TicketRecordReasoned.model_fields = {
    "reasoning": TicketRecordReasoned.model_fields["reasoning"],
    **{k: v for k, v in TicketRecordReasoned.model_fields.items()
       if k != "reasoning"},
}
TicketRecordReasoned.model_rebuild(force=True)


def few_shot_reasoned(ticket: str, tier: str = "SMALL") -> dict:
    """TODO B: few_shot with TicketRecordReasoned.

    Identical to few_shot except the schema carries a leading `reasoning`
    field and the worked examples show it too (with_reasoning=True), so the
    example format still matches the requested format.
    """
    return _extract(_few_shot_prompt(ticket, with_reasoning=True), ticket,
                    tier, schema=TicketRecordReasoned)


_CASCADE_FIELDS = ("category", "urgency", "sentiment", "product")


def cascade(ticket: str) -> dict:
    """TODO C: SMALL first; escalate to MAIN on a trigger you choose.

    Triggers, roughly in ascending order of how well they work:
      - validation failed                      (free, weak: misses confident errors)
      - evidence field empty or very short     (free, surprisingly decent)
      - urgency >= 4                           (free, but it is not a confidence signal)
      - two SMALL samples at T=0.7 disagree    (2x small cost, much the best)

    Record which path each ticket took -- set rec['_path'] = 'small' | 'large'
    so grid.py can report the escalation rate.

    ---
    Implementation. Escalate to MAIN when ANY of:
      1. the SMALL call fails validation (structured() raises after repairs)
      2. the SMALL `evidence` span is empty
      3. a SECOND SMALL sample, drawn at temperature 0.7, disagrees with the
         first on category / urgency / sentiment / product

    (3) is drawn at T>0 on purpose. At T=0 the second call is the *same
    request* as the first, the response cache serves it, the two answers are
    byte-identical, disagreement is never seen, and the escalation rate is a
    silent 0.00 (README Part C). T=0.7 changes the sampling AND the cache key.

    Checks 1-2 are done before drawing sample 2, so a ticket that already
    fails them costs one SMALL call, not two.

    rec['_path']        = 'small' | 'large'   -> grid.py escalation rate
    rec['_small_agree'] = True | False | None -> trigger-signal analysis
    rec['_escalate_reason'] = 'accepted' | 'small_failed' | 'no_evidence' | 'disagree'
    """
    prompt = f"Ticket:\n{ticket}"

    # 1. cheap first pass -- identical call to zero_shot(SMALL), reuses cache
    try:
        s1 = structured(prompt, schema=TicketRecord, system=SYSTEM_PROMPT,
                        tier="SMALL")
    except Exception:  # noqa: BLE001 - a raise here just means "not confident"
        s1 = None

    agree: bool | None = None
    if s1 is None:
        reason = "small_failed"
    elif not (s1.evidence or "").strip():
        reason = "no_evidence"
    else:
        # 2. confidence probe: an independent SMALL sample at T>0
        try:
            s2 = structured(prompt, schema=TicketRecord, system=SYSTEM_PROMPT,
                            tier="SMALL", temperature=0.7)
            agree = all(getattr(s1, f) == getattr(s2, f)
                        for f in _CASCADE_FIELDS)
        except Exception:  # noqa: BLE001 - a failed probe -> treat as disagreement
            agree = False
        reason = "accepted" if agree else "disagree"

    if reason == "accepted":
        fields = s1.model_dump()
        path = "small"
    else:
        # 3. escalate
        try:
            fields = structured(prompt, schema=TicketRecord, system=SYSTEM_PROMPT,
                                tier="MAIN").model_dump()
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: cascade MAIN call failed -> {type(exc).__name__}: "
                  f"{str(exc)[:140]}", file=sys.stderr)
            fields = (s1.model_dump() if s1 is not None else
                      dict(evidence="", category="information", urgency=1,
                           sentiment="neutral", product="unknown", language="en"))
            fields["needs_human_review"] = True
            fields["review_reason"] = "cascade: SMALL uncertain, MAIN unavailable"
        path = "large"

    fields.update(extract_deterministic(ticket))
    rec = apply_business_rules(fields, ticket)
    rec["_path"] = path
    rec["_small_agree"] = agree
    rec["_escalate_reason"] = reason
    return rec


VARIANTS = {
    "zero_shot": lambda t: zero_shot(t, "SMALL"),
    "zero_shot_main": lambda t: zero_shot(t, "MAIN"),
    "few_shot": lambda t: few_shot(t, "SMALL"),
    "few_shot_main": lambda t: few_shot(t, "MAIN"),
    "few_shot_reasoned": lambda t: few_shot_reasoned(t, "SMALL"),
    "few_shot_reasoned_main": lambda t: few_shot_reasoned(t, "MAIN"),
    "cascade": cascade,
}
