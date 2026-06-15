---
name: research-cycle
description: >
  Run the autonomous research loop (PROCESS.md) iteratively without human
  supervision: harvest results -> record -> MANDATORY failure-specific literature
  search -> redesign -> pre-register -> implement -> dispatch -> milestone ->
  repeat. Use when the maintainer says "continue the research cycle", "keep researching
  overnight", or similar standing instructions.
---

# Research cycle

This is the NORMAL research loop — identical whether supervised or autonomous.
The only difference: in autonomous mode, do not await the maintainer's go between
iterations (everything else, including every gate and guardrail, is unchanged).

One iteration (repeat until a stop condition):

1. **Harvest**: collect finished runs (local logs/monitors), record results +
   lessons into the experiment page exactly as observed (negatives included).
2. **Literature gate (MANDATORY, 2026-06-12)**: for every new learning —
   *especially every specific failure* — run a targeted search BEFORE designing
   the next iteration: WebSearch for the phenomenon + the hitting-a-wall protocol
   (scout skill: OpenAlex/S2 citation walk from the nearest anchor paper) when the
   failure blocks progress. Queue + ingest the key find at method depth if it
   changes the design. Rationale: exp 0011 was redesigned twice by papers found
   this way; never burn GPU on a wall the literature has mapped.
3. **Redesign + pre-register**: next experiment page with falsifiable hypothesis,
   named counter-outcomes, and the **tagged bar** (`LADDER-EXIT` | `EXTRA-RIGOR`,
   PROCESS.md §Efficiency guardrails); cite what was ingested. If this is a redesign
   of a stalled thread, note the stall count and apply the stall rule below.
4. **Implement minimally** (+ smoke test locally), keep experiment code thin.
5. **Dispatch**: run locally, parallel seeds as concurrent processes on the one
   GPU (<=6-wide; the GPU idles on our latency-bound step — see [[compute-strategy]]),
   arm a Monitor per batch. Keep any single run under ~1 h (laptop thermal cap);
   larger jobs get flagged to the maintainer for a rented-GPU burst.
6. **Milestone** per the contract: pytest/ruff/publish-check, reviewer agent on
   code-bearing diffs, KB curation (INDEX/LOG/routing), commit everything.
7. Repeat from 1.

## Guardrails (autonomous mode)

- **Scope**: stay within the current rung's envs (currently MiniGrid <=6x6 /
  Crafter prep); no new dependencies heavier than a pip package; no env-budget
  inflation beyond the standing protocol without recording justification.
- **Spend**: local GPU only; no cloud rentals without the maintainer's explicit
  go-ahead (flag the cost, don't spend). Nothing leaves the local box.
- **Stall rule (countable, tightened 2026-06-15)**: count consecutive
  pre-registered redesigns whose *headline metric does not move*. At **2**, the
  thread is a wall — a 3rd attempt is allowed ONLY with a written justification
  that it is not a known-class repeat (a genuinely new mechanism or a found paper).
  Otherwise STOP: write a consolidated status note and switch threads or hand back
  to the maintainer. Precedent this would have caught: retention 0013–0015,
  length-2 0042–0044.
- **Wall-relocation tripwire (2026-06-15)**: if a localized wall keeps *moving*
  under successive probes (actor → objective → execution → WM-fidelity, exp
  0040→0044) OR two independent threads converge on the same missing capability
  (e.g. 0031 and 0043 both → hierarchy), the next step is a **committed build of
  that capability**, not another diagnostic — unless a probe is <1 day AND
  decision-changing. Diagnostics localize; past two relocations, build.
- **Honesty**: counter-outcomes recorded with the same care as wins; no frontier
  claims without the deep-search protocol; decisions/schema/results only get proposals.
- **Report**: each milestone commit message is the journal; on session end leave
  a "state of the night" LOG entry: threads advanced, results, what's running,
  recommended next decision for the maintainer.
- **PAID-RESOURCE IMPACT flag (2026-06-12)**: whenever a planned sweep or
  run would gain >~3x wall-clock from rented GPUs (vast.ai), or env throughput
  becomes the bottleneck (Craftax/ADR-0001 revisit trigger), flag it explicitly
  in the report with a cost estimate — spend decisions stay the maintainer's.
