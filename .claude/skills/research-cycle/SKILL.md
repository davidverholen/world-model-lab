---
name: research-cycle
description: >
  Run the autonomous research loop (PROCESS.md) iteratively without human
  supervision: harvest results -> record -> MANDATORY failure-specific literature
  search -> redesign -> pre-register -> implement -> dispatch -> milestone ->
  repeat. Use when Dave says "continue the research cycle", "keep researching
  overnight", or similar standing instructions.
---

# Autonomous research cycle

One iteration (repeat until a stop condition):

1. **Harvest**: collect finished runs (remote logs/monitors), record results +
   lessons into the experiment page exactly as observed (negatives included).
2. **Literature gate (MANDATORY, Dave 2026-06-12)**: for every new learning —
   *especially every specific failure* — run a targeted search BEFORE designing
   the next iteration: WebSearch for the phenomenon + the hitting-a-wall protocol
   (scout skill: OpenAlex/S2 citation walk from the nearest anchor paper) when the
   failure blocks progress. Queue + ingest the key find at method depth if it
   changes the design. Rationale: exp 0011 was redesigned twice by papers found
   this way; never burn GPU on a wall the literature has mapped.
3. **Redesign + pre-register**: next experiment page with falsifiable hypothesis,
   named counter-outcomes, and the standing bar; cite what was ingested.
4. **Implement minimally** (+ smoke test locally), keep experiment code thin.
5. **Dispatch**: remote desktop, parallel seeds (remote.sh shell pattern,
   <=6-wide), arm a Monitor per batch.
6. **Milestone** per the contract: pytest/ruff/publish-check, reviewer agent on
   code-bearing diffs, KB curation (INDEX/LOG/routing), commit everything.
7. Repeat from 1.

## Guardrails (autonomous mode)

- **Scope**: stay within the current rung's envs (currently MiniGrid <=6x6 /
  Crafter prep); no new dependencies heavier than a pip package; no env-budget
  inflation beyond the standing protocol without recording justification.
- **Spend**: desktop GPU only; no cloud rentals; nothing outside the repo +
  desktop.
- **Stall rule**: if the same failure mode survives 2 consecutive redesigns with
  no new literature insight, STOP that thread, write a consolidated status note,
  and switch to the next queued thread (retention -> ignition/exploration ->
  text staircase prep) or pause with a summary for Dave.
- **Honesty**: counter-outcomes recorded with the same care as wins; no frontier
  claims without the deep-search protocol; owner:human pages only get proposals.
- **Report**: each milestone commit message is the journal; on session end leave
  a "state of the night" LOG entry: threads advanced, results, what's running,
  recommended next decision for Dave.
