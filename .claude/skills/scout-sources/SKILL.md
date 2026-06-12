---
name: scout-sources
description: >
  Search for new world-model research (papers, releases, code) and feed the intake
  queue. Use when asked to "find new papers", "what's new in world models",
  "check AMI labs", or periodically to keep the KB current.
---

# Scout for sources

Goal: find and *verify existence of* new material; ingestion is a separate step
(`/ingest-source`). Never write findings directly into wiki pages.

1. **Watch-list** (search these first): AMI Labs / LeCun / Balestriero (LeJEPA line);
   Meta FAIR JEPA family; Hafner / DeepMind Dreamer + Genie; Wayve GAIA; NVIDIA
   Cosmos; TD-MPC line; arXiv listings for "world model" in cs.LG/cs.AI; the
   awesome-list github.com/LMD0311/Awesome-World-Model. Also check open items in
   `knowledge/sources/QUEUE.md` marked unverified.
2. **Verify** every candidate: WebFetch the arXiv/page, confirm title + authors.
   No unverified ids enter the registry as verified. Batch trick (2026-06-12): one
   call to `export.arxiv.org/api/query?id_list=<id1>,<id2>,...` verifies dozens of
   ids at once (id→title match) — use it for backlog sweeps; it caught a wrong id
   that pointed at an astronomy paper.
3. **Triage** into `knowledge/sources/QUEUE.md` (high/medium/low) with a one-line
   why-it-matters. Register clearly-relevant items in SOURCES.md (`verified` per
   what you actually checked).
4. **Report** back: what's new, what it might change in our current pages (flag
   pages that may now be `stale` — don't mark them without checking), what to
   ingest first. LOG.md entry: `## [date] scout | <n> found, <m> queued`.
