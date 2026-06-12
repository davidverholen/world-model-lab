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

## Hitting-a-wall protocol (deep search beyond WebSearch; added 2026-06-12)

When a blocking problem needs exhaustive prior-art search (the research-loop
horizon check), layer these:
1. WebSearch — recency + news (what we do by default).
2. **OpenAlex** (primary programmatic; truly open, no key, no affiliation):
   `api.openalex.org/works?search=...&mailto=...` for metadata;
   `?filter=cites:W...` / `cited_by` for citation-graph walks. Weak semantic
   ranking — compensate by walking citations of a known-good anchor paper
   instead of keyword-searching cold.
3. **Semantic Scholar API, anonymous best-effort** — better semantic ranking +
   `/paper/arXiv:<id>/citations|references`; shared anonymous pool 429s under
   load → retry off-peak with backoff, treat as bonus not dependency. (API key
   requires institutional affiliation — gate rejected independent research,
   2026-06-12; re-check occasionally, policies change.)
4. Google Scholar — manual only (no API; ToS); best human tool for cited-by
   exploration of a key paper.

Local index (deferred, trigger recorded): building our own embeddings DB over
external papers duplicates S2 for no curation gain. The version worth building
later is semantic search over OUR corpus (knowledge/ pages + the
sources/files/ PDF archive) once the KB outgrows INDEX-based navigation —
revisit at ~150 pages or when "which of OUR pages covers X" misses start hurting.
