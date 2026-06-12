---
name: ingest-source
description: >
  Properly ingest a paper, blog post, repo, or talk into the knowledge base
  (knowledge/). Use when asked to "ingest", "read this paper into the KB",
  "add this source", or when processing items from knowledge/sources/QUEUE.md.
---

# Ingest a source

Follow `knowledge/_schema/PROCESS.md` §1 (authoritative). Checklist:

1. **Verify**: resolve the URL/arXiv id (WebFetch). Confirm title/authors match what we
   think it is. Only then set `verified: yes (date)` in `knowledge/sources/SOURCES.md`;
   register the source there (and remove from QUEUE.md if queued).
2. **Read** the source — actually fetch and read it, not just the abstract. If only
   the abstract is feasible now, the page stays `status: stub` with
   `Read state: abstract-only`.
3. **Write the page** from `knowledge/_schema/templates/paper.md` into the right
   directory. Key ideas with section/figure refs; explicit "Relevance to our
   experiments" section.
4. **Propagate**: update every affected concept/lab/environment page (a good source
   touches several) and cross-link both directions. Check INDEX's "Wanted pages" —
   does this ingest let us create one?
5. **Bookkeep**: INDEX.md line, LOG.md entry (`## [date] ingest | <title>`, pages touched).

Authority reminders: claims cite source ids; this page may not cite stubs/stale pages;
never edit `owner: human` pages without proposing.
