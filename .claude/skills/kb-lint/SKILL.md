---
name: kb-lint
description: >
  Health-check the knowledge base: contradictions, stale claims, orphan pages,
  broken links, overdue reviews, missing concept pages. Use when asked to "lint the
  wiki/kb", periodically, or after every ~5 ingests.
---

# Knowledge-base lint

Follow `knowledge/_schema/PROCESS.md` §3 (authoritative). Sweep all of `knowledge/`
(except `sources/files/`) and check:

1. Contradictions between pages (compare claims on linked pages).
2. Claims invalidated by newer sources → mark the page `status: stale` with a note.
3. Orphans: pages no other page links to.
4. `[[links]]` pointing nowhere — either create the page, queue it in INDEX
   "Wanted pages", or fix the link.
5. `status: current` pages with `last_reviewed` > 90 days → flag for review.
6. Concepts mentioned on ≥3 pages without their own page → propose creation.
7. Violations: stub/stale/`verified: false` pages cited as authority → fix the citing page.
8. INDEX.md complete and accurate (every page listed, one line, status current).
9. SOURCES.md ids all used somewhere; QUEUE.md items not cited by pages.

Output: fix what is mechanical (links, INDEX); list what needs judgment. Lint may
downgrade status; only the human upgrades `owner: human` pages. Finish with a LOG.md
entry: `## [date] lint | <n> issues, <m> fixed`.
