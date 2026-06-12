---
name: context-architecture
description: >
  Set up and operate a curated, agent-maintained knowledge base (Karpathy LLM Wiki
  structure × Context Architecture operating model). SELF-CONTAINED and PORTABLE —
  copy this skill directory into any project to instantiate the same system. Use when
  asked to "set up a knowledge base", "bootstrap the wiki", "context architecture",
  or when a project needs durable, curated memory across sessions.
---

# Context Architecture Knowledge Base

A knowledge system where an agent does the bookkeeping and a human owns the meaning.
Two parents, one hybrid:

- **Structure** from Karpathy's LLM Wiki gist
  (https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f): three layers —
  immutable raw sources, an LLM-maintained wiki of markdown pages, and a schema that
  makes the LLM "a disciplined wiki maintainer rather than a generic chatbot". Three
  operations: ingest, query, lint. The wiki is a persistent, compounding artifact:
  cross-references are already there, synthesis already reflects everything read.
- **Operating model** from the Context Architecture whitepaper (D. Verholen, v1.0.0):
  knowledge is only trustworthy when curated, owned, scoped, reviewed, and kept
  current. Every page carries authority metadata; signals pass a curation gate before
  becoming durable memory; humans stay accountable for meaning, decisions, and risk;
  stale or unowned text must not silently become policy.

Why the hybrid: a wiki that only grows still decays (knowledge drift), and an agent is
more likely than a human to treat stale text as instruction (authority drift). The
metadata + curation gates are what keep a compounding wiki *safe to act on*.

## Part 1 — Setup (new project)

Create this tree (rename `knowledge/` to taste):

```
knowledge/
├── INDEX.md          # catalog: one line per page, grouped by directory
├── LOG.md            # append-only journal of ingests/queries/lints/curation
├── _schema/
│   ├── SCHEMA.md     # layout, page anatomy, frontmatter, authority rules
│   ├── PROCESS.md    # ingest/query/lint procedures + learning-signal routing
│   └── templates/    # one template per page type
├── sources/
│   ├── SOURCES.md    # registry: id | title | url | type | verified | added
│   ├── QUEUE.md      # unread material; NOT citable until ingested
│   └── files/        # optional: local copies of raw sources (gitignore if heavy)
├── concepts/         # one page per concept        (kebab-case.md)
├── papers/           # one page per ingested paper (<name>-<year>.md)
├── decisions/        # ADRs, human-owned, numbered (NNNN-<slug>.md)
└── experiments/      # one page per experiment     (NNNN-<slug>.md)
```

Add domain directories as needed (this project added `labs/`, `environments/`; a
product project might add `domains/`, `customers/`, `incidents/`). A directory earns
existence at ~3 pages; before that, file pages under `concepts/`.

**Frontmatter — every wiki page starts with:**

```yaml
---
status: stub | draft | current | stale | deprecated
owner: human | agent        # who may change the page's MEANING
scope: local | shared       # shared = reusable beyond this project
sources: [arxiv:XXXX.XXXXX] # ids from sources/SOURCES.md
verified: true | false      # key claims checked against the source itself?
last_reviewed: YYYY-MM-DD
---
```

**Page anatomy:** What it is → Why it matters here → Details (claims cite source ids)
→ Open questions → Links (`[[wiki-links]]`; links to not-yet-existing pages are
markers for future work, not errors).

**Authority rules (non-negotiable):**

1. `stub`/`stale`/`verified: false` pages must not be cited as authority anywhere.
2. `owner: human` pages: agent proposes (draft/PR), never silently rewrites meaning.
   Accepted ADRs are immutable — supersede, don't edit.
3. Precedence: raw source > paper page > concept page > INDEX summary.
4. `scope: local` lessons need an explicit curation step (logged) to become `shared`.
5. Unread material lives in QUEUE.md only. The queue may be long; the wiki may not be
   unverified.

**Bootstrap steps:** create tree → write SCHEMA.md + PROCESS.md (copy from this
project or restate the rules above) → templates for concept/paper/decision/experiment →
seed SOURCES.md with whatever the project already trusts → record the adoption itself
as ADR 0001 or 0002 (`owner: human`, get it accepted) → first LOG.md entry → add the
**binding rules** section to the project's CLAUDE.md:

> 1. Start from INDEX.md; load only task-relevant pages (minimum needed context).
> 2. Respect frontmatter (status/owner/verified) as binding.
> 3. Unread material → QUEUE.md, never into pages.
> 4. Route learnings before session end (see PROCESS.md table).
> 5. LOG.md entry whenever the wiki changed.

## Part 2 — Operations

**Ingest** (per source): register in SOURCES.md (id, url, `verified` only after the
URL actually resolved and claims were checked) → read it → paper/source page from
template → update every affected concept page (a good source touches several; update
cross-links both directions) → INDEX line → LOG entry. If you only read the abstract,
the page is `status: stub` and says so.

**Query**: answer from wiki pages first (INDEX → pages), raw sources second. If you
needed the raw source, the page was incomplete — file the finding back. Valuable
explorations become new pages.

**Lint** (periodic; after every ~5 ingests): contradictions; claims newer sources
invalidate (→ `stale`); orphan pages; broken `[[links]]`; pages with
`last_reviewed` > 90 days; concepts mentioned ≥3× without a page; stubs cited as
authority (violation — fix the citer). Lint may downgrade status; only the human
upgrades `owner: human` pages to `current`.

**Learning-signal routing** (the curation gate — run before ending any session that
produced learnings):

| Signal | Route |
|---|---|
| Experiment/work result | experiments/ page (always) + update affected concepts |
| Choice with rationale | propose ADR; human accepts |
| Human correction | fix the wrong page now + LOG entry |
| Recurring confusion | candidate concept page |
| Local trick that looks general | stays `scope: local` until curation promotes it |
| Process lesson | edit `_schema/` (human reviews) |

Not every observation deserves durability. Default: LOG.md (cheap), promote on
recurrence.

**Milestone checkpoint** (curate-then-commit): after every meaningful unit of work —
experiment concluded, capability working, decision accepted — run, in order:
(1) verify the project's checks pass; (2) curate: route learning signals, lint the
pages touched since the last checkpoint, LOG entry
`## [date] curation | milestone: <name>`; (3) commit *everything* in one commit
(`milestone: <name>`, body lists code and knowledge changes separately). The invariant:
every commit carries a knowledge base consistent with its code, and git history
doubles as the project journal. Name milestones like results, not activities.

## Part 3 — Companion pieces

For day-to-day use, pair this skill with thin per-operation skills that just point
here and to `_schema/` (this project: `/ingest-source`, `/kb-lint`,
`/run-experiment`, `/scout-sources`) — keep them small; the schema is the single
source of truth. Optionally add a `curator` subagent for bulk maintenance passes.
Obsidian opens the wiki directly (`[[links]]` + frontmatter work natively); git
provides versioning and review.
