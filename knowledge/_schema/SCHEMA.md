# Knowledge Base Schema

This wiki follows a three-layer architecture (adapted from Karpathy's LLM Wiki pattern,
operated under the context-architecture model from the whitepaper — see
`knowledge/decisions/0002-knowledge-architecture.md`):

1. **Raw sources** (immutable) — papers, posts, docs. Registered in `sources/SOURCES.md`,
   never edited, only referenced.
2. **The wiki** (agent-maintained) — everything under `knowledge/` except `sources/`.
   Synthesis, concept pages, paper pages, comparisons. The agent owns the bookkeeping;
   the human owns meaning, decisions, and risk.
3. **The schema** (this directory) — conventions and processes that make the agent a
   disciplined wiki maintainer instead of a generic chatbot.

## Directory layout

| Directory | Page type | Naming |
|---|---|---|
| `concepts/` | One page per **field** concept (technique, architecture family, phenomenon) — knowledge that holds independent of this project | `kebab-case.md`, noun phrase |
| `design/` | This project's own synthesis — system description, roadmaps, operating strategy (our architecture, capability map, env ladder, compute strategy). `scope: local`; ADRs in `decisions/` record the frozen calls these elaborate | `kebab-case.md`, noun phrase |
| `papers/` | One page per ingested paper | `<firstauthor-or-name>-<year>.md` |
| `labs/` | Research groups and their agendas | `<lab-name>.md` |
| `environments/` | RL environments / benchmarks we can train in | `<env-name>.md` |
| `decisions/` | ADRs — human-owned decisions with rationale | `NNNN-<slug>.md`, numbered |
| `experiments/` | One page per experiment: hypothesis → setup → result → lesson | `NNNN-<slug>.md` |
| `sources/` | Raw-source registry (`SOURCES.md`) + intake queue (`QUEUE.md`) | — |
| `INDEX.md` | Catalog of every page with one-line summary | — |
| `LOG.md` | Append-only journal of ingests, queries, lint passes | — |

## Page anatomy

Every wiki page starts with YAML frontmatter:

```yaml
---
status: stub | draft | current | stale | deprecated
owner: human | agent          # who may change the *meaning* of this page
scope: local | shared          # local = this project only; shared = reusable beyond it
sources: [arxiv:2506.09985]    # verified source ids from sources/SOURCES.md
verified: true | false         # were the key claims checked against the source itself?
last_reviewed: 2026-06-12
---
```

Status semantics (these drive curation, see PROCESS.md):

- **stub** — placeholder from intake; metadata + summary only, paper not yet read.
- **draft** — substantial content, not yet reviewed by the human.
- **current** — reviewed; safe for agents to treat as authoritative.
- **stale** — flagged by lint or contradicted by newer source; do not build on it.
- **deprecated** — superseded; kept for the record, links to successor.

Body sections (in order, omit what's empty):

1. **What it is** — 2–5 sentences, plain language.
2. **Why it matters here** — relevance to *this project's* goal (game-playing world-model agents).
3. **Details** — free-form, but every load-bearing claim cites a source id.
4. **Open questions** — what we don't know yet; feeds the experiment queue.
5. **Links** — `[[wiki-links]]` to related pages. Link liberally; a link to a page that
   doesn't exist yet marks ingestion-worthy material, not an error.

## Authority rules (the bounded-context part)

- `decisions/` and anything `owner: human` — the agent may **propose** edits (as draft
  or PR), never silently change meaning. ADRs are immutable once accepted; supersede instead.
- `experiments/` results — facts; the agent records, never retro-edits outcomes.
- A page with `verified: false` or `status: stub/stale` must **not** be cited as
  authority in code comments, decisions, or other pages' load-bearing claims.
- Source-of-truth precedence: paper itself > paper page > concept page > INDEX summary.
- Local lessons (`scope: local`) do not get generalized into `scope: shared` pages
  without an explicit curation step recorded in LOG.md.
