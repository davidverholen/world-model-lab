---
status: current
owner: world-model
scope: shared
sources: [pdf:context-arch-v1, web:karpathy-llm-wiki]
verified: true
last_reviewed: 2026-06-12
---

# 0002: Knowledge architecture — Karpathy LLM Wiki × Context Architecture

**Status:** accepted (2026-06-12)

## Context

The project needs a constantly evolving, curated knowledge structure on world-model
research, maintainable primarily by agents. Two source models: Karpathy's LLM Wiki gist
(three layers: immutable raw sources / LLM-maintained wiki / schema; ingest–query–lint
operations) and the Context Architecture whitepaper v1.0.0 (ownership metadata, status
and review state on every page, curation gates before memory promotion, learning-signal
routing, human accountability for meaning and decisions).

## Decision

Adopt the Karpathy three-layer wiki as the structure and the whitepaper's operating
model as the process, scaled to a solo research project:

- Layout, page anatomy, frontmatter (status/owner/scope/verified): `_schema/SCHEMA.md`
- Ingest/query/lint + learning-signal routing: `_schema/PROCESS.md`
- Bounded contexts kept lightweight: `knowledge/` (curated meaning, this model),
  `src/` (engineering, owned by code review + tests), `experiments/` pages (immutable
  results). Authority boundaries via frontmatter rather than separate wiki areas.
- Human owns: decisions/, page meaning upgrades to `current`, schema changes, risk.
  Agent owns: bookkeeping, cross-references, INDEX/LOG, drafts, lint.

## Consequences

- Every page carries trust metadata; stale/stub pages cannot be cited as authority.
- Slight overhead per ingest (registry + queue discipline) buys auditability and
  reusability: the whole setup is packaged as the `context-architecture` skill for
  future projects.

## Links

[[0003-environment-ladder]] · `_schema/SCHEMA.md` · `_schema/PROCESS.md`
