# Knowledge Base Operating Process

The wiki only stays trustworthy if signals are curated, routed, and re-checked.
Three recurring operations (Karpathy's ingest/query/lint, run under the whitepaper's
curation model):

## 1. Ingest (skill: `/ingest-source`)

When a new source (paper, post, repo, talk) enters:

1. Register it in `sources/SOURCES.md` with id, URL, verification status.
2. Read it (or its abstract, marking the page `status: stub`).
3. Write/update the paper page using `_schema/templates/paper.md`.
4. Update every affected concept/lab/environment page — a single paper typically
   touches several pages. Update cross-links in both directions.
5. Update `INDEX.md` (one line per new page).
6. Append a LOG.md entry: `## [YYYY-MM-DD] ingest | <title>` with pages touched.

Intake discipline: unread material goes to `sources/QUEUE.md`, not into wiki pages.
The queue is allowed to be long; the wiki is not allowed to be unverified.

## 2. Query

Answer questions from wiki pages first (INDEX.md → relevant pages), raw sources second.
If an answer required going back to a raw source, that's a signal the wiki page is
incomplete — file the finding back into the page. Valuable explorations become new pages.

## 3. Lint (skill: `/kb-lint`)

Periodic health check (also run after every ~5 ingests):

- contradictions between pages
- claims that newer sources invalidate → mark `stale`
- orphan pages (nothing links to them)
- broken/missing `[[links]]`, including links to pages that should now be created
- pages past review age (`last_reviewed` > 90 days and `status: current`)
- concepts mentioned on ≥3 pages that lack their own page
- stubs that have been cited as authority anywhere (violation — fix the citing page)

Lint output is a LOG.md entry plus fixes or a fix-list. Lint may downgrade status
(current → stale) on evidence; only the human upgrades to `current` for
`owner: human` pages.

## Learning-signal routing (from experiments and sessions)

Work produces signals; signals only become memory through this gate:

| Signal | Route |
|---|---|
| Experiment result | `experiments/NNNN-*.md` page (always), + update affected concept pages |
| Architectural choice with rationale | propose ADR in `decisions/` (human accepts) |
| Correction from the human | update the wrong page now; note in LOG.md |
| Recurring confusion / repeated lookup | candidate for new concept page |
| Local trick that seems general | stays `scope: local` until curation promotes it |
| Lesson about the *process* itself | edit `_schema/` (human reviews) |

Not every observation deserves to become durable knowledge. When in doubt:
record in LOG.md (cheap, append-only), promote later if it recurs.

## Session protocol for agents

- Start: read `INDEX.md`; load only pages relevant to the task (minimum needed
  context — do not bulk-load the wiki).
- During: treat `status`/`verified`/`owner` metadata as binding (see SCHEMA.md).
- End: route any learning signals per the table above; append LOG.md entry if the
  session changed the wiki.

## Literature-first rule (added 2026-06-12, Dave)

Before designing an experiment that attacks a *known class* of problem (collapse,
exploration, interference, ...), check QUEUE/scout for prior art and ingest the key
source first — implement the published protocol, then deviate deliberately. Cheap
diagnostic re-derivations are fine; full experiments against a wall the literature
has already mapped are not. (Origin: exps 0009/0010 re-derived primacy bias before
reading Nikishin; exp 0011 fetched the protocol first and was better for it.)

## Milestone checkpoint (skill: `/milestone`)

A milestone is any state worth returning to: experiment concluded, rung exit
criterion met, new capability working, KB restructure, ADR accepted. After every
milestone, in this order — curate first, commit second, so every commit contains a
knowledge base consistent with its code:

1. **Verify** — `uv run pytest -q` and `uv run ruff check .` pass. Broken states
   don't get checkpointed (WIP commits on a branch are fine, they just aren't
   milestones).
2. **Curate** — the knowledge step:
   - route all learning signals from the work since the last checkpoint
     (table above); experiments get their Result/Lesson sections filled now;
   - touched-page lint: INDEX.md lists everything, statuses/frontmatter on changed
     pages are honest (`last_reviewed` bumped), new `[[links]]` resolve or are
     registered under "Wanted pages", QUEUE/SOURCES consistent;
   - LOG.md entry: `## [date] curation | milestone: <name>` — what changed, what
     was learned, what's next.
3. **Commit** — everything, one commit per milestone:
   - message: `milestone: <name>` + body listing code changes and knowledge changes
     separately (so history doubles as a project journal);
   - nothing stays uncommitted: stragglers either belong to the milestone (include),
     the next one (stash/branch), or nowhere (delete).

Full `/kb-lint` stays a separate periodic pass; the checkpoint lint covers only
pages touched since the last checkpoint.
