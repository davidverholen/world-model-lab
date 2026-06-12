---
name: milestone
description: >
  Milestone checkpoint: verify → knowledge curation → commit everything. Use when a
  meaningful unit of work is done ("milestone", "checkpoint this", "wrap this up",
  experiment concluded, feature working, ADR accepted) so that every commit carries
  a curated, consistent knowledge base.
---

# Milestone checkpoint

Follow `knowledge/_schema/PROCESS.md` § "Milestone checkpoint" (authoritative).
Condensed:

1. **Verify**: `uv run pytest -q` && `uv run ruff check .` &&
   `scripts/publish_check.sh` — must pass; otherwise report and stop (no broken
   checkpoints, no personal machine references entering history).
   For milestones with non-trivial code changes, additionally spawn the
   **reviewer** agent (`.claude/agents/reviewer.md`, opus) on the diff since the
   last milestone; fix blockers before committing, record important-but-deferred
   findings in the LOG entry.
2. **Curate**:
   - route learning signals from work since the last checkpoint (PROCESS.md table);
     fill Result/Lesson on any experiment pages this work concluded;
   - touched-page lint: INDEX complete, frontmatter honest (`last_reviewed` bumped),
     `[[links]]` resolve or are queued, SOURCES/QUEUE consistent;
   - LOG.md entry: `## [date] curation | milestone: <name>`.
3. **Commit**: stage everything (`git add -A`), review `git status` for files that
   don't belong (delete or defer them deliberately — nothing stays uncommitted),
   single commit:

   ```
   milestone: <name>

   Code: <summary>
   Knowledge: <pages added/updated, decisions, lessons>
   ```

Name the milestone like a result, not an activity ("naive JEPA collapses on
MiniGrid", not "worked on training").
