# Knowledge Base Log

Append-only. Entry format: `## [YYYY-MM-DD] <ingest|query|lint|curation> | <title>`

## [2026-06-12] curation | Knowledge base bootstrapped

Initial setup per ADR 0002 (Karpathy LLM Wiki structure × Context Architecture
whitepaper process). Created: schema (+4 templates), source registry (27 sources, 7
verified), intake queue, 5 concept pages, 4 paper stubs, 1 lab page, 2 environment
pages, 3 ADRs, 1 planned experiment, INDEX.

Research provenance: web verification done for V-JEPA 2 (2506.09985), Dreamer 4
(2509.24527), LeJEPA (2511.08544), When-Does-LeJEPA-Learn-a-World-Model (2605.26379),
AMI Labs funding (TechCrunch 2026-03-09), Karpathy LLM Wiki gist. All other arXiv ids
come from a training-data compilation and are marked `verified: no` in SOURCES.md —
verify at ingest time.

## [2026-06-12] curation | Smoke run confirms collapse hypothesis (exp 0001)

End-to-end pipeline verified on CUDA (`world_model.collect`, reduced settings):
naive joint latent regression collapses (latent_std 0.0005 @ 100 updates).
Preliminary result recorded on experiments/0001; formal run with default settings
still pending. Next: experiment 0002 (SIGReg per [[lejepa-2025]]).

## [2026-06-12] curation | milestone: project bootstrap

Checkpoint protocol adopted (PROCESS.md § Milestone checkpoint, /milestone skill):
after every meaningful milestone, curate the KB first, then commit everything in one
commit. Touched-page lint for this checkpoint: INDEX complete, frontmatter consistent,
known dangling links registered under "Wanted pages" (latent-collapse,
world-models-1803.10122). First commit follows this entry.
