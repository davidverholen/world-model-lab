# Training Asset Registry (datasets / pretrained models / corpora)

Heavyweight artifacts for future rungs — registered like sources (verify, then
retain by tier). Retained files live OUTSIDE the repo on the desktop
(`~/minecraft-assets/`); this registry is the map. Rationale: research blob links
and YouTube corpora rot; retention is availability insurance (Dave, 2026-06-12).

## Retention tiers

- **T1 retain now**: small, load-bearing, rot-exposed (weights, indexes, code)
- **T2 retain at rung-5b entry**: bulky but bounded (contractor data subsets)
- **T3 never bulk-retain**: unbounded/link-only corpora (note risk, sample later)

## VPT (OpenAI, MIT license; arXiv:2206.11795 — id verified; repo github.com/openai/Video-Pre-Training)

| asset | tier | status | notes |
|---|---|---|---|
| IDM 4x (weights+model) | T1 | **retained 2026-06-12** → `~/minecraft-assets/vpt/idm/` | the video→actions labeler; crown jewel for the 2b pipeline |
| Foundation models 1x/2x/3x (+ .model archs) | T1 | retained | behavioral-cloning base from 70k h pseudo-labeled video |
| bc-early-game 2x/3x, rl-from-early-game-2x, rl-from-foundation-2x | T1 | retained | fine-tuned + RL agents (diamond-pickaxe lineage) |
| Contractor data index JSONs (v6–v10 + 4 BASALT tasks) | T1 | retained → `vpt/indexes/` | maps of the full datasets |
| Repo snapshot (code, MIT) | T1 | retained → `vpt/repo/` | NOTE: pinned torch 1.9 — porting needed, architecture docs in code |
| Contractor demonstrations (videos+actions+checkpoints) | T2 | NOT retained (~150 GB per BASALT task; v6–v10 sets large) | curate early-game subsets at rung-5b entry; desktop has 1.28 TB free |
| 70k h YouTube corpus | — | **never released** (recipe only) | the IDM + own scraping reproduces the approach if ever needed |

## Other (register + verify at rung-5b entry)

| asset | tier | notes |
|---|---|---|
| MineRL human demo datasets | T2 | served via minerl pip infra; availability has had outages historically — verify + retain at entry |
| MineDojo corpus (wiki/Reddit dumps; YouTube index) | T2/T3 | text dumps retainable (T2); YouTube index is links → rot risk, T3 (sample only) |
| Dreamer 4 / reference checkpoints | T1 at entry | check what's published when we get there |

Re-check availability of all T2 items at rung-5b entry; broken links here are an
early-warning signal worth a LOG entry.
