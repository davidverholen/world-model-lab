# Knowledge Base Log

Append-only. Entry format: `## [YYYY-MM-DD] <type> | <title>`

## [2026-06-16] scout | OpenReview 6fDZYJYYgu — identified as SimuRA (Deng, Hou, Hu & Xing); queued

OpenReview forum id 6fDZYJYYgu: API returned empty notes (double-blind embargo likely); forum
WebFetch returned a plausible but unverifiable description. Cross-verified via WebSearch + arXiv
fetch: the id belongs to arxiv:2507.23773 "SimuRA / General Agentic Planning Through Simulative
Reasoning with World Models" (Deng, Hou, Hu, Xing; Jul 2025, v3 May 2026) — same authorship
and theme as "Critiques of World Models" (arxiv:2507.05169, already queued). SimuRA is an
LLM-orchestration paper (LLM as world model, natural-language belief states, System II planning)
— not a latent-predictive RL paper. No RSSM/imagination/manual-reading/intrinsic-motivation
content. Verdict: broad-context read, low priority relative to our active wall. 1 source added
to SOURCES.md; 1 entry appended to QUEUE.md (LLM-as-latent-backbone / PAN section).

## [2026-06-16] experiment | exp0051 budget probe — flat-VR ~0.10 is a REAL ceiling (not compute-bound)

Post-hierarchy, optimizing the FLAT validated-reading architecture on the 2-step tutorial. Budget probe
(coef 0.3, 2x training = 40 rounds/800 ac-updates vs the 30r/400u 0.105 baseline): length-2 swap_follow
0.115 — FLAT, no take-off → the ~0.10 grounding ceiling is real, not compute-bound at this scale
(correct rose 0.073→0.115, but grounding pinned). Lever = stronger/denser grounding incentive (the
maintainer's insight: the incentive is what works; every acting level needs it). Running coef 0.5 (cheap
dose-response gap-fill, low-odds). On-deck: decoupled actor-VR (own reward head, actor optimizes
task+λ·VR_head — dedicated reality-judged grounding signal on the acting policy). exp0051 page added.

## [2026-06-16] experiment | exp0050d VQ autoencoder ALSO negative — hierarchy thread CONCLUDED

Built Director's load-bearing VQ-VAE goal autoencoder (the principled fix for the diagnosed
uninformative-goal-space cause; reviewer-cleared). Result: correct 0.005, swap_follow 0.000 across 4
seeds — STILL zero. VQ healthy (vq_used 23-30/64, recon ~0.45) but worker_sim stayed ~0.5 (not goal-
directed even in the learned latent) + manager collapsed on 2/4 seeds. Hierarchy now gives ~0 across
BOTH goal representations (raw-belief + VQ) and every stability fix (adv-norm, grad-clip, flat-collect,
hindsight, success-bias). CONCLUSION: hierarchy concluded negative for this wall. Leading read (well-
motivated): Director hierarchy is the WRONG TOOL for a 2-step gesture — nothing to decompose; it solves
long-horizon decomposition, our wall is short-horizon grounding+credit-assignment. RECOMMEND: bank
validated-reading (exp0048, the confirmed 0->0.10 objective-gap crack) as the rung-4 deliverable; if
execution is revisited, attack it as a short-horizon problem (manual-directed planner / better short-
horizon credit-assignment / scaling), not hierarchy. All hierarchy code banked (reviewer-cleared).
Surfaced to maintainer. GPU free; tree clean after milestone.

## [2026-06-16] experiment | exp0050 hierarchy STALLED (2 attempts) — handed back to maintainer

Phase A v2 (advantage-norm + grad-clip + --hier-flat-collect) ALSO NEGATIVE: length-2 correct/swap_follow
0.00 on all 4 seeds even with clean flat-collection (good WM); worker_loss still oscillates ±30-90
(policy collapse). Two attempts (v1 divergent, v2 stable-but-zero), headline did not move off ~0 → STALL
RULE: stop, hand back, no 3rd attempt without justification. Consolidated diagnosis: the minimal first
cut (raw belief-space goals + K-means codebook + cosine worker reward, NO learned goal autoencoder) is
the likely culprit — exactly Director's load-bearing ablation (raw-RSSM cosine goals are uninformative;
the worker never learns to reach subgoals). Hand-back options for the maintainer: (A) build the real
VQ-VAE goal autoencoder (principled, bigger); (B) recursive decompose-or-execute MPC over the VR-WM
(no trained manager, sidesteps co-training instability); (C) bank validated-reading (the confirmed
0->0.10 objective-gap crack) and pause hierarchy. NOT Phase B (manual-directed manager) — the executor
must work first. Hierarchy code (reviewer-cleared) banked. Autonomous loop PAUSED at this fork. Nothing
running; GPU free; tree clean after milestone.

## [2026-06-16] experiment | exp0050 Phase A v1 DIVERGENT → v2 (stability fixes) running

The Director hierarchy built today (4 commits, reviewer-cleared) ran Phase A: NEGATIVE as a BUG, not a
clean test. Partially worked at length-1 (correct→0.35) then collapsed+diverged at length-2 (correct
0.00 all rounds vs flat 0.073; worker_loss exploded to ±80 = unbounded PG; manager imagined macro_r
0.3-0.7 vs real ~0 = imagination exploitation). Compounded by the hierarchy driving its own collection
from round 3 → poisoned the buffer once degenerate. NOT "grounding is the ceiling". Fixes → Phase A v2
(running): normalize worker+manager advantages + clip_grad_norm(100) (kills divergence; v2 smoke
worker_loss bounded); --hier-flat-collect keeps the flat VR-actor collecting competent data while the
hierarchy trains in imagination over the good WM (isolates execution from collection-poisoning). If v2
still ~= flat → Phase B (manual-directed manager); if v2 diverges → escalate to maintainer (recursive
decompose-or-execute fallback). 0050 page updated with the v1 result+diagnosis.

## [YYYY-MM-DD] <ingest|query|lint|curation> | <title>`

## [2026-06-16] decision | exp0050 pre-registered — manual-directed hierarchy (Director x manual) to break the ~0.10 ceiling

Maintainer chose the full hierarchy build (over a minimal manual-directed-MPC tweak) — "we wanted to
build this anyway" ([[hierarchical-imagination-agent]] is its home). Lit gate (scout): Director (anchor,
ingested), FuN (manager/worker origin, continuous-vs-discrete goal trade-off), HAC (co-training
instability + hindsight-relabel fix — the top risk), THICK (ICLR2024, hierarchy-in-WM; Director more
portable for us). Novelty: the manual-DIRECTED manager (subgoals soft-conditioned on read text) is
relatively novel; the Director-hierarchy-in-WM and Dynalang-style language-WM binding are not. Key
design call (scout): TEST HIERARCHY WITHOUT THE MANUAL FIRST (Phase A) to isolate credit-assignment vs
grounding, THEN add manual conditioning (Phase B). Minimal first cut: goal-conditioned worker
(similarity reward, K=8 imagination) + manager MLP over a K-means RSSM-state codebook (no full VQ yet),
pre-seeded from the VR-trained checkpoints (which reach the gesture endpoint ~10%). Stability designed in
(hindsight relabel, separate buffers, Dreamer alternation). exp0050 pre-registered with phased
implementation plan + named counter-outcomes. NEXT SESSION = the careful build (reviewer-gated). Nothing
running; tree clean.

## [2026-06-16] scout | hierarchy / manual-directed subgoals lit gate (exp0050 design) — 4 found, 4 queued

Focused failure-specific literature gate for the manual-directed Director build. Four clusters
searched: (1) language-conditioned manager subgoals, (2) subgoal representation trade-offs,
(3) co-training instability, (4) hierarchy over learned WM post-Director. Verified via arXiv API
batch + abstract fetch. Findings: FuN (1703.01161) and HAC (1712.00948) are the mandatory
background for subgoal space and co-training stability respectively; THICK (openreview:TjCDNssXKU,
ICLR 2024) and Hieros (2310.05167) are the post-Director hierarchy-in-WM papers. Novelty verdict:
the exact combination of manual/language-DIRECTED manager subgoal proposals inside a learned WM
appears unpublished — the two axes (language→subgoal and hierarchy-in-WM) exist separately but not
combined with anti-baking conditioning. QUEUE.md: new section "Hierarchy / manual-directed subgoals
(exp0050 design; scouted 2026-06-16)" added above the existing "Hierarchy / subgoal emergence"
section. SOURCES.md: 4 entries added (FuN, HAC, Hieros, THICK). Pages potentially affected:
papers/director-2022.md (open questions now partially addressable from FuN/HAC/THICK reads),
knowledge/INDEX.md (new SOURCES entries — no INDEX change needed, scout doesn't create pages).

## [2026-06-16] experiment | exp0049 NULL — VR cracked the objective gap; ~0.10 is the EXECUTION ceiling → design fork

Shaping-floor A/B (α=0.3, floor 0.2 vs the floor=0 control, 4 seeds): NO lift — swap_follow 0.094 vs
0.105, flat. Length-2 swap_follow is a robust **~0.10** across α∈{0.1,0.3}, floor∈{0,0.2}, AND MPC
(0.06–0.08 ≈ reactive — exp0044 reconfirmed on the VR-trained WM). Two-part conclusion: (1) the
validated-reading reward SOLVED its target — moved swap_follow from a flat 0.00 (all 0042–0047, the
[[0040-rtfm-actor-conditioning]] objective gap) to a confirmed ~0.10, anti-baking clean (the actor now
follows the displayed text); (2) the new ~0.10 ceiling is the EXECUTION wall (exp0043/0044) re-met from
above — invariant to reward strength, scaffold, and search, so it is "can't reliably execute the 2-step
gesture," not "won't obey." KEY insight: reward-maximizing MPC can't break it because swap_follow is
scored in SWAPPED mode where obeying earns no reward → a reward-maximizer won't follow the text. Past
~0.10 needs manual-DIRECTED execution (plan toward what the manual predicts, not toward reward) =
[[hierarchical-imagination-agent]] territory. This is a genuine DESIGN/SCOPE FORK → autonomous loop
PAUSES here (per guardrail: next step isn't a clear one-flag experiment). Hand-back to maintainer with
3 options: (A) manual-directed execution / hierarchy build, (B) scale up (compute — rented GPU, the
maintainer's earlier idea; ~0.10 may climb with training), (C) accept VR as a confirmed partial result
and bank the rung. 0049 finalized (verified). Nothing running; tree clean after milestone.

## [2026-06-16] experiment | exp0048 4-seed control — WEAK CONFIRM (phase-1 was inflated)

The 4-seed control (α=0.3) at the comparable final window (rounds 25–29): length-2 swap_follow mean
**0.105** (range 0.05–0.13), ALL 4 seeds >0 vs the α=0 control's flat 0.00, swapped≈0 (anti-baking
clean every seed), events 5.8 (~2× control). This CLEARS the pre-registered bar (>0 across seeds, mean
≳0.10–0.15) at its lower edge → the validated-reading reward is the FIRST confirmed lever to move
length-2 swap_follow off zero in the whole rung-4 thread. HONEST CORRECTION: phase-1's single-seed
numbers were inflated — partly 1-seed variance, partly GPU non-determinism (re-running *seed 0* gave
0.09 not 0.18). Magnitude is ~half what phase-1 advertised, and the oracle_pct elevation (0.92) WASHES
OUT to baseline (0.88) across seeds → that claim retracted. Verdict: effect REAL but UNDERPOWERED (a
chip, not yet a crack). 0048 finalized (verified). Methodology lesson reinforced: never trust 1-seed +
GPU runs; the multi-seed control did exactly its job tonight (caught a 3× inflation). Next:
[[0049-rtfm-sustained-vr]] — can it be made strong (shaping-floor so VR replaces the annealing scaffold
+ ridge), multi-seed from the start. (Also running: exp0048ctl01 = α=0.1 × 2 seeds, a comparison point.)

## [2026-06-16] experiment | exp0048 validated-reading PHASE-1 POSITIVE — first lever to move length-2 swap_follow off zero

The validated-reading intrinsic reward WORKS at 1 seed. Phase-1 coef sweep (5×1, fixed seed): the
sweet-spot VR arms (α=0.1, 0.3) lift held-out length-2 `swap_follow` from a flat 0.00 (every prior
experiment 0042–0047) to **0.16–0.18** with `swapped`=0.00 (clean anti-baking), while the α=0 control
stays dead-zero. Coherent across FIVE metrics on the same arms: swap_follow↑, events 2.8→7.4 (~2.6×
control, the flywheel turning), oracle_pct→0.92 (beats the entire 0044–0047 calibration thread that
broke trying to push it there directly), oracle_ret ~2×, length-1 swap_follow ~2.6×. Clean inverted-U
dose–response (α=1.0 dark-rooms — highest raw reward, task collapses), and the effect GREW over the
length-2 phase (c03 0.10→0.18, peaks 0.25) rather than fading. The right lever (reward validated
reading) did gracefully what the wrong levers (direct reward calibration) broke themselves on; the
exp-0040 objective gap is addressable by changing the OBJECTIVE, not the architecture. Reality-as-judge
keeps it anti-baking by construction. CAVEAT: 1 seed — the cross-arm dose–response can't be seed-luck
(shared seed) but the trajectory could be a favourable draw → 4-seed control at α=0.3 (`runs/exp0048ctl`)
is RUNNING; phase-1 = candidate until it lands. Wrote 0048 Result + compare plots (swap-follow/events/
oracle), pre-registered [[0049-rtfm-sustained-vr]] (push past the plateau: α-ridge 0.2–0.5 + shaping
floor), captured the [[mentored-learning-loop]] north-star design note. INDEX updated. Dashboards:
sweep on :8000, live control on :8001.

## [2026-06-15] milestone | exp0047 NEGATIVE closes calibration thread → exp0048 validated-reading built + dispatched

exp0047 (conservative reward head, CROP/CQL) concluded NEGATIVE: 6-coef sweep showed monotonic
over-suppression (oracle_ret +0.13→−0.13 as α 0.1→3.0), no coef beats the 0.88 baseline — conservatism
can't separate OOD junk from the sparse true gesture in our starved regime. This CLOSES the
execution/calibration thread (0044–0047); the durable signal `swap_follow≈0` is the [[0040-rtfm-actor-conditioning]]
objective/identifiability gap, not execution. Wall-relocation tripwire fired → redirect to the
objective. Designed [[validated-reading-reward]] (intrinsic reward for the manual's marginal next-state
predictive value, validated by the real transition — reality as judge, anti-baking + un-wireheadable;
marginal framing dodges dark-room/noisy-TV). Lit gate (scout): VIME (IG formula), Marino-2020
(act-to-verify, real-env judge — closest neighbour), ICM (foil), RND (anti-wirehead); novelty:
relatively novel combination. Ingested VIME + Marino. Built exp0048 (`validated_reading_reward` +
`--validated-reading-coef`, computed at collection vs `deter_noctx` baseline, detached/no_grad). 42
tests pass, reviewer (opus) SHIP no-blockers. Dispatched a coef sweep overnight; swap_follow is the
headline (does it finally lift off 0). Also queued Aleph/Kona (energy-based reasoning, LeCun/Freedman)
+ CQL/CROP papers from the prior thread.

## [2026-06-15] ingest | VIME + Marino (hypothesis-verification) — exp0048 validated-reading lit anchors

Ingested two papers at METHOD depth as the literature anchors for exp0048 (validated-reading intrinsic
reward). Both sources were already verified in SOURCES.md and queued in the "Intrinsic-motivation /
validated-reading" section of QUEUE.md; no re-verification performed.

- **papers/vime-2016.md** (arxiv:1605.09674, Houthooft et al., NeurIPS 2016): VIME — information-gain
  intrinsic reward via BNN dynamics model. Method depth on: the KL(posterior||prior) over BNN
  parameter space as the intrinsic reward formula; mean-field variational inference approximation of
  the posterior (diagonal Gaussian, SGVB updates); why this is an information-gain signal rather than
  a prediction-error signal (the ICM inversion); dark-room and noisy-TV partial-fix properties.
  Relevance section maps the connection to exp0048: VIME is the single-source IG ancestor; our
  mechanism is VIME applied contrastively across the manual condition (marginal IG). Gaps documented:
  no language/reading, single-source not contrastive, no marginal framing.

- **papers/marino-hypothesis-2020.md** (arxiv:2006.15762, Marino et al., 2020): act-to-verify
  hypothesis verification. Method depth on: the pre-condition/action-sequence/post-condition triplet
  structure; two-phase policy (setup to reach pre-condition, then execute); resolving reward +-1 gated
  on real-environment post-condition (not a model); anti-wireheading property (environment is the
  incorruptible judge). Relevance section includes a structural comparison table (Marino vs exp0048
  mechanism on 7 dimensions). Gaps documented: no language/reading step, no marginal-value framing,
  no contrastive dual pass, resolving (+-1) vs confirming (clip>=0).

Both pages cross-link [[icm-2017]] and [[rnd-2018]] as the foil (prediction ERROR = inverted sign)
and anti-wirehead pattern (frozen external judge) respectively, using unresolved wiki-link notation
since those pages do not yet exist. Cross-links to [[validated-reading-reward]], [[0048-rtfm-validated-reading]],
[[vime-2016]], and [[marino-hypothesis-2020]] in the existing design/experiment pages now resolve.

Pages changed: papers/vime-2016.md (created), papers/marino-hypothesis-2020.md (created),
sources/QUEUE.md (two [x] marks in the Intrinsic-motivation section), INDEX.md (two lines added
under Papers).

## [2026-06-15] scout | intrinsic-motivation / validated-reading lit gate — 6 found, 6 queued (exp0048 design)

Targeted search for the proposed "intrinsic reward = manual's marginal predictive value, confirmed by
acting." Search covered: ICM/curiosity prediction-error, VIME/Plan2Explore information-gain,
Oudeyer-Kaplan learning progress, empowerment (Klyubin / Mohamed-Rezende), Friston dark-room /
active inference, and specifically: any prior art on intrinsic reward for instruction-following /
manual-derived prediction confirmation. Full analysis in scout return message (2026-06-15).

QUEUE.md: new section "Intrinsic-motivation / validated-reading (exp0048 design)" added with 4 HIGH
and 2 MEDIUM items; 3 context items cross-referenced without duplication. SOURCES.md: 6 entries added
(all ids verified 2026-06-15 via arXiv API or abstract fetch). No wiki pages edited.

NOVELTY VERDICT: the specific mechanism (intrinsic reward = real-environment confirmation of a
manual-derived dynamics prediction, with marginal-value framing) appears RELATIVELY NOVEL. The closest
prior art is arXiv:2006.15762 (Marino et al. 2020 — hypothesis verification via acting), which shares
the "act to test a claim about dynamics" structure but lacks the language-reading / manual-derived
prediction component and the marginal-value framing. The full combination has not been found.

Pages potentially affected: knowledge/experiments/0047-rtfm-conservative-reward.md (provides forward
context for the next experiment design), any future exp0048 page.

## [2026-06-15] ingest | CQL + CROP (conservative reward/value) — exp0047 lit anchor

Ingested two papers at METHOD depth as the literature anchor for exp0047 (conservative reward head):

- **papers/cql-2020.md** (arxiv:2006.04779, Kumar et al., NeurIPS 2020): conservative Q-learning;
  method depth on the α·(E_{a~μ}[Q] − E_{a~π_β}[Q]) regularizer, logsumexp variant, the offline
  distributional-shift problem, and portability of the regularizer to a reward head (substituting
  R_hat for Q, one-sided relu for sparse rewards).
- **papers/crop-2023.md** (arxiv:2310.17245, Li et al., 2023 preprint): CQL push-down applied
  directly to the learned REWARD estimator in MBRL; L_CROP objective (MSE + α·E_{OOD}[R_hat]);
  conservative Q lower-bound theory; D4RL dense-reward evaluation; open questions on sparse-reward
  regime and push-down target (uniform-random vs replay-supported); exp0047 online variant documented
  (detached belief, one-sided relu clamp, α=3.0, M=16).

Both pages include a "Relevance to our work" section connecting to exp0025, exp0045, exp0046, and
exp0047. The [[cql-2020]]/[[crop-2023]] links already present in 0047-rtfm-conservative-reward.md
now resolve to the new pages.

QUEUE.md: two items marked [x] (ingested). INDEX.md: two lines added under Papers. No concept stub
created (see decision note below).

Decision note — conservative/pessimistic value learning concept stub: NOT created this session.
Rationale: the field concept is real and linkable (CQL, CROP, MOPO, COMBO, MOReL, CBOP, COPlanner
all belong to it), but (a) several of those anchors are still queued (MOPO, COMBO, COPlanner), so
a concept page written now would be partially sourced and would need immediate stub status; (b) the
two paper pages carry enough cross-linking to serve exp0047 without a concept page; (c) the right
time to write concepts/conservative-value-learning.md is after the remaining cluster (MOPO, COMBO,
COPlanner) is ingested and exp0047 is harvested. Flagged for the next curation pass.

## [2026-06-15] scout | reward-head OOD conservatism lit gate — 7 found, 5 newly queued (QUEUE + SOURCES updated)

Targeted search triggered by exp0045/0046 failure: reward head overrates ~12% of OOD action sequences;
sampled-rollout MPC only partially corrects (oracle pct 0.88→0.92, not 1.0). Root diagnosis: sparse
positives (~5/60) + no training-time OOD penalty. Search question: what does the literature say about
conservative/pessimistic reward/value heads that push down estimates on OOD action sequences?

**Papers found and verified (all ids confirmed via arXiv API batch fetch or abstract fetch):**

1. arxiv:2006.04779 CQL (Kumar et al., NeurIPS 2020) — foundational OOD conservatism: push-down Q on
   random actions, push-up on data. Directly portable to a REWARD head.
2. arxiv:2005.13239 MOPO (Yu et al., NeurIPS 2020) — already queued; verified id + formula:
   r_tilde = r_hat - λ * max_i ||Σ_φ^i(s,a)||_F. Targets reward at step level via ensemble.
3. arxiv:2005.05951 MOReL (Kidambi et al., NeurIPS 2020) — was queued without id; id now verified.
   Pessimistic MDP that absorbs policy into penalty when leaving data support.
4. arxiv:2102.08363 COMBO (Yu, Kumar et al., NeurIPS 2021) — CQL-style push-down on value function
   for model-generated (OOD) rollouts; no ensemble uncertainty required.
5. arxiv:2210.03802 CBOP (Jeong et al., ICLR 2023) — was queued without id; id now verified.
   Bayesian posterior lower-bound on value expansion; weights model-free/model-based by uncertainty.
6. arxiv:2310.17245 CROP (Li et al., 2023 preprint) — NEW. The direct "conservative REWARD head"
   paper: trains reward estimator to simultaneously minimize prediction error AND predicted reward on
   random actions (CQL-style push-down on reward head, not Q). The closest published instantiation
   of our exp0047 target design.
7. arxiv:2310.07220 COPlanner (Wang et al., 2023) — NEW. DreamerV3 + uncertainty-aware MPC:
   dynamics uncertainty penalizes reward during imagined rollouts (conservative imagination) while
   acting as exploration bonus in real env. Direct Dreamer-family analogue to our planning-time
   pessimism lever.

**SOURCES.md:** 8 rows added (CQL, MOReL, MOPO, COMBO, CBOP, CROP, COPlanner — all verified).
**QUEUE.md:** MOReL entry updated with verified id (2005.05951); CBOP entry updated with verified id
(2210.03802) + MOPO formula added; new section "Reward-head OOD conservatism" added with CQL (HIGH),
CROP (HIGH), COPlanner (HIGH), COMBO (MEDIUM), triaged in priority order for exp0047 design.

**Wiki pages that may now be stale or require updates:**
- knowledge/design/hierarchical-imagination-agent.md §5 (calibration; CROP/CQL mechanisms directly
  relevant to the conservative reward head design flagged there)
- knowledge/experiments/0046-rtfm-robust-planning.md lesson "next" section (now has a concrete
  lit anchor: CQL+CROP for the reward-head regularizer, COPlanner for the planning-time penalty)
- Any concept page on "reward modeling" or "value calibration" (none currently exist; candidate
  for creation once CQL/CROP are ingested)

**Single most actionable mechanism:** CROP (arxiv:2310.17245) = CQL push-down applied directly to
the reward head. Implement: add α * E_{a~uniform}[R_hat(z,a)] - E_{a~replay}[R_hat(z,a)] to the
world-model reward-prediction loss. This requires zero architectural change (reward head already
exists as a two-hot distributional head in our WM); OOD actions are sampled uniformly or from the
MPC candidate set; α is the single tunable hyperparameter. Ingest CROP before coding exp0047.

## [2026-06-15] experiment | rung-4 exp0046 — sampled-rollout MPC: PARTIAL (lever real but under-powered)

Ran the pre-registered robust-planning v1: K=10 sampled-rollout MPC (average) vs the reward head's
prior-mean optimism (`--mpc-rollout-samples`, commit 7901aaa; 4 seeds, 30 rounds, ~1h45m local).
Result is a **partial positive**: oracle pct 0.88 → ~0.92 (all seeds >0.90, tight band), MPC correct
0.05 → ~0.11 (~2× floor), and `swapped` ≈ 0 so anti-baking holds cleanly — the predicted direction,
confirming averaging shaves *some* OOD false positives. But it does NOT clear the bar: oracle pct
plateaus ~0.91, not ~1.0, and MPC correct is noisy/seed-dependent. The collection-signal panel
explains why — length-2 positives are starved (~5/60), so the reward head never gets data to sharpen
OOD calibration; robust *planning* can only discount optimism the head already shows, not fix a
starved head. Per the pre-registered counter-outcome → escalate by COMPOSING v1 with pessimism
(two-hot spread) / continue-gating / reward-head OOD regularization (the root fix). Wrote
0046 page with trajectory plots (oracle-probe, mpc-eval, collection-signal); INDEX updated. The oracle
probe stays the readout for every next variant (works iff pct → 1). Empirically reinforces the
design's §5 calibration dependency ([[hierarchical-imagination-agent]]).

## [2026-06-15] SESSION HANDOFF | rung-4 execution thread → next = robust-planning / reward-calibration lever

**Pick-up point for a fresh session.** Read order: this entry → experiments 0043/0044/0045 →
design/hierarchical-imagination-agent.md (§5) → decisions/0007.

**The arc this session (all committed, working tree clean, 38 tests pass, nothing running):**
reading-to-learn-dynamics is DEMONSTRATED (exp0036, swap-tested). Then the multi-step-EXECUTION wall:
exp0043 (WM reads a length-2 gesture, reactive actor can't execute) → exp0044 (CEM-MPC ≈ reactive,
planning alone doesn't crack it) → exp0045 oracle probe DECISIVE: the WM READS AND VALUES the true
gesture (oracle pct ≈ 0.88, return 5–6× random) — NOT a fidelity wall — but ~12% of wrong sequences
are overrated (reward-head OOD false positives) and the planner chases them. **= exp0017–0025
imagination-exploitation at the PLANNING layer.** Architecture's read→imagine→value chain is intact.

**EXACT next experiment (exp0046): a robust-planning / reward-calibration lever (NOT a new WM).**
Tools: (a) pessimistic/value-aware MPC — penalize predicted reward by WM uncertainty (two-hot spread /
ensemble), use the continue head, average over SAMPLED rollouts instead of the prior-mean argmax;
and/or (b) reward-head OOD regularization so the gesture becomes ~argmax. Re-run the exp0044 MPC A/B
+ the `--oracle-probe`; SUCCESS = oracle pct → 1 AND MPC `correct` lifts off the floor (~0.05). All
machinery exists: `agents/rtfm_mpc.py` (planner), `--mpc-eval`/`--oracle-probe` flags, curriculum+aux
config from exp0044/0045 (`dispatch_rtfm.sh exp0046 4 --rounds 30 --curriculum-rounds 10 ... --length
2 --one-shot --reading-shaping-coef 1.0 --manual-aux-coef 1.0`).

**Parked for the maintainer (owner pages — propose, don't rewrite):** ratify decisions/0007
(Crafter-mastery milestone) + design/hierarchical-imagination-agent.md (both proposed/draft).

**Playable artifact:** best watchable agent = rung-3 Crafter `runs/_legacy/crafter_rew_s2.pt`
(`uv run python -m world_model.play --checkpoint runs/_legacy/crafter_rew_s2.pt`) — survival breadth,
not deep tree. The rtfm reading agents are NOT wired into `play` (would need text-enc + ConditionedRSSM
+ manual rendering); optional small build if a visual of the reading agent is wanted.

## [2026-06-15] curation | exp 0045: oracle probe — WM reads+values the gesture; wall is reward-head OOD calibration

exp0045 (`--oracle-probe`, commit 692c233): scores the TRUE 2-action gesture (privileged manual_facts,
eval-only) vs 200 random sequences by the same planner objective, reports the oracle's percentile.
Result: oracle pct ≈ 0.82–0.93 (stable ~0.88), oracle return ~5–6× random — so the WM genuinely
READS AND VALUES the gesture; NOT a fundamental fidelity wall (which would be pct≈0.5). But pct is
0.88 not 1.0: ~12% of (wrong) sequences are scored ABOVE the only sequence that earns reward =
reward-head FALSE POSITIVES on OOD action sequences. Naive MPC maximizes → chases those overestimates
→ exactly why exp0044 MPC matched the reactive actor. This is the exp0017–0025 imagination-exploitation
problem at the PLANNING layer (planner exploits reward-head OOD overestimation). Architecture's
read→imagine→value chain is intact; gap = planner robustness to reward-head OOD false positives.
Empirically motivates the calibrated-uncertainty/confidence-gating flagged in the hierarchical-
imagination design §5. Next (calibration/robust-planning lever, NOT a new WM): reward-head OOD
regularization to make the gesture ~argmax, and/or pessimistic/value-aware planning + sampled rollouts;
re-run the MPC A/B expecting oracle pct→1 and MPC correct lifting. Maintainer checkpoint.

## [2026-06-15] curation | workflow retro → efficiency guardrails + remote-GPU retired

Neutral retro of how we work (188 commits / 44 experiments / 17 sessions, all 06-12→06-15),
done via two analysis subagents (transcripts + experiment record). Findings: the experiment
engine is strong (pre-registration ~universal, healthy ~49% null rate, swap-tests/probes catch
false wins, honest losses) but the meta-loop leaks — ~2 experiments slow to leave a wall
(retention 0013–0015; length-2 0042–0044), answers walls with another diagnostic instead of
building the capability the diagnostics keep pointing at (hierarchy, flagged by both 0031 and
0043), and carries a heavy workflow tax (commit/verify cadence, post-compaction file-state
desync = #1 wasted-call class). Acted on it:
- **Efficiency guardrails** (PROCESS.md, new section + CLAUDE.md rule 7 + research-cycle skill):
  countable stall rule (2 flat redesigns ⇒ wall), wall-relocation tripwire (diagnose twice then
  build), `LADDER-EXIT`|`EXTRA-RIGOR` bar tagging (added to experiment template), lit-first
  mandatory for borrowed mechanisms, pinned per-rung eval protocol, session-hygiene (fresh
  session per milestone — the KB is the durable memory).
- **Remote-GPU retired**: removed scripts/remote.sh, docs/REMOTE.md, .env.remote*, the win32
  torch index in pyproject.toml, sweep.py --remote. ADR 0004 superseded; CLAUDE.md hardware +
  compute-strategy + director/dreamer4 pages now read "local + on-demand rented cloud GPU"
  (maintainer-triggered spend). Measured desktop benchmarks kept as reference data.

## [2026-06-15] curation | exp 0044: MPC ≈ reactive at length-2 — planning doesn't crack it; wall relocates to WM rollout fidelity

exp0044 (CEM-MPC over the manual-conditioned WM vs the reactive actor, same trained WM, length-2,
commit c679a4a). Final: reactive mean correct 0.06, MPC 0.09 — MPC consistently but MARGINALLY ahead
(≥reactive on 3/4 seeds across rounds), within eval noise (n=20), and NEITHER cracks length-2 (both
~0.05–0.10 vs length-1's 0.4). WM reads throughout (inv_ratio 1.10–1.23). Verdict = the 2nd
pre-registered branch: swapping a search planner for the reactive policy barely moves the needle, so
the binding constraint is NOT the policy's credit assignment but the WORLD MODEL's multi-step
rollout/reward FIDELITY — the belief reads the manual but the imagined 2-step payoff isn't sharp
enough to plan through. Implication: WM short-horizon fidelity is UPSTREAM of all planning/hierarchy
(every planner inherits it as a ceiling) — sharpen WM predictive/reward accuracy before more planning
machinery. Echoes exp0017–0025 (imagination calibration is the recurring limiter) + the design's §5
calibration dependency. Confounds to rule out first (prior-mean vs sampled rollout, no value tail,
shaping-shaped reward head, receding replan). Next: oracle-gesture reward-prediction probe to localize
WM-fidelity vs MPC-search — maintainer checkpoint (meatier fork than another rtfm lever).

## [2026-06-15] curation | schema change: `owner:` is now a domain; discussion de-attributed

Two schema/voice changes (no research content altered). (1) The `owner:` frontmatter
field now names the owning **domain** (`world-model` | `crafter-rtfm`), not an
authorship role — everything in this repo is `owner: world-model` (88 pages + 4
templates). The old human/agent owner distinction is gone; the
edit-protection it used to gate is re-anchored on page **type/status** instead:
`decisions/` (ADRs, immutable once accepted) and experiment **results** (facts) are
propose-only; `current` design/concept meaning changes via a logged curation step.
SCHEMA.md authority rules, PROCESS.md, CLAUDE.md, the curator/ingest/kb-lint/
research-cycle/context-architecture skill+agent docs, and inline LOG references all
updated to match. (2) Discussion in the documents is now framed as **project thoughts**,
not attributed to a person/role: ~190 "the maintainer"/"(maintainer)" attributions
across LOG.md and the wiki bodies were de-attributed (decisions stated as the project's;
"(maintainer)" credit tags dropped). Genuinely operational human-in-the-loop references
(delegation authorization, autonomous-mode go/spend guardrails) are kept. Verify: 38
tests pass, ruff clean, publish_check clean.

## [2026-06-15] curation | design doc: hierarchical imagination agent (dual-process planning) — drafted from a design conversation

New `design/hierarchical-imagination-agent.md` (draft — proposed for review). The
agent-side design for the next major capability after reading, targeting the multi-step-EXECUTION wall
(exp0043) and ADR-0007. Architecture co-designed in a design session: **System-1 reactive default**
(amortized policy, automatic, no language) / **System-2 deliberate fallback** (recursive read-grounded
decomposition + imagination/MPC, sketched in language) / **compilation** (System-2 distills into
System-1 — "done 100× → automatic", reuses exp0016 actor-distillation) / **confidence-gated
arbitration** (escalate when the WM can't confidently imagine a path to the goal → fetch language
read/LLM → re-plan). Key principles: depth via RECURSIVE decomposition (HTN), not fixed N-levels;
the recipe DAG is READ, not learned (Director's VQ-manager = fallback); text is the System-2 *sketch*
interface (read now, frozen/commodity LLM far seam) NOT the runtime default; reading becomes
demand-driven (read your knowledge gaps). LOAD-BEARING RISK called out: arbitration needs CALIBRATED
uncertainty — the exp0017–0025 overconfidence dragon — else an overconfident WM never asks for help.
Near-term staircase pinned to exp0044 (CEM-MPC executes the length-2 leaf the reactive actor couldn't),
then compile-back, then 1-level decomposition on a Crafter state-chain. Folds into ADR-0007 later
pillars. Next: pre-register + run exp0044.

## [2026-06-15] ingest | Director: Deep Hierarchical Planning from Pixels (Hafner, Lee, Fischer, Abbeel — NeurIPS 2022, arxiv:2206.04114)

Verified: arXiv abstract page confirms exact title, all four authors, submission 2022-06-08, NeurIPS 2022
main conference. Source registered in SOURCES.md. Two QUEUE entries marked ingested (temporal-abstraction
thread + hierarchy/subgoal thread). Paper page created at papers/director-2022.md (method depth).

Key mechanism captured: DreamerV2 RSSM backbone; goal autoencoder is a VQ-VAE compressing RSSM states to
discrete codes; manager selects goal code every K=8 steps; worker conditioned on decoded goal feature vector
and trained on cosine/feature similarity reward (no task reward); manager trained in imagination on task
reward + reconstruction-error exploration bonus; all training in imagined RSSM rollouts. Why raw latents fail:
high-dimensional continuous goal space → intractable search and instability (ablation confirmed).

Results: competitive on dense-reward tasks (Control Suite, Atari, DMLab); qualitative gain on sparse
long-horizon tasks — Ant Maze XL (egocentric camera, no global position) solved by Director, failed by flat
DreamerV2 and Plan2Explore; Visual Pin Pad solved by large margin.

Pages touched: papers/director-2022.md (created), sources/SOURCES.md (new row), sources/QUEUE.md (two [x]
marks), INDEX.md (new Papers entry), concepts/hierarchy-and-credit.md (Director section updated + [[director-2022]]
link), concepts/temporal-abstraction.md (table row updated with link).

Design section in paper page covers: (a) feasibility at our scale — FEASIBLE on 16 GB desktop (RSSM backbone
already in stack; VQ-VAE + second actor-critic is modest overhead; 8× longer imagination horizon is
wall-clock not memory); (b) reading-grounded manager — two strategies: hard manual→goal-code substitution
(simpler, no manager RL) vs soft manual-conditioned manager (learns language→codebook); (c) minimal first
experiment framed as exp 0044 pre-registration candidate: Director-style hierarchy on rtfm length-2 WITHOUT
reading conditioning first (pure credit-assignment isolation test), then add reading conditioning in follow-up.

## [2026-06-15] curation | exp 0043: DECISIVE — length-2 is a PURE EXECUTION wall (WM reads it, actor can't do it)

exp0043 (length-2 curriculum + aux ON to read inv_ratio): at length-2, inv_ratio = 1.13–1.25 on
every seed — INDISTINGUISHABLE from the length-1 phase (1.13–1.27) — so the WM reads the 2-step
gesture just as well; but correct collapses ~0.35→0.05. Confounder-free localization: **the WM reads
the multi-step recipe, the actor cannot execute it.** Closes the diagnosis with 4 converging results
(swap_follow ~0.25 / len-2 cold fail / len-2 curriculum fail / len-2 reads-but-can't-execute):
**reading is SOLVED; multi-step/compositional EXECUTION is THE wall.** Unification: this is the SAME
bottleneck as the rung-3 Crafter depth plateau (exps 0026–0031: actor/discovery-bound, not
perception) — plain Crafter stalls on the deep tech tree, rtfm stalls on the 2-step gesture; both are
multi-step-execution / credit-assignment / HIERARCHY. The next major capability (hierarchy / temporal
abstraction / goal-conditioned credit assignment) pays off on BOTH fronts at once; echoes Dreamer-4's
per-subtask staging. PARK rtfm length-scaling (reading demonstrated). Next = strategic call
on the execution/hierarchy bet + ADR-0007 ratification (this is its later-pillar work).

## [2026-06-15] curation | exp 0042: length curriculum fails — the wall is multi-step EXECUTION, not reading

exp0042 (length-1→2 curriculum, --curriculum-rounds 10 then length-2, commit 991ed32): the len-1
warm-up ignites (2/4 seeds ground, swapped≪correct — replicates exp0036), but at the switch to
length-2 `correct` collapses 0.35→0.05 and never recovers over 20 length-2 rounds. 2nd failed
length-2 lever (after exp0041 cold-start). Converges with the parked swap_follow result (0037–0040)
into ONE diagnosis: **the WM reads (length-1 grounds, inv_ratio>1), but the actor under-executes** —
swap_follow ~0.25 even at length-1, and a 2-step gesture won't ignite cold OR warm-started. Bottleneck
= multi-step/compositional EXECUTION of read content, not reading. This is on the ADR-0007 critical
path (Crafter deep tree is all multi-step). exp0042 had aux OFF so no inv_ratio at length-2 → can't
yet tell reading-compositionality vs execution apart. Next: exp0043 (length-2 curriculum + aux ON) to
localize the wall (inv_ratio>1 + correct≈0 ⇒ pure execution failure). Then the fix targets execution
= the milestone's later pillars (goal-conditioning, hierarchy) — bigger bets (open).

## [2026-06-15] curation | public-ready pass: owned-GPU model names and personal profile URL removed

Mechanical depersonalization pass (no research content changed). Owned GPU model names
genericized in all remaining tracked files → "laptop GPU" / "desktop GPU" (capacity
numbers preserved where inline). The special-case 2026-06-14 LOG entry's parenthetical
model names were reworded to "(owned laptop/desktop GPUs)". The personal profile URL in
sources/SOURCES.md replaced with the public GitHub repo URL for
the context-architecture whitepaper. Historical LOG narrative "profile page" phrasing
neutralized to "the author's profile page". Rental/market GPUs (4090, 5090, H100, 3090,
vast.ai) left untouched throughout. 14 files changed; `last_reviewed` dates not bumped
(mechanical edits, not content re-review).

Files changed: knowledge/LOG.md, knowledge/INDEX.md, knowledge/sources/SOURCES.md,
knowledge/decisions/0001-pytorch-over-jax.md, knowledge/decisions/0004-remote-dispatch.md,
knowledge/experiments/0001-naive-latent-regression.md,
knowledge/experiments/0002-sigreg-anti-collapse.md,
knowledge/experiments/0003-probe-protocol-8x8.md,
knowledge/experiments/0004-mpc-agent-empty8x8.md,
knowledge/experiments/0005-doorkey-memory.md,
knowledge/experiments/0006-value-head-doorkey.md,
knowledge/experiments/0008-doorkey6x6-vs-ppo.md,
knowledge/experiments/0009-ignition-mechanics.md,
knowledge/experiments/0010-retention-mechanics.md,
knowledge/experiments/0019-stochastic-latents.md,
.claude/skills/run-experiment/SKILL.md.

## [2026-06-15] ingest | Dreamer 4: Training Agents Inside of Scalable World Models (Hafner, Yan, Lillicrap, 2025) → papers/dreamer4-2025.md (method depth)

Trigger: high priority (direct successor to DreamerV3; transformer WM, shortcut forcing,
offline imagination RL, and Minecraft diamonds all bear on our rung-3 decision and ADR-0007).
Source arxiv:2509.24527 already verified + registered in SOURCES.md (2026-06-12). Primary source:
TalkRL podcast transcript with Danijar Hafner (direct author); secondary: Harold Benoit technical
writeup + EmergentMind summary + unofficial PyTorch implementation README. PDF compressed/inaccessible.

Pages changed:
- REPLACED knowledge/papers/dreamer4-2025.md stub → method-depth draft. Sections: full
  transformer WM architecture (tokenization, spatial/temporal factored attention, context window,
  action conditioning, comparison to RSSM); shortcut forcing mechanism (step-size conditioning,
  x-prediction vs v-prediction, ramp weighting, 16× speedup from 64→4 denoising steps);
  offline agent training pipeline (3 phases: WM pretraining → BC finetuning with agent tokens →
  imagination RL with PMPO + KL-to-BC prior); compute/data table; explicit strategic analysis for
  our program on all three questions (RSSM vs transformer, shortcut forcing adoption,
  offline-imagination-to-diamond implications for ADR-0007).
- UPDATED knowledge/INDEX.md — promoted dreamer4-2025.md line from stub to draft with
  substantive description.
- UPDATED knowledge/sources/QUEUE.md — marked dreamer4 ingested in High priority section.
- UPDATED knowledge/concepts/imagination-training.md — expanded Dreamer 4 entry in lineage
  section with technical detail; promoted verified: false → true + bumped last_reviewed.

Key findings for our program:
1. RSSM is NOT obsolete at our scale. The transformer WM (2B params, 256–1024 TPUs, H100 for
   inference) is a scale result; the ideas (sparse temporal attention, x-prediction) are
   portable but the architecture itself isn't home-lab viable. The small-transformer WM line
   (arxiv:2502.01591, arxiv:2605.16457) is the right comparison before any rung-3 pivot.
2. Shortcut forcing does NOT subsume our exps 0017–0025 fixes. It addresses denoising artifacts
   in diffusion WMs; our fixes (continue predictor, critic-on-replay, two-hot) address
   actor-calibration in RSSM WMs. The problems are at different layers; both sets of fixes remain
   valid in their respective architectures.
3. The offline-imagination-to-diamond pipeline validates ADR-0007's structure, specifically the
   multitask achievement-hierarchy training (their ~20 subtasks = our directable-competence goal
   conditioning). Their KL-to-BC prior serves the same exploitation-guard role as our
   critic-on-replay, but is only available when an offline dataset with BC prior exists.

## [2026-06-15] scout | DeepMind world-model landscape 2025-2026: 4 verified items, 3 queued

Scout triggered by a possible breakthrough video (noted, pending verification). Searched the Genie family,
Dreamer line, and adjacent players. Findings:

- Dreamer 4 (arxiv:2509.24527) — already in SOURCES.md (added 2026-06-12). REAL paper, Hafner+Yan+
  Lillicrap, Sep 2025. First agent to get Minecraft diamonds from offline data. Relevant to our stack.
- Genie 2 (Dec 2024, blog-only) — added as web:genie2-blog to SOURCES.md. Generative pixel WM, no
  paper, no planning substance. NOTE TO: hype / not our architecture.
- Genie 3 (Aug 2025, blog-only) — added as web:genie3-blog to SOURCES.md. Real-time 720p/24fps,
  no paper. Generative pixel WM + SIMA 2 integration. Substance is real but it is a video generator,
  not a latent-predictive planner; not relevant to our architecture.
- SIMA 2 (arxiv:2512.04797, Nov 2025) — added to SOURCES.md and QUEUE.md (Medium). Real paper.
  Gemini agent + Genie 3 worlds + self-generated task/reward loop. Closest published
  instantiation of "agent trains in dreamed worlds" thesis, but at pixel not latent level.

No "Genie 4" or other 2026 DeepMind WM announcement found. The likely video noted was
either Genie 3 (Aug 2025) or the SIMA 2 + Genie 3 demo. See final report for substance breakdown.

## [2026-06-15] curation | ADR 0007 PROPOSED: Crafter mastery = "directable competence" (rung-3 exit milestone)

From a design conversation. Set the major milestone: master Crafter (the
purpose-built 2D env) before advancing to 3D/Phase 4 — no-rung-skipping (ADR-0003). "Finish" defined
as DIRECTABLE COMPETENCE: reliably reach ANY target achievement on demand (per-achievement
success-rate across all 22, deep tree included), NOT aggregate score (gameable by easy-achievement
farming) nor all-22-in-one-episode (impossible — survival vs deep-crafting compete for episode time).
Mechanism = the thesis in 3 pillars: dynamics-manual (read the rules) + goal-conditioning (the target,
NOT yet built) + imagination (plan through the grounded WM) → execution. Surpassing DreamerV3 on the
deep tree IS the thesis ablation (Dreamer has no reading). Two tiers: WE set goal = directable
(measurable milestone); ACTOR sets goal = autonomous (stretch). Staircase with per-rung
swap/ablation checkpoints so we never grind blind. Drafted as knowledge/decisions/0007 (status:
proposed — ratification pending); ROADMAP Phase 3 refined to point at it. Also dispatched a scout for
a possible recent DeepMind world-model breakthrough (noted from a video, pending verification).

## [2026-06-15] STATE OF THE NIGHT | rung-4 reading-to-learn-dynamics: DEMONSTRATED (genuine) + swap_follow strengthening PARKED

**Headline: first genuine reading-to-learn-dynamics on this stack.** From the exp-0035 sparse-reward
ignition wall, the autonomous loop went: HO-0007 (crafter-rtfm shipped dense reading-gated shaping,
`reading_shaping_coef`) → exp0036 IGNITES genuine reading (3/4 seeds correct≫none AND swapped≪correct
on held-out, no LRS-brittleness shortcut; skill persists at coef=0). HO-0006 + HO-0007 both ACCEPTED.

**Then a 4-experiment strengthening investigation of the one weakness — swap_follow only ~0.25
(reads enough to be disrupted by a wrong manual, but doesn't reliably EXECUTE the displayed gesture):**
- exp0037 (2× training): NULL — not undertraining.
- exp0038/0039 (Dynalang masked-manual auxiliary, coef 1 & 3; new code `manual_aux.py`, anti-baking
  `deter_noctx` + invariance diagnostic): the aux GROUNDS the WM (inv_ratio 0.87→1.1–1.4, all seeds
  ground, worst seed fixed) but swap_follow stays flat → bottleneck is NOT WM reading.
- exp0040 (actor-conditioning, §6.2 fallback; new code `ConditionedActor`): does NOT lift swap_follow
  (mean ~0.18, high variance, 2 seeds collapse, 1 seed best-ever 0.40); no baking.

**Diagnosis (3 architectural levers exhausted → it's the OBJECTIVE):** swap_follow ~0.25–0.40 is
bounded by correct-mode reward training, where "follow the displayed manual" ≡ "do the true recipe"
(the reward can't separate them; they diverge only at the swap). Reward-RL has no obedience pressure;
a scripted reader hits 1.0 because it's hardwired to obey. So this is a task/objective question, not
a model defect. **PARKED per the stall rule.** Call (see experiments/0040 §Decision):
(1) partial-anneal the shaping (retain follow-displayed pressure), (2) train on mixed/swapped modes
(needs a crafter-rtfm handoff), or (3) accept ~0.3 as "partial grounding," declare the reading thesis
demonstrated, and move to length-2 / next rung. **My recommendation: (3)** — the reading result is
secured; swap_follow→1.0 is a separate instruction-OBEDIENCE question.

**Committed this session:** wm 51199b3 (exp0034 base-reward fix + housekeeping), 4696a71 (exp0036 +
Dynalang ingest), 0b72bcd (aux infra), 209e19b (exp0037/0038), 43a77c5 (actor-cond infra), + this
milestone; commons 93385fb/2edaa76 (HO-0006/0007 accepted). New infra (aux, actor-cond) is flag-gated
OFF — exp0036 recipe is the default. Also: lit-gate (RTFM/Messenger/Dynalang/LRS-brittleness queued;
Dynalang ingested method-depth). Cross-domain subagent autonomous-mode idea recorded to memory.
**Update — length-2 ladder step also tested (exp0041):** the validated length-1 recipe does NOT
generalize to length-2 (events 3–6/round vs 12–27, correct ~0, no grounding). Length-1 is the
current ceiling. This is the lit-backed curriculum case (RTFM/Messenger both needed complexity
staging) — a length-1→length-2 curriculum is the principled next step but a new design direction for
sign-off required, not another autonomous lever. **Stopping here: clean wall reached.**
Three items now wait for decision: (a) swap_follow objective fork; (b) length curriculum for
length-2; (c) stale-remote-docs cleanup (touches accepted ADR-0004 → propose, don't rewrite).
**Nothing running; GPU idle; all work committed.**

## [2026-06-15] curation | exp 0038: masked-manual aux grounds the WM, but swap_follow is ACTOR-bound

Two swap_follow-strengthening levers (exp page 0038). exp0037 (2× training, 40 rounds): NULL —
swap_follow unchanged ~0.25, same scatter; more training is not the lever. exp0038 (Dynalang
masked-manual auxiliary ON, --manual-aux-coef 1.0, commit 0b72bcd): the aux is a GENUINE grounding
signal — inv_ratio (wrong/correct manual reconstruction loss) rose 0.87 (untrained) → 1.1–1.4, i.e.
the belief is manual-specific, NOT the copy-through failure (deter_noctx anti-baking held). All 4
seeds now ground (s0, which FAILED in 0036, recovers grounding 0.00→0.30); swapped≪correct clean
everywhere. BUT swap_follow did NOT break its ~0.25–0.35 ceiling. Re-diagnosis: strengthening the
WM's reading (proven via inv_ratio) does not move swap_follow → the bottleneck is the ACTOR's
execution, not WM reading. Likely root cause: correct-mode-only training, where "follow displayed
manual" ≡ "do true recipe" (identifiability gap the reward can't separate). Stall-rule: swap_follow
resisted 2 levers, but 0038 gave a NEW diagnosis (actor-bound), so next redesign targets the actor
(stronger-aux disambiguator exp0039, then actor-conditioning §6.2), not a stalled repeat. Next: exp0039.

## [2026-06-14] curation | exp 0036: reading-shaping IGNITES genuine (partial) reading-to-learn-dynamics → HO-0007 accepted

The HO-0007 dense reading-gated shaping (crafter-rtfm 809d44f, `reading_shaping_coef`, annealed
1.0→0, eval at c=0) broke the exp-0035 ignition deadlock. exp 0036 (4 seeds, length-1, held-out):
tutorial events climbed 0–6 → 12–27/round and HELD at coef=0 (skill persists without the
training-wheels); on 3/4 seeds `correct ≫ none` AND **swapped ≪ correct** (s1 0.00/s2 0.10/s3 0.15
vs correct ~0.4) — genuine content-reading, the signal exp 0033 failed. The LRS-brittleness shortcut
(swapped≈correct) did NOT fire. FIRST genuine reading-to-learn-dynamics result on this stack. Caveat:
grounding is PARTIAL — swap_follow 0.15–0.40 (vs scripted reader 1.0), s0 didn't ground. → HO-0007
ACCEPTED (env mechanism validated; strengthening swap_follow is our side). Next: strengthen
swap_follow (more-training test, then Dynalang-aux / actor-conditioning), re-verifying swapped≪correct.

## [2026-06-14] ingest | Dynalang: Learning to Model the World with Language (Lin et al., 2023) → papers/dynalang-2023.md (method depth)

Trigger: rung-4 lit gate (HO-0007 sparse-reward ignition wall). Source arxiv:2308.01399
already verified + registered in SOURCES.md. Full HTML paper read (two passes).

Pages changed:
- CREATED knowledge/papers/dynalang-2023.md — method depth including: exact RSSM
  conditioning (token-by-token streaming, concatenation not cross-attention, T5-small per
  token), full aux-loss formulas (β_reg=0.1, β_pred=0.5, language recon at unit weight),
  training regime (online RL, optional text pretraining, no curriculum), Messenger results
  (qualitative; beats EMMA on S3, no published table), dense-gradient mechanism (language
  reconstruction fires at every token timestep, decoupled from sparse task reward), and
  CRITICAL ANALYSIS of static-manual mismatch with four candidate adaptations.
- UPDATED knowledge/sources/QUEUE.md — marked Dynalang ingested in three places
  (Rung-4 anchor, Minecraft milestone, Language/imagination sections).
- UPDATED knowledge/INDEX.md — added dynalang-2023.md entry under Papers.
- UPDATED knowledge/concepts/language-grounding.md — added [[dynalang-2023]] wiki-link
  in installation architectures table and LLM-as-aligned-source section.

Key finding for HO-0007 decision: Dynalang's dense-gradient mechanism (stream text
token-by-token through RSSM; reconstruct from belief) does NOT transfer directly to our
static-manual / cross-attention design. The copy-through triviality risk is real: if M is
already a cross-attention input, the decoder can route M to output without the RSSM
encoding anything meaningful. Recommended analog: masked manual reconstruction from belief
with cross-attention blocked at decode time (option A in the paper page), combined with
the curriculum shaping from HO-0007. See [[dynalang-2023]] Critical analysis section.

## [2026-06-14] scout | rung-4 reading-ignition lit gate: 9 found, 8 queued (High), 1 background

Triggered by HO-0007 (sparse-reward ignition wall). Anchored on RTFM/Messenger citation walk;
supplemented with S2+OpenAlex leads. Sources registered: arxiv:1910.08210 (RTFM), 2101.07393
(Messenger/EMMA), 2308.01399 (Dynalang), 2511.22904 (LED-WM), 2210.00066 (LDD), 2305.16621
(LRS brittleness), 2110.10661 (SILG), 1707.01495 (HER), icml:ng1999shaping (Ng et al.).
New queue section: "Rung-4: reading-ignition literature gate". Cross-references updated in
Minecraft/Language sections. Pages potentially affected: experiments/0035*, design/architecture-strategy.md,
concepts/grounded-language-game.md (if it exists). Ingest order: RTFM → Messenger → Dynalang → LED-WM → LRS-brittleness.

## [2026-06-14] curation | exp 0034 (one_shot re-test): base-reward farming fixed → sparse-reward ignition wall → HO-0007

Re-ran the rung-4 manual-conditioned agent under crafter-rtfm's `one_shot` fix (HO-0006, which
killed the within-episode SEARCH leak 0033 exposed). Two findings, recorded in
experiments/0034-rtfm-oneshot-ignition.md. (1) **Confound:** the env step reward `r` MIXES
farmable base-Crafter achievements with the sparse tutorial bonus (both 1.0); `collect_rtfm` trained
on `r`, so the agent farmed readingless base reward (all-zeros eval while ~120 "events"/round).
**Fixed:** train on tutorial-only reward `len(info["tutorial_newly"])` so reading is the sole reward
source. (2) **Residual wall:** with the honest reward, one_shot length-1 is too SPARSE to ignite —
reward events collapse to chance (0–6/60, no climb), no gradient toward reading, eval stays at
chance across all rounds (run runs/exp0035, 4 seeds). one_shot is correct (search/no-text provably
fail; a scripted reader still grounds, swap_follow=1.0) so **HO-0006 ACCEPTED** — but a from-scratch
LEARNER can't bootstrap. → **HO-0007 opened** (requirement: a reading-gated training curriculum/
shaping signal that's dense enough to bootstrap without reopening the search leak; eval stays honest
one_shot + swap_follow on held-out). Also: reward fix + run housekeeping (runs/<exp>/s<seed>.{pt,log}
convention, scripts/dispatch_rtfm.sh, *.log gitignored). Next: await crafter-rtfm on HO-0007, re-test.

## [2026-06-14] ingest | Achievement Distillation (Moon et al., NeurIPS 2023) → papers/achievement-distillation-2023.md (full method depth: achievement segmentation rule, intra/cross-trajectory InfoNCE losses with formulas, Crafter per-achievement results, RSSM portability note)

## [2026-06-14] ingest | Curious Replay for Model-based Adaptation (Kauvar et al., ICML 2023) → papers/curious-replay-2023.md (method depth; priority formula, update rule, Crafter results, frozen-encoder applicability)

## [2026-06-14] curation | milestone: repo made public-ready — leaks scrubbed, README opened for third parties

Publication-prep pass (on request; no research content changed). Internal/
personal leaks removed from tracked files: `.vscode/settings.json` (bypass-permissions
config) untracked + `.vscode/` gitignored; author email dropped from `pyproject.toml`;
README "whitepaper" link repointed from the author's profile page to the context-architecture
GitHub repo. Depersonalized 142 first-name mentions → project-neutral phrasing across 28 files (LOG,
CLAUDE.md, skills, experiment/concept/design pages) + softened one personal-email mention here.
Owned-GPU model names (owned laptop/desktop GPUs) genericized to capacity/role descriptors
in CLAUDE.md and design/compute-strategy.md (all measurements/conclusions preserved; rental/
market GPUs 4090/5090/H100/3090 left intact). Added MIT LICENSE. README quick-start
restructured around the real third-party arc (collapse→SIGReg fix, then train DoorKey agent
→ watch it play via `--checkpoint`). `baselines/ppo/` kept — load-bearing for exp 0007/0008
(the rung-2 sample-efficiency claim). Verify: 21 tests pass, ruff clean, publish_check clean.
Note: `last_reviewed` dates deliberately NOT bumped on depersonalized pages — the edits were
mechanical (name swap), not content re-review.

## [2026-06-13] milestone+curation | exp 0021 CONFIRMED — critic-on-replay kept as recipe default

Critic-on-replay (DreamerV3 β_repval 0.3) confirmed on our stack: imagined_return pulled
from 0019's inflated 2–7 down to ~1.0 (s1 clean 3.2→1.0) — independent validation of the
Dreamer mechanism AND our implementation. Ignition still occurs (s2 0.55, s0 0.15) but
LATE (round 6 vs 0019's 3–5) and at honest value; endpoint trajectories opposite (0019
inflated→collapse, 0021 calibrated→rising). Best evals seed-reshuffled wash
(0019 0/0.55/0.35; 0021 0.15/0/0.55). DECISION: mechanism confirms the Dreamer
design → keep it; default flipped `--repval` 0.0→0.3 (recipe default now); bank the
MiniGrid imagination loop as good-enough+calibrated, don't over-polish the stepping stone,
move to recipe-hardening (two-hot/symlog/percentile-norm) → Crafter. Caveat logged: core
mechanism confirmed on a sparse task; full DreamerV3 recipe still to harden on denser
rewards. Process meta-lesson (2nd time): I read rounds 0–5 as "ignition vanished"; the
final round flipped it — partial-data conclusions burned us again (cf. 0019 smoke-test).
Pages: experiments/0021 Result+Lesson (status CONFIRMED); code default flip.

## [2026-06-13] curation | language-grounding: symbols-as-thought ideas

Added an "Ideas to look into later" section to concepts/language-grounding.md capturing
three forward ideas from a conversation (flagged not-decisions): (1) pixels-for-percept /
symbols-for-thought boundary — text→pixels→frozen vision encoder is right for text the
agent SEES in-world, wrong for text-as-thought (rendering discards symbolic structure the
tokens already carry); distinct from text→video→latent which stays legit (generating
experience ≠ binding meaning). (2) The deep question is symbol→belief binding (a word =
pointer into a belief region), which makes language a capability jump (ape↔human gap):
compositional abstraction lets the agent imagine in abstractions not pixels (counterfactuals,
"what if I had a pickaxe") ⇒ systematic generalization. (3) LLM-as-aligned-source-into-belief
= PAN's leverage done experience-primary: keep frozen vision, add a language pathway mapping
LLM-seeded tokens into the SAME belief space, alignment learned by grounding; beyond Dynalang
on deep binding + LLM-prior reuse. Also doubles as why [[frozen-encoder-lean]] survives the
language rung. Rides into next milestone commit (with the exp 0021 result). No page cites
PAN/Dynalang as authority (QUEUE-only).

## [2026-06-13] curation | Queued imagination-module substrate thread

Conversation on modular reuse of big pretrained world models (could a Cosmos-scale
video model serve as the Dreamer imagination loop?). Verdict in-conversation: no for
the inner loop (latency/pixel/non-differentiable — wrong complexity class), but the
question reframes usefully as "reuse the representation, not the simulator." Added a
neutral OPEN-QUESTION block to sources/QUEUE.md bracketing the design space with two
already-in-KB items: DIAMOND (2405.12399, diffusion WM) vs V-JEPA 2-AC (in 2506.09985,
frozen-encoder + predictor). Explicitly flagged not-a-decision to avoid biasing the
future read. No wiki page asserts anything yet.

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

## [2026-06-12] ingest | LeJEPA method sections (arXiv:2511.08544 + reference repo)

SIGReg details extracted (Epps-Pulley on random slices, 17-point trapezoid, λ=0.05,
batch≥128, bounded gradients). lejepa-2025 page stub→draft, read state: skimmed.
Pages touched: papers/lejepa-2025, concepts/latent-collapse (new), concepts/jepa.

## [2026-06-12] curation | milestone: SIGReg fixes latent collapse on MiniGrid

Exp 0001 formal run recorded (full collapse, commit 6692aa2). Exp 0002 run and
recorded: SIGReg λ=0.05 holds latent_std at ~0.75 over 1000 updates (control: 0.0003);
final probe protocol reads SIGReg 0.25 vs control −0.49. Probe target 0.8 not met —
representation-quality measurement itself was the main lesson (amplitude vs
information; standardization pitfall) → documented in latent-collapse page; proper
probe protocol is exp 0003. INDEX updated (latent-collapse off wanted list, exps
done); touched-page frontmatter checked. Next: exp 0003 probe protocol + Empty-8x8,
then memory for partial observability (rung 2).

## [2026-06-12] curation | milestone: rung 1 complete — world model beats copy baseline

Exp 0003 run (Empty-8x8, 3 seeds x 2 arms): SIGReg world model predicts held-out
transitions at 0.20-0.30x the copy-baseline error (controls: 1.5-1.7x). Rung-1 exit
criterion marked met on environment-ladder (flagged for review). Linear probe demoted to diagnostic-only after persistent seed noise;
dynamics-vs-copy ratio promoted to primary metric (latent-collapse page updated).
New tooling: world_model.play viewer (human render / GIF record), checkpoint saving
(--save). Next: rung 2 — DoorKey partial obs, memory, then first acting agent.

## [2026-06-12] curation | milestone: first acting agent — world model plays Empty-8x8 at 19/20

Exp 0004 closed after three iterations (6/20 -> 0/20 -> 12/20 -> 19/20 with stronger
eval-time planning; random baseline 3/20). Root causes found by diagnostics, not
tuning: (1) compounding rollout error from 1-step-only training -> multi-step rollout
training (replay sequence windows, losses at every imagined step, reward head trained
on drifted latents); (2) reward-magnitude starvation under sparse rewards ->
pos_weight 100. New code: RewardHead, MPCAgent (CEM), play --checkpoint,
ReplayBuffer.sample_sequences, --rollout-length. ROADMAP phase-1 complete + first
acting agent ticked. Lessons routed to exp page; "compounding rollout error" is now
mentioned on 2 pages — concept-page candidate if it recurs. Next: DoorKey + memory
(rung 2), PPO baseline comparison.

## [2026-06-12] curation | milestone: recurrent world model — DoorKey negative result, flywheel signal

Exp 0005 closed as an instructive negative: belief-state CEM-MPC 0/20 on DoorKey-5x5
(hypothesis >=40% refuted), but the collect->train->collect flywheel showed its first
life (random 8.5% -> round-0-model+eps-MPC 14.3% during collection) and the failure
decomposed cleanly: (a) pure MPC cannot span DoorKey's ~25-step reward chain with
horizon 20 -> value head needed (exp 0006, TD-MPC/Dreamer lineage answer); (b)
continued training across rounds destabilized (pred_loss 0.46->0.59) -> per-round
checkpoints/eval + optimizer handling next time. New infra committed: RecurrentDynamics
GRU belief model, burn-in/open-loop sequence training, RecurrentMPCAgent,
play --epsilon + recurrent checkpoint auto-detection, RGB partial-obs wrapper.
Also: compute-strategy concept page (local vs desktop GPU vs vast.ai prediction, prices
checked 2026-06-12). Next: exp 0006 value head + per-round eval.

## [2026-06-12] curation | thermal measurement: laptop GPU power-capped + throttling

90s load test after noticing heat: laptop GPU capped at ~45W (TGP floor), 62->78C in
90s, clocks ~1.2GHz vs 3.1 max, SW thermal slowdown already active ~396s cumulative
today. compute-strategy page updated (desktop GPU advantage revised 2.5-3x -> 4-6x;
desktop-dispatch trigger lowered 4h -> 1h); CLAUDE.md hardware note updated.

## [2026-06-12] curation | milestone: remote GPU dispatch live + benchmark validates compute strategy

Windows desktop (desktop GPU) wired up end-to-end: OpenSSH + Git Bash default
shell (cmd.exe breaks git transport), bare-repo push dispatch (scripts/remote.sh
setup/gpu/run/pull), uv sync with marker-gated cu130 torch wheels, CUDA verified.
First real dispatch = gpu_bench.py: matmul 4.5x (prediction 4-6x confirmed),
our recurrent train-step 0.93x (latency-bound prediction confirmed) ->
compute-strategy page updated with measured table. Publishability pass (on request):
no machine names/keys/paths in tracked files; config via env vars +
gitignored .env.remote (.env.remote.example committed); ADR 0004 records the
design. Setup gotchas (bare HEAD main-vs-master, administrators_authorized_keys,
DefaultShell) documented in docs/REMOTE.md.

## [2026-06-12] curation | milestone: DoorKey solved — value head + flywheel (exp 0006)

Exp 0006 closed: 90% per-round eval / 75% post-hoc (20 eps) vs 8% random — value
head makes beyond-horizon reward visible; collect->train->collect compounds
(collection 8.5%->5%->36%). First training dispatched through the remote pipeline
(desktop GPU, ~25 min). Eval-hygiene incident found and fixed: play.py tile_size=16
changed agent observations vs training tile 8 (4/20 -> 75% after fix); exp 0004
re-verified at 20/20 matched; obs_shape now stored in checkpoints + asserted at
load; minigrid.md gotcha section added. Process additions (on request):
reviewer agent (opus, research-code charter — first pass verified 0006 paths
correct + caught sweep-scrapability gap), /sweep skill + scripts/sweep.py,
scripts/publish_check.sh wired into /milestone verify. Next: PPO baseline
(closes rung 2), then Crafter prep.

## [2026-06-12] curation | milestone: PPO baseline wins on DoorKey-5x5 — rung 2 stays open (exp 0007)

Pre-registered counter-outcome occurred: sb3 PPO (CnnPolicy, 8 envs, no tricks)
hits 100% on all 3 seeds by 80k steps, ahead of our agent at every budget; our
agent burns 60k steps on random collection before the flywheel spins. Honest
verdict recorded; rung-2 sample-efficiency criterion NOT met on this env.
Escalation pre-registered in exp page: DoorKey-6x6/8x8 + earlier/smaller
collection rounds (exp 0008); ladder-criterion split proposed for sign-off. Infra: baselines/ppo as isolated uv project (sb3 caps
gymnasium <1.3); first /sweep dogfood produced the results table directly.

## [2026-06-12] curation | milestone: 6x6 ignition failure — the real blockers identified (exp 0008)

Exp 0008 closed via pre-registered alternative (b): round-0 random collection drew
1 reward event in 20k steps and the flywheel never ignited (best 10% vs PPO mean
~37% at 110k; PPO breakthrough moved 60-80k on 5x5 -> 100k+ on 6x6). Second
confirmed sighting of round-over-round training instability (round 6: 28%
collection success -> 0% eval after retraining). Exp 0009 agenda: ignition
(adaptive round 0, success-episode oversampling in replay, intrinsic signal) +
stability (success-balanced sampling, lr schedule, EMA agent weights); world-model
runs need >=2-3 seeds (round-0 luck is decisive). Rung 2b open; scoreboard PPO 2:1.

## [2026-06-12] curation | milestone: ignition solved, retention isolated (exp 0009)

3 parallel seeds on the desktop GPU (first remote.sh shell use; ~2h wall, 82% util,
49C). Adaptive round 0 + success-window oversampling fixed ignition everywhere
(seed 1: 6 success examples -> 60% greedy straight after round 0 — vs PPO 0% at
that budget). But continued round training destroyed and only partly rebuilt that
competence (60->0->50%): catastrophic interference under distribution shift is now
THE isolated bottleneck (3rd sighting, first clean). Best-checkpoint means: WM 40%
@ 115-130k vs PPO 37% @ 110k — parity via guard, not a win; rung 2b open. Exp 0010
(pre-registered): retention mechanics — EMA/snapshot acting weights, lr decay after
round 0, value target network, or frozen-trunk/head-only later rounds. Crafter
source verified (abstract) during the wait. remote.sh gained shell+kill; killed
runs survive ssh death on Windows — kill subcommand is the off switch.

## [2026-06-12] curation | timing correction: exp 0009 parallel sweep was 35 min, not ~2h

Log timestamps (12:04 launch -> 12:37-12:39 per-seed finish): 3 parallel seeds in
35 min wall, contention nil. Estimate was 3.5x pessimistic (MPC collection per-step
cost on the desktop GPU overestimated). exp 0009 page + compute-strategy corrected;
rung-2 sweeps are coffee-break scale.

## [2026-06-12] curation + scout | agent-architecture page; plasticity-loss literature found

New concept page: agent-architecture — our 4 optimization layers (CEM planning /
belief-model / Adam joint training / data flywheel) as a mermaid diagram with a
per-layer failure->experiment diagnosis table. Scout: our retention
problem IS the literature's "plasticity loss / primacy bias / churn" — 6 sources
queued (Nikishin resets = top exp-0011 candidate: reset last layers, keep replay).
QUEUE gained a themed retention section.

## [2026-06-12] curation | milestone: retention 2x2 negative — primacy bias confirmed, resets next (exp 0010)

Full factorial (ctrl/lr03/ema/both x 3 seeds, 2x 6-wide parallel batches): no arm
prevents the post-round-0 crash; seed 1 crashes in ALL arms (deterministic
interference, not noise); EMA damps peaks (60->40) without protecting them; lr
decay kills late recovery. Conclusion: smoothing the weight trajectory can't fix
directional interference — matches primacy-bias literature; exp 0011 = Nikishin
resets (reinit heads each round, keep replay + oversampling). Ops lessons: 6-wide
batches page heavily (replay ~10GB/process -> uint8 storage queued); block-buffered
logs look empty mid-run; laptop suspend pauses monitors (runs unaffected).
Also today: agent-architecture diagram iterated to extension-safe form (no init
directive, classDef styling).

## [2026-06-12] ingest | Nikishin primacy bias (protocol depth) + LeCun path paper (skim) + retention concept page

papers/nikishin-primacy-2022 (implementation-grade protocol; underpins running exp
0011), papers/lecun-2022-path (skim depth, module<->our-layer mapping table; full
read queued), concepts/retention (fix-family table: smoothing ruled out by 0010,
resets running, churn/continual-backprop/frozen-trunk untested). DINOv3 queued with
frozen-encoder framing. PROCESS gained the literature-first rule (flagged here since _schema is human-owned).

## [2026-06-12] scout | AMI pair completed (WBench id), Causal-JEPA + VL-JEPA queued

Watch-list sweep (3 searches): WBench arxiv:2605.25874 confirmed as the late-May
AMI-circle companion benchmark (theorem 2605.26379 + stress test pair; "current
models collapse under minor visual shifts") -> ami-labs page updated, QUEUE open
item resolved. New: Causal-JEPA (2602.11389, object-level latent interventions),
VL-JEPA (2512.10942, low priority). Dreamer/DeepMind side: nothing new beyond
known Genie/Dreamer state; 2026 survey blogs noted but not queued (secondary).

## [2026-06-12] curation | intellectual-lineage page (decades of background)

New concept page organizing 4 threads: mental models (Craik 1943, Tolman, Kahneman
S1/S2 -> Mode-1/2), predictive brain (Rao-Ballard, Friston), LeCun's arc (LeNet ->
contrastive -> EBM -> cake -> 2022 path), model-based RL (Dyna 1991 — our flywheel's
true name — Schmidhuber 90/91, POMDP belief states, 2018 revival). 6 lineage
sources queued (background section). Page verified:false until ids checked on ingest.

## [2026-06-12] curation + scout | QUEUE ids batch-verified via arXiv API; rung-3 SOTA line queued

All ~34 queued arXiv ids verified in one API call (id->title). Catches: DINO-WM id
was WRONG (2411.04958 = astronomy survey; corrected to 2411.04983, ICML 2025) —
the "id uncertain" flag from intake did its job; two title corrections (General
agents CONTAIN world models; MAINTAINING Plasticity). SOURCES verified flags
flipped (25 rows). New rung-3 prep section: 2502.01591 (Craftax SOTA jump),
2605.16457 ITC (May 2026, current Craftax SOTA 72.5%, identifiability framing
converging with LeJEPA theorem), delta-IRIS + DART (low).

## [2026-06-12] ingest | DreamerV3 (implementation depth) + I-JEPA (method depth); local PDF archive

dreamerv3-2023 stub->draft: full trick kit extracted (symlog, two-hot K=255,
KL balancing 1/0.5/0.1 + free bits 1 nat, return percentile norm, imagination
H=15/lambda .95) — retention-relevant find: their critic-EMA is a TARGET
(regularize toward slow copy), not acting weights like our failed 0010 arm; +
training ratio up to 16 vs our 0.1. New page ijepa-2023: multi-block masking
params, EMA-teacher 0.996->1.0 — the heuristic LeJEPA replaced. QUEUE marks.
On request: scripts/fetch_sources.sh -> 21 arXiv PDFs (183MB) in gitignored
knowledge/sources/files/ for personal reading; idempotent, documented in SOURCES.

## [2026-06-12] lint | 6-ingest checkpoint: graph closed, 1 INDEX gap fixed

Scripted pass over 45 pages: zero dangling [[links]] (cross-link graph fully
closed); INDEX was missing exp 0011 (fixed); remaining stubs (dreamer4, vjepa2)
honestly labeled, not cited as authority. Skills updated with today's craft:
ingest-source gained archive step + read-depth labels + ar5iv route;
scout-sources gained the arXiv-API batch-verification trick. Assessment: no new
skills/agents needed at current KB size; ingestion stays on-demand per the
literature-first rule (exceptions queued: Dyna, WBench).

## [2026-06-12] curation | research loop codified as top-level process

PROCESS.md gains "The research loop": experiment-verify <-> literature-first <->
route-to-KB <-> skills-carry-procedure-not-knowledge. Point 4 sharpened from
a working draft: maintenance trigger is process drift, not knowledge growth (thin
skills + INDEX navigation make new knowledge reachable without skill edits).

## [2026-06-12] curation | research-loop horizon defined

Process holds until a blocking problem has no published solution; then step 2
transforms (nearest-neighbor mapping + novel mechanism + our pages as primary
record). Noted: the model-based retention transfer (exp 0011) may already sit
on that edge.

## [2026-06-12] curation | deep-search protocol added to scout skill

Tested live: Semantic Scholar API = the right semantic+citation tool (anonymous
tier congested; free key recommended -> env S2_API_KEY); OpenAlex = instant but
weak ranking, good citation graphs; Google Scholar = manual-only (no API/ToS).
Local embeddings index DEFERRED with trigger: build it over OUR corpus (pages +
PDF archive) at ~150 pages, not over external papers (duplicates S2).

## [2026-06-12] curation | deep-search stack re-ranked: OpenAlex primary (S2 key gated)

S2 API-key form requires institutional affiliation + rejected a personal (non-institutional) email —
independent research not in their concept. Re-ranked: OpenAlex (truly open) as
programmatic primary with anchor-paper citation-walking to compensate ranking;
S2 anonymous as best-effort bonus. Project principle reinforced: the KB/process
depends on no gated service (arXiv, ar5iv, OpenAlex all open).

## [2026-06-12] curation | milestone: naive reset transfer fails — frontier confirmed (exp 0011)

9 runs, 3 arms, all below ctrl: resetting "heads" includes next-latent = world-model
amputation each round at 25x-too-low replay ratio (per-round amnesia). Bonus
finding: hr arm's s1 round-0 60->20 under doubled early training = primacy bias
reproduced in-setting (diagnosis confirmed, cure mis-mapped). Exp 0012
pre-registered in the page: value/reward-only reset, once at round 3, 4x post-reset
updates, shrink-perturb arm. Horizon rule active: no published work on this
question — our pages are now the primary literature for it. uint8 buffers held
6-wide at 3.7GB (paging fixed). retention.md fix-table updated.

## [2026-06-12] review | exp 0011 reviewer pass: latent reset+EMA bug found, negative result verified clean

Opus reviewer on the milestone diff: uint8 roundtrip provably exact, reinit
coverage complete, optimizer rebuild correct — and one important latent bug:
--reset never reaches the EMA shadow (acting weights), so reset+EMA combined
would silently no-op. Checked all 9 exp-0011 checkpoint configs: ema_decay=0
throughout -> negative result stands clean. Guard added (mutually exclusive
flags); lr-rebuild fragility commented.

## [2026-06-12] scout+ingest+correction | frontier claim overturned by the wall protocol

Ran the hitting-a-wall protocol properly for the first time (OpenAlex anchor walk +
S2 citation pages of Nikishin, ~300 citations triaged): found arXiv:2310.15017
"Mind the Model, Not the Agent" (2023) — directly on-topic, ingested at method
depth. It independently confirms our 0011 negative (agent resets harm MBRL),
locates MBRL primacy bias in the world model, and predicts reset failure at low
model-UTD (ours: ~0.1) — reframing our crashes as possible UNDER-training. Exp
0011 page corrected (frontier claim narrowed to sparse-reward low-UTD belief-MPC
flywheel); exp 0012 redesigned (Qiao shrink-perturb arm, UTD-scaling diagnostic
arm, surgical-heads arm); PROCESS horizon rule hardened (deep-search mandatory
for frontier claims); 6 more citation-walk finds queued. ./scripts/fetch_sources.sh
to be re-run for the new PDF.

## [2026-06-12] curation | ADR 0005: Minecraft milestone — three-signal training

New rung 5b before real-world transfer: self-play + VPT-style action-labeled video
+ text-in-world-model (open research; Dynalang/VL-JEPA direction). Ladder + ROADMAP
amended; 5 sources queued (ids to verify). Rationale: forces multi-modal/language
integration without robotics hardware; resource-rich (VPT corpus, MineRL/MineDojo,
published reference agents incl. Dreamer 4 offline diamonds).

## [2026-06-12] curation | text-signal staircase added to ADR 0005

Messenger/RTFM (text necessary) -> text-augmented Crafter (text helpful; metric =
sample-efficiency delta) -> Minecraft tutorials. Steps i-ii run on current hardware.

## [2026-06-12] curation | language-grounding concept page (binding/installation insight)

New page: text->world-model = binding (shared embedding geometry) + installation
(declarative -> dynamics belief). Three architectures: text-as-context (Dynalang),
text-as-data (text-induced imagination training — possibly unexplored; wall
protocol required before claiming), text-as-weights (Schmidhuber fast weights /
model editing). Pre-registered prediction: context suffices at Messenger scale;
data/weights needed at wiki scale. Linked into ADR 0005 + lineage + retention.

## [2026-06-12] curation | language-grounding 2b: external imagination engine

Text -> domain video generator (Oasis/Genie-3/Dreamer-4-WM class) -> VPT-IDM action
labels -> trust-weighted synthetic replay -> corroboration gate (real play
validates/discounts). Breaks 2a's circularity; unifies ADR-0005 signals 2+3;
cognitive analogy: Craik's mental simulation + belief corroboration. Gate is
mandatory (video hallucination) and retention-adjacent (synthetic distribution
shift).

## [2026-06-12] curation | unified trust-weighted replay: source priors + corroboration

Real vs imagined experience unified on one trust scale: source sets the prior,
corroboration updates it, loss scales with it; imagination earns what real
experience gets at birth. Human analogy: imagination is inexact AND functional via
constant verification. Prototype path: corrupted-synthetic injection on Crafter.
Symmetry noted with the KB's own verified-flag epistemics.

## [2026-06-12] curation | horizon: self-generated hypotheses — agent internalizes the research loop

Text-installed beliefs -> imagination-generated beliefs -> active validation, all on
the trust-gate substrate. Near-term echo: hypothesis-driven exploration as the
principled ignition fix (exp 0008 thread); Plan2Explore queued.

## [2026-06-12] scout | language/imagination design space swept (4 searches, 14 queued)

Solved-vs-open verdict per thread: binding-by-conditioning substantially explored
(2511.22904 reads dynamics descriptions — read before Messenger work); generated-
video-as-experience active at platform level (survey 2603.28489); synthetic-
transition reweighting mature BUT all generator-self-confidence based — the
reality-corroborated cross-source gate stays novel; uncertainty-driven exploration
covered (Plan2Explore, DreamerV3-XP) — compositional hypothesis generation beyond
state-novelty stays open. language-grounding page to absorb refs at ingest time.

## [2026-06-12] curation | paper-readiness check: provenance chain confirmed, venue column added

Claim -> page -> source id -> verified registry row chain is manuscript-grade by
construction; read-depth labels gate citability (only `read` sources citable in a
manuscript). Gap closed: venue/peer-review status now recorded at ingest (backfill
at pre-paper lint); BibTeX export planned (mechanical from arXiv ids).

## [2026-06-12] curation | /research-cycle skill: autonomous loop with mandatory per-failure literature gate

Prepared for overnight autonomous operation (a standing instruction, to be
activated later): harvest -> record -> mandatory failure-specific search ->
pre-register -> implement -> dispatch -> milestone; guardrails (scope, spend,
2-strike stall rule, honesty, state-of-the-night report). PROCESS gained the
per-iteration literature gate.

## [2026-06-12] curation | training-asset registry + tier-1 VPT retention (availability insurance)

New ASSETS.md registry (tiered retention: T1 now / T2 at rung-5b entry / T3 never-
bulk). Tier-1 retained to desktop ~/minecraft-assets/ (~10GB, 1.28TB free): IDM 4x
(the video->actions labeler — load-bearing for architecture 2b), all VPT
foundation/fine-tuned/RL weights, contractor-data index JSONs, repo snapshot (MIT).
Key finding: the 70k-h YouTube corpus was NEVER released (recipe only) —
contractor data + IDM are the retainable substance. VPT id verified:
arXiv:2206.11795. T2 (contractor subsets, MineRL, MineDojo dumps) deferred to
rung-5b entry with availability re-check.

## [2026-06-12] curation | tier-1 retention confirmed: 21/21 OK, 9.8 GB

All VPT tier-1 assets verified on desktop ~/minecraft-assets/vpt/ (first detached
attempt died with its ssh session — Git Bash quoting; foreground-over-held-ssh
pattern worked). ASSETS.md claims now confirmed-true.

## [2026-06-12] curation | temporal-abstraction concept page (via Robbins/Bergson)

Events-not-ticks critique routed: converging lines table (options, H-JEPA,
Director, Zacks event segmentation, action chunking); Zacks mechanism directly
implementable (segment at prediction-error spikes — boundaries for free);
identified as the DEEP fix for exp-0005 horizon blindness and as the same
binding problem as language-grounding. Cheap diagnostic probe pre-sketched
(boundary alignment on DoorKey trajectories). 5 sources queued.

## [2026-06-12] curation | milestone: UTD x4 — first PPO defeat at equal env budget (exp 0012)

9 runs close the literature-recipe arc: resets conclusively dead (5 variants <=
ctrl across 0011/0012 — both Nikishin and Qiao assume high-UTD overfitting; we
were UNDER-trained); UTD x4 (no resets) = 63% mean / 80% peak vs PPO 37% and ctrl
40% — bar (ii) met, rung 2b performance criterion cleared at equal env steps.
Bar (i) no-crash still fails (80->30): interference persists at higher amplitude;
trunk remains the only unprotected component. Exp 0013 pre-registered: UTD
baseline + trunk-freeze arm (+optional round-0-at-x1 arm). retention.md fix table
updated. No paid resources needed (~80-min desktop batches).

## [2026-06-12] curation | AUTONOMOUS MODE ACTIVATED (conditional)

Standing instruction: if there is no response after exp 0013 concludes, continue
the research cycle autonomously (/research-cycle skill — identical loop, no
per-iteration go). Stop gates: (1) hard wall that survives the
literature gate (no resolution found AND no further research to ingest — i.e.,
the skill's 2-strike stall rule across ALL queued threads), (2) token exhaustion.
All other guardrails unchanged: desktop-only spend, current-rung scope, honesty
rules, milestone discipline, PAID-RESOURCE flags recorded but not acted on,
decisions/schema/results get proposals only. Reports: milestone commits as journal +
state-of-the-night LOG entry.

## [2026-06-13] curation | milestone: interference localized to the encoder (exp 0013) [autonomous]

9 runs: encoder-freeze@r2 stops the crash phenomenon in all 6 frozen-arm seeds
(fenc monotone-rising, peaks at final round, mean 47% unconverged; ftrunk stable
but capped 40% — GRU plasticity needed; warm 43% — round-0 primacy dodged, crashes
return with encoder free). The 5-experiment retention arc resolves: interference =
ENCODER DRIFT. Literature gate: 2310.07418 (ICLR24) localizes to the critic in
model-free visual RL — tension recorded, ingestion queued (augmentation lever
noted). Exp 0014 pre-registered + launching: freeze-round sweep (fenc@3, fenc@4,
3 seeds each, same budget) — if both bars clear, propose rung-2b closure (decision pending)
and pivot to the actor thread.

## [2026-06-13] ingest | Ma et al. ICLR24 plasticity paper — the 0013 tension resolves [autonomous]

Their encoder-stays-healthy + frozen-pretrained-encoders-suffice findings
COMPLEMENT our encoder-drift result (different pathology — dormancy vs drift —
same prescription: stop training the encoder once competent). Critic-bottleneck
mechanism is TD-specific (our MC values dodge it). Imports queued for exp 0015:
FAU per-module logging, Adaptive-RR scheduling atop our UTD finding.

## [2026-06-13] STATE OF THE NIGHT [autonomous]

Threads advanced since last check-in:
- exp 0013 CLOSED: interference localized to the ENCODER (freeze @r2 -> 6/6 seeds
  crash-free; GRU must stay plastic; warm arm confirms round-0 primacy dodgeable).
  Milestone 70ff51d.
- Literature gates run: Ma ICLR24 ingested (tension resolved — frozen encoders
  legitimized; FAU + Adaptive-RR levers queued); progressive-freezing literature
  imported (FreezeOut/LayerLock principle).
- exp 0014 CLOSED: freeze-timing frontier mapped (40/47/53/60/63% across
  ftrunk/fenc@2/@3/@4/free); residual crashes = GRU drift. Milestone 6833495.
- exp 0015 RUNNING: staged freeze (enc@2+gru@4, enc@2+gru@5), both bars targeted;
  --freeze2 mechanics shipped + smoke-verified.

Decision queued: if 0015 clears both bars -> rung-2b closure proposal
(ladder is a roadmap-level decision) + pivot to the actor thread. If it caps like ftrunk ->
budget-tier extension + PPO re-baseline proposal instead.
No paid resources used or needed. No stop gates approached.

## [2026-06-13] curation | milestone: staged freeze counter-outcome — frontier confirmed (exp 0015) [autonomous, pre-return]

g4/g5 staged enc+GRU freeze caps at 30-37% (< ftrunk 40% < fenc@4 60% < utd 63%):
freezing the GRU always recovers the ftrunk ceiling. Stability<->performance is a
genuine frontier at this budget, not out-tunable. Desktop suspended mid-run
overnight, resumed clean (both freezes fired, no corruption). KEY STRATEGIC NOTE:
rung-2b exit criterion (beat model-free on sample efficiency) was already MET at
exp 0012 (63% vs PPO 37%); crash-free "both bars" was self-imposed extra rigor,
now characterized. Recommendation logged: declare 2b met, pivot to actor
thread. Autonomous mode PAUSED — awaiting strategic call.

## [2026-06-13] curation | hierarchy-and-credit concept page + rung-2b MET accepted

A credit-assignment question routed to a new concept page: flat value head
already does implicit local credit (exp-0004 monotonic value gradient = 63%), but
explicit reusable subgoals need hierarchy (H-JEPA/options). Conditional-subgoal
constraint -> unsupervised discovery (bottlenecks/Director/empowerment), never
hardcoded. KEY DESIGN DIRECTION: the Mode-1 actor we distill should be hierarchical
(manager-worker), unifying actor + hierarchy + temporal-abstraction + language
threads. 5 sources queued (Director, FuN, Option-Critic, empowerment, DIAYN).
Rung-2b accepted as MET (63% > PPO 37%, stable) — pivot to actor confirmed.

## [2026-06-13] curation | architecture-testing tier ladder added

hierarchy-and-credit page gains a 3-tier testing ladder: (1) Empty/DoorKey =
mechanism debugging (fast loop); (2) MiniGrid KeyCorridor/ObstructedMaze/MultiRoom
= cheap flat-vs-hierarchical credit A/B (negative = cheap kill, positive =
necessary-not-sufficient); (3) Crafter = the decisive skill-reuse claim. Principle:
cheapest env that reveals the effect; never debug architecture on Crafter.

## [2026-06-13] curation | compute efficiency elevated to stated operating principle

Generalized the tier-ladder into a lab strategy: architecture and compute-efficiency
are the same axis (judge architectures by capability-per-FLOP slope; our world-model
bet IS an efficiency bet). For a home lab efficiency is the entire moat. Rules:
minimum-SUFFICIENT-scale (compute analog of whitepaper's minimum-needed-context),
measure slopes not endpoints, literature-first/pre-registration as efficiency gates.
Added to compute-strategy.md + PROCESS.md research loop.

## [2026-06-13] curation | language-grounding sharpened: order of operations + verification battery

Two refinements: (1) language is NOT the cause of the missing "get the key" subgoal
— temporal abstraction is; language LABELS pre-existing nameless abstractions, so
grounding rides on top of the hierarchical actor (order: concepts/events first,
words attach). (2) real language = learned binding to OUR latent geometry, not a
plugged-in LLM (whose words are grounded in text stats, not this agent's
experience). Added the grounding-verification battery (imagination match,
cross-modal probe, REFERENT SWAP = gold standard / why Messenger-RTFM shuffle,
compositional zero-shot, modality transfer). First grounding experiment = built
around referent swap on a cheap env; Crafter is payoff not proving-ground.

## [2026-06-13] curation | capability-map page — locking the capability separation

Consolidation: single orientation index of the 7 distinct capabilities (world
modeling, flat credit, temporal abstraction/hierarchy, language grounding, language
installation, trust-weighted imagination, self-generated hypotheses) — problem each
solves, dependency order, status, test env. Explicitly records the separations that
keep blurring (#3 subgoals != #4 language; #4 grounding != plugged LLM; #5
installation != #4 grounding). A roadmap-level artifact.

## [2026-06-13] curation | run profiled; acceleration verdict

profile_run.py: training 79% (launch-bound), MPC collection 16%, eval 4%, env 0.4%.
Verdict: faster env is pointless (0.4%); no big easy wins (TF32 1.02x, bf16 1.23x,
compile 1.33x but CUDA-graph capture fights our freeze/reset optimizer rebuilds).
Already fleet-efficient via 6-wide parallelism. Shipped TF32 default + --amp opt-in
(off by default for fp32 comparability); deferred torch.compile to the actor/Crafter
phase (bigger models, no reset/freeze pattern). compute-strategy.md updated.

## [2026-06-13] milestone | exp 0016: planner distillation fails (BC compounding error) → on-policy

Flat actor distilled from the 80% planner via BC: train_acc ~54% (stochastic CEM
target), eval ~0% (O(eT^2) compounding error on the 25-step brittle chain). Clean
negative; literature gate confirms (DAgger/Ross-Bagnell). Also surfaced an
eval-rigor finding: teacher 80%(10 eps, seeds 10000+) vs 25-30%(20 eps, seeds 0-19)
— headline numbers are high-variance point estimates; standardize >=20 eval eps on
a fixed seed set. Verdict: go on-policy. Exp 0017 = imagination actor-critic
(Dreamer-style; on-policy, no teacher — fixes all three failure modes, and it's the
Crafter-path actor anyway). New code: Actor, ActorAgent, distill_actor.

## [2026-06-13] milestone | exp 0017: imagination AC exploits the model — missing continue predictor

Dreamer-style actor-critic flywheel built + ran (3 seeds, fast — actor-collection
cheap). Clean negative: imagined_return 2.0/3.0/0.48 vs eval 0% all seeds = model
exploitation. Root cause: we omitted Dreamer's CONTINUE PREDICTOR — imagination has
no termination, so the actor farms goal reward across the un-terminated horizon
(imagined return >> real max 1.0). Literature gate confirms (DreamerV2/3 discount/
continue head). The rest works (WM, on-policy AC, fast actor-collection). Exp 0018:
add continue predictor, train on replay dones, discount imagined returns by
cumulative continue prob. Single-variable fix.

## [2026-06-13] curation | generative-vs-predictive concept page (LeCun vs Xing debate)

Sharpening: generative-vs-latent (what you produce) and hallucination/exploitation
(intrinsic to optimizing any learned model) are ORTHOGONAL axes. Empirical anchor:
exp 0017 proves latent-only imagination ALSO hallucinates (no decoder, still farmed
fake reward) → verification mandatory regardless. Generation isn't needed to ground
words (contrastive/CLIP suffices) but IS needed to imagine experience from text
(arch 2b); the latent→decode→re-encode round-trip's only honest justification is
knowledge import. Unifying principle "real outweighs imagined" links trust-weighted
replay + continue predictor + corroboration. Roadmap: LeCun-pure rungs 1-4,
generative/Xing layer at 5b gated by verification.

## [2026-06-13] curation | imagination-as-novelty: hallucination is the raw material of ideas

Reframe added to generative-vs-predictive: an original idea = a hallucination that
survives verification (variation+selection = creativity; Popper/Campbell/Dennett/
Schmidhuber). Value is in what survives, not the generation (exp-0017 bad-idea
killed by 0018 = refutation). Creativity-safety tradeoff: never-hallucinating =
uncreative; target = regulated imagination (entropy/curiosity + verify). Closure:
agent creativity loop ≡ our research loop; capability #7 is this internalized.
Suppression and creativity are ONE mechanism, two aims.

## [2026-06-13] curation | architecture-strategy concept page: modes, interfaces, reusable components

Design brainstorm routed. Key reframes: (1) external-learning is a MODE (source +
verification regime), not the default loop = LeCun's configurator; (2) monolithic-
vs-modular is the wrong axis — the real decision is the stable INTERFACE
(belief-state), which makes module-count reversible/per-component; (3) reuse a
pretrained Minecraft WM as a frozen module — backed by knowledge-import+compute AND
our own encoder-freeze finding (components want different lifecycles). Decide
joint-vs-staged empirically at rung 3/5b. Flagged: FULL read of lecun-2022-path is
now design-critical (its modular architecture has been re-derived 3x in design sessions).

## [2026-06-13] curation | generative-vs-predictive: does imagination need generative AI?

Goals-divergence with LeCun (control-accuracy vs human-like-cognition). Sharpened:
imagination != pixel generation (exp 0017 imagines latent, no decoder); generative
is PRACTICALLY valuable for knowledge-import/tutorials + interpretability, and
STRUCTURALLY for 3D occlusion/amodal-completion (object permanence) — the strongest
real case. Corrected over-attribution (3D-hard != generative-key; DreamerV3 does 3D
latent). Resolution: latent core + optional generative module per mode; testable
(text->latent direct vs render-and-re-encode at Messenger scale).

## [2026-06-13] curation | dreaming concept + lazy-generation refinement + text→video→latent

Dreaming page (backlog): relaxed-constraints + weak-writeback framing = Hoel
overfitted-brain (verified arXiv:2007.09560) — dreams = anti-overfitting augmentation;
over-training triggers dreams = OUR primacy bias. Two functions: creativity-seeding +
latent-dream-augmentation (cheap testable retention regularizer, no gen-AI). Also:
generative-vs-predictive refined — generation is LAZY/attention-gated (latent tracks
persistence/object-permanence; render on demand only), demoting last night's
continuous-amodal-generative claim (corrected); text→latent solved by
composition text→video→encoder (seams: domain video-gen, encoder robustness to
generated frames, trust-gate). Hoel queued.

## [2026-06-13] milestone+curation | exp 0018 partial (deeper exploitation) + word-associations

Exp 0018: continue predictor partially tamed exploitation (s0 2.06->1.13) but s1
worsened (5.4) and eval still ~0 — actor exploits OTHER off-distribution model
inaccuracies; root = deterministic WM can't imagine the 25-step chain (compounding
error), no honest success gradient. Lack Dreamer's stochastic latents. Fork logged
for decision: (a) deepen (stochastic latents / H=5 / uncertainty penalty) vs (b) DAgger
on-policy distillation (reuse working planner, sidesteps imagination exploitation) —
recommend cheap H=5 check then DAgger. Also: language-grounding gains word-association
section — association matrix = word embedding (Levy-Goldberg, queued); don't rebuild
as metadata (import + bind to grounded latent); associations enable TRANSITIVE
grounding (smelt↔furnace bridges ungrounded tutorial words to experience).

## [2026-06-13] lint | full-KB curation sweep | 11 issues, 9 fixed, 2 flagged

Full sweep (53 pages; no >90-day staleness — KB is ~1 day old). Concept clusters
audited for duplication/contradiction via 3 parallel readers (architecture /
imagination / grounding-hierarchy): **no contradictions; clean abstraction-level
separation** — the rapid 06-12/13 brainstorm pages do not duplicate each other
(agent-architecture=descriptive, architecture-strategy=design, capability-map=roadmap;
generative-vs-predictive=philosophy anchor, dreaming=testable retention regularizer).
Link graph re-closed: every wikilink resolves except the registered wanted page
world-models-1803.10122; only remaining orphan is the meta-ADR 0002 (acceptable).
Fixed: INDEX (added 0019, corrected 0018 running→done-partial, reordered 0016);
retention.md (fix-table stopped at 0013 → extended with 0014/0015 frontier + Ma 2024;
both "open questions" answered by 0011-0013; added 7 missing links incl. orphaned
[[ma-plasticity-2024]] and [[dreaming]]); agent-architecture L3 interference "(open)"
→ localized-to-encoder/0013; orphan inbound links for 0019 (from 0018), 0015 (from
0014), [[ijepa-2023]] (from jepa); language-grounding Links gained
[[hierarchy-and-credit]]/[[temporal-abstraction]]. FLAGGED (meaning-bearing, no
silent rewrite): (1) "pension" terminology in exps 0014/0015 — reconstructed as a
deliberate metaphor from the [autonomous]-run agent ("pension off" = retire a network
component = freeze it permanently; encoder + GRU are the "two plasticity taps / drift
sources" each needing to be pensioned). Coherent but obscure; keep-or-normalize is
an open call. (2)
capability-map world-modeling status says "rungs 1-2 / SIGReg" — predates the
0017-0019 imagination thread, may want an exp-range refresh.

## [2026-06-13] curation | structural: new design/ directory

Resolved a shelving seam: `concepts/` conflated field-knowledge
explainers ("what is X") with this project's own design/strategy artifacts ("OUR X"),
so titles like "capability-map" mismatched the generic-concept expectation. The
`scope: local|shared` flag was meant to carry this distinction but isn't visible when
browsing by directory. Fix: new top-level `knowledge/design/` for project synthesis.
`git mv`'d 5 unambiguous pages out of concepts/ → design/ (capability-map,
agent-architecture, architecture-strategy, compute-strategy, environment-ladder);
wikilinks survived (basename-resolved), 6 path-based refs updated (INDEX ×5 → new
"Design & strategy" section, ROADMAP, sweep SKILL, gpu_bench.py, CLAUDE.md). SCHEMA.md
directory table gained a `design/` row and `concepts/` was narrowed to
"field concept". Hybrid pages (retention, temporal-abstraction, generative-vs-predictive,
hierarchy-and-credit, language-grounding, dreaming) deliberately LEFT in concepts/ —
reclassifying them is a separate judgment call (deferred). environment-ladder keeps its
ADR-0003 (frozen decision) + design-page (living spec) split. Also resolved flag (1)
above: "pension" wording normalized to "freeze/frozen" across exps 0014/0015 (clearer word adopted). Link graph re-verified closed post-move.

## [2026-06-13] curation | agent-architecture synced to code: 4 → 5 layers

Checked design/agent-architecture against the actual src/world_model/ (Explore map).
The page was an accurate snapshot of the exp-0001–0010 MPC+value system but stale
across all layers vs the 0016–0019 code: missing the learned actor (second acting
mode), the stochastic RSSM world model, the continue + reconstruction heads, and the
entire imagination actor-critic training loop. Per design intent ("the model should follow the
architecture, not stick to 4 layers"), promoted the imagination actor-critic to a real
**L4 BEHAVIOR** layer (policy gradient + λ-return critic w/ EMA target); the flywheel
moved L4 → **L5**. Updates: title now count-agnostic ("Nested Optimization Layers");
mermaid redrawn (5 subgraphs, two acting modes L1-search + L4-amortized feeding L5);
algorithm table gained the L4 row + RSSM/continue in L2/L3 + 0017-0019 failure modes;
prose contrasts Dreamer(no L1)/TD-MPC2/PPO(model-free) against our layering and locates
model-exploitation at L4. Renumber rippled to one cross-ref: capability-map "4 → 5
optimization layers" (the (L1)/(L2) maps in lecun-2022-path are stable, untouched).
FLAGGED (separate, not fixed): lecun-2022-path:38 calls stochastic latents "future" —
now built (RSSM, 0019); that page wants a refresh too.

## [2026-06-13] queue | PAN / GLP (Eric Xing) — LLM-as-latent-backbone

Queued two papers after a discussion sparked by an Eric Xing talk:
PAN (arxiv:2511.09057) and its position paper "Critiques of World Models"
(arxiv:2507.05169), under a new QUEUE subsection "LLM-as-latent-backbone (PAN/GLP)".
Why they matter to us: PAN's Generative Latent Prediction (encoder→latents, an LLM
backbone as the latent dynamics model conditioned on language actions, diffusion
decoder→video) is the published answer to the LLM↔latent integration we'd only
sketched — and Xing argues FOR a generative decoder, the explicit counter to JEPA on
our generative-vs-predictive axis. Logged the key ingest question so it isn't lost:
does PAN show only (a) static pretrained knowledge in the backbone, or (b) test-time
acquisition (read a novel tutorial → new latents → changed behavior)? (b) is our edge
case; if PAN demonstrates it the roadmap shifts. No wiki page cites these yet (queue
only). Pairs flagged: concepts/generative-vs-predictive, design/architecture-strategy.

## [2026-06-13] queue | Teams to watch — Hassabis/DeepMind + Fan-Yun Sun/Moonlake

Added a new QUEUE subsection "Teams to watch" after noting who Xing credits in the
DataCamp podcast "Will World Models Bring us AGI?" (youtube VNyLNZunv9E). Transcript-
confirmed: Xing names **Demis Hassabis / DeepMind** as near-perfectly aligned on what a
world model (and virtual cell) is and how to build/test it, and expects "something
fancier and disruptive in the next few months" (public releases trail internal work).
Second lead surfaced in the same discussion: **Fan-Yun Sun / Moonlake AI** (ex-Stanford
SAIL; Manning/Goodfellow orbit) — causal, multimodal, interactive, EFFICIENT world
models over blind scaling; aligns with our generative-vs-predictive + Causal-JEPA
threads. Both are groups-to-track, not papers; ingest specific outputs as they land.
First-ingest pointers logged (Genie line for DeepMind; latent.space/p/moonlake for
Moonlake).

## [2026-06-13] milestone+curation | exp 0019 RSSM — eval off zero (partial win)

Exp 0019 (RSSM stochastic latents) closed as a **partial success**, 3 seeds DoorKey-6x6,
commit b8fdb62. Eval rose off zero for the first time on the imagination line: best
s0=0.00, s1=0.55, s2=0.35 (both winners peak round 5, regress round 6). s1 checkpoint
reproduces 10/20 under play (new reactive RSSM agent path in play.py + RSSMActorAgent.
from_checkpoint). Key finding written to experiments/0019: **RSSM fixed policy quality,
not value calibration** — imagined_return stayed inflated 0.7–4.3 (hypothesis predicted
≤1.0; refuted), the SAME range as exploited 0017/0018, yet now coexists with real
competence. Stochasticity stops collapse onto one fake trajectory (relative action
ordering tracks reality) without bounding the absolute value scale. Two open problems:
peak-then-collapse round 6 (retention signature; freeze-round 2 insufficient → next
fork a) and seed variance (s0 never ignited). Process lesson recorded: a smoke confirms
mechanism wiring, not steady-state behaviour — the earlier "imagined_return ~0.1, killed"
claim was a premature smoke-test promotion (inflation re-emerges after 4k×7 AC updates).
Pages: experiments/0019 Result+Lesson filled (pre-registered hypothesis left intact).
Code: play.py + agents/rssm_agent.py (watchable checkpoints). Next: fork (a), the
round-6 collapse, via retention/plasticity levers.

## [2026-06-13] milestone+curation | exp 0020 actor-critic reset — REFUTED (fork a closed)

Fork (a) tested and falsified fast. Per-round shrink-perturb reset of the actor-critic
(α=0.5, `--ac-reset`) to attack 0019's round-6 collapse instead **suppressed learning**:
best_eval s0/s1/s2 = 0.00/0.05/0.10 vs 0019's 0.00/0.55/0.35 — the reset erased the
competence on the two seeds that had it. Mechanism: 0019's policy *consolidates*
gradually across rounds (s1 0.15→0.20→0.55), and an α=0.5 reset halves that each round.
So the round-6 collapse is NOT behaviour-layer plasticity loss (clean falsification).
Critic reset made `imagined_return` inflation worse (s2 → 7.3), confirming **value
miscalibration** is the load-bearing problem → redirect to fork (b): ground/bound the
imagined value (DAgger real-rollout anchoring / KL / shorter horizon), not the behaviour
layer. Pages: experiments/0020 Result+Lesson (status REFUTED); pre-reg hypothesis intact.
Caveats logged not chased: only aggressive α tested; 0019's 0.55→0 may be partly eval
variance. Code already committed d03eeba (the --ac-reset flag); this entry records the
outcome. Live logs worked this run (PYTHONUNBUFFERED in the dispatched commit).

## [2026-06-13] scout+queue+curation | value-calibration toolkit; exp 0021 critic-on-replay

After exp 0020 redirected to value miscalibration, lit sweep run to accelerate by reusing
established learnings. Lit sweep (4 searches) mapped the published toolkit for inflated
imagined value / model exploitation — THE central MBRL failure mode, not exotic:
b1 DreamerV3 **critic-on-replay** (β_repval 0.3, critic grounded in real returns) ·
b2 MBPO short *branched* rollouts (1906.08253) · b3 DreamerV3 percentile return-norm +
two-hot critic · b4 MOPO/MOReL/CBOP uncertainty pessimism. New QUEUE subsection
"Value calibration / model exploitation" (MBPO, MOPO, MOReL, CBOP queued; ids from search,
verify at ingest). **Curation fix:** papers/dreamerv3-2023.md had MISSED the critic-on-replay
detail (had two-hot/EMA-target/replay-ratio but not β_repval) — added the bullet; that was
the load-bearing fact for our fix. Pre-registered + implemented exp 0021 (b1): `--repval`
flag in train_rssm, critic also regressed on real burn-in returns (reuses beliefs already
computed; real_bel detached → grad to critic only). Flag defaults off → baseline ≡ 0019.
Prediction: imagined_return falls 2–4 → ~1 and ignition broadens. Smoke-confirmed wiring.

## [2026-06-13] decision | ADR 0006 accepted — Crafter (original) over Craftax for rung 3

ADR 0006 accepted. Crafter (original, PyTorch-native) for rung-3 first contact, NOT
Craftax. Three deciding facts: (1) Craftax 257x is a model-FREE PPO (1B-step) figure; we
are model-based/sample-efficient (~1e6 steps) so env-stepping is not our wall; (2) Craftax
is JAX-only, vs ADR 0001 PyTorch (interop friction / full rewrite), and 0001 pre-registered
the only revisit trigger as heavy parallel-env training; (3) we want pixel obs (frozen-
encoder thesis), and Craftax fast mode is symbolic. Craftax kept as documented escape hatch
with a measured trigger (env-stepping >=~30% wall-clock, or sample-hungry pivot). Next:
Crafter env wrapper behind the MiniGrid interface -> hello-Crafter baseline.

## [2026-06-13] curation | JAX rewrite revisit-triggers consolidated

Added a consolidated "JAX rewrite revisit triggers" subsection to design/compute-strategy.md
(extends ADR 0001; cross-links ADR 0006). Key framing: JAX advantage SHRINKS up the ladder —
real envs cannot be JAX-fused (Atari C++, Minecraft Java, real world), so the parallel-env
speedup exists only at the Crafter rung; after that the question is moot. Triggers (none
current): TPU access, sample-hungry pivot, env-stepping >=30% wall-clock. Even then: surgical
(per-experiment JAX env/agent via dlpack hybrid), not a stack rewrite. JEPA/WM stays PyTorch.

## [2026-06-13] milestone | rung-3 first contact — Crafter env wrapper

ADR 0006 accepted -> built the Crafter env wrapper (src/world_model/envs/crafter.py): a
gymnasium adapter over old-gym crafter.Env emitting CHW float32 obs in [0,1] like the
MiniGrid wrapper (same encoder/agents/training consume it unchanged). done split into
terminated (death, info.discount==0) vs truncated (length); seeded reset rebuilds the env
for reproducible eval worlds (~0.02s); info.achievements (22-dict) is the score basis.
crafter 1.8.3 added (light deps, no second CUDA framework). Smoke test added (19/19 pass).
Hello-Crafter random rollout: 283 steps to death, 1/22 achievements (wake_up), reward ~0.1
-- the expected random floor; env+reward+achievement+death/timeout all flow. New page
environments/crafter.md (incl. the Crafter->Minecraft proxy->target transfer relationship
Addressed question: method+frozen-encoder transfer, WM weights do not; JAX corollary). INDEX
+ environment-ladder additive pointer. Next: 64x64 encoder (frozen DINO/JEPA per lean) +
port the calibrated imagination loop; harden DreamerV3 recipe here (denser rewards).

## [2026-06-13] experiment+curation | exp 0022 — frozen DINOv2 encodes Crafter state (encoder bet validated)

Before building the frozen-encoder pipeline, validated the gating risk (does natural-image
DINO transfer to Crafter pixel-art?). scripts/probe_dino_crafter.py: linear probe on frozen
DINOv2-S CLS features (600 random-play frames) -> "which materials in the 9x9 view" (free
labels from info.semantic). Result: mean test acc 0.981 vs majority 0.804 (+0.177 lift),
per-material 0.93-1.00; shuffled-label control collapses to 0.748 (signal real). Frozen DINO
VALIDATED as the Crafter encoder, no fine-tuning. Greenlights DINO-WM-style build (frozen
DINOv2 + small RSSM dynamics). Frozen => drop SIGReg, cache embeddings in replay, immune to
primacy/drift. Pages: experiments/0022, crafter.md encoder bullet, INDEX (also caught up
0019-0021 statuses). Next: FrozenDinoEncoder module + wire into the RSSM flywheel for Crafter.

## [2026-06-13] milestone | frozen DINO + RSSM imagination loop wired on Crafter (train_crafter v1)

Wired the validated frozen DINOv2 encoder (exp 0022) into the calibrated RSSM imagination
loop (0019/0021) for Crafter: new src/world_model/train_crafter.py reuses wm_train/imagine_ac/
critic-on-replay unchanged, swaps in FrozenDinoEncoder (embed_dim 384, frozen -> excluded from
opt), make_crafter_env (17 actions), and an achievement/reward eval. SIGReg dropped (frozen
cant collapse; guarded wm_train with lam>0). v1 encodes in the training loop (correctness
first); embedding caching (encode once at collection, skip encoder in WM updates) is the next
optimization and is needed before a full-scale run. Tiny CPU smoke: full flywheel runs end to
end, imagined_return sane (0.10/0.18 = calibrated, critic-on-replay carries to Crafter), ckpt
saved. MiniGrid train_rssm untouched (lam>0 guard preserves it). Next: embedding cache -> first
real desktop Crafter run; then harden DreamerV3 recipe (two-hot/symlog/percentile-norm).

## [2026-06-13] milestone | embedding cache for the frozen-encoder Crafter loop

Implemented freezing payoff: train_crafter now stores DINO EMBEDDINGS (float32) in replay,
not pixels, encoding each frame ONCE at collection (new collect_embed) and passing an Identity
encoder to wm_train/imagine_ac -> zero DINO forwards in WM/AC updates (~98% of forwards
eliminated; training-update cost no longer DINO-bound). ReplayBuffer gained an obs_dtype param
(float32 stores embeddings verbatim, no *255 round-trip; default uint8 unchanged -> MiniGrid
path untouched, 20/20 tests). Correctness-equivalent: cached round-0 smoke is NUMERICALLY
IDENTICAL to the v1 uncached run (0.393/0.1687/0.096) -- same embeddings, same training, just
encoded at collection. Float-buffer roundtrip test added. Embeddings are ~8x smaller than the
image buffer too. Makes a full-scale desktop Crafter run feasible. Next: first real run +
harden DreamerV3 recipe.

## [2026-06-13] validation | first Crafter learning signal + value inflation surfaced

Local-GPU validation of the full train_crafter pipeline (cached frozen DINO + RSSM + critic-
on-replay), rounds 0->1 in 75s on the throttled laptop: eval_reward 0.10->1.10, achievements
1->2 -- LEARNS above the random floor; cache works (no OOM, training not DINO-bound). BUT value
inflation resurfaced at Crafter scale: imagined_return 9.95->21.67, critic_loss 14->35 (vs ~1
calibrated on MiniGrid). Denser/larger-scale rewards overwhelm plain-MSE critic + critic-on-
replay (beta_repval 0.3) -- model exploitation (cf 0017/0018) at Crafter scale. Decision: do
NOT dispatch a long run yet (would just exploit the model); harden the DreamerV3 recipe FIRST
(symlog + two-hot distributional critic + percentile return-norm) -- the deferred step, now
empirically justified by this run. crafter.md pipeline-status + what-next updated.

## [2026-06-13] experiment | exp 0023 two-hot distributional critic — Crafter value bounded

Value inflation on Crafter (imagined_return 21, critic_loss 35) attacked with DreamerV3 two-hot
distributional critic (models/twohot.py: 255 symlog bins, two-hot CE; forward=symexp expectation,
drop-in for ValueHead). imagine_ac gated via _critic_loss (twohot_loss if available else MSE) ->
MiniGrid path byte-identical (21/21 tests), Crafter gets distributional loss. Local smoke: same
config that gave MSE imagined_return 21 now gives 6.1->5.4 STABLE, critic_loss 2.5->2.2 -- blow-up
gone. Eval flat in 2 noisy rounds (calibration-delays-ignition, cf 0021); climb test needs a long
run -> dispatching overnight. Pages: experiments/0023.

## [2026-06-13] experiment | exp 0023 result — first Crafter LEARNING run (two-hot critic)

Overnight desktop run (2 seeds, 10 rounds) of the full rung-3 pipeline (frozen DINO + RSSM +
critic-on-replay + two-hot critic + embedding cache). RESULT: learns and climbs off the random
floor (1 ach) to ~2.5-3 achievements / reward ~2 (20x random reward), no collapse, NO ignition
stall (the 0021 calibration-kills-exploration risk did not materialize -- Crafter reward dense
enough). Both seeds plateau at the shallow tree (~3 ach). Two-hot fixed the CRITIC (loss 1.5-2.7
stable vs MSE 35) but imagined_return stayed inflated/noisy (s0 20->5-10 self-calibrating; s1
17-22 w/ a round-8 collapse to 0.56) -- the REWARD head is the remaining over-prediction source.
Best reward s0 2.10@r6, s1 1.97@r2. Next (reordered by data): exp 0024 DINO patch tokens (richer
spatial features for navigation/gathering, attack the plateau) BEFORE the reward-head fix, since
the plateau (not value) is the headline blocker. Pages: experiments/0023 Result+Lesson, INDEX.

## [2026-06-13] diagnostic | exp 0023 agent is STATIONARY — reward-head exploitation (live viewer observation)

Observation in the live viewer: the player never moves relative to the (egocentric-scrolling)
world. Quantified: 3 episodes, unique_tiles=1, player_pos bbox_span=(0,0) -- the agent NEVER
changes position. Action histogram: do (chop adjacent), mk_iron_sword spammed 38-48x/ep (a no-op:
no iron), sleep, place_plant. All 3 achievements are reachable without moving (wood/table/plant
from spawn). The mk_iron_sword spam = MODEL EXPLOITATION: the reward head over-predicts reward
for that useless action in imagination -> THIS is the inflated imagined_return (17-31); the
actor maximizes broken imagined reward by spamming the fake-rewarding action instead of exploring.
RE-PRIORITIZES the plateau diagnosis: it is a degenerate non-exploratory policy driven by
reward-head inflation, NOT primarily representation. => reward-head fix (symlog/two-hot REWARD
predictor) jumps to top priority (kill spurious action value -> remove spam incentive -> free
exploration), likely + stronger entropy. exp 0024 (patch tokens) still a clean representation-axis
test; keep it running. Movement is not broken (random agent moves; actions pass through).

## [2026-06-13] experiment | exp 0024 patch tokens — navigation unlocked (seed-dependent); overturns the null

DINO patch tokens (cls+patch) vs the 0023 plateau. Aggregate looked like a wash (best reward
s0 2.47, s1 1.60) BUT behavior_report revealed a huge split: s0 MOVES (moved_frac 0.12, span
17.7 tiles, move_actions 0.42) and builds a TABLE (place_table -- real tech-tree progress);
s1 collapsed to the 0023 stationary policy (do:0.96, moves 0
## [2026-06-13] experiment | exp 0024 patch tokens — navigation unlocked (seed-dependent); overturns the null

DINO patch tokens (cls+patch) vs the 0023 plateau. Aggregate looked like a wash (best reward
s0 2.47, s1 1.60) BUT behavior_report revealed a huge split: s0 MOVES (moved_frac 0.12, span
17.7 tiles, 42pct movement) and builds a TABLE (place_table -- real tech-tree progress); s1
collapsed to the 0023 stationary policy (do:0.96, moves 0pct). => Representation IS part of the
bottleneck (overturns pre-reg null): global CLS cannot localize resources to navigate; spatial
patch tokens give the where-signal and s0 learned to walk + craft. Seed-dependent because the
reward exploitation is unfixed (imagined_return 8-22) and tips s1 into collapse. BOTH levers
matter -> exp 0025 combines patch tokens + two-hot reward head, 3 seeds. PROCESS WIN: the
behavior QA gate caught the s0/s1 split that aggregate eval masked. Pages: 0024, INDEX.

## [2026-06-13] curation | why Crafter onboards easier than DoorKey + plateau is a scaling story

Recorded an honest decomposition on crafter.md of why Crafter feels easier than the DoorKey
struggle: (1) reward DENSITY (built-in achievement curriculum vs DoorKey single sparse goal --
biggest factor), (2) frozen pretrained encoder (skips the from-scratch representation instability
that was half the DoorKey fight), (3) our algorithmic improvements (prevent failures, not the main
ease driver). Plus a correction of an over-dramatic "DoorKey wall returns" framing: the plateau
is NOT a fundamental wall -- (a) current ~3 plateau = the exploitation BUG (0025 fixes), (b)
mid-tree = mostly COMPUTE SCALING (DreamerV3 reaches it at ~1M steps; we are ~10x under-trained),
(c) only deepest chains might want hierarchy, as an efficiency lever. For a compute-efficiency lab
efficient scaling IS the research. Discipline: MEASURE the scaling slope once 0025 lands.

## [2026-06-14] curation | ROADMAP.md refreshed to current state (rung 3 / exp 0025 era)

ROADMAP was stale (stopped ~exp 0008). Rewrote to reflect the journey: rungs 1-2 done
(incl. the imagination actor-critic line + retention thread), rung 3 (Crafter) current
with the frozen-encoder + calibrated-imagination + value-hardening + behavior-QA progress
and the next steps (consistent navigation, scaling slope, hierarchy). Depersonalized for
the public repo (no identity/hardware leaks; proper KB links). Phases map to ladder rungs.

## [2026-06-14] scout+queue | does our grounding testbed already exist? — CrafText/RTFM/SILG/LDD

Checked whether a custom Crafter-with-tutorials env exists or needs to be built.
the literature first (frontier-claim protocol). Found: CrafText (2505.11962, Craftax instruction-
following — "Crafter+language" exists, but instructions not read-to-learn-dynamics, and JAX/Craftax
per ADR-0006 wall); RTFM (right read-to-learn-dynamics paradigm, toy scale); Messenger (referent-swap
gold standard); SILG (2110.10661, unified grounded-language-game benchmark); LDD (2210.00066, method).
KEY DISTINCTION recorded: instruction-following (text=goal) vs reading-to-learn-DYNAMICS (text=rules;
our installation interest). LIKELY GAP (UNVERIFIED, needs deep protocol): rich-env read-to-learn-
dynamics. Recommendation: literature-first — ingest this cluster BEFORE building any custom env
(gives task designs + baselines + decides reuse-vs-build). Queued under a new QUEUE subsection. No
build started.

## [2026-06-14] design | two-domain split: CrafterManual grounding-env as a separate repo; this project authors requirements

The grounding testbed was framed as a SECOND context-architecture DOMAIN: a separate
standalone GitHub repo (CrafterManual / Crafter-RTFM), built by its own Claude to spec, with THIS
research project iteratively AUTHORING requirements for it. Clean inter-domain contract:
requirements flow research->tool, capability reports flow tool->research; neither reaches into
the others internals. Wrote v0 requirements catalog at knowledge/design/grounding-env-spec.md
(gate=verify-gap-first; mandatory-read-to-learn-DYNAMICS, rich+PyTorch, built-in referent-swap
verification harness, phased build with a P1 mandatory-reading validation gate). The spec is the
boundary object, owned/versioned here; implementation lives in the separate repo. Downstream of
the Crafter foundation; not started.

## [2026-06-14] milestone | exp 0025 SUCCESS — combined fix resolves the exploitation (all seeds move/craft)

Combined fix (DINO patch tokens + two-hot reward head + two-hot critic + critic-on-replay), 3
seeds, 10 rounds. best_eval_reward s0 2.35 / s1 2.47 / s2 3.47. DECISIVE: behavior_report PASSES
on ALL 3 seeds (vs 0023 all-stationary, 0024 seed-dependent) — moved_frac 0.11-0.18 (all move,
span 18-22 tiles), action entropy 1.5-2.0 (no collapse), real tech-tree (wood/stone pickaxes,
furnace), s2 6 unique achievements / reward 3.77. imagined_return calibrated 1.3-1.7 (vs 8-22),
trended DOWN. Both levers needed: patch tokens (navigate) + bounded reward (dont hallucinate
reward). The exploitation arc 0017->0025 is CLOSED. Residual: s0 minor make_iron_sword spam.
Ceiling (~3-6 ach) now a COMPUTE-SCALING question (~10x under-trained vs DreamerV3) -> exp 0026
scaling slope. Process win: behavior_report (from the it-doesnt-move live observation) was the
decisive metric; eval_reward alone looked flat. Pages: experiments/0025 Result+Lesson, INDEX.

## [2026-06-14] experiment | exp 0026 replay-ratio scaling — breadth not depth (mixed)

2x updates/round (replay ratio), 0025 recipe, 2 seeds. End achievements ROSE (s0 3.88, s1 4.00
vs 0025 ~2.25-3.4) BUT the names show it is BREADTH: union all shallow/survival (collect_wood/
drink/sapling, place_plant, eat_cow, defeat_zombie, wake_up) -- NO crafting tech-tree (no
pickaxe/table/furnace that 0025 reached). Movement DROPPED (moved_frac 0.07-0.15 vs 0.11-0.18,
top place_plant/sleep) = mild over-training toward in-place farming (retention/primacy thread).
=> replay-ratio lifts breadth/reliability (we were mildly under-trained there) but is NOT the
depth lever; dont push it higher. Depth (crafting->stone->iron) is data/exploration-bound (the
DoorKey long-horizon credit-assignment problem returning), not under-training. Caveat: crafting
rare + 2 seeds = partial seed confound. -> exp 0027 step/data scaling (more rounds at 1x replay):
does more exploration data unlock depth? If not -> exploration/hierarchy-bound. Pages: 0026, INDEX.

## 2026-06-14 — exp 0027 dispatched + lit-gate prep for the depth fork
- Dispatched exp 0027 (step/data scaling: 20 rounds, 1× replay, 2 seeds) at commit 755d902 on the desktop GPU. Tests whether crafting DEPTH is data/exploration-bound (the 0026 follow-up: replay-ratio bought breadth not depth).
- Literature gate (autonomous research-cycle step 2) for the likely 0027 fork: queued 3 new on-target anchors under sources/QUEUE.md → new section "Crafter DEPTH / achievement-hierarchy fork (exp 0027 thread)": Achievement Distillation (2307.03486, contrastive achievement hierarchy — most on-target), Learning Achievement Structure (2305.00508, structured exploration via the dependency graph), Curious Replay (2306.15934, novelty-prioritized DreamerV3 replay, +1.33× Crafter — the contrast to 0026's failed uniform 2× replay). Ingest waits until 0027 picks the fork.

## [2026-06-14] state-of-the-night | autonomous research-cycle: depth thread (0026→0027→0028)

**Threads advanced tonight (all milestoned):**
- **exp 0026** (replay-ratio scaling, 2× updates) → MIXED: breadth/reliability up, crafting
  DEPTH unchanged, mild over-training (less movement). Milestone 207ff4a.
- **exp 0027** (step/data scaling, 2× data @ 1× replay, 20 rounds) → NULL for depth: reward/
  count rose (best 3.47/3.72, count 4.38/4.62 > 0025) with movement HEALTHY (behavior_report
  PASS both; 1× replay avoided 0026's collapse), but frontier did NOT exceed 0025 (s0 stalls
  at place_table, s1 one transient make_wood_pickaxe; no stone/furnace). Milestone b9fa258.
- **Verdict:** both compute axes (gradient-steps AND env-steps) are ruled out as the depth
  lever → depth is **exploration/hierarchy-bound**, not compute-bound. Stall rule fired
  (same failure survived 2 redesigns) → switched thread from scaling to structured exploration.

**Running now:** **exp 0028 — Curious Replay** (novelty/surprise-prioritized WM replay;
arxiv:2306.15934, ingested at method depth → papers/curious-replay-2023.md). 2 seeds,
dispatched at commit 997183a on the desktop GPU. One variable vs 0027: `--curious` (WM-train
windows sampled by p=c·β^visits+(|recon+KL|+ε)^α instead of uniform). Reviewer-confirmed the
optimized loss is identical to the 0027 baseline → clean attribution. Chosen because the paper
reports its Crafter gains SPECIFICALLY on the deep nodes we're stuck on (stone pickaxe, iron).

**Recommended next decision:**
- If 0028's eval union gains frontier achievements (collect_stone / make_stone_pickaxe /
  place_furnace) with behavior_report still PASS → curiosity-on-WM IS a depth lever →
  exp 0029 = extend Curious Replay to the imagination-AC burn-in sampling too (compounding).
- If 0028 is NULL (frontier unchanged) → the bottleneck is the ACTOR not discovering the path,
  not the WM not learning it → escalate to structured exploration / hierarchy: Achievement
  Distillation (2307.03486) or achievement-graph structured exploration (2305.00508), both
  queued. This is a bigger design step (achievement-conditioned policy/contrastive head) and
  is a reasonable point to want input on direction.
- Caveat to watch: DINO-embedding recon error may be lower-variance than the paper's pixel
  loss → the curiosity signal could be weak. If 0028 shows priorities barely differentiating
  (recon+KL near-uniform), that's the likely culprit, not the method.

No paid resources used (desktop GPU only). No PAID-RESOURCE flag this cycle.

## [2026-06-14] experiment | exp 0029 stat-perception probe — frozen encoder NOT blind; drink-spam is actor-side

Triggered by observing crafter_steps_s1 spam collect_drink at the cap. Probed
linear decodability of Crafter vitals from the frozen DINO embedding AND the trained RSSM
belief. Result: drink decodes at 0.95 (embedding, cls+patch) and 0.94 (belief) — the level
survives end-to-end to the actor. Refutes the perception hypothesis: it is an actor/reward/
credit problem (drinking at cap is a no-op with no penalty), not encoder blindness. Vindicates
frozen-encoder-lean. Converges with 0026/0027 (depth is actor/exploration-bound). Side finding:
health drops 0.79->0.35 in the belief (RSSM compresses the slow health stat). New diagnostic
tool: world_model.stat_probe.

## [2026-06-14] experiment | exp 0028 Curious Replay — NULL for depth (3rd confirmation actor-side)

Curious Replay (WM-side novelty-prioritized replay) improved exploration (moved_frac 0.23/0.19
vs 0027's 0.13/0.14, s1 behavior reward 4.10 = best yet) but did NOT move the crafting frontier:
no stone/furnace; s1 reached one transient make_wood_sword (= 0027's wood-pickaxe depth). best
3.22/3.73 ~= 0027. behavior_report PASS both; drink/do-spam persists. Three depth levers now NULL
(0026 replay-ratio, 0027 data, 0028 curiosity) + 0029 cleared perception => actor/credit-side is
the frontier. Scout launched on the Crafter leaderboard (is the deep tree solved by ANY SOTA, or
universal ceiling?) to calibrate next-fork expectations.

## [2026-06-14] scout | Crafter leaderboard — deep tree near-universally unsolved; SOTA is actor-side

Resolves the 0028 tension (CR is "champion" yet NULL for depth in our hands). Crafter Score =
geom-mean of 22 achievement rates. Human 50.5%. Achievement Distillation (2307.03486) 21.8%
(best from-scratch), Curious Replay 19.4, PPO-ResNet 15.6, DreamerV3 14.5. KEY: the deep tree
is near-universally UNSOLVED from scratch — AD collects iron ~3% (= 20x DreamerV3's ~0.15%),
diamond ~0% for all. The ~14-22% scores are dominated by shallow/mid breadth. So our depth
plateau is MOSTLY the universal ceiling at iron/diamond; real headroom is the STONE tier where
SOTA reaches moderate rates and we get ~0. The method credited with depth gains is Achievement
Distillation = ACTOR-SIDE contrastive self-imitation on the achievement hierarchy, NOT a WM
trick -> vindicates the 0029/0028 actor-side diagnosis. (scout agent stalled on a fetch; synthesis
done inline via search.) Queued: 2507.04075 MLT, 2406.07381 LLM-hint WM. Next fork decision pending
open fork: actor-side AD-style capstone (target stone tier, NOT diamond) vs consolidate the rung.

## [2026-06-14] experiment | exp 0030 self-imitation (SIL) implemented + Crafter Score metric

Actor-side depth capstone. Added SIL to imagine_ac (reinforce real actions whose return beat the
EMA-critic baseline, (R-V)_+ weighted; oversample achievement windows; opt-in, sil=0 byte-identical
per reviewer). Added the official Crafter Score (geom-mean of 22 achievement rates) to evaluate().
ANCHOR: our best agent (crafter_steps_s1) scores only 2.61% vs DreamerV3 14.5% / AD 21.8% / human
50.5% -- the gap is BREADTH (we touch 8/22). Ingested Achievement Distillation (2307.03486) at method
depth -> papers/achievement-distillation-2023.md (AD is contrastive representation, NOT SIL; its L_pred
is the 0031 fallback). Reviewer SHIP. Dispatching 0030 = 0028 (curious) + SIL, 2 seeds.

## [2026-06-14] infra | vectorized parallel env collection (train_crafter --n-envs)

Crafter collection is CPU-bound (single-thread env.step) → GPU idles during collect/eval.
Added collect_embed_vec: N Crafter workers in parallel (AsyncVectorEnv NEXT_STEP autoreset),
batched encoder+policy on GPU, per-env streams appended sequentially (contiguity preserved,
boundaries force-terminated). Opt-in --n-envs (1 = unchanged serial). Measured 1.6x faster
collection at n=6 on the laptop (scales better on many-core boxes/rentals). Unit-tested
invariants (no cross-boundary windows) + integration smoke; reviewer SHIP (one robustness fix
applied: loop bounds total recorded to ~steps). Enables keeping rented GPUs ~100% utilized.
Design note: design/compute-strategy.md.

## [2026-06-14] experiment | exp 0030 self-imitation (SIL) — NULL; cheap actor-side lever sweep exhausted

SIL (reinforce real achievement trajectories, (R-V)_+ weighted) did NOT lift the Crafter Score
(30-ep: s0 2.06%, s1 2.77% vs 2.61% baseline) or breadth (7-8 distinct, wood-tier ceiling, no
stone). behavior_report PASS both; drink-spam persists. Diagnostic failure: SIL can only imitate
successes that EXIST in replay, and the agent rarely reaches stone-tier states -> nothing to
consolidate. This is the DISCOVERY problem, not credit-propagation. Fourth NULL lever (after 0026
replay-ratio, 0027 data, 0028 curious) + 0029 cleared perception => plain-Crafter depth ~2.6%/wood
is at/near our ceiling. Standing instruction reached its "if-we-cannot -> tutorial-driven" branch.
Open fork: (a) 0031 AD L_pred for closure (low EV, 0029 cleared representation) vs (b)
declare rung-3 done + pivot to reading-to-learn (crafter-rtfm).

## [2026-06-14] experiment | exp 0031 semantic foundation probe — GREEN, frozen encoder validated

Before the rung-4 pivot, stress-tested the frozen-encoder foundation: can the embedding AND the
RSSM belief decode achievement-critical MATERIALS (stone/coal/iron/tree/table)? Decisive GREEN:
emb AUC 0.99-1.00, and crucially they SURVIVE to the belief at 0.94-0.98 (stone 0.942, iron 0.979)
-- unlike health which the RSSM crushed to 0.35 in 0029. The actor has the full tech-tree
perception end-to-end; the plateau is exploration/discovery, NOT perception or architecture. Closes
the rung-3 loop (0026/0027 compute, 0028 WM-curiosity, 0030 self-imitation, 0029+0031 perception all
ruled out -> discovery wall). No adapter/custom encoder needed; frozen-encoder-lean holds. Foundation
stable -> safe to build rung-4 reading-to-learn. 1M run unnecessary. New tool: world_model.semantic_probe.

## [2026-06-14] design | rung-4 manual-conditioned agent — proposed; anti-baking as the spine

Pivoted to rung-4 (reading-to-learn-dynamics) after rung-3 closed + foundation validated (0031).
New design page design/rung4-manual-conditioned-agent.md (proposed). Core choices:
(1) condition the WORLD MODEL (RSSM dynamics + reward head) on the manual, NOT the policy -- the
grounded "read-to-learn-dynamics" choice AND the strongest anti-baking lever (a WM can only
shortcut by correctly predicting per-episode dynamics from the manual, which IS reading). (2)
Frozen/small text encoder mirroring the visual side. (3) Anti-baking is LAYERED: env per-episode
randomization (memorization useless) + referent-swap & held-out-config eval (the unfakeable metric)
+ belief-probe diagnostic (mechanistic, our 0029/0031 method) + no architecture backdoor. Success =
swap-following on held-out manuals, NOT task reward. Next: derive the crafter-rtfm handoff (env API
needs) via commons once design settles.

## [2026-06-14] rung-4 | encoder pre-check — pooled text vector INSUFFICIENT → token-level cross-attention

First concrete rung-4 step (diagnose-before-build, our 0029/0031 habit). Pre-check via
world_model.text_probe on held-out r1 recipe manuals (crafter-rtfm): a POOLED MiniLM-L6 sentence
vector decodes the verbatim-in-text recipe gesture at only 0.23-0.37 (10-class, chance 0.17);
token-level mean+max lifts to 0.42-0.55 (info is in the tokens, pooling dilutes it) but still not
clean because the gesture is ORDERED and mean+max is order-invariant. Swap-tracking is directionally
right (displayed >> correct) but weak. Decision: condition on frozen TOKEN embeddings via
cross-attention (order/token-aware), NOT FiLM/concat on a pooled sentence vector. Updated
design/rung4-manual-conditioned-agent.md (§2 architecture + §5 handoff DELIVERED + §6 resolved).
Also: HO-0005 accepted (env consumable), HO-0004 acceptance confirmed valid.

## [2026-06-14] rung-4 | manual-conditioning mechanism built (cross-attention into the RSSM prior)

Core rung-4 research piece, built + tested. models/text_encoder.py (FrozenTextEncoder: frozen MiniLM,
TOKEN-level, cached per manual) + models/manual_conditioning.py (ManualConditioner cross-attention;
ConditionedRSSM whose PRIOR is conditioned on the manual tokens via the RSSM deter state as query).
The manual conditions the DYNAMICS (prior), not the policy. Tests (tests/test_rung4.py): shapes, mask,
gradient flow, and the LOAD-BEARING property -- swapping the manual CHANGES the predicted prior (so
conditioning is used, not ignored). Real-MiniLM integration smoke: manual text -> 384-d tokens ->
cross-attn -> conditioned prior, correct vs swapped manual differ. Next: rung-4 training loop on
crafter-rtfm + swap-following eval on held-out r1.

## [2026-06-14] rung-4 | manual-conditioned flywheel (train_rtfm.py) runs end-to-end

The rung-4 training pipeline is built + integrated. world_model.train_rtfm: collect from crafter-rtfm
(r1 recipes, capped at the task horizon; DINO frame embeds + per-episode manual id as a replay tag) →
manual-conditioned WM train (ConditionedRSSM, KL(post||manual-conditioned-prior) + recon + reward +
continue) → imagination AC planning through the conditioned WM → four-mode eval (correct/none/swapped +
swap_follow) REUSING crafter-rtfm's own harness (run_episode/swap_follow_rate; our agent wrapped as a
crafter-rtfm Policy). Added: ReplayBuffer per-transition tag channel (manual id), crafter-rtfm as the
optional `rtfm` extra. 2-round toy smoke runs clean (scores 0.00 as expected, untrained). 27 tests
pass. Next: a real training run on the desktop GPU to see if it LEARNS to ground (swap-following > 0
on held-out r1) — the first reading-to-learn-dynamics result.

## [2026-06-14] experiment | rung-4 first grounding attempt (0032/0033) — agent does NOT read content

exp0032 (length-3): correct never lifted off zero, but a productive debugging arc calibrated the
manual-conditioned WM training (value inflation->repval; sparse reward->AC success-oversampling;
reward-head hallucination->WM trains unbiased). exp0033 (length-1 diagnostic, 4 seeds): DECISIVE
NEGATIVE caught by the swap test. correct-none "grounding" appeared (s1 0.40 vs 0.05) but
swapped~=correct (0.30-0.35) and swap_follow~=0 across all seeds => the agent ignores manual CONTENT,
exploits manual PRESENCE + a VISION shortcut (staged visual state leaks the recipe). The swap-following
metric caught a confound correct-none missed (methodology win, validates the anti-baking design).
Two-domain finding: crafter-rtfm's reading-necessity gate (text-only stub grounds, swap_follow=1.0)
does NOT hold for a vision+text learner -> handoff to crafter-rtfm (need a recipe-not-visually-inferable
mode). Next: that handoff, then re-test (consider direct actor-conditioning). Run stopped, GPU free.

## [2026-06-15] tooling | per-panel trajectory plots for experiment reports

Added `scripts/plot_experiment.py`: parses the rung-3/rung-4 training logs generically (every
`key=value` per `round N:` block; multi-seed `s*.log` aggregated to mean ± min–max band).
Theme-agnostic (transparent background + neutral-gray chrome + saturated palette → readable on light
or dark); unknown metrics auto-bucket into their own panel, so non-RTFM logs degrade gracefully.
**Always** renders the 3-column `overview.png` (all metrics — goes at the bottom of the report for
at-a-glance reference) and renders full-width `<panel>.png` ONLY for the slugs passed via `--panels`,
so committed assets are exactly the figures the report uses (overview + featured panels), nothing
else. Wired into the `/run-experiment` skill (Record step) and the experiment-page template
(**Trajectory** = featured panels + readings; **All-metrics overview** = the contact sheet at the
bottom). Backfilled exp0036/0038/0040/0041/0042/0043/0044/0045. exp0046 left un-plotted — its 4-seed
run is still training (round 12/30); regenerate at write-up. Note: the 4-seed aggregate corrected a
single-seed misread of exp0046 (grounding ignites ~0.3 at length-1 then collapses at the length-2
switch, not pinned-near-zero) — read the aggregate, not one seed.

## 2026-06-16 — exp0052 built: decoupled validated-reading reward head

Built the on-deck lever from [[0051-rtfm-flat-vr-optimization]] (coef lever exhausted: budget-2× and
coef-0.5 both ≈0.10). exp0052 applies the maintainer's insight — *every acting level needs the
grounding incentive directly* — by decoupling validated-reading (VR) from the WM's blended reward
head: new `Transition.vr` buffer channel (default 0 → all existing callers byte-for-byte unchanged),
RAW VR stored separately (task reward stays clean), a dedicated two-hot `vr_head` trained as a
DETACHED readout (WM representation identical to the validated baseline), and the imagination actor
optimises `task + λ·VR_head` (`--vr-head-coef`). Added `test_vr_channel_roundtrip`. Reviewer (opus):
SHIP (decoupling correct, detach preserves WM, two-hot represents small VR, backward-compatible, no
train/eval leakage). CPU-smoked end-to-end (vr_loss trains, raw VR stored, actor picks it up). GPU
dispatch pending — waiting for exp0051 coef-0.5 to free the 8 GB GPU. New page
[[0052-rtfm-decoupled-vr-head]] (pre-registration), INDEX updated.

## 2026-06-16 — hardware correction: local machine is a 16 GB desktop GPU

Maintainer corrected the hardware assumptions: local compute is a **16 GB desktop GPU**, not the
old 8 GB laptop (4070). CLAUDE.md hardware section updated — 16 GB, desktop, multi-hour local
training now viable (no laptop power-cap/thermal throttle), ~1.2 GB/seed so many parallel seeds fit.
compute-strategy.md: added a 2026-06-16 status note + relabelled the option table; the laptop
thermal/power-cap/benchmark measurements are now marked HISTORICAL (retired laptop), kept as
reference data. Kept tracked files depersonalized (no GPU model name). Confirmed in practice: 4
concurrent training runs (exp0051c05 ×2 + exp0052 ×2) sat at ~3.4 GB used / ~12.4 GB free.

## 2026-06-16 — new tool: visualize_rtfm.py (watchable gameplay GIFs)

Added scripts/visualize_rtfm.py — renders a watchable GIF of a trained rung-4 agent playing one
crafter-rtfm episode, with the displayed manual + per-step action + tutorial score overlaid (left:
upscaled 64×64 game frame; right: telemetry panel). SWAPPED mode is the diagnostic case (does it
follow the displayed-but-wrong manual?). Reusable helpers compose_frame()/write_gif() for a later
imagination-viewer. Built by an opus impl subagent, reviewed in the main loop. GIFs live in runs/
(gitignored); for report embedding they go to assets/. First assets rendered from exp0051c05.

## 2026-06-16 — exp0052 NEGATIVE, exp0053 localizes the wall, exp0054 dispatched

- **exp0052 (decoupled VR head) CONCLUDED NEGATIVE.** λ=1.0 collapsed the policy to manual-blind
  (swap_follow 0.00–0.02 frozen 13 rounds, dead flywheel, imagined_return inflated 15–25). Diagnosis:
  the actor reward-hacks the VR head's OOD over-predictions — exp0045/0047 redux on a new channel, no
  conservative guard. Page [[0052-rtfm-decoupled-vr-head]] finalized (verified:true).
- **exp0053 imagination-fidelity probe** (new tool `scripts/imagine_report.py`). Finding: the WM
  imagines FAITHFULLY (deterministic-h divergence only ~1.5–2.7× the real-vs-real sampling floor, for
  BOTH baseline and collapsed models) → the ~0.10 wall is NOT the world model; it's DOWNSTREAM
  (policy / reward-readout). Baseline WM is manual-sensitive (CORRECT 1.5× vs SWAPPED 2.7×); the
  collapsed exp0052 model lost that asymmetry. New page [[0053-rtfm-imagination-fidelity]].
- **exp0051 finalized**: coef-0.5 ≈0.10 → coef lever exhausted; whole grounding-incentive axis
  saturates/collapses across the lever ladder. Pivot recorded.
- **Save format**: train_rtfm now persists recon_head/rew/cont/vr_head (extra keys; older loaders
  unaffected) so the imagined-vs-real REWARD-gap metric works on future checkpoints.
- **exp0054 dispatched** (`runs/exp0054`, 2 seeds): working baseline (folded VR coef 0.3) WITH the new
  save format → then reward-gap probe to decide reward-readout-overoptimism vs pure credit-assignment.
- Scout queued the Eric Xing cluster paper (SimuRA, arxiv:2507.23773, inferred from OpenReview id
  6fDZYJYYgu which would not resolve) — verified:false, low priority; LLM-as-world-model, tangential.

## 2026-06-16 — exp0054 pre-registered (reward-gap probe)

Added the missing definition page for exp0054 (was running on the dashboard without a page). exp0054
re-runs the working ~0.10 baseline (folded VR 0.3) WITH the new save format (rew head persisted) so
the imagined-vs-real REWARD-gap can be measured — decides the downstream fork from
[[0053-rtfm-imagination-fidelity]]: reward-head over-optimism (→ conservatism) vs pure
credit-assignment (→ planning/horizon). Note: exp0053 is a DIAGNOSTIC over existing checkpoints (no
runs/exp0053 train dir); exp0054 is the fresh training run.
