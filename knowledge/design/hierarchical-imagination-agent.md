---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-15
---

# Hierarchical imagination agent — dual-process planning over the world model

**Status: draft, for review.** The agent-side design for the next major capability
after reading-to-learn-dynamics. Motivated by [[0043-rtfm-execution-wall]]: the world model *reads*
a multi-step recipe but the flat reactive actor *cannot execute* it — the same bottleneck as the
rung-3 Crafter depth plateau (exps 0026–0031, actor/discovery-bound). Targets the directable-
competence milestone ([[0007-crafter-mastery-milestone]]); this is its execution + abstraction
pillars. Built from a design conversation (2026-06-15).

## 1. The wall this solves

Flat imagination fails at multi-step execution. Two independent demonstrations: CEM-MPC's horizon was
too short for long chains (rung 2, exp 0016–0017); the reactive imagination-actor can't sequence even
a 2-action gesture to a sparse one-shot reward (exp 0043, with the WM provably reading it). The
missing capability is **multi-step / compositional execution** — temporal abstraction, hierarchy,
sequential credit assignment.

## 2. The architecture — a dual-process loop (System 1 / System 2)

Modelled on human cognition: practiced tasks run automatically; only novel/hard ones invoke
deliberate planning; deliberation compiles into automaticity with repetition.

- **System 1 — reactive default.** A well-practiced task is grounded in latent space and runs
  automatically: belief → action, fast, no planning, no language. This is the amortized policy /
  Director-style worker. **Most behaviour lives here.** (We already have this: the imagination-AC
  reactive actor.)
- **System 2 — deliberate fallback.** Invoked only when System 1 can't (see §5 arbitration). Lay out
  a rough plan — partly inner *language* (the subgoal skeleton), partly *imagination* (the concrete
  steps) — then **recursively decompose** (§3) and execute the leaves by imagination-planning (MPC).
  Slow, occasional, expensive. Text is the interface **here only — NOT the default runtime path.**
- **Compilation (System 2 → System 1).** Deliberate solutions are distilled into the reactive policy,
  so a task done many times becomes automatic and stops needing planning. Deliberation is therefore
  both a runtime fallback *and a teacher* generating targets for System 1. (We built actor-distillation
  once — exp 0016 — the mechanism exists.)

## 3. Recursive read-grounded decomposition (not a fixed N-level hierarchy)

Director is a hardcoded 2-level manager/worker; HAC generalizes to *k* levels but deeper *trained*
hierarchies are unstable (each level is a moving target for the one above). Instead of a fixed tower,
use a single **decompose-or-execute** operator applied **recursively**:

- *Can imagination-planning (MPC) reach this goal within its horizon, confidently?* → **execute**.
- *Too far?* → **decompose into subgoals, recurse on each.**

Depth then **adapts to task complexity automatically**: "place a table" bottoms out at once; "get a
diamond" recurses several levels (diamond → iron pickaxe → furnace → iron → stone → wood), each level
smaller and more detailed, until leaves are short enough for imagination. One recursive planner, not N
trained levels — so it sidesteps the deep-hierarchy instability.

**Where the decomposition comes from — the thesis payoff.** The Crafter recipe DAG *is* a recursive
task structure (an HTN — hierarchical task network). The manual **is** the decomposition tree, so we
**read** the hierarchy instead of learning it (the fragile part). The [[rung4-manual-conditioned-agent]]
binding machinery (manual → world-model conditioning) is exactly what turns a read text-subgoal into a
latent goal the planner can reach. Director's learned VQ-VAE goal manager is the **fallback** for the
parts of a task not covered by a manual.

### 3a. The decomposition operator runs BACKWARD from the goal — and it's the SAME primitive as the training curriculum

The decompose-or-execute operator (§3) is naturally **backward**: start at the final goal and ask *"what
must be true just before this?"* — goal regression through **preconditions**. Each precondition is the
next subgoal back; recurse. Depth **adapts to complexity** by the §5 confidence gate: recurse only until a
leaf is reactively (System-1) executable with confidence — a simple goal bottoms out immediately (no
hierarchy at all), a complex goal regresses several levels. So the hierarchy stays **lazy / only-when-we-
need-it**, never a fixed tower (consistent with the §7 "don't build the tower early" discipline).

**The payoff — one backward primitive, two uses (training + inference).** This is the *same* operation as
the [[0066-rtfm-imagination-backward-curriculum]] training lever:
- **Training time** (learn the chain): seed actor-critic imagination from buffered **prefix-complete
  latents** and learn to complete the tail — manufacturing the deep-step experience that exploration
  reaches only ~p^(k-1) of the time ([[0065-rtfm-per-step-plateau-large-budget]] mechanism).
- **Inference time** (plan the chain): regress backward from the goal through preconditions to emit the
  subgoal skeleton, then execute the leaves by imagination/MPC.

Both are "work backward from the goal." Building the backward curriculum first therefore *also* exercises
the substrate the recursive decomposition will reuse — the same lazy escalation as the read↔LLM seam (§4).

**Preconditions are both the obstacle and the scaffold.** The precondition relation (step k-1 enables
step k) is exactly what *blocks* a real-env backward curriculum — you can't reset reality into a step-k-
ready state without first doing step k-1 — which is why the curriculum lives in **imagination** (seed a
*believed*-done latent). That same precondition relation is the structure the decomposition **regresses
over** to build the subgoal tree. The thing that forces us into imagination at training time is the thing
that gives the hierarchy its edges at inference time.

**Lit-grounded design constraints for exp0066** (from [[romi-2021]] / [[florensa-2017]] / [[lexa-2021]]):
- **Seed only from REAL buffered prefix-done latents — never imagination-of-imagination — and walk back
  one step at a time** (ROMI's compounding-error argument: backward-imagination error grows with rollout
  depth). We get this guarantee by construction; corollary: **no learned reverse-dynamics model needed**,
  the replay buffer's real latents are the targets.
- **Advance the frontier by a difficulty gate, not a fixed schedule** (Florensa's Starts-of-Intermediate-
  Difficulty): only add step-(k-2) seeds once step-k success is *stable but non-trivial* (~10–90%). The
  open risk is whether the buffer accumulates enough prefix-complete latents to supply the frontier — if
  step-(k-j) latents go too sparse, a more active seeding mechanism is needed.
- **Actor conditioning (resolved):** the imagined-from-frontier actor conditions on the **frontier latent
  + the recipe TEXT** (our existing [[rung4-manual-conditioned-agent|ConditionedRSSM]]) — i.e. "you are
  here, here's the recipe, finish it." A LEXA-style explicit **goal-latent** signal is a later upgrade,
  not the first cut (isolate one lever). LEXA is the existence proof that an achiever trained *purely in
  imagination* transfers to real multi-step execution, so the bet is sound; the risk is buffer supply,
  not whether imagination transfers.

**Discipline (when, not now):** the flat backward curriculum (exp0066) comes first; the recursive backward
*decomposition* is the escalation for when goals get genuinely complex (branching sub-recipes, depth ≥3) —
triggered by the same calibrated-confidence gate (§5), not built ahead of need.

## 4. Text is the System-2 *sketch* interface — read now, LLM later

The abstraction layer's input is a **textual plan** (an ordered list of subgoal descriptions), and its
*source is swappable*:

- **Read** (now): the env's manual / a wiki — grounded, swap-testable. This is what we have.
- **Reason** (far seam): an internal "plan mode" — a **frozen/API LLM** describes a plan for a novel
  goal before decomposition. The LLM is a **commodity** (same stance as the frozen text encoder); we
  train nothing there. The moat stays in the binding + imagination layer.

Critically, reading and LLM-reasoning are **two sources of the same thing** (a text plan) feeding the
**same binding layer**. So building reading-grounded decomposition now *also* builds the substrate the
LLM-plan-mode reuses later — zero re-architecting; design the seam now, build the LLM layer many rungs
out. And text only *sketches* the skeleton; **imagination fills the concrete steps** — the plan is
sparse structure, not a full action script.

## 5. Arbitration — confidence-gated, and the calibration dependency

**The trigger to escalate System 1 → System 2 → seek-language is the world model's confidence in its
own imagined path to the goal.** The planner imagines a path; if the best path reaches the goal with
confidence **above threshold**, execute. If **no confident path exists**, the agent has discovered a
gap in its own knowledge ("I don't know how to do this") → fetch language (read the relevant text /
query the LLM) → re-plan with the new information → if now confident, execute (and compile to System 1).

This makes reading **demand-driven, not always-on** — the agent consults text only at its knowledge
gaps (more efficient, more human, and *self-limiting* on text-reliance, which is good for anti-baking).

**LOAD-BEARING RISK — calibrated uncertainty.** The whole trigger rests on the model *knowing when it
doesn't know*. That is exactly the failure mode of exps 0017–0025: our imagination was *overconfident*
(hallucinated reward/value). An overconfident world model imagines a confident-but-wrong path and
**never triggers learning**. So **calibration is the prerequisite** for arbitration, not a nice-to-have.
Candidate confidence signals we partly have: the **two-hot distributional** reward/value heads (spread =
uncertainty); ensemble-disagreement (Plan2Explore) if needed; the imagined plan's predicted
goal-success probability. The open work is making these *trustworthy enough to gate on*.

This is no longer hypothetical at the rung-4 execution layer: [[0045-rtfm-oracle-probe]] showed the
reward head ranks the true gesture only ~88th percentile (~12% of OOD action sequences overrated above
it), and [[0046-rtfm-robust-planning]] found robust *planning* (K=10 sampled-rollout averaging) only
lifts that to ~0.92 — it discounts the optimism but cannot fix a head starved of length-2 positives.
Empirical confirmation that calibration is a *training-time* prerequisite, not a planning-time patch.

## 6. Precedents (lineage)

- [[director-2022]] — 2-level manager(VQ subgoal)/worker, both trained in imagination. Our structural
  base; we make decomposition recursive + read-grounded, and worker via MPC first.
- HAC (Levy) — multi-level goal-conditioned hierarchy; cautionary on deep-hierarchy instability.
- [[imagination-training]] — our reactive imagination-AC = System 1.
- exp 0016 actor-distillation — the System-2 → System-1 compilation mechanism.
- [[dreamer4-2025]] — needed explicit per-subtask staging for *its* deep tree; independent evidence
  that the deep tree needs decomposition, not aggregate reward.
- Plan2Explore / [[retention|curious replay]] — disagreement/uncertainty as the confidence signal.

## 7. Near-term staircase (each rung isolates one pillar; build bottom-up)

1. **exp 0044 — execution pillar, via MPC, on our exact wall.** CEM-MPC over the manual-conditioned WM
   on length-2: search 2-action sequences in imagination, pick the one the (reading) reward head
   predicts earns the gesture. Hypothesis: planning **cracks length-2 where the reactive actor
   flatlined (correct≈0)** — MPC *searches* the sequence instead of learning it by policy-gradient
   through a sparse one-shot reward. Sharp diagnostic: MPC works ⇒ wall was credit-assignment, planning
   is the fix; MPC *also* fails ⇒ wall is WM rollout-accuracy (a different problem). Uses rung-2 CEM-MPC.
2. **Compile** the MPC solution back into the reactive policy (System 2 → System 1) so length-2 becomes
   automatic — first instance of the full default/fallback/compile cycle.
3. **One decomposition level** on a short Crafter state-chain (e.g. get-stone), Director-style
   state-subgoals (the natural fit for Crafter, unlike rtfm's action-gesture); read the subgoal plan.
4. **Recursion** when a deeper goal (diamond) demands it; **calibrated confidence** as the arbitration
   gate; **confidence-gated demand-driven reading**.
5. **LLM plan-mode** at the same text-plan seam — far end.

**Discipline:** do not build the recursive tower / LLM layer before the bottom rung (MPC executes a
leaf) works. Each rung pre-registers a falsifiable hypothesis and a kill signal.

## 8. Open questions (for review)

- Low-level execution: **MPC (inference-time search)** vs a **trained goal-conditioned worker**
  (Director). MPC first (matches "imagine actions", cheap, no new training); worker for scale.
- Subgoal representation: latent target state (Director VQ) vs read symbolic step — and how a *read*
  subgoal binds to a worker/MPC-reachable target.
- The calibration signal (§5): is the two-hot spread enough, or do we need an ensemble?
- Arbitration threshold: too high → always reading (inefficient, baking risk); too low → never asks
  for help (fails on novel goals).

## Links

[[0007-crafter-mastery-milestone]] · [[0043-rtfm-execution-wall]] · [[0065-rtfm-per-step-plateau-large-budget]] · [[0066-rtfm-imagination-backward-curriculum]] · [[director-2022]] · [[hierarchy-and-credit]] · [[temporal-abstraction]] · [[imagination-training]] · [[rung4-manual-conditioned-agent]] · [[language-grounding]] · [[dreamer4-2025]]
