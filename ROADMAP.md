# Roadmap: autonomous game-playing world-model agents

The project climbs the environment ladder (authoritative version with exit criteria:
[environment-ladder](knowledge/design/environment-ladder.md),
ADR [0003](knowledge/decisions/0003-environment-ladder.md)). Distinct capabilities and
their dependency order live in [capability-map](knowledge/design/capability-map.md); the
durable experiment record is under [knowledge/experiments/](knowledge/experiments/).

**Where we are:** rung 3 (Crafter). The agent learns and plays; current focus is fixing
*reward exploitation* (the model imagining reward that isn't real) so it explores and
climbs the tech tree instead of farming the trivial achievements.

## Phase 1 — Pipeline + representation (rung 1) ✅

- [x] Project + knowledge-base scaffold (Karpathy-wiki × context-architecture operating model)
- [x] Reproduce latent collapse with naive JEPA; fix with SIGReg/LeJEPA (exp 0001–0002)
- [x] Multi-step latent rollouts — required for planning (compounding error, exp 0004)
- [x] Rung-1 exit: world model predicts held-out transitions far better than a copy baseline (exp 0003)

## Phase 2 — Model-based control on MiniGrid (rung 2) ✅

- [x] First *acting* agent: latent CEM-MPC plays Empty-8x8 at 20/20 vs random 3/20 (exp 0004)
- [x] Partial observability (DoorKey): GRU belief memory + value head + the
      collect→train→collect flywheel (exp 0005–0006)
- [x] Honest PPO comparison: model-free PPO wins on raw sample-efficiency at DoorKey-5x5
      (exp 0007) — recorded as a loss, not papered over; escalated to DoorKey-6x6
- [x] **The imagination actor-critic line** — the core of the project's method, built
      incrementally on DoorKey-6x6: stochastic latents / RSSM (exp 0019), the
      model-exploitation problem diagnosed and the value calibrated via DreamerV3
      critic-on-replay (exp 0017–0021). Banked as a *calibrated* imagination loop — the
      reusable foundation carried up the ladder.
- Cross-cutting result: **retention / plasticity** characterized (encoder drift =
      interference; freeze fixes it) — exp 0009–0013,
      [retention](knowledge/concepts/retention.md).

## Phase 3 — Crafter (rung 3) ⏳ current

- [x] Decision: original Crafter over Craftax/JAX
      (ADR [0006](knowledge/decisions/0006-crafter-vs-craftax.md)); env wrapper + first contact
- [x] **Frozen pretrained encoder**: DINOv2 features validated to encode Crafter state
      (linear probe 0.98 vs 0.80 baseline, exp 0022) → FrozenDinoEncoder + embedding cache.
      Sidesteps the from-scratch representation instability that dominated rung 2.
- [x] Calibrated imagination loop ported to Crafter (`train_crafter.py`); it *learns*
      (reward/achievements climb off the random floor, exp 0023)
- [x] **Value-scale hardening** for Crafter's denser reward: two-hot distributional critic
      + two-hot reward head (exp 0023–0025) — bounds the imagined value so the agent can't
      chase hallucinated reward.
- [x] **Behavioral QA gate** ([scripts/behavior_report.py](scripts/behavior_report.py)):
      auto-flags degenerate policies (stationary, action-collapse) that eval-reward alone
      hides — catches *how* it plays, not just the score.
- [ ] **Consistent navigation + tech-tree climb** (in progress): patch tokens unlock
      movement (exp 0024); the reward fix should make it consistent across seeds and stop
      the exploitation (stationary / drink-at-cap / no-op spam).
- [ ] **Measure the scaling slope**: does 2×/4× compute climb the mid-tree? (we are ~10×
      under-trained vs DreamerV3's reference; the plateau is largely a scaling story, not a
      wall — see [crafter](knowledge/environments/crafter.md))
- [ ] **Hierarchy** for the deep tree (stone→iron→diamond), as a compute-*efficiency* lever
      ([hierarchy-and-credit](knowledge/concepts/hierarchy-and-credit.md))
- [ ] DreamerV3-class score comparison; optionally Atari100k as a second rung-3 data point

## Phase 4 — Continuous control / 3D (rung 4)

- [ ] dm_control / Miniworld — parked until rung 3 earns it

## Phase 5 — Minecraft (rung 5b, ADR [0005](knowledge/decisions/0005-minecraft-milestone.md)): the pre-real-world milestone

- [ ] MineRL/MineDojo up; baseline study (DreamerV3 online / Dreamer 4 offline / VPT)
- [ ] Offline world-model pretraining from action-labeled human video (VPT corpus)
- [ ] Self-play flywheel at Minecraft scale (retention fixes must hold)
- [ ] **Language into the world model** — the text-staircase (Messenger/RTFM → text-Crafter →
      Minecraft wiki); binding + installation
      ([language-grounding](knowledge/concepts/language-grounding.md)). Open research; the
      project's differentiated bet.
- [ ] "Successfully play": diamond at a published sample-efficiency tier + task breadth

## Phase 6 — Real-world transfer

- [ ] sim2real study; camera-feed world model. Parked until the ladder earns it.

---

**Operating strategy** (the home-lab moat): compute *efficiency*, not scale — judge
architectures by capability-per-FLOP slope, pay minimum-sufficient scale, measure slopes
not endpoints ([compute-strategy](knowledge/design/compute-strategy.md)). **Principle:**
never hardcode environment structure into the model — the agent must *learn* the env's
caps, reward, and dynamics from experience (bias the learning process, not the content).
