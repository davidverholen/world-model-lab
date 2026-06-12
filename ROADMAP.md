# Roadmap: autonomous game-playing world-model agents

The project climbs the environment ladder (authoritative version with exit criteria:
[knowledge/concepts/environment-ladder.md](knowledge/concepts/environment-ladder.md),
ADR [0003](knowledge/decisions/0003-environment-ladder.md)).

## Phase 1 — Pipeline bring-up (now)

- [x] Project + knowledge-base scaffold
- [x] Experiment 0001: reproduce latent collapse with naive JEPA (`world_model.collect`)
- [x] Experiment 0002: add SIGReg (LeJEPA); latents survive (linear probe demoted, exp 0003)
- [x] Multi-step latent rollouts (exp 0004: required for planning — compounding error)

## Phase 2 — Model-based control on MiniGrid

- [x] First *acting* agent: latent MPC/CEM plays Empty-8x8 at 20/20 vs random 3/20
      (exp 0004; `python -m world_model.play --checkpoint ...`)
- [x] Partial observability (DoorKey): GRU belief memory + value head solves
      DoorKey-5x5 at ~80% vs 8% random (exp 0006, trained on the remote GPU)
- [ ] Beat a model-free baseline (PPO) on sample efficiency → closes rung 2
      (exp 0007: PPO wins on DoorKey-5x5 @100k — escalate to DoorKey-6x6/8x8 and
      spin the flywheel earlier; exp 0008)

## Phase 3 — Crafter, then Atari100k

- [ ] DreamerV3-class baseline on Crafter; our architecture vs it
- [ ] Atari100k subset; compare against published world-model agents (IRIS, DIAMOND, DreamerV3)

## Phase 4 — Continuous control / 3D

- [ ] dm_control or Miniworld

## Phase 5 — Minecraft (ADR 0005): the pre-real-world milestone

- [ ] MineRL env up; baseline study (DreamerV3 online / Dreamer 4 offline / VPT)
- [ ] Offline world-model pretraining from action-labeled video (VPT corpus)
- [ ] Self-play flywheel at Minecraft scale (retention fixes must hold here)
- [ ] Language into the world model (Dynalang/VL-JEPA direction) — open research
- [ ] "Successfully play": diamond at a published efficiency tier + task breadth

## Phase 6 — Real-world transfer

- [ ] sim2real study; camera-feed world model
- Parked until the ladder earns it (ADRs 0003, 0005).
