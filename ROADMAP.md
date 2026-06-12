# Roadmap: autonomous game-playing world-model agents

The project climbs the environment ladder (authoritative version with exit criteria:
[knowledge/concepts/environment-ladder.md](knowledge/concepts/environment-ladder.md),
ADR [0003](knowledge/decisions/0003-environment-ladder.md)).

## Phase 1 — Pipeline bring-up (now)

- [x] Project + knowledge-base scaffold
- [ ] Experiment 0001: reproduce latent collapse with naive JEPA (`world_model.collect`)
- [ ] Experiment 0002: add SIGReg (LeJEPA) or EMA target encoder; latents survive,
      linear-probe for agent position works
- [ ] Multi-step latent rollouts; prediction-error curves vs horizon

## Phase 2 — Model-based control on MiniGrid

- [ ] Partial observability (DoorKey): add memory (GRU/transformer) to the latent model
- [ ] First *acting* agent: planning in latent space (MPC/MCTS) or small actor-critic
      trained on imagined rollouts
- [ ] Beat a model-free baseline (PPO) on sample efficiency

## Phase 3 — Crafter, then Atari100k

- [ ] DreamerV3-class baseline on Crafter; our architecture vs it
- [ ] Atari100k subset; compare against published world-model agents (IRIS, DIAMOND, DreamerV3)

## Phase 4+ — Continuous control / 3D, real-world transfer

- [ ] dm_control or Miniworld; sim2real study; camera-feed world model
- Parked until the ladder earns it (see ADR 0003).
