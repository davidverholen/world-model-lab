---
status: current
owner: agent
scope: local
sources: [arxiv:2511.08544, arxiv:1811.04551]
verified: true
last_reviewed: 2026-06-12
---

# Our Agent: the Four Optimization Layers

## What it is

The system we actually built (exps 0001–0010), as nested optimization loops — each
layer optimizes something different, on a different timescale, with a different
algorithm:

```mermaid
flowchart TB
    subgraph L4["L4 - DATA FLYWHEEL: optimizes the data distribution (per round, ~15k env steps)"]
        direction LR
        collect["collect with eps-greedy MPC agent<br/>(round 0: random + adaptive ignition >=5 successes)"]
        trainstep["train all modules on replay"]
        evalstep["eval greedy, fixed seeds<br/>+ best-checkpoint guard"]
        collect --> trainstep --> evalstep --> collect
    end

    subgraph L3["L3 - MODEL TRAINING: optimizes weights (Adam 3e-4, per update)"]
        loss["joint loss = (1-lambda) multi-step latent prediction<br/>+ lambda SIGReg (anti-collapse, 0.05)<br/>+ reward loss (pos_weight 100)<br/>+ value loss (MC return-to-go, pos_weight 20)"]
        tricks["success-window oversampling 25%<br/>grad-clip 10 - exp 0010: EMA shadow / lr decay"]
        loss --- tricks
    end

    subgraph L2["L2 - BELIEF and MODEL: optimizes state estimation (per env step)"]
        enc["ConvEncoder: 56x56 RGB to z (128)"]
        gru["GRU belief s (256): closed-loop on real z,<br/>open-loop on imagined z-hat"]
        heads["heads: next z-hat, reward r(s,a), value V(s)"]
        enc --> gru --> heads
    end

    subgraph L1["L1 - PLANNING: optimizes the action (CEM, per env step)"]
        cem["sample 512 action sequences (H=20)<br/>imagine rollouts in latent space<br/>score: sum gamma^t r + gamma^H V(s_H)<br/>refit to elites x2-3, act, re-plan"]
    end

    L4 -->|replay windows| L3
    L3 -->|weights or EMA shadow| L2
    L2 -->|belief s_t| L1
    L1 -->|actions / env steps| L4
```

## Which algorithm optimizes what

| Layer | Optimizer | Objective | Timescale | Known failure (exp) |
|---|---|---|---|---|
| L1 planning | CEM (sampling) | discounted imagined return | per action | horizon blindness → value head (0005→0006); argmax oscillation (0004) |
| L2 belief/model | — (forward pass) | state estimation | per step | obs-scale mismatch (0006); collapse w/o SIGReg (0001) |
| L3 training | Adam on joint loss | model+heads fit | per update | **catastrophic interference (0009/0010, open)**; reward starvation (0004); compounding rollout error (0004) |
| L4 flywheel | greedy data loop | data quality | per round | ignition (0008→0009, solved); instability amplification (0005) |

## Why it matters here

Every experiment so far has been a failure at exactly one layer, fixed by a
mechanism at that layer. When something breaks, locate the layer first — the table
doubles as a diagnosis guide. Contrast with the published systems: Dreamer replaces
L1 with a learned actor (amortized planning — our likely path for Crafter speed);
TD-MPC2 keeps L1 but learns value by TD instead of MC; PPO has no L2/L4 and fuses
L1 into the policy.

## Links

[[jepa]] · [[latent-collapse]] · [[imagination-training]] · [[value-equivalent-planning]] ·
experiments 0001–0010 · `src/world_model/` (modules link back here)
