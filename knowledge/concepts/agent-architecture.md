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
    subgraph L4["L4 FLYWHEEL"]
        direction LR
        collect["collect<br/>eps-greedy MPC"] --> trainstep["train on replay"] --> evalstep["eval + best-<br/>checkpoint guard"] --> collect
    end

    subgraph L3["L3 TRAINING"]
        direction LR
        loss["prediction + SIGReg<br/>+ reward + value loss"] --- tricks["success oversampling<br/>EMA / lr-decay arms"]
    end

    subgraph L2["L2 BELIEF"]
        direction LR
        enc["encoder<br/>RGB to z"] --> gru["GRU belief s"] --> heads["heads:<br/>z-hat, r, V"]
    end

    subgraph L1["L1 PLANNING"]
        cem["imagine 512 futures, score<br/>sum gamma^t r + gamma^H V"]
    end

    L4 -->|replay windows| L3
    L3 -->|weights / EMA shadow| L2
    L2 -->|belief s_t| L1
    L1 -->|actions| L4

    classDef box fill:#475569,stroke:#94a3b8,color:#f8fafc
    class collect,trainstep,evalstep,loss,tricks,enc,gru,heads,cem box
    style L4 fill:none,stroke:#64748b
    style L3 fill:none,stroke:#64748b
    style L2 fill:none,stroke:#64748b
    style L1 fill:none,stroke:#64748b
```

Full per-layer specifics (loss weights, planner params, ignition rules) are in the
table below and the module docstrings.

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
