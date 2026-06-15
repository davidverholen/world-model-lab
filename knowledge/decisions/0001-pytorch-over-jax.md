---
status: current
owner: human
scope: local
sources: []
verified: true
last_reviewed: 2026-06-12
---

# 0001: PyTorch over JAX

**Status:** accepted (2026-06-12)

## Context

The two serious ecosystems for world-model research are PyTorch and JAX. Official
DreamerV3 and Craftax are JAX; I-JEPA, V-JEPA 2, LeJEPA, TD-MPC2, IRIS, DIAMOND are
PyTorch. We have one consumer GPU (laptop GPU, 8 GB) and value iteration speed
over peak throughput.

## Decision

PyTorch as the default framework (`torch>=2.12`, with `torch.compile` when it helps).

## Consequences

- Direct reuse of the JEPA-family reference code we care about most.
- DreamerV3 baselines run via community PyTorch ports or by treating published numbers
  as the baseline; if exact reproduction matters later, running the JAX original in a
  separate venv is acceptable and does not change this decision.
- Revisit only if the project pivots to heavy parallel-env training where JAX's
  end-to-end JIT (à la Craftax) is decisive.

## Links

[[0003-environment-ladder]] · [[imagination-training]]
