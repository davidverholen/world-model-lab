---
status: draft
owner: world-model
scope: local
sources: [arxiv:2310.07418]
verified: true
last_reviewed: 2026-06-13
---

# Revisiting Plasticity in Visual RL: Data, Modules, Training Stages (Ma et al., ICLR 2024)

**Read state:** skimmed (method+evidence via full text, 2026-06-13) · ingested
during the exp-0013→0014 literature gate [autonomous]

## One-paragraph summary

Module-level analysis of plasticity loss in model-free visual RL using **FAU**
(fraction of active units). Findings: the ENCODER's plasticity stays naturally
healthy; the CRITIC's plasticity collapses early under bootstrapped targets and is
the true sample-efficiency bottleneck; data augmentation works mainly by
preserving critic plasticity (not representation quality); early intervention is
critical (late rescue impossible); **frozen ImageNet-pretrained encoders work
fine** as their isolation probe. Remedy: Adaptive RR (start replay ratio 0.5,
raise to 2 when critic FAU stabilizes).

## Relevance to our experiments — the exp-0013 "tension" resolves

Not a contradiction but a complement: they show encoders don't need plasticity
(frozen pretrained ones suffice); we show (exp 0013) our encoder's continued
training actively destroys downstream function (drift-interference, not
dormancy-plasticity-loss — different pathology, same prescription: STOP TRAINING
THE ENCODER once competent). Their critic-bottleneck mechanism (TD bootstrapping
non-stationarity) doesn't apply to our MC value targets. Imports for us:
1. **FAU logging per module per round** — cheap instrumentation to distinguish
   dormancy from drift in our stack (exp 0015 candidate);
2. their frozen-pretrained-encoder probe independently legitimizes the
   DINOv3/frozen-trunk line ([[retention]]);
3. early-stage criticality matches our round-0/1 crash timing;
4. Adaptive-RR-style scheduling (UTD as a *schedule*, not a constant) is an
   unexplored lever atop our exp-0012 UTD finding.

## Links

[[retention]] · [[0013-trunk-freeze]] · [[0012-utd-and-targeted-resets]] ·
[[nikishin-primacy-2022]] · [[qiao-model-primacy-2023]]
