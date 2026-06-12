# Raw Source Registry

Immutable layer: sources are referenced by id, never edited. `verified` means the URL
was resolved and key claims checked during this project (not taken from memory).

Local reading copies: `scripts/fetch_sources.sh` downloads every arXiv source below
to `knowledge/sources/files/arxiv-<id>.pdf` (gitignored; idempotent; re-run after
new ingests). Non-arXiv sources (books, web docs) are not auto-fetched.

Paper-readiness (2026-06-12): record the VENUE/peer-review status in the title
column when known — e.g. "(ICML 2022)", "(Nature 2025)", "(preprint)" — captured
at ingest time going forward; backfill at the pre-paper lint. BibTeX export is
mechanical from arXiv ids (planned script, not needed until manuscript time).
Citation rule: a manuscript may cite only sources whose page is at read depth
`read` (read-depth labels in paper pages).

| id | title | url | type | verified | added |
|---|---|---|---|---|---|
| arxiv:2506.09985 | V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | https://arxiv.org/abs/2506.09985 | paper | yes (2026-06-12) | 2026-06-12 |
| arxiv:2509.24527 | Dreamer 4: Training Agents Inside of Scalable World Models | https://arxiv.org/abs/2509.24527 | paper | yes (2026-06-12) | 2026-06-12 |
| arxiv:2511.08544 | LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics | https://arxiv.org/abs/2511.08544 | paper | yes (2026-06-12) | 2026-06-12 |
| arxiv:2605.26379 | When Does LeJEPA Learn a World Model? | https://arxiv.org/abs/2605.26379 | paper | yes (2026-06-12) | 2026-06-12 |
| openreview:lecun-path | A Path Towards Autonomous Machine Intelligence (v0.9.2) | https://openreview.net/pdf?id=BZ5a1r-kVsf | position paper | yes (2026-06-12, title+abstract) | 2026-06-12 |
| arxiv:2205.07802 | The Primacy Bias in Deep RL (Nikishin et al.) | https://arxiv.org/abs/2205.07802 | paper | yes (2026-06-12, protocol sections) | 2026-06-12 |
| arxiv:2310.15017 | Mind the Model, Not the Agent: The Primacy Bias in Model-Based RL (Qiao et al.) | https://arxiv.org/abs/2310.15017 | paper | yes (2026-06-12, full-text method) | 2026-06-12 |
| arxiv:2508.10104 | DINOv3 (Meta) | https://arxiv.org/abs/2508.10104 | paper | yes (2026-06-12, abstract) | 2026-06-12 |
| arxiv:2301.08243 | I-JEPA: Self-Supervised Learning from Images with a JEPA | https://arxiv.org/abs/2301.08243 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2404.08471 | V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video | https://arxiv.org/abs/2404.08471 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:1803.10122 | World Models (Ha & Schmidhuber) | https://arxiv.org/abs/1803.10122 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:1811.04551 | PlaNet: Learning Latent Dynamics for Planning from Pixels | https://arxiv.org/abs/1811.04551 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2010.02193 | DreamerV2: Mastering Atari with Discrete World Models | https://arxiv.org/abs/2010.02193 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2301.04104 | DreamerV3: Mastering Diverse Domains through World Models | https://arxiv.org/abs/2301.04104 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:1911.08265 | MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model | https://arxiv.org/abs/1911.08265 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2310.16828 | TD-MPC2: Scalable, Robust World Models for Continuous Control | https://arxiv.org/abs/2310.16828 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2209.00588 | IRIS: Transformers are Sample-Efficient World Models | https://arxiv.org/abs/2209.00588 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2405.12399 | DIAMOND: Diffusion for World Modeling | https://arxiv.org/abs/2405.12399 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2402.15391 | Genie: Generative Interactive Environments | https://arxiv.org/abs/2402.15391 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2501.03575 | NVIDIA Cosmos World Foundation Model Platform | https://arxiv.org/abs/2501.03575 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2506.01622 | General agents contain world models (Richens et al.; title corrected at verification) | https://arxiv.org/abs/2506.01622 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2109.06780 | Crafter: Benchmarking the Spectrum of Agent Capabilities (Hafner) | https://arxiv.org/abs/2109.06780 | paper | yes (2026-06-12, abstract) | 2026-06-12 |
| arxiv:2411.14499 | Survey: Understanding World or Predicting Future? | https://arxiv.org/abs/2411.14499 | survey | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| web:gym-docs | Gymnasium documentation | https://gymnasium.farama.org | docs | no | 2026-06-12 |
| web:minigrid-docs | MiniGrid documentation | https://minigrid.farama.org | docs | no | 2026-06-12 |
| web:worldmodels-site | World Models interactive article | https://worldmodels.github.io/ | article | no | 2026-06-12 |
| web:karpathy-llm-wiki | Karpathy: LLM Wiki gist | https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f | article | yes (2026-06-12) | 2026-06-12 |
| pdf:context-arch-v1 | Context Architecture for Enterprise Agentic Software Delivery v1.0.0 (D. Verholen) | local PDF (not in repo); via author: https://www.linkedin.com/in/david-verholen-14aa23aa/ | whitepaper | yes (2026-06-12, read in full) | 2026-06-12 |
| news:ami-funding | AMI Labs raises $1.03B (TechCrunch, 2026-03-09) | https://techcrunch.com/2026/03/09/yann-lecuns-ami-labs-raises-1-03-billion-to-build-world-models/ | news | yes (2026-06-12) | 2026-06-12 |
