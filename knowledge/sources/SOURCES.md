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
| arxiv:2306.15934 | Curious Replay for Model-based Adaptation (Kauvar et al., ICML 2023) | https://arxiv.org/abs/2306.15934 | paper | yes (2026-06-14, HTML full-text) | 2026-06-14 |
| arxiv:1911.08265 | MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model | https://arxiv.org/abs/1911.08265 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2310.16828 | TD-MPC2: Scalable, Robust World Models for Continuous Control | https://arxiv.org/abs/2310.16828 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2209.00588 | IRIS: Transformers are Sample-Efficient World Models | https://arxiv.org/abs/2209.00588 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2405.12399 | DIAMOND: Diffusion for World Modeling | https://arxiv.org/abs/2405.12399 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2402.15391 | Genie: Generative Interactive Environments | https://arxiv.org/abs/2402.15391 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2501.03575 | NVIDIA Cosmos World Foundation Model Platform | https://arxiv.org/abs/2501.03575 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2506.01622 | General agents contain world models (Richens et al.; title corrected at verification) | https://arxiv.org/abs/2506.01622 | paper | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2109.06780 | Crafter: Benchmarking the Spectrum of Agent Capabilities (Hafner) | https://arxiv.org/abs/2109.06780 | paper | yes (2026-06-12, abstract) | 2026-06-12 |
| arxiv:2411.14499 | Survey: Understanding World or Predicting Future? | https://arxiv.org/abs/2411.14499 | survey | yes (2026-06-12, title via arXiv API) | 2026-06-12 |
| arxiv:2512.04797 | SIMA 2: A Generalist Embodied Agent for Virtual Worlds (DeepMind, Dec 2025) | https://arxiv.org/abs/2512.04797 | paper | yes (2026-06-15, title+authors+abstract via arXiv page) | 2026-06-15 |
| web:genie2-blog | Genie 2: A large-scale foundation world model — Google DeepMind blog (Dec 2024, no paper) | https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/ | blog | yes (2026-06-15, fetched) | 2026-06-15 |
| web:genie3-blog | Genie 3: A new frontier for world models — Google DeepMind blog (Aug 2025, no paper) | https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/ | blog | yes (2026-06-15, fetched) | 2026-06-15 |
| web:gym-docs | Gymnasium documentation | https://gymnasium.farama.org | docs | no | 2026-06-12 |
| web:minigrid-docs | MiniGrid documentation | https://minigrid.farama.org | docs | no | 2026-06-12 |
| web:worldmodels-site | World Models interactive article | https://worldmodels.github.io/ | article | no | 2026-06-12 |
| web:karpathy-llm-wiki | Karpathy: LLM Wiki gist | https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f | article | yes (2026-06-12) | 2026-06-12 |
| pdf:context-arch-v1 | Context Architecture for Enterprise Agentic Software Delivery v1.0.0 (D. Verholen) | local PDF (not in repo); via author: https://github.com/davidverholen/context-architecture | whitepaper | yes (2026-06-12, read in full) | 2026-06-12 |
| news:ami-funding | AMI Labs raises $1.03B (TechCrunch, 2026-03-09) | https://techcrunch.com/2026/03/09/yann-lecuns-ami-labs-raises-1-03-billion-to-build-world-models/ | news | yes (2026-06-12) | 2026-06-12 |
| arxiv:1910.08210 | RTFM: Generalising to Novel Environment Dynamics via Reading (Zhong et al., ICLR 2020) | https://arxiv.org/abs/1910.08210 | paper | yes (2026-06-14, title+authors via arXiv API) | 2026-06-14 |
| arxiv:2101.07393 | Grounding Language to Entities and Dynamics for Generalization in RL — Messenger/EMMA (Hanjie et al., ICML 2021) | https://arxiv.org/abs/2101.07393 | paper | yes (2026-06-14, title+authors via arXiv API) | 2026-06-14 |
| arxiv:2308.01399 | Dynalang: Learning to Model the World with Language (Lin et al., 2023) | https://arxiv.org/abs/2308.01399 | paper | yes (2026-06-14, title+authors confirmed) | 2026-06-14 |
| arxiv:2511.22904 | Language-conditioned world model improves policy generalization by reading environmental descriptions — LED-WM (Nguyen & Lee, 2025) | https://arxiv.org/abs/2511.22904 | paper | yes (2026-06-14, title+authors+Messenger eval confirmed) | 2026-06-14 |
| arxiv:2210.00066 | Improving Policy Learning via Language Dynamics Distillation — LDD (Zhong et al., 2022) | https://arxiv.org/abs/2210.00066 | paper | yes (2026-06-14, title+authors via arXiv API) | 2026-06-14 |
| arxiv:2305.16621 | A Reminder of its Brittleness: Language Reward Shaping May Hinder Learning for Instruction Following Agents (Huang et al., 2023) | https://arxiv.org/abs/2305.16621 | paper | yes (2026-06-14, title+authors+abstract confirmed) | 2026-06-14 |
| arxiv:2110.10661 | SILG: The Multi-environment Symbolic Interactive Language Grounding Benchmark (Zhong et al., NeurIPS 2021) | https://arxiv.org/abs/2110.10661 | paper | yes (2026-06-14, title+authors via arXiv API) | 2026-06-14 |
| icml:ng1999shaping | Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping (Ng, Harada & Russell, ICML 1999) | https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html | paper | yes (2026-06-14, title+authors confirmed via S2) | 2026-06-14 |
| arxiv:1707.01495 | Hindsight Experience Replay — HER (Andrychowicz et al., NeurIPS 2017) | https://arxiv.org/abs/1707.01495 | paper | yes (2026-06-14, title+authors via arXiv API) | 2026-06-14 |
| arxiv:2206.04114 | Director: Deep Hierarchical Planning from Pixels (Hafner, Lee, Fischer, Abbeel — NeurIPS 2022) | https://arxiv.org/abs/2206.04114 | paper | yes (2026-06-15, title+authors+venue confirmed via arXiv abstract page) | 2026-06-15 |
| arxiv:2006.04779 | Conservative Q-Learning for Offline Reinforcement Learning — CQL (Kumar, Zhou, Tucker & Levine, NeurIPS 2020) | https://arxiv.org/abs/2006.04779 | paper | yes (2026-06-15, title+authors via arXiv API batch fetch) | 2026-06-15 |
| arxiv:2005.05951 | MOReL: Model-Based Offline Reinforcement Learning (Kidambi, Rajeswaran, Netrapalli & Joachims, NeurIPS 2020) | https://arxiv.org/abs/2005.05951 | paper | yes (2026-06-15, title+authors via arXiv API batch fetch) | 2026-06-15 |
| arxiv:2005.13239 | MOPO: Model-based Offline Policy Optimization (Yu, Thomas, Yu et al., NeurIPS 2020) | https://arxiv.org/abs/2005.13239 | paper | yes (2026-06-15, title+authors via arXiv API batch fetch) | 2026-06-15 |
| arxiv:2102.08363 | COMBO: Conservative Offline Model-Based Policy Optimization (Yu, Kumar, Rafailov et al., NeurIPS 2021) | https://arxiv.org/abs/2102.08363 | paper | yes (2026-06-15, title+authors via arXiv API batch fetch) | 2026-06-15 |
| arxiv:2210.03802 | Conservative Bayesian Model-Based Value Expansion for Offline Policy Optimization — CBOP (Jeong, Wang, Gimelfarb et al., ICLR 2023) | https://arxiv.org/abs/2210.03802 | paper | yes (2026-06-15, title+authors via arXiv API batch fetch) | 2026-06-15 |
| arxiv:2310.17245 | CROP: Conservative Reward for Model-based Offline Policy Optimization (Li, Zhou, Li et al., 2023, preprint) | https://arxiv.org/abs/2310.17245 | paper | yes (2026-06-15, title+authors via arXiv API batch fetch + abstract confirmed) | 2026-06-15 |
| arxiv:2310.07220 | COPlanner: Plan to Roll Out Conservatively but to Explore Optimistically for Model-Based RL (Wang, Zheng, Sun et al., 2023) | https://arxiv.org/abs/2310.07220 | paper | yes (2026-06-15, title+authors via arXiv API batch fetch) | 2026-06-15 |
| arxiv:1705.05363 | Curiosity-driven Exploration by Self-supervised Prediction — ICM (Pathak, Agrawal, Efros & Darrell, ICML 2017) | https://arxiv.org/abs/1705.05363 | paper | yes (2026-06-15, title+authors via arXiv API) | 2026-06-15 |
| arxiv:1605.09674 | VIME: Variational Information Maximizing Exploration (Houthooft, Chen, Duan, Schulman, De Turck & Abbeel, NeurIPS 2016) | https://arxiv.org/abs/1605.09674 | paper | yes (2026-06-15, title+authors+abstract confirmed) | 2026-06-15 |
| arxiv:1810.12894 | Exploration by Random Network Distillation — RND (Burda, Edwards, Storkey & Klimov, ICLR 2019) | https://arxiv.org/abs/1810.12894 | paper | yes (2026-06-15, title+authors+abstract confirmed) | 2026-06-15 |
| arxiv:2006.15762 | Empirically Verifying Hypotheses Using Reinforcement Learning (Marino, Fergus, Szlam & Gupta, 2020) | https://arxiv.org/abs/2006.15762 | paper | yes (2026-06-15, title+authors+abstract confirmed via arXiv API) | 2026-06-15 |
| arxiv:2102.04399 | How to Stay Curious while Avoiding Noisy TVs using Aleatoric Uncertainty Estimation (Mavor-Parker, Young, Barry & Griffin, ICML 2022) | https://arxiv.org/abs/2102.04399 | paper | yes (2026-06-15, title+authors+abstract confirmed) | 2026-06-15 |
| arxiv:2211.10515 | Curiosity in Hindsight: Intrinsic Exploration in Stochastic Environments (Jarrett, Tallec, Altché, Mesnard, Munos & Valko, ICML 2023) | https://arxiv.org/abs/2211.10515 | paper | yes (2026-06-15, title+authors confirmed via ICML proceedings) | 2026-06-15 |
| arxiv:1703.01161 | FeUdal Networks for Hierarchical Reinforcement Learning (Vezhnevets, Osindero, Schaul, Heess et al., ICML 2017) | https://arxiv.org/abs/1703.01161 | paper | yes (2026-06-16, title+authors via arXiv API batch fetch) | 2026-06-16 |
| arxiv:1712.00948 | Learning Multi-Level Hierarchies with Hindsight — HAC (Levy, Konidaris, Platt & Saenko, ICLR 2019) | https://arxiv.org/abs/1712.00948 | paper | yes (2026-06-16, title+authors via arXiv API batch fetch) | 2026-06-16 |
| arxiv:2310.05167 | Hieros: Hierarchical Imagination on Structured State Space Sequence World Models (Mattes, Schlosser & Herbrich, 2024) | https://arxiv.org/abs/2310.05167 | paper | yes (2026-06-16, title+authors via arXiv API batch fetch) | 2026-06-16 |
| openreview:TjCDNssXKU | Learning Hierarchical World Models with Adaptive Temporal Abstractions from Discrete Latent Dynamics — THICK (Gumbsch, Sajid, Martius & Butz, ICLR 2024) | https://openreview.net/forum?id=TjCDNssXKU | paper | yes (2026-06-16, title+authors confirmed via ICLR 2024 proceedings page + GitHub CognitiveModeling/THICK) | 2026-06-16 |
| arxiv:2507.23773 | SimuRA / General Agentic Planning Through Simulative Reasoning with World Models (Deng, Hou, Hu & Xing, 2025; v3 May 2026) | https://arxiv.org/abs/2507.23773 | paper | yes (2026-06-16, title+authors+abstract confirmed via arXiv HTML; likely OpenReview id 6fDZYJYYgu) | 2026-06-16 |
| arxiv:1707.05300 | Reverse Curriculum Generation for Reinforcement Learning (Florensa, Held, Wulfmeier, Zhang & Abbeel, CoRL 2017) | https://arxiv.org/abs/1707.05300 | paper | yes (2026-06-17, title+authors via arXiv API batch fetch) | 2026-06-17 |
| arxiv:1812.03381 | Learning Montezuma's Revenge from a Single Demonstration (Salimans & Chen, 2018) | https://arxiv.org/abs/1812.03381 | paper | yes (2026-06-17, title+authors via arXiv API batch fetch + abstract confirmed) | 2026-06-17 |
| arxiv:1807.06919 | Backplay: "Man muss immer umkehren" (Resnick, Raileanu, Kapoor, Peysakhovich, Cho & Bruna, 2018) | https://arxiv.org/abs/1807.06919 | paper | yes (2026-06-17, title+authors via arXiv API batch fetch) | 2026-06-17 |
| arxiv:2405.03379 | Reverse Forward Curriculum Learning for Extreme Sample and Demonstration Efficiency in RL — RFCL (Tao, Shukla, Chan & Su, ICLR 2024) | https://arxiv.org/abs/2405.03379 | paper | yes (2026-06-17, title+authors+venue via arXiv abstract page) | 2026-06-17 |
| arxiv:2004.12919 | First return, then explore (Ecoffet, Huizinga, Lehman, Stanley & Clune, Nature 2021) | https://arxiv.org/abs/2004.12919 | paper | yes (2026-06-17, title+authors via arXiv API batch fetch) | 2026-06-17 |
| arxiv:1901.10995 | Go-Explore: a New Approach for Hard-Exploration Problems (Ecoffet, Huizinga, Lehman, Stanley & Clune, 2019) | https://arxiv.org/abs/1901.10995 | paper | yes (2026-06-17, title+authors via arXiv API batch fetch) | 2026-06-17 |
| arxiv:2110.00188 | Offline Reinforcement Learning with Reverse Model-based Imagination — ROMI (Wang, Li, Jiang, Zhu, Li & Zhang, NeurIPS 2021) | https://arxiv.org/abs/2110.00188 | paper | yes (2026-06-17, title+authors via arXiv API batch fetch + abstract confirmed) | 2026-06-17 |
| arxiv:2509.13341 | Imagined Autocurricula — IMAC (Güzel, Jackson, Liesen, Rocktäschel, Foerster, Bogunovic & Parker-Holder, NeurIPS 2025) | https://arxiv.org/abs/2509.13341 | paper | yes (2026-06-17, title+authors+NeurIPS venue confirmed via arXiv abstract + OpenReview) | 2026-06-17 |
| arxiv:2110.09514 | Discovering and Achieving Goals via World Models — LEXA (Mendonca, Rybkin, Daniilidis, Hafner & Pathak, NeurIPS 2021) | https://arxiv.org/abs/2110.09514 | paper | yes (2026-06-17, title+authors+abstract confirmed via arXiv abstract page) | 2026-06-17 |
| openreview:B1gqipNYwH | Option Discovery using Deep Skill Chaining (Bagaria & Konidaris, ICLR 2020; no arXiv preprint) | https://openreview.net/forum?id=B1gqipNYwH | paper | yes (2026-06-17, title+authors confirmed via OpenReview ICLR 2020 proceedings + ICLR virtual poster page) | 2026-06-17 |
