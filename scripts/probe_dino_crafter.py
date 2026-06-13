"""Probe: do FROZEN DINOv2 features encode Crafter game state? (gates the frozen-encoder bet)

Linear-probe DINOv2-S CLS features against "which materials are in the agent's 9x9 view"
(free labels from info['semantic'] cropped around player_pos). If a linear map predicts
material presence well above the majority baseline (and a shuffled-label control collapses
to baseline), frozen DINO features transfer to Crafter's out-of-distribution pixel-art and
the DINO-WM-style recipe is viable. See knowledge/decisions/0006 + frozen-encoder-lean.

    uv run python -m scripts.probe_dino_crafter   # or: uv run python scripts/probe_dino_crafter.py
"""

import numpy as np
import torch
import torch.nn.functional as F

from world_model.envs import make_crafter_env

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_FRAMES = 600


def main() -> None:
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval().to(DEV)
    for p in dino.parameters():
        p.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(obs_chw: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(obs_chw.to(DEV), size=98, mode="bilinear", align_corners=False)
        x = (x - mean) / std
        return torch.cat(
            [dino.forward_features(x[i : i + 64])["x_norm_clstoken"] for i in range(0, len(x), 64)]
        )

    rng = np.random.default_rng(0)
    obs_list, sem_list, pos_list = [], [], []
    for seed in range(8):
        env = make_crafter_env(seed=seed)
        o, info = env.reset(seed=seed)
        for t in range(250):
            if len(obs_list) >= N_FRAMES:
                break
            o, r, term, trunc, info = env.step(int(rng.integers(17)))
            if t % 2 == 0:
                obs_list.append(o)
                sem_list.append(info["semantic"])
                pos_list.append(info["player_pos"])
            if term or trunc:
                o, info = env.reset()
        if len(obs_list) >= N_FRAMES:
            break

    def view_ids(sem: np.ndarray, pos, half: int = 4) -> set:
        r, c = int(pos[0]), int(pos[1])
        return set(np.unique(sem[max(0, r - half) : r + half + 1, max(0, c - half) : c + half + 1]))

    vocab = sorted(set().union(*[view_ids(s, p) for s, p in zip(sem_list, pos_list, strict=False)]))
    Y = np.zeros((len(obs_list), len(vocab)), dtype=np.float32)
    for i, (s, p) in enumerate(zip(sem_list, pos_list, strict=False)):
        ids = view_ids(s, p)
        Y[i] = [1.0 if v in ids else 0.0 for v in vocab]
    base = Y.mean(0)
    keep = [j for j in range(len(vocab)) if 0.05 < base[j] < 0.95]  # informative labels only
    Y, vocab_k = Y[:, keep], [int(vocab[j]) for j in keep]
    print(f"frames={len(obs_list)} informative-material-labels(ids)={vocab_k}")

    X = encode(torch.tensor(np.stack(obs_list))).float()
    n = len(X)
    idx = rng.permutation(n)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n) :]
    Xt, Xe = X[tr], X[te]
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True) + 1e-6
    Xt, Xe = (Xt - mu) / sd, (Xe - mu) / sd
    Yt = torch.tensor(Y[tr], device=DEV)
    Ye = torch.tensor(Y[te], device=DEV)

    def fit(targets: torch.Tensor) -> torch.Tensor:
        lin = torch.nn.Linear(X.shape[1], Y.shape[1]).to(DEV)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
        for _ in range(800):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(lin(Xt), targets).backward()
            opt.step()
        with torch.no_grad():
            return ((lin(Xe) > 0).float() == Ye).float().mean(0).cpu()

    acc = fit(Yt).numpy()
    maj = np.maximum(Ye.mean(0).cpu().numpy(), 1 - Ye.mean(0).cpu().numpy())  # per-label majority
    acc_sh = fit(Yt[torch.randperm(len(Yt))]).mean().item()  # shuffled-label control

    print("per-material test acc:", dict(zip(vocab_k, np.round(acc, 2), strict=False)))
    print("per-material majority :", dict(zip(vocab_k, np.round(maj, 2), strict=False)))
    print(
        f"\nMEAN test acc={acc.mean():.3f}  majority-baseline={maj.mean():.3f}  "
        f"LIFT=+{acc.mean() - maj.mean():.3f}"
    )
    print(f"shuffled-label control={acc_sh:.3f} (should collapse to ~majority {maj.mean():.3f})")


if __name__ == "__main__":
    main()
