"""exp 0029: linear-probe decodability of the agent's own stats from the FROZEN DINO embedding.

Motivation: the 0027 agent keeps spamming collect_drink at the cap (a no-op) — hypothesis is
it is perceptually blind to its vitals because the frozen encoder washes out the HUD stat-bars.
This probes it directly: if a linear map embedding->stat recovers the true stat with high test
R^2, the info is present (actor's problem); if not, the encoder drops it (architecture signal).
Does NOT hardcode any env value — it only MEASURES what the embedding carries.
See knowledge/experiments/0029-stat-perception-probe.md.
"""

import argparse

import numpy as np
import torch

from world_model.agents.rssm_agent import RSSMActorAgent
from world_model.envs import make_crafter_env
from world_model.models import FrozenDinoEncoder

STAT_KEYS = ["health", "food", "drink", "energy", "wood", "stone"]


def collect(n_frames: int, agent_ckpt: str | None, device: str, seed: int):
    """Frames + true stats + the action that produced each frame + episode id, from random
    episodes (vital variance) + the trained agent (material variance). Returns obs
    (N,3,H,W), stats (N,K), actions (N,), ep_ids (N,)."""
    frames, stats, actions, ep_ids = [], [], [], []
    rng = np.random.default_rng(seed)
    agent = None
    if agent_ckpt:
        agent = RSSMActorAgent.from_checkpoint(agent_ckpt, device, epsilon=0.0, seed=seed)

    def record(obs, info, a, ep):
        frames.append(obs.astype(np.float32))
        inv = info["inventory"]
        stats.append([float(inv[k]) for k in STAT_KEYS])
        actions.append(a)
        ep_ids.append(ep)

    ep = 0
    while len(frames) < n_frames:
        use_agent = agent is not None and ep % 2 == 1  # alternate random / trained episodes
        env = make_crafter_env(length=400, seed=seed + ep)
        obs, info = env.reset(seed=seed + ep)
        if use_agent:
            agent.reset()
        done = False
        while not done and len(frames) < n_frames:
            a = agent.act(obs) if use_agent else int(rng.integers(env.action_space.n))
            obs, _, term, trunc, info = env.step(a)
            record(obs, info, a, ep)
            done = term or trunc
        ep += 1
    return (
        np.stack(frames),
        np.asarray(stats, dtype=np.float64),
        np.asarray(actions, dtype=np.int64),
        np.asarray(ep_ids, dtype=np.int64),
    )


@torch.no_grad()
def roll_beliefs(obs, actions, ep_ids, ckpt: str, device: str) -> np.ndarray:
    """Roll the TRAINED RSSM forward per episode (exactly as RSSMActorAgent.act: prev_action
    lags one frame) and return the belief concat(h,z) the actor actually sees at each frame."""
    agent = RSSMActorAgent.from_checkpoint(ckpt, device, epsilon=0.0, seed=0)
    enc, rssm = agent.encoder, agent.rssm
    beliefs = []
    for ep in np.unique(ep_ids):
        m = np.flatnonzero(ep_ids == ep)
        state = rssm.initial(1, device)
        prev_a = torch.full((1,), rssm.no_action, device=device)
        for t in m:
            embed = enc(torch.as_tensor(obs[t], dtype=torch.float32, device=device).unsqueeze(0))
            state, _, _ = rssm.obs_step(state, prev_a, embed)
            beliefs.append(rssm.belief(state)[0].cpu().numpy())
            prev_a = torch.tensor([int(actions[t])], device=device)
    return np.asarray(beliefs, dtype=np.float64)


@torch.no_grad()
def encode(obs: np.ndarray, pool: str, device: str, batch: int = 256) -> np.ndarray:
    enc = FrozenDinoEncoder(pool=pool).to(device).eval()
    out = []
    for i in range(0, len(obs), batch):
        x = torch.as_tensor(obs[i : i + batch], dtype=torch.float32, device=device)
        out.append(enc(x).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def ridge_r2(X: np.ndarray, Y: np.ndarray, lam: float, test_frac: float, rng) -> np.ndarray:
    """Standardize X (train stats), closed-form ridge, return per-target test R^2."""
    n = len(X)
    idx = rng.permutation(n)
    n_te = int(n * test_frac)
    te, tr = idx[:n_te], idx[n_te:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])  # bias
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    ytr_mu = Y[tr].mean(0)
    d = Xtr.shape[1]
    reg = lam * np.eye(d)
    reg[-1, -1] = 0.0  # don't penalize bias
    W = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ Y[tr])
    pred = Xte @ W
    ss_res = ((Y[te] - pred) ** 2).sum(0)
    ss_tot = ((Y[te] - ytr_mu) ** 2).sum(0) + 1e-9
    return 1.0 - ss_res / ss_tot


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, default=5000)
    p.add_argument("--agent", type=str, default=None, help="trained ckpt for material variance")
    p.add_argument("--belief", type=str, default=None, help="ckpt to also probe the RSSM belief")
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    src = "trained" if args.agent else "no"
    print(f"collecting {args.frames} frames (random + {src} agent)...")
    obs, stats, actions, ep_ids = collect(args.frames, args.agent, args.device, args.seed)
    sd_by_key = dict(zip(STAT_KEYS, stats.std(0).round(2), strict=True))
    print(f"  stats variance (std per key): {sd_by_key}")

    def probe_row(label, X):
        r2 = ridge_r2(X, stats, args.lam, args.test_frac, np.random.default_rng(args.seed))
        shuf = ridge_r2(  # shuffled control: break frame<->stat correspondence → chance floor
            X, stats[rng.permutation(len(stats))], args.lam, args.test_frac,
            np.random.default_rng(args.seed),
        )
        print(label.ljust(14) + "".join(f"{v:.2f}".ljust(9) for v in r2) + f"{shuf.mean():.2f}")

    print("\ntest R^2 by stat (1.0=perfect, ~0=carries nothing):")
    print("repr".ljust(14) + "".join(k.ljust(9) for k in STAT_KEYS) + "shuffled")
    for pool in ["cls", "patch_mean", "cls+patch"]:
        probe_row(f"emb:{pool}", encode(obs, pool, args.device))
    if args.belief:
        # the belief the actor actually consumes — does the RSSM RETAIN what the embedding has?
        probe_row("belief(rssm)", roll_beliefs(obs, actions, ep_ids, args.belief, args.device))


if __name__ == "__main__":
    main()
