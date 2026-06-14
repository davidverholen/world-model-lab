"""exp 0031: does the frozen DINO encoder see Crafter's achievement-critical visual structure?

0029 proved DINO encodes the stat HUD (drink 0.94). This probes the harder, depth-relevant
question: can a LINEAR map from the frozen embedding decode the *materials in the agent's view*
— stone/coal/iron/tree/water/table/furnace — the tiles the tech tree depends on? If stone (the
first deep-tree step) and friends decode well, the frozen-encoder foundation is sound and our
plateau is exploration/scale, not perception → build rung-4 on it. If they don't, the encoder
is dropping task-critical structure → a learned ADAPTER (not a from-scratch encoder) is the fix.
See knowledge/experiments/0031-semantic-probe.md. Does NOT hardcode env content — only MEASURES.
"""

import argparse

import numpy as np
import torch

from world_model.agents.rssm_agent import RSSMActorAgent
from world_model.envs import make_crafter_env
from world_model.models import FrozenDinoEncoder

# crafter world._mat_ids — the achievement-critical materials (ids verified 2026-06-14)
MATERIALS = {"water": 1, "stone": 3, "sand": 5, "tree": 6, "lava": 7,
             "coal": 8, "iron": 9, "diamond": 10, "table": 11, "furnace": 12}
VIEW = 9  # egocentric crop (tiles) around player_pos approximating the agent's view


def collect(n_frames, agent_ckpt, device, seed):
    """Frames + binary 'material present in the agent's local view' labels + actions + ep ids."""
    frames, labels, actions, ep_ids = [], [], [], []
    rng = np.random.default_rng(seed)
    agent = (
        RSSMActorAgent.from_checkpoint(agent_ckpt, device, epsilon=0.0, seed=seed)
        if agent_ckpt else None
    )
    r = VIEW // 2
    ep = 0
    while len(frames) < n_frames:
        use_agent = agent is not None and ep % 2 == 1
        env = make_crafter_env(length=400, seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        if use_agent:
            agent.reset()
        done = False
        while not done and len(frames) < n_frames:
            a = agent.act(obs) if use_agent else int(rng.integers(env.action_space.n))
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            sem = np.asarray(info["semantic"])
            px, py = info["player_pos"]
            crop = sem[max(0, px - r): px + r + 1, max(0, py - r): py + r + 1]
            present = set(np.unique(crop).tolist())
            frames.append(obs.astype(np.float32))
            labels.append([1.0 if mid in present else 0.0 for mid in MATERIALS.values()])
            actions.append(a)
            ep_ids.append(ep)
        ep += 1
    return (
        np.stack(frames),
        np.asarray(labels, dtype=np.float64),
        np.asarray(actions, dtype=np.int64),
        np.asarray(ep_ids, dtype=np.int64),
    )


@torch.no_grad()
def encode(obs, pool, device, batch=256):
    enc = FrozenDinoEncoder(pool=pool).to(device).eval()
    out = [enc(torch.as_tensor(obs[i:i + batch], dtype=torch.float32, device=device)).cpu().numpy()
           for i in range(0, len(obs), batch)]
    return np.concatenate(out).astype(np.float64)


@torch.no_grad()
def roll_beliefs(obs, actions, ep_ids, ckpt, device):
    """Roll the TRAINED RSSM per episode (as RSSMActorAgent.act) → the belief the actor sees.
    The decisive foundation test: does the material info survive the RSSM, not just the encoder?"""
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


def _auc(scores, y):
    """Mann-Whitney AUC: P(score(pos) > score(neg)). 0.5 = chance, 1.0 = perfectly decodable."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    npos = y.sum()
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ridge_probe_auc(X, y, lam, test_frac, rng):
    """Linear (ridge) probe → test AUC for a binary target."""
    n = len(X)
    idx = rng.permutation(n)
    te, tr = idx[: int(n * test_frac)], idx[int(n * test_frac):]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xtr = np.hstack([(X[tr] - mu) / sd, np.ones((len(tr), 1))])
    Xte = np.hstack([(X[te] - mu) / sd, np.ones((len(te), 1))])
    w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ y[tr])
    return _auc(Xte @ w, y[te])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, default=8000)
    p.add_argument("--agent", type=str, default=None)
    p.add_argument("--pool", default="cls+patch")
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    src = "trained" if args.agent else "no"
    print(f"collecting {args.frames} frames (random + {src} agent)...")
    obs, labels, actions, ep_ids = collect(args.frames, args.agent, args.device, args.seed)
    emb = encode(obs, args.pool, args.device)
    belief = roll_beliefs(obs, actions, ep_ids, args.agent, args.device) if args.agent else None

    def auc_for(X, y):
        a = ridge_probe_auc(X, y, args.lam, args.test_frac, np.random.default_rng(args.seed))
        s = ridge_probe_auc(X, y[rng.permutation(len(y))], args.lam, args.test_frac,
                            np.random.default_rng(args.seed))
        return a, s

    print(f"\ntest AUC of 'material in view' (1.0=decodable, 0.5=chance) | pool={args.pool}:")
    head = "material".ljust(10) + "support".ljust(9) + "base%".ljust(8) + "emb_AUC".ljust(9)
    print(head + ("belief_AUC".ljust(12) + "shuffled" if belief is not None else "shuffled"))
    for name, j in zip(MATERIALS, range(len(MATERIALS)), strict=True):
        y = labels[:, j]
        support = int(y.sum())
        row = name.ljust(10) + f"{support}".ljust(9) + f"{100 * y.mean():.1f}".ljust(8)
        if support < 30 or support > len(y) - 30:
            print(row + "LOW-SUPPORT")
            continue
        ea, es = auc_for(emb, y)
        row += f"{ea:.3f}".ljust(9)
        if belief is not None:
            ba, _ = auc_for(belief, y)
            row += f"{ba:.3f}".ljust(12)
        print(row + f"{es:.3f}")


if __name__ == "__main__":
    main()
