"""exp 0053: imagination-fidelity probe + viewer for the rung-4 reading agent.

Measures AND visualizes how faithful the world model's imagined rollout is on length-2 tutorials.
From a real start state (right after the manual is read), roll the greedy actor forward in
IMAGINATION (``rssm.img_step``), then REPLAY that exact imagined action plan in the REAL env and
compare. The dream vs reality under the dream's own plan.

Quantitative — per imagined step, the belief divergence between the imagined latent trajectory and
the real one under the same actions:

    div[k] = || belief(imagined s_k) - belief(real s_k) ||      (L2, and cosine)

div[0] == 0 by construction (same start state); how fast it grows is the model's open-loop
multi-step accuracy. If it diverges by step 1-2, the actor plans in a fantasy → the ~0.10 wall is
the MODEL, not the policy. Needs only rssm + the frozen encoder, so it runs on any rtfm checkpoint.

Visual — a side-by-side GIF: left = the imagined path (each imagined belief mapped to its nearest
real frame by retrieval; the WM predicts latents, we have no pixel decoder), right = reality
executing the same actions. You see where they diverge.

If the checkpoint carries the WM heads (exp0053+ also saves ``rew``), the imagined-vs-real per-step
REWARD gap is reported too (the over-optimism signal behind exp0052's inflated imagined_return).

Usage::

    uv run python scripts/imagine_report.py runs/exp0051c05/s0_s0.pt \\
        --mode CORRECT --length 2 --horizon 8 --seeds 30 \\
        --out-dir runs/exp0053/baseline --gif-seeds 0,1,2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import crafter_rtfm as C
import matplotlib
import numpy as np
import torch
from crafter_rtfm import ManualMode

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# House plot style — mirror scripts/plot_experiment.py so these diagnostic figures match the other
# experiment-report assets: transparent background + neutral-gray chrome (reads on light or dark),
# saturated palette legible against either. Saved with transparent=True so the page theme shows.
_GRAY = "#888888"
plt.rcParams.update(
    {
        "text.color": _GRAY,
        "axes.labelcolor": _GRAY,
        "axes.titlecolor": _GRAY,
        "axes.edgecolor": _GRAY,
        "xtick.color": _GRAY,
        "ytick.color": _GRAY,
        "grid.color": _GRAY,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
    }
)
# Shared palette with plot_experiment.py (blue, red, green, amber, purple, cyan).
_BLUE, _RED, _GREEN, _AMBER = "#3b82f6", "#ef4444", "#22c55e", "#f59e0b"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image, ImageDraw  # noqa: E402
from visualize_rtfm import _FONT, ACTION_NAMES, GAME_PX, write_gif  # noqa: E402

from world_model.models import Actor, ConditionedActor, FrozenDinoEncoder  # noqa: E402
from world_model.models.manual_conditioning import ConditionedRSSM  # noqa: E402
from world_model.models.text_encoder import FrozenTextEncoder  # noqa: E402
from world_model.models.twohot import TwoHotRewardHead  # noqa: E402
from world_model.train_rtfm import RTFMAgent, actor_logits, img_to_chw  # noqa: E402

BG = (18, 18, 22)
FG = (225, 225, 230)
ACCENT = (120, 220, 140)
WARN = (235, 170, 90)


def build_rtfm_modules(checkpoint: str | Path, device: str):
    """Load an rtfm checkpoint → (RTFMAgent, heads dict). heads carries recon_head/rew/vr_head when
    the checkpoint saved them (exp0053+); empty for older rssm/actor-only checkpoints."""
    ckpt = torch.load(checkpoint, map_location=device)
    if not ckpt.get("rtfm_agent"):
        raise ValueError(f"{checkpoint} is not an rtfm_agent checkpoint")
    cfg = ckpt["config"]
    ed, td, n_act = cfg["embed_dim"], cfg["text_dim"], cfg["num_actions"]
    enc = FrozenDinoEncoder(pool=cfg["pool"]).to(device)
    text_enc = FrozenTextEncoder(device=device)
    rssm = ConditionedRSSM(embed_dim=ed, num_actions=n_act, text_dim=td).to(device)
    rssm.load_state_dict(ckpt["rssm"])
    if cfg["actor_cond"]:
        actor: torch.nn.Module = ConditionedActor(
            belief_dim=rssm.state_dim, text_dim=td, num_actions=n_act
        ).to(device)
    else:
        actor = Actor(state_dim=rssm.state_dim, num_actions=n_act).to(device)
    actor.load_state_dict(ckpt["actor"])
    rssm.eval()
    actor.eval()
    agent = RTFMAgent(enc, text_enc, rssm, actor, n_act, device, epsilon=0.0)

    heads: dict[str, torch.nn.Module] = {}
    if ckpt.get("rew") is not None:
        rew = TwoHotRewardHead(state_dim=rssm.state_dim, num_actions=n_act).to(device)
        rew.load_state_dict(ckpt["rew"])
        rew.eval()
        heads["rew"] = rew
    return agent, heads


def _embed(enc, image, device):
    """obs['image'] (HWC uint8) → frozen-encoder embedding (1, D)."""
    return enc(torch.as_tensor(img_to_chw(image), device=device).unsqueeze(0))


@torch.no_grad()
def build_frame_library(agent, mode, length, one_shot, n_episodes, max_steps, device):
    """Roll greedy real episodes, collecting (belief, render-frame) pairs for retrieval. Returns
    (beliefs (N, D) tensor, frames list of (64,64,3) uint8)."""
    beliefs, frames = [], []
    for ep in range(n_episodes):
        env = C.make_recipe_env(mode, length=length, one_shot=one_shot)
        obs, info = env.reset(seed=10_000 + ep)  # disjoint from the eval seeds
        agent.reset(obs, info)
        for _t in range(max_steps):
            a = agent.act(obs, info)  # greedy; advances the agent's own belief over this obs
            beliefs.append(agent.rssm.belief(agent._state))
            frames.append(np.asarray(env.render(), dtype=np.uint8))
            obs, _r, term, trunc, info = env.step(a)
            if term or trunc:
                break
    return torch.cat(beliefs, dim=0), frames


@torch.no_grad()
def imagine_vs_real(agent, rew_head, mode, length, one_shot, seed, horizon, device):
    """From a SHARED real start state s0, (a) imagine `horizon` greedy steps (img_step), and (b)
    replay that exact plan in the real env (obs_step). Both rollouts start from the SAME s0, so the
    divergence is anchored at 0. The RSSM belief carries a SAMPLED stochastic latent, so we also run
    a real-vs-real NOISE FLOOR: a second real pass over the SAME observations (z resampled) — the
    divergence the metric would show even for a perfect model. img-vs-real ABOVE the floor = genuine
    open-loop model error."""
    rssm, actor, enc = agent.rssm, agent.actor, agent.enc
    env = C.make_recipe_env(mode, length=length, one_shot=one_shot)
    obs0, info0 = env.reset(seed=seed)
    agent.reset(obs0, info0)
    tok, mask = agent._tok, agent._mask
    # Shared start state s0 = belief after seeing obs0 (mirrors RTFMAgent.act's first closed step).
    s0, _, _ = rssm.obs_step(
        agent._state, agent._prev_a, _embed(enc, obs0["image"], device), tok, mask
    )

    # --- (a) Imagined rollout (open loop): greedy actor + img_step from s0 ---
    # Track the full belief (h+z, what the policy/heads consume) AND the deterministic h alone —
    # h is where recurrent-dynamics error shows cleanly, un-inflated by per-step z sampling.
    s = s0
    img_beliefs, img_h, plan, img_rews = [rssm.belief(s0)], [s0[0]], [], []
    for _k in range(horizon):
        bel = rssm.belief(s)
        a = int(actor_logits(actor, bel, tok, mask).argmax(-1).item())  # greedy (matches eval)
        at = torch.tensor([a], device=device)
        plan.append(a)
        if rew_head is not None:
            img_rews.append(float(rew_head(bel, at).item()))
        s, _ = rssm.img_step(s, at, tok, mask)
        img_beliefs.append(rssm.belief(s))
        img_h.append(s[0])

    # --- (b) Real rollout from the SAME s0, applying the imagined plan; cache embeds + frames ---
    sr = s0
    real_beliefs, real_h = [rssm.belief(s0)], [s0[0]]
    real_frames = [np.asarray(env.render(), dtype=np.uint8)]
    real_embeds, real_rews, newly = [], [], []
    for k in range(horizon):
        obs, _r, term, trunc, info = env.step(plan[k])
        e = _embed(enc, obs["image"], device)
        real_embeds.append(e)
        real_rews.append(float(len(info.get("tutorial_newly", []))))
        newly.append(bool(info.get("tutorial_newly")))
        sr, _, _ = rssm.obs_step(sr, torch.tensor([plan[k]], device=device), e, tok, mask)
        real_beliefs.append(rssm.belief(sr))
        real_h.append(sr[0])
        real_frames.append(np.asarray(env.render(), dtype=np.uint8))
        if term or trunc:
            break

    # --- noise floor: a 2nd real pass over the SAME cached embeds (z resampled), from s0 ---
    sn = s0
    noise_beliefs, noise_h = [rssm.belief(s0)], [s0[0]]
    for k in range(len(real_embeds)):
        sn, _, _ = rssm.obs_step(
            sn, torch.tensor([plan[k]], device=device), real_embeds[k], tok, mask
        )
        noise_beliefs.append(rssm.belief(sn))
        noise_h.append(sn[0])

    n = min(len(img_beliefs), len(real_beliefs))
    ib, rb = torch.cat(img_beliefs[:n], 0), torch.cat(real_beliefs[:n], 0)
    nb = torch.cat(noise_beliefs[:n], 0)
    ih, rh, nh = torch.cat(img_h[:n], 0), torch.cat(real_h[:n], 0), torch.cat(noise_h[:n], 0)
    return {
        "plan": plan,
        "l2": (ib - rb).norm(dim=-1).cpu().numpy(),
        "cos": torch.nn.functional.cosine_similarity(ib, rb, dim=-1).cpu().numpy(),
        "l2_noise": (nb - rb).norm(dim=-1).cpu().numpy(),
        "l2_h": (ih - rh).norm(dim=-1).cpu().numpy(),
        "l2_h_noise": (nh - rh).norm(dim=-1).cpu().numpy(),
        "img_beliefs": ib,
        "real_frames": real_frames,
        "img_rews": img_rews,
        "real_rews": real_rews,
        "newly": newly,
        "n": n,
    }


def _nearest_frames(lib_beliefs, lib_frames, query_beliefs):
    """For each query belief, the nearest library frame by L2 (retrieval-based imagined view)."""
    idx = torch.cdist(query_beliefs, lib_beliefs).argmin(dim=-1).cpu().numpy()
    return [lib_frames[i] for i in idx]


def compose_pair(img_frame, real_frame, step, action, img_r, real_r, newly):
    """Side-by-side: imagined (left) vs real (right), upscaled, with a caption strip."""

    def up(f):
        return Image.fromarray(np.ascontiguousarray(f)).resize((GAME_PX, GAME_PX), Image.NEAREST)

    cap_h = 92
    canvas = Image.new("RGB", (GAME_PX * 2 + 24, GAME_PX + cap_h), BG)
    canvas.paste(up(img_frame), (8, 20))
    canvas.paste(up(real_frame), (GAME_PX + 16, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), "IMAGINED (retrieval)", fill=WARN, font=_FONT)
    draw.text((GAME_PX + 16, 4), "REAL (same actions)", fill=ACCENT, font=_FONT)
    y = GAME_PX + 26
    aname = ACTION_NAMES[action] if action is not None else "(start)"
    draw.text((8, y), f"step {step}   action: {action} {aname}", fill=FG, font=_FONT)
    if img_r is not None:
        draw.text(
            (8, y + 16),
            f"imagined reward {img_r:+.3f}   real reward {real_r:+.0f}",
            fill=FG,
            font=_FONT,
        )
    if newly:
        draw.text((8, y + 32), "REAL: tutorial achievement!", fill=ACCENT, font=_FONT)
    return np.asarray(canvas, dtype=np.uint8)


def _stack(rows, width):
    m = np.full((len(rows), width), np.nan)
    for i, x in enumerate(rows):
        m[i, : len(x)] = x
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=str)
    p.add_argument("--mode", default="CORRECT", choices=["CORRECT", "NONE", "SWAPPED"])
    p.add_argument("--length", type=int, default=2)
    p.add_argument("--one-shot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--horizon", type=int, default=8, help="imagined rollout length")
    p.add_argument("--seeds", type=int, default=30, help="eval seeds for divergence stats")
    p.add_argument(
        "--lib-episodes", type=int, default=12, help="episodes for the retrieval library"
    )
    p.add_argument("--max-steps", type=int, default=48)
    p.add_argument(
        "--gif-seeds", type=str, default="0,1,2", help="comma seeds rendered side-by-side"
    )
    p.add_argument("--fps", type=float, default=1.0, help="side-by-side GIF speed (lower = slower)")
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent / f"imagine_{ckpt.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = ManualMode[args.mode]

    agent, heads = build_rtfm_modules(ckpt, device)
    rew_head = heads.get("rew")
    print(f"device={device}  rew_head={'yes' if rew_head else 'no'}  out={out_dir}")

    lib_beliefs, lib_frames = build_frame_library(
        agent,
        ManualMode.CORRECT,
        args.length,
        args.one_shot,
        args.lib_episodes,
        args.max_steps,
        device,
    )
    print(f"retrieval library: {len(lib_frames)} frames")

    all_l2, all_cos, all_noise, all_h, all_h_noise, gaps = [], [], [], [], [], []
    gif_seeds = {int(s) for s in args.gif_seeds.split(",") if s != ""}
    for seed in range(args.seeds):
        r = imagine_vs_real(
            agent, rew_head, mode, args.length, args.one_shot, seed, args.horizon, device
        )
        all_l2.append(r["l2"])
        all_cos.append(r["cos"])
        all_noise.append(r["l2_noise"])
        all_h.append(r["l2_h"])
        all_h_noise.append(r["l2_h_noise"])
        if rew_head is not None and r["real_rews"]:
            m = min(len(r["img_rews"]), len(r["real_rews"]))
            gaps.append(float(np.sum(r["img_rews"][:m]) - np.sum(r["real_rews"][:m])))
        if seed in gif_seeds:
            img_frames = _nearest_frames(lib_beliefs, lib_frames, r["img_beliefs"][: r["n"]])
            frames = []
            for k in range(r["n"]):
                a = r["plan"][k - 1] if k > 0 else None
                ir = (
                    r["img_rews"][k - 1]
                    if (rew_head is not None and 0 < k <= len(r["img_rews"]))
                    else None
                )
                rr = r["real_rews"][k - 1] if 0 < k <= len(r["real_rews"]) else 0.0
                nw = r["newly"][k - 1] if 0 < k <= len(r["newly"]) else False
                frames.append(compose_pair(img_frames[k], r["real_frames"][k], k, a, ir, rr, nw))
            gif_path = out_dir / f"imagine_seed{seed}.gif"
            write_gif(frames, gif_path, fps=args.fps)
            print(f"  wrote {gif_path} ({len(frames)} frames)")

    H = max(len(x) for x in all_l2)
    l2m, cosm, noisem = _stack(all_l2, H), _stack(all_cos, H), _stack(all_noise, H)
    hm, hnoisem = _stack(all_h, H), _stack(all_h_noise, H)
    l2_mean, cos_mean, noise_mean = np.nanmean(l2m, 0), np.nanmean(cosm, 0), np.nanmean(noisem, 0)
    h_mean, h_noise_mean = np.nanmean(hm, 0), np.nanmean(hnoisem, 0)
    steps = np.arange(H)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    # full belief (h+z) — what the policy/heads actually consume
    ax[0].plot(steps, l2_mean, "-o", color=_RED, label="imagined vs real")
    ax[0].fill_between(
        steps,
        np.nanpercentile(l2m, 25, 0),
        np.nanpercentile(l2m, 75, 0),
        alpha=0.2,
        color=_RED,
    )
    ax[0].plot(steps, noise_mean, "--s", color=_GRAY, label="real vs real (sampling floor)")
    ax[0].legend(fontsize=8)
    ax[0].set(title="belief (h+z) divergence L2", xlabel="imagined step", ylabel="L2")
    # deterministic h only — clean recurrent-dynamics fidelity (un-inflated by z sampling)
    ax[1].plot(steps, h_mean, "-o", color=_AMBER, label="imagined vs real")
    ax[1].fill_between(
        steps, np.nanpercentile(hm, 25, 0), np.nanpercentile(hm, 75, 0), alpha=0.2, color=_AMBER
    )
    ax[1].plot(steps, h_noise_mean, "--s", color=_GRAY, label="real vs real (floor)")
    ax[1].legend(fontsize=8)
    ax[1].set(title="deterministic h divergence L2", xlabel="imagined step", ylabel="L2")
    ax[2].plot(steps, cos_mean, "-o", color=_BLUE)
    ax[2].fill_between(
        steps,
        np.nanpercentile(cosm, 25, 0),
        np.nanpercentile(cosm, 75, 0),
        alpha=0.2,
        color=_BLUE,
    )
    ax[2].set(
        title="belief cosine: imagined vs real",
        xlabel="imagined step",
        ylabel="cos",
        ylim=(0, 1.02),
    )
    fig.suptitle(f"{ckpt.name}  mode={args.mode}  {args.seeds} seeds  H={args.horizon}")
    fig.tight_layout()
    plot_path = out_dir / "fidelity.png"
    fig.savefig(plot_path, dpi=110, transparent=True)
    print(f"wrote {plot_path}")

    print("\n=== imagination fidelity summary (imagined-vs-real | sampling floor) ===")
    for k in range(min(H, args.horizon + 1)):
        print(
            f"  step {k}: belief L2={l2_mean[k]:.2f} (floor {noise_mean[k]:.2f})  "
            f"h L2={h_mean[k]:.2f} (floor {h_noise_mean[k]:.2f})  cos={cos_mean[k]:.3f}"
        )
    if gaps:
        print(
            f"  imagined-minus-real reward (sum over H): mean={np.mean(gaps):+.3f} "
            f"(>0 = imagination over-optimistic)"
        )


if __name__ == "__main__":
    main()
