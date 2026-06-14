"""RSSM Dreamer flywheel on Crafter (rung 3) with a FROZEN DINOv2 encoder.

Reuses the calibrated rung-2 imagination loop (RSSM + critic-on-replay; exps 0019/0021) and
the validated frozen DINO encoder (exp 0022). Differences vs train_rssm (MiniGrid): frozen
DINOv2 encoder (embed_dim 384; no SIGReg — a frozen encoder can't collapse), the Crafter env
(17 actions), and an achievement/reward eval (Crafter has no binary success).

v1 encodes inside the training loop (CORRECTNESS first). Embedding caching — freezing's real
payoff (encode once at collection, skip the encoder in every WM update) — is the next
optimization. See knowledge/environments/crafter.md, experiments/0022-frozen-dino-probe.md.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from world_model.agents.rssm_agent import RSSMActorAgent
from world_model.envs import make_crafter_env
from world_model.models import Actor, FrozenDinoEncoder, ValueHead
from world_model.models.continue_head import ContinueHead
from world_model.models.rssm import RSSM
from world_model.models.twohot import TwoHotRewardHead, TwoHotValueHead
from world_model.models.value import value_loss
from world_model.train_rssm import imagine_ac, kl_balanced, wm_train
from world_model.training import ReplayBuffer, Transition


@torch.no_grad()
def collect_embed(buffer, env, enc, rssm, actor, n_act, steps, epsilon, device, rng, seed):
    """Collect into the buffer storing FROZEN-ENCODER EMBEDDINGS (not pixels): encode each
    frame ONCE here, so WM updates never re-run DINO. actor=None → pure random (round 0)."""

    def emb(o):
        return enc(torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0))

    obs, _ = env.reset(seed=seed)
    embed = emb(obs)
    state = rssm.initial(1, device)
    prev_a = torch.full((1,), rssm.no_action, device=device)
    reward_events = episodes = 0
    for _ in range(steps):
        if actor is not None:
            state, _, _ = rssm.obs_step(state, prev_a, embed)
        if actor is None or rng.random() < epsilon:
            a = int(rng.integers(n_act))
        else:
            a = int(actor(rssm.belief(state)).argmax(dim=-1))
        next_obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
        next_embed = emb(next_obs)
        buffer.add(
            Transition(embed[0].cpu().numpy(), a, float(r), next_embed[0].cpu().numpy(), done)
        )
        reward_events += int(r > 0)
        prev_a = torch.tensor([a], device=device)
        embed = next_embed
        if done:
            episodes += 1
            obs, _ = env.reset()
            embed = emb(obs)
            state = rssm.initial(1, device)
            prev_a = torch.full((1,), rssm.no_action, device=device)
    return reward_events, episodes


def wm_train_curious(
    buffer,
    rssm,
    recon_head,
    reward_head,
    value_head,
    continue_head,
    opt,
    updates,
    seq_batch,
    window,
    free_bits,
    device,
    cr,
):
    """Crafter WM training with Curious Replay (exp 0028; arxiv:2306.15934).

    Identical objective to train_rssm.wm_train (two-hot reward, no SIGReg — frozen encoder,
    cached embeddings → identity encode), but the replay windows are sampled by per-start
    priority and rescored each step. The per-item curiosity signal `L_i` is recon+KL summed
    over the window (the RSSM dynamics-prediction error) — surprising/rare transitions get
    replayed more so the WM learns their dynamics. `cr` = dict(alpha,beta,c,eps,p_max).
    See knowledge/experiments/0028-curious-replay.md and papers/curious-replay-2023.md.
    """
    buffer.prepare_prioritized(window, p_max=cr["p_max"])
    for step in range(updates):
        batch, starts = buffer.sample_sequences_prioritized(seq_batch, window)
        embed = torch.as_tensor(batch["obs"], device=device)  # cached embeddings (B,W,E)
        actions = torch.as_tensor(batch["action"], device=device)
        rewards = torch.as_tensor(batch["reward"], device=device)
        returns = torch.as_tensor(batch["return"], device=device)
        dones = torch.as_tensor(batch["done"], device=device, dtype=torch.float32)
        b, w = actions.shape

        state = rssm.initial(b, device)
        prev_a = torch.full((b,), rssm.no_action, dtype=torch.long, device=device)
        per_item = torch.zeros(b, device=device)  # recon+KL per window → CR priority
        recon = kl = r_l = v_l = c_l = torch.zeros((), device=device)
        for k in range(w):
            state, prior, post = rssm.obs_step(state, prev_a, embed[:, k])
            belief = rssm.belief(state)
            kl_k = kl_balanced(post, prior, free_bits, reduce=False)  # (B,)
            recon_k = F.mse_loss(recon_head(belief), embed[:, k].detach(), reduction="none").mean(
                -1
            )
            per_item = per_item + recon_k.detach() + kl_k.detach()
            recon = recon + recon_k.mean()
            kl = kl + kl_k.mean()
            r_l = r_l + reward_head.twohot_loss(belief, actions[:, k], rewards[:, k])
            v_l = v_l + value_loss(value_head(belief), returns[:, k])
            c_l = c_l + F.binary_cross_entropy_with_logits(continue_head(belief), 1.0 - dones[:, k])
            prev_a = actions[:, k]
        loss = (recon + kl + r_l + v_l + c_l) / w
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 100.0)
        opt.step()
        buffer.update_priorities(
            starts, per_item.cpu().numpy(), cr["alpha"], cr["beta"], cr["c"], cr["eps"]
        )
        if (step + 1) % 200 == 0:
            print(
                f"  wm {step + 1}: recon={recon.item() / w:.4f} kl={kl.item() / w:.3f} "
                f"r={r_l.item() / w:.4f}"
            )


def crafter_score(per_ach_unlock_counts: dict[str, int], episodes: int) -> float:
    """Official Crafter Score (Hafner 2021): geometric mean of the 22 per-achievement success
    PERCENTAGES, S = exp(mean_i ln(1 + s_i)) - 1, s_i in [0,100]. Rewards BREADTH (any nonzero
    rate on many achievements) over spamming a few — the leaderboard metric (human 50.5%,
    Achievement-Distillation 21.8%, DreamerV3 14.5%). Needs many episodes for stable rare rates."""
    rates = [100.0 * c / episodes for c in per_ach_unlock_counts.values()]
    return float(np.exp(np.mean(np.log1p(rates))) - 1.0)


@torch.no_grad()
def evaluate(enc, rssm, actor, n_act, episodes, length, device, seed_base=10_000):
    """Mean reward, mean #achievements, the unlocked SET, and the official Crafter Score (%).
    The set reveals DEPTH (count can't tell {wake_up} from {collect_stone}); the score makes us
    directly comparable to the published leaderboard."""
    agent = RSSMActorAgent(enc, rssm, actor, n_act, device, epsilon=0.0, seed=seed_base)
    rewards, achievements = [], []
    unlocked: set[str] = set()
    counts: dict[str, int] = {}
    for ep in range(episodes):
        env = make_crafter_env(length=length, seed=seed_base + ep)
        obs, info = env.reset(seed=seed_base + ep)
        agent.reset()
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(agent.act(obs))
            total += r
            done = term or trunc
        rewards.append(total)
        if not counts:  # seed all 22 achievement keys (env exposes the full set every step)
            counts = dict.fromkeys(info["achievements"], 0)
        ep_unlocked = {k for k, v in info["achievements"].items() if v > 0}
        for k in ep_unlocked:
            counts[k] += 1
        achievements.append(len(ep_unlocked))
        unlocked |= ep_unlocked
    score = crafter_score(counts, episodes) if counts else 0.0
    return float(np.mean(rewards)), float(np.mean(achievements)), unlocked, score


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--round0-steps", type=int, default=10000)
    p.add_argument("--actor-steps", type=int, default=10000)
    p.add_argument("--updates-per-round", type=int, default=2000)
    p.add_argument("--ac-updates-per-round", type=int, default=2000)
    p.add_argument("--seq-batch", type=int, default=16)
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--burn-in", type=int, default=8)
    p.add_argument("--free-bits", type=float, default=1.0)
    p.add_argument("--success-frac", type=float, default=0.0)  # Crafter reward not sparse-binary
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--ent-coef", type=float, default=3e-3)
    p.add_argument("--repval", type=float, default=0.3)  # critic-on-replay (exp 0021), kept on
    p.add_argument("--pool", default="cls", choices=["cls", "patch_mean", "cls+patch"])  # exp 0024
    # Curious Replay (exp 0028; arxiv:2306.15934) — prioritize WM-train windows by recon+KL
    # surprise. Defaults are the paper's fixed values. --curious off → uniform WM replay.
    p.add_argument("--curious", action="store_true")
    p.add_argument("--cr-alpha", type=float, default=0.7)
    p.add_argument("--cr-beta", type=float, default=0.7)
    p.add_argument("--cr-c", type=float, default=1e4)
    p.add_argument("--cr-eps", type=float, default=0.01)
    p.add_argument("--cr-pmax", type=float, default=1e5)
    # Self-imitation (exp 0030; AD-spirit actor-side lever): reinforce real achievement
    # trajectories the imagined PG drowns out. --self-imitation on → sil-weight active.
    p.add_argument("--self-imitation", action="store_true")
    p.add_argument("--sil-weight", type=float, default=0.5)
    p.add_argument("--sil-success-frac", type=float, default=0.5)
    p.add_argument("--ep-length", type=int, default=2000)  # collection episode cap
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--eval-length", type=int, default=1000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_crafter_env(length=args.ep_length, seed=args.seed)
    n_act = int(env.action_space.n)
    enc = FrozenDinoEncoder(pool=args.pool).to(device)  # frozen; not in any optimizer
    ed = enc.latent_dim
    rssm = RSSM(embed_dim=ed, num_actions=n_act).to(device)
    sd = rssm.state_dim
    recon_head = torch.nn.Linear(sd, ed).to(device)
    rew = TwoHotRewardHead(state_dim=sd, num_actions=n_act).to(device)  # bounded reward (exp 0025)
    val = ValueHead(state_dim=sd).to(device)
    cont = ContinueHead(state_dim=sd).to(device)
    actor = Actor(state_dim=sd, num_actions=n_act).to(device)
    critic = TwoHotValueHead(state_dim=sd).to(device)  # bounded distributional critic (exp 0023)
    target_critic = copy.deepcopy(critic)

    wm_mods = [rssm, recon_head, rew, val, cont]  # enc frozen → excluded from the optimizer
    opt_wm = torch.optim.Adam([q for m in wm_mods for q in m.parameters()], lr=3e-4)
    opt_ac = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=3e-4)

    # buffer stores DINO EMBEDDINGS (float32), not pixels; identity "encoder" in the WM/AC
    # loops so the cached embeddings pass through untouched (zero DINO forwards in training).
    cap = args.round0_steps + args.rounds * args.actor_steps + 1000
    buffer = ReplayBuffer(cap, (ed,), seed=args.seed, obs_dtype=np.float32)
    train_enc = torch.nn.Identity()
    rng = np.random.default_rng(args.seed)
    best = {"reward": None, "round": -1, "state": {}}

    for rnd in range(args.rounds):
        for m in wm_mods + [actor, critic]:
            m.train()
        act_fn = None if rnd == 0 else actor
        steps = args.round0_steps if rnd == 0 else args.actor_steps
        print(f"round {rnd}: collecting {steps} ...", flush=True)
        r, e = collect_embed(
            buffer, env, enc, rssm, act_fn, n_act, steps, args.epsilon, device, rng, args.seed + rnd
        )
        print(f"  -> {e} episodes, {r} reward events")
        buffer.compute_returns(args.gamma)

        print(f"round {rnd}: wm train {args.updates_per_round} ...", flush=True)
        if args.curious:
            wm_train_curious(
                buffer,
                rssm,
                recon_head,
                rew,
                val,
                cont,
                opt_wm,
                args.updates_per_round,
                args.seq_batch,
                args.window,
                args.free_bits,
                device,
                cr={
                    "alpha": args.cr_alpha,
                    "beta": args.cr_beta,
                    "c": args.cr_c,
                    "eps": args.cr_eps,
                    "p_max": args.cr_pmax,
                },
            )
        else:
            wm_train(
                buffer,
                train_enc,
                rssm,
                recon_head,
                rew,
                val,
                cont,
                None,
                opt_wm,
                args.updates_per_round,
                args.seq_batch,
                args.window,
                0.0,
                args.free_bits,
                args.success_frac,
                device,
            )  # lam=0 → SIGReg skipped (frozen encoder)
        print(f"round {rnd}: imagine AC {args.ac_updates_per_round} ...", flush=True)
        al, cl, ir = imagine_ac(
            buffer,
            train_enc,
            rssm,
            rew,
            cont,
            actor,
            critic,
            target_critic,
            opt_ac,
            args.ac_updates_per_round,
            args.seq_batch,
            args.window,
            args.burn_in,
            args.horizon,
            args.gamma,
            args.lam,
            args.ent_coef,
            device,
            beta_repval=args.repval,
            sil_weight=args.sil_weight if args.self_imitation else 0.0,
            sil_success_frac=args.sil_success_frac,
        )
        mean_r, mean_ach, unlocked, score = evaluate(
            enc, rssm, actor, n_act, args.eval_episodes, args.eval_length, device
        )
        print(
            f"  actor_loss={al:.3f} critic_loss={cl:.4f} imagined_return={ir:.3f} "
            f"eval_reward={mean_r:.2f} eval_achievements={mean_ach:.2f} "
            f"crafter_score={score:.2f}% unlocked=[{','.join(sorted(unlocked))}]",
            flush=True,
        )
        if best["reward"] is None or mean_r >= best["reward"]:
            best = {
                "reward": mean_r,
                "round": rnd,
                "state": {
                    "rssm": copy.deepcopy(rssm.state_dict()),
                    "actor": copy.deepcopy(actor.state_dict()),
                },
            }

    path = Path(args.save)
    path = path.with_name(f"{path.stem}_s{args.seed}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rssm_agent": True,
            "encoder": "dino",
            **best["state"],
            "config": {
                **vars(args),
                "num_actions": n_act,
                "obs_shape": tuple(env.observation_space.shape),
            },
            "metrics": {"eval_reward": best["reward"], "best_round": best["round"]},
        },
        path,
    )
    print(f"best_eval_reward={best['reward']:.2f} round={best['round']} -> {path}", flush=True)


if __name__ == "__main__":
    main()
