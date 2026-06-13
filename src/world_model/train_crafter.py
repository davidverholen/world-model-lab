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

from world_model.agents.rssm_agent import RSSMActorAgent
from world_model.envs import make_crafter_env
from world_model.models import Actor, FrozenDinoEncoder, RewardHead, ValueHead
from world_model.models.continue_head import ContinueHead
from world_model.models.rssm import RSSM
from world_model.train_rssm import imagine_ac, wm_train
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


@torch.no_grad()
def evaluate(enc, rssm, actor, n_act, episodes, length, device, seed_base=10_000):
    """Mean episode reward + mean #achievements unlocked (greedy RSSMActorAgent)."""
    agent = RSSMActorAgent(enc, rssm, actor, n_act, device, epsilon=0.0, seed=seed_base)
    rewards, achievements = [], []
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
        achievements.append(sum(1 for v in info["achievements"].values() if v > 0))
    return float(np.mean(rewards)), float(np.mean(achievements))


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
    enc = FrozenDinoEncoder().to(device)  # frozen; not in any optimizer
    ed = enc.latent_dim
    rssm = RSSM(embed_dim=ed, num_actions=n_act).to(device)
    sd = rssm.state_dim
    recon_head = torch.nn.Linear(sd, ed).to(device)
    rew = RewardHead(latent_dim=sd, num_actions=n_act).to(device)
    val = ValueHead(state_dim=sd).to(device)
    cont = ContinueHead(state_dim=sd).to(device)
    actor = Actor(state_dim=sd, num_actions=n_act).to(device)
    critic = ValueHead(state_dim=sd).to(device)
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
        )
        mean_r, mean_ach = evaluate(
            enc, rssm, actor, n_act, args.eval_episodes, args.eval_length, device
        )
        print(
            f"  actor_loss={al:.3f} critic_loss={cl:.4f} imagined_return={ir:.3f} "
            f"eval_reward={mean_r:.2f} eval_achievements={mean_ach:.2f}",
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
