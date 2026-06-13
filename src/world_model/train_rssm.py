"""RSSM Dreamer flywheel: stochastic-latent world model + imagination actor-critic
(exp 0019 — the exploitation fix for 0017/0018).

World model = encoder + RSSM (stochastic latent). WM loss = reconstruction
(belief→embed) + balanced KL(post‖prior) with free bits + reward/value/continue, plus
SIGReg on the embedding for anti-collapse. Imagination samples z~prior (img_step) — so
the actor cannot drive to a single deterministic fake-reward state. Reuses the
settled recipe (encoder freeze @round 2, UTD, success oversampling, continue predictor
0018, actor-critic 0017). See knowledge/experiments/0019-stochastic-latents.md.

    uv run python -m world_model.train_rssm --save runs/dk6_rssm.pt
"""

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, kl_divergence

from world_model.agents import RandomAgent
from world_model.agents.rssm_agent import RSSMActorAgent
from world_model.envs import make_minigrid_env
from world_model.models import Actor, ConvEncoder, RewardHead, SIGReg, ValueHead
from world_model.models.continue_head import ContinueHead
from world_model.models.reward import reward_loss
from world_model.models.rssm import RSSM
from world_model.models.value import value_loss
from world_model.train_recurrent import collect_round, shrink_perturb
from world_model.training import ReplayBuffer


def _sg(dist: Normal) -> Normal:
    return Normal(dist.mean.detach(), dist.stddev.detach())


def kl_balanced(post: Normal, prior: Normal, free_bits: float) -> torch.Tensor:
    """DreamerV3 KL balancing: beta_dyn*KL(sg(post)||prior) + beta_rep*KL(post||sg(prior))."""
    kl_dyn = kl_divergence(_sg(post), prior).sum(-1).clamp(min=free_bits)
    kl_rep = kl_divergence(post, _sg(prior)).sum(-1).clamp(min=free_bits)
    return (0.5 * kl_dyn + 0.1 * kl_rep).mean()


def wm_train(
    buffer,
    encoder,
    rssm,
    recon_head,
    reward_head,
    value_head,
    continue_head,
    sigreg,
    opt,
    updates,
    seq_batch,
    window,
    lam,
    free_bits,
    success_frac,
    device,
):
    for step in range(updates):
        batch = buffer.sample_sequences(seq_batch, window, success_frac=success_frac)
        obs = torch.as_tensor(batch["obs"], device=device)
        actions = torch.as_tensor(batch["action"], device=device)
        rewards = torch.as_tensor(batch["reward"], device=device)
        returns = torch.as_tensor(batch["return"], device=device)
        dones = torch.as_tensor(batch["done"], device=device, dtype=torch.float32)
        b, w = actions.shape
        embed = encoder(obs.flatten(0, 1)).unflatten(0, (b, w))  # (B,W,E)

        state = rssm.initial(b, device)
        prev_a = torch.full((b,), rssm.no_action, dtype=torch.long, device=device)
        kl = recon = r_l = v_l = c_l = torch.zeros((), device=device)
        for k in range(w):
            state, prior, post = rssm.obs_step(state, prev_a, embed[:, k])
            belief = rssm.belief(state)
            kl = kl + kl_balanced(post, prior, free_bits)
            recon = recon + F.mse_loss(recon_head(belief), embed[:, k].detach())
            r_l = r_l + reward_loss(reward_head(belief, actions[:, k]), rewards[:, k])
            v_l = v_l + value_loss(value_head(belief), returns[:, k])
            c_l = c_l + F.binary_cross_entropy_with_logits(continue_head(belief), 1.0 - dones[:, k])
            prev_a = actions[:, k]
        loss = (recon + kl + r_l + v_l + c_l) / w
        if lam > 0:  # frozen encoder (exp 0022) can't collapse → SIGReg unnecessary, pass lam=0
            loss = loss + lam * sigreg(embed.flatten(0, 1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 100.0)
        opt.step()
        if (step + 1) % 200 == 0:
            print(
                f"  wm {step + 1}: recon={recon.item() / w:.4f} kl={kl.item() / w:.3f} "
                f"r={r_l.item() / w:.4f}"
            )


def _critic_loss(critic, beliefs, targets):
    """Two-hot cross-entropy if the critic supports it (DreamerV3 distributional, bounded);
    else plain MSE. Lets the same imagine_ac serve MiniGrid (ValueHead) and Crafter (TwoHot)."""
    if hasattr(critic, "twohot_loss"):
        return critic.twohot_loss(beliefs, targets)
    return (critic(beliefs) - targets).pow(2).mean()


def imagine_ac(
    buffer,
    encoder,
    rssm,
    reward_head,
    continue_head,
    actor,
    critic,
    target_critic,
    opt_ac,
    updates,
    seq_batch,
    window,
    burn_in,
    horizon,
    gamma,
    lam,
    ent_coef,
    device,
    beta_repval=0.0,
):
    stats = (0.0, 0.0, 0.0)
    for _ in range(updates):
        batch = buffer.sample_sequences(seq_batch, window)
        obs = torch.as_tensor(batch["obs"], device=device)
        actions = torch.as_tensor(batch["action"], device=device)
        ret_real = torch.as_tensor(batch["return"], device=device)  # real MC returns
        b, w = actions.shape
        with torch.no_grad():
            embed = encoder(obs.flatten(0, 1)).unflatten(0, (b, w))
            state = rssm.initial(b, device)
            prev_a = torch.full((b,), rssm.no_action, dtype=torch.long, device=device)
            real_bels = []  # beliefs over REAL states (for the replay-grounded critic, b1)
            for k in range(burn_in):
                state, _, _ = rssm.obs_step(state, prev_a, embed[:, k])
                real_bels.append(rssm.belief(state))
                prev_a = actions[:, k]
            real_bel = torch.stack(real_bels)  # (burn_in,b,S), detached
            beliefs, acts, rews, conts = [], [], [], []
            s = state
            for _t in range(horizon):
                bel = rssm.belief(s)
                a = Categorical(logits=actor(bel)).sample()
                beliefs.append(bel)
                acts.append(a)
                rews.append(reward_head(bel, a))
                conts.append(torch.sigmoid(continue_head(bel)))
                s, _ = rssm.img_step(s, a)
            bel_s = torch.stack(beliefs)  # (H,b,S)
            a_s = torch.stack(acts)
            r_s = torch.stack(rews)
            c_s = torch.stack(conts)
            v_tgt = target_critic(bel_s.flatten(0, 1)).unflatten(0, (horizon, b))
            v_H = target_critic(rssm.belief(s))
            returns = torch.empty(horizon, b, device=device)
            g = v_H
            for t in reversed(range(horizon)):
                nextv = v_tgt[t + 1] if t + 1 < horizon else v_H
                g = r_s[t] + gamma * c_s[t] * ((1 - lam) * nextv + lam * g)
                returns[t] = g

        flat = bel_s.flatten(0, 1)
        adv = (returns - v_tgt).flatten().detach()
        dist = Categorical(logits=actor(flat))
        actor_loss = -(dist.log_prob(a_s.flatten()) * adv).mean() - ent_coef * dist.entropy().mean()
        critic_loss = _critic_loss(critic, flat, returns.flatten().detach())
        if beta_repval > 0:
            # DreamerV3 critic-on-replay (exp 0021): ground the critic in REAL returns so
            # the EMA target — and thus the actor's advantage baseline — can't drift with
            # the inflated imagined value. real_bel is detached, so grad flows to critic only.
            rb = real_bel.flatten(0, 1)
            rt = ret_real[:, :burn_in].t().reshape(-1).detach()
            critic_loss = critic_loss + beta_repval * _critic_loss(critic, rb, rt)
        opt_ac.zero_grad()
        (actor_loss + critic_loss).backward()
        opt_ac.step()
        with torch.no_grad():
            for pt, pc in zip(target_critic.parameters(), critic.parameters(), strict=True):
                pt.mul_(0.98).add_(pc, alpha=0.02)
        stats = (actor_loss.item(), critic_loss.item(), returns.mean().item())
    return stats


def evaluate(encoder, rssm, actor, env_id, episodes, device, seed_base=10_000):
    env = make_minigrid_env(env_id, fully_observable=False)
    agent = RSSMActorAgent(encoder, rssm, actor, int(env.action_space.n), device)
    succ = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        agent.reset()
        done, reward = False, 0.0
        while not done:
            obs, reward, term, trunc, _ = env.step(agent.act(obs))
            done = term or trunc
        succ += int(reward > 0)
    for m in (encoder, rssm, actor):
        m.train()
    return succ / episodes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default="MiniGrid-DoorKey-6x6-v0")
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--round0-steps", type=int, default=20000)
    p.add_argument("--actor-steps", type=int, default=15000)
    p.add_argument("--updates-per-round", type=int, default=4000)
    p.add_argument("--ac-updates-per-round", type=int, default=4000)
    p.add_argument("--seq-batch", type=int, default=32)
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--burn-in", type=int, default=8)
    p.add_argument("--sigreg-weight", type=float, default=0.05)
    p.add_argument("--free-bits", type=float, default=1.0)
    p.add_argument("--success-frac", type=float, default=0.25)
    p.add_argument("--ignition-events", type=int, default=5)
    p.add_argument("--freeze-round", type=int, default=2)
    p.add_argument(
        "--ac-reset",
        action="store_true",
        help="exp 0020: shrink-perturb actor+critic at the start of each round >=1 "
        "(Nikishin/Qiao primacy fix on the behaviour layer) — attacks the round-6 collapse",
    )
    p.add_argument("--ac-reset-alpha", type=float, default=0.5)
    p.add_argument(
        "--repval",
        type=float,
        default=0.3,
        help="critic-on-replay weight (DreamerV3 beta_repval). Now recipe-DEFAULT (0.3) after "
        "exp 0021 confirmed it calibrates imagined value (inflated->~1). Set 0.0 to disable.",
    )
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--ent-coef", type=float, default=3e-3)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", required=True)
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_minigrid_env(args.env_id, fully_observable=False)
    n_act = int(env.action_space.n)

    enc = ConvEncoder().to(device)
    rssm = RSSM(embed_dim=enc.latent_dim, num_actions=n_act).to(device)
    sd = rssm.state_dim
    recon_head = torch.nn.Linear(sd, enc.latent_dim).to(device)
    rew = RewardHead(latent_dim=sd, num_actions=n_act).to(device)
    val = ValueHead(state_dim=sd).to(device)
    cont = ContinueHead(state_dim=sd).to(device)
    sig = SIGReg().to(device)
    actor = Actor(state_dim=sd, num_actions=n_act).to(device)
    critic = ValueHead(state_dim=sd).to(device)
    target_critic = copy.deepcopy(critic)

    wm_mods = [enc, rssm, recon_head, rew, val, cont]
    opt_wm = torch.optim.Adam([q for m in wm_mods for q in m.parameters()], lr=3e-4)
    opt_ac = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=3e-4)

    cap = 2 * args.round0_steps + args.rounds * args.actor_steps
    buffer = ReplayBuffer(cap, env.observation_space.shape, seed=args.seed)
    best = {"rate": None, "round": -1, "state": {}}
    total = 0
    frozen = False

    for rnd in range(args.rounds):
        if rnd == 0:
            agent, steps = RandomAgent(env.action_space, seed=args.seed), args.round0_steps
        else:
            agent = RSSMActorAgent(
                enc, rssm, actor, n_act, device, epsilon=args.epsilon, seed=args.seed + rnd
            )
            for m in wm_mods + [actor, critic]:
                m.train()
            steps = args.actor_steps
        print(f"round {rnd}: collecting {steps} ...")
        r, e = collect_round(buffer, env, agent, steps, seed=args.seed + rnd)
        total += steps
        while rnd == 0 and r < args.ignition_events and total < 2 * args.round0_steps:
            chunk = min(5000, 2 * args.round0_steps - total)
            r2, _ = collect_round(buffer, env, agent, chunk, seed=args.seed + 100)
            r += r2
            total += chunk
        print(f"  -> {e} episodes, {r} reward events (total {total})")
        buffer.compute_returns(args.gamma)

        if not frozen and rnd >= args.freeze_round:
            for q in enc.parameters():
                q.requires_grad_(False)
            opt_wm = torch.optim.Adam(
                [q for m in wm_mods for q in m.parameters() if q.requires_grad], lr=3e-4
            )
            frozen = True
            print(f"  encoder frozen at round {rnd}")

        print(f"round {rnd}: wm train {args.updates_per_round} ...")
        wm_train(
            buffer,
            enc,
            rssm,
            recon_head,
            rew,
            val,
            cont,
            sig,
            opt_wm,
            args.updates_per_round,
            args.seq_batch,
            args.window,
            args.sigreg_weight,
            args.free_bits,
            args.success_frac,
            device,
        )
        if args.ac_reset and rnd >= 1:
            # primacy/plasticity reset of the behaviour layer (exp 0020). Resync the EMA
            # target to the perturbed critic and rebuild opt_ac so stale Adam moments
            # don't corrupt the freshly perturbed params.
            shrink_perturb(actor, args.ac_reset_alpha)
            shrink_perturb(critic, args.ac_reset_alpha)
            target_critic.load_state_dict(critic.state_dict())
            opt_ac = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=3e-4)
            print(f"  ac reset (shrink-perturb alpha={args.ac_reset_alpha})")

        print(f"round {rnd}: imagine AC {args.ac_updates_per_round} ...")
        al, cl, ir = imagine_ac(
            buffer,
            enc,
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
        rate = evaluate(enc, rssm, actor, args.env_id, args.eval_episodes, device)
        print(
            f"  actor_loss={al:.3f} critic_loss={cl:.4f} imagined_return={ir:.3f} "
            f"eval_success={rate:.2f}"
        )
        if best["rate"] is None or rate >= best["rate"]:
            best = {
                "rate": rate,
                "round": rnd,
                "state": {
                    "encoder": copy.deepcopy(enc.state_dict()),
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
            **best["state"],
            "config": {
                **vars(args),
                "num_actions": n_act,
                "obs_shape": tuple(env.observation_space.shape),
            },
            "metrics": {"eval_success": best["rate"], "best_round": best["round"]},
        },
        path,
    )
    print(f"best_eval={best['rate']:.2f} best_round={best['round']} saved to {path}")


if __name__ == "__main__":
    main()
