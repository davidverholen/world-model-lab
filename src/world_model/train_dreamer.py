"""Dreamer-style flywheel: world model + actor-critic trained in imagination (exp 0017).

Replaces the slow CEM planner with a fast reactive actor learned on-policy in the
world model's imagination — the fix exp 0016 (BC distillation) pointed to, and the
Mode-1 policy needed for Crafter-scale collection. Reuses the settled world-model
recipe (train()/collect_round from train_recurrent; encoder freeze after round 2);
adds only the actor-critic. Recipe follows DreamerV3 (knowledge/papers/dreamerv3):
imagine H steps under the actor, lambda-returns, REINFORCE actor + value-baseline +
entropy, EMA target critic. Discrete actions → REINFORCE (no dynamics backprop).

    uv run python -m world_model.train_dreamer --save runs/dk6_dreamer.pt
"""

import argparse
import copy

import torch

from world_model.agents import RandomAgent
from world_model.agents.actor_agent import ActorAgent
from world_model.envs import make_minigrid_env
from world_model.models import Actor, ConvEncoder, RecurrentDynamics, RewardHead, SIGReg, ValueHead
from world_model.models.continue_head import ContinueHead
from world_model.train_recurrent import collect_round
from world_model.train_recurrent import train as wm_train
from world_model.training import ReplayBuffer


def train_continue(
    buffer, encoder, dynamics, continue_head, opt_c, updates, seq_batch, window, device
):
    """Train the continue predictor on real (belief, 1-done) pairs along the window."""
    no_act = dynamics.no_action
    last = 0.0
    for _ in range(updates):
        batch = buffer.sample_sequences(seq_batch, window, success_frac=0.5)  # need terminations
        obs_seq = torch.as_tensor(batch["obs"], device=device)
        actions = torch.as_tensor(batch["action"], device=device)
        dones = torch.as_tensor(batch["done"], device=device, dtype=torch.float32)  # (B, W)
        b, w = actions.shape
        with torch.no_grad():
            z = encoder(obs_seq.flatten(0, 1)).unflatten(0, (b, w))
        s = dynamics.initial_state(b, device)
        pa = torch.full((b,), no_act, dtype=torch.long, device=device)
        logits = []
        for k in range(w):
            s = dynamics.update(z[:, k].detach(), pa, s)
            logits.append(continue_head(s))
            pa = actions[:, k]
        cont_logits = torch.stack(logits, dim=1)  # (B, W)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(cont_logits, 1.0 - dones)
        opt_c.zero_grad()
        loss.backward()
        opt_c.step()
        last = loss.item()
    return last


def imagine_ac(
    buffer,
    encoder,
    dynamics,
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
):
    """Train actor+critic on imagined rollouts. World model is frozen here (no grad);
    only actor/critic update. Returns last (actor_loss, critic_loss, mean_imagined_return)."""
    no_act = dynamics.no_action
    stats = (0.0, 0.0, 0.0)
    for _ in range(updates):
        batch = buffer.sample_sequences(seq_batch, window)
        obs_seq = torch.as_tensor(batch["obs"], device=device)
        actions = torch.as_tensor(batch["action"], device=device)
        b, w = actions.shape
        with torch.no_grad():
            z = encoder(obs_seq.flatten(0, 1)).unflatten(0, (b, w))
            s = dynamics.initial_state(b, device)
            pa = torch.full((b,), no_act, dtype=torch.long, device=device)
            for k in range(burn_in):
                s = dynamics.update(z[:, k], pa, s)
                pa = actions[:, k]
            # imagine H steps under the current actor
            states, acts, rews, conts = [], [], [], []
            si = s
            for _t in range(horizon):
                a = torch.distributions.Categorical(logits=actor(si)).sample()
                states.append(si)
                acts.append(a)
                rews.append(reward_head(si, a))
                conts.append(torch.sigmoid(continue_head(si)))  # P(episode continues)
                si = dynamics.update(dynamics.predict_next(si, a), a, si)
            s_states = torch.stack(states)  # (H, b, S)
            a_acts = torch.stack(acts)  # (H, b)
            r_rews = torch.stack(rews)  # (H, b)
            c_cont = torch.stack(conts)  # (H, b) in [0,1]
            v_tgt = target_critic(s_states.flatten(0, 1)).unflatten(0, (horizon, b))  # (H, b)
            v_H = target_critic(si)  # (b,)
            # lambda-returns with the continue predictor: the per-step discount is
            # gamma*c_t, so reward past a predicted termination (c->0) is zeroed.
            # G_t = r_t + gamma*c_t[(1-lam)V(s_{t+1}) + lam G_{t+1}], G_H=V(s_H)
            returns = torch.empty(horizon, b, device=device)
            g = v_H
            for t in reversed(range(horizon)):
                nextv = v_tgt[t + 1] if t + 1 < horizon else v_H
                g = r_rews[t] + gamma * c_cont[t] * ((1 - lam) * nextv + lam * g)
                returns[t] = g

        flat_s = s_states.flatten(0, 1)
        adv = (returns - v_tgt).flatten().detach()
        dist = torch.distributions.Categorical(logits=actor(flat_s))
        logp = dist.log_prob(a_acts.flatten())
        actor_loss = -(logp * adv).mean() - ent_coef * dist.entropy().mean()
        critic_loss = (critic(flat_s) - returns.flatten().detach()).pow(2).mean()

        opt_ac.zero_grad()
        (actor_loss + critic_loss).backward()
        opt_ac.step()
        with torch.no_grad():
            for pt, pc in zip(target_critic.parameters(), critic.parameters(), strict=True):
                pt.mul_(0.98).add_(pc, alpha=0.02)
        stats = (actor_loss.item(), critic_loss.item(), returns.mean().item())
    return stats


def evaluate_actor(encoder, dynamics, actor, env_id, episodes, device, seed_base=10_000):
    env = make_minigrid_env(env_id, fully_observable=False)
    agent = ActorAgent(encoder, dynamics, actor, int(env.action_space.n), device)
    succ = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        agent.reset()
        done, reward = False, 0.0
        while not done:
            obs, reward, term, trunc, _ = env.step(agent.act(obs))
            done = term or trunc
        succ += int(reward > 0)
    encoder.train()
    dynamics.train()
    actor.train()
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
    p.add_argument("--success-frac", type=float, default=0.25)
    p.add_argument("--ignition-events", type=int, default=5)
    p.add_argument("--freeze-round", type=int, default=2, help="freeze encoder from this round")
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--ent-coef", type=float, default=3e-3)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save", required=True)
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_minigrid_env(args.env_id, fully_observable=False)
    n_act = int(env.action_space.n)

    enc = ConvEncoder().to(device)
    dyn = RecurrentDynamics(num_actions=n_act).to(device)
    rew = RewardHead(latent_dim=dyn.state_dim, num_actions=n_act).to(device)
    val = ValueHead(state_dim=dyn.state_dim).to(
        device
    )  # for wm_train (real-data MC), unused by actor
    sig = SIGReg().to(device)
    actor = Actor(state_dim=dyn.state_dim, num_actions=n_act).to(device)
    critic = ValueHead(state_dim=dyn.state_dim).to(device)
    target_critic = copy.deepcopy(critic)
    cont = ContinueHead(state_dim=dyn.state_dim).to(device)
    wm_mods = [enc, dyn, rew, val]
    opt_wm = torch.optim.Adam([q for m in wm_mods for q in m.parameters()], lr=3e-4)
    opt_ac = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=3e-4)
    opt_c = torch.optim.Adam(cont.parameters(), lr=3e-4)

    mpc_steps = args.actor_steps
    buffer = ReplayBuffer(
        2 * args.round0_steps + args.rounds * mpc_steps, env.observation_space.shape, seed=args.seed
    )
    best = {"rate": None, "round": -1, "state": {}}
    total_steps = 0
    enc_frozen = False

    for rnd in range(args.rounds):
        if rnd == 0:
            agent = RandomAgent(env.action_space, seed=args.seed)
            steps = args.round0_steps
        else:
            agent = ActorAgent(
                enc, dyn, actor, n_act, device, epsilon=args.epsilon, seed=args.seed + rnd
            )
            for m in wm_mods + [actor, critic]:
                m.train()
            steps = mpc_steps
        print(f"round {rnd}: collecting {steps} steps ...")
        r, e = collect_round(buffer, env, agent, steps, seed=args.seed + rnd)
        total_steps += steps
        while rnd == 0 and r < args.ignition_events and total_steps < 2 * args.round0_steps:
            chunk = min(5000, 2 * args.round0_steps - total_steps)
            r2, _ = collect_round(buffer, env, agent, chunk, seed=args.seed + 100)
            r += r2
            total_steps += chunk
        print(f"  -> {e} episodes, {r} reward events (total {total_steps})")
        buffer.compute_returns(args.gamma)

        if not enc_frozen and rnd >= args.freeze_round:
            for q in enc.parameters():
                q.requires_grad_(False)
            opt_wm = torch.optim.Adam(
                [q for m in wm_mods for q in m.parameters() if q.requires_grad], lr=3e-4
            )
            enc_frozen = True
            print(f"  encoder frozen at round {rnd}")

        print(f"round {rnd}: world-model train {args.updates_per_round} ...")
        wm_train(
            buffer,
            enc,
            dyn,
            rew,
            val,
            sig,
            opt_wm,
            args.updates_per_round,
            args.seq_batch,
            args.window,
            args.burn_in,
            args.sigreg_weight,
            device,
            success_frac=args.success_frac,
            amp=args.amp,
        )
        cont_loss = train_continue(
            buffer,
            enc,
            dyn,
            cont,
            opt_c,
            args.ac_updates_per_round,
            args.seq_batch,
            args.window,
            device,
        )
        print(f"round {rnd}: actor-critic imagine {args.ac_updates_per_round} ...")
        al, cl, ir = imagine_ac(
            buffer,
            enc,
            dyn,
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
        )
        rate = evaluate_actor(enc, dyn, actor, args.env_id, args.eval_episodes, device)
        print(
            f"  cont_loss={cont_loss:.4f} actor_loss={al:.3f} critic_loss={cl:.4f} "
            f"imagined_return={ir:.3f} eval_success={rate:.2f}"
        )
        if best["rate"] is None or rate >= best["rate"]:
            best = {
                "rate": rate,
                "round": rnd,
                "state": {
                    "encoder": copy.deepcopy(enc.state_dict()),
                    "dynamics": copy.deepcopy(dyn.state_dict()),
                    "actor": copy.deepcopy(actor.state_dict()),
                    "critic": copy.deepcopy(critic.state_dict()),
                    "continue_head": copy.deepcopy(cont.state_dict()),
                },
            }

    from pathlib import Path

    path = Path(args.save)
    path = path.with_name(f"{path.stem}_s{args.seed}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dreamer": True,
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
