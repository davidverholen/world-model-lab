"""Train a recurrent world model on a partially observable MiniGrid env.

Pipeline (exp 0005): collect rounds of experience -> train GRU belief-state world
model on sequence windows (burn-in closed-loop, then open-loop rollout losses) ->
optionally collect the next round WITH the trained agent (epsilon-greedy MPC), which
concentrates data near rewarding behavior far better than random play in
sparse-reward tasks.

Run: uv run python -m world_model.train_recurrent --env-id MiniGrid-DoorKey-5x5-v0
         --rounds 2 --save runs/doorkey5x5_rwm.pt
"""

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from world_model.agents import RandomAgent
from world_model.agents.mpc import RecurrentMPCAgent
from world_model.envs import make_minigrid_env
from world_model.models import ConvEncoder, RecurrentDynamics, RewardHead, SIGReg, ValueHead
from world_model.models.reward import reward_loss
from world_model.models.value import value_loss
from world_model.training import ReplayBuffer, Transition


def collect_round(buffer: ReplayBuffer, env, agent, steps: int, seed: int) -> tuple[int, int]:
    """Roll `agent` for `steps`; returns (reward_events, episodes)."""
    rewards, episodes = 0, 0
    obs, _ = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()
    for _ in range(steps):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(Transition(obs, action, float(reward), next_obs, done))
        rewards += int(reward > 0)
        if done:
            episodes += 1
            obs, _ = env.reset()
            if hasattr(agent, "reset"):
                agent.reset()
        else:
            obs = next_obs
    return rewards, episodes


def train(
    buffer: ReplayBuffer,
    encoder: ConvEncoder,
    dynamics: RecurrentDynamics,
    reward_head: RewardHead,
    value_head: ValueHead,
    sigreg: SIGReg,
    opt: torch.optim.Optimizer,
    updates: int,
    seq_batch: int,
    window: int,
    burn_in: int,
    lam: float,
    device: str,
    success_frac: float = 0.0,
    ema: tuple[list, list, float] | None = None,  # (online_modules, ema_modules, decay)
    amp: bool = False,
) -> None:
    no_act = dynamics.no_action
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda")
    for step in range(updates):
        batch = buffer.sample_sequences(seq_batch, window, success_frac=success_frac)
        obs_seq = torch.as_tensor(batch["obs"], device=device)
        actions = torch.as_tensor(batch["action"], device=device)  # (B, W)
        rewards = torch.as_tensor(batch["reward"], device=device)
        returns = torch.as_tensor(batch["return"], device=device)
        final_obs = torch.as_tensor(batch["next_obs"], device=device)

        b, w = actions.shape
        with autocast:  # no-op unless --amp (bf16 forward; params stay fp32)
            z_seq = encoder(obs_seq.flatten(0, 1)).unflatten(0, (b, w))
            z_final = encoder(final_obs)
            targets = torch.cat([z_seq[:, 1:], z_final.unsqueeze(1)], dim=1)  # (B, W, D)

            # burn-in: build belief closed-loop from real latents
            s = dynamics.initial_state(b, device)
            prev_a = torch.full((b,), no_act, dtype=torch.long, device=device)
            for k in range(burn_in):
                s = dynamics.update(z_seq[:, k], prev_a, s)
                prev_a = actions[:, k]

            # closed-loop 1-step losses + open-loop imagination losses share the rollout
            pred_loss = torch.zeros((), device=device)
            r_loss = torch.zeros((), device=device)
            v_loss = torch.zeros((), device=device)
            s_open = s
            z_hat = None
            for k in range(burn_in, w):
                if z_hat is not None:  # open-loop: belief updated with imagined latent
                    s_open = dynamics.update(z_hat, actions[:, k - 1], s_open)
                r_loss = r_loss + reward_loss(reward_head(s_open, actions[:, k]), rewards[:, k])
                v_loss = v_loss + value_loss(value_head(s_open), returns[:, k])
                z_hat = dynamics.predict_next(s_open, actions[:, k])
                pred_loss = pred_loss + F.mse_loss(z_hat, targets[:, k])
            n = w - burn_in
            pred_loss, r_loss, v_loss = pred_loss / n, r_loss / n, v_loss / n

            if lam > 0:
                loss = (1 - lam) * pred_loss + lam * sigreg(z_seq.flatten(0, 1)) + r_loss + v_loss
            else:
                loss = pred_loss + r_loss + v_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], max_norm=10.0
        )
        opt.step()
        if ema is not None:
            online, shadow, decay = ema
            with torch.no_grad():
                for m_on, m_sh in zip(online, shadow, strict=True):
                    for p_on, p_sh in zip(m_on.parameters(), m_sh.parameters(), strict=True):
                        p_sh.mul_(decay).add_(p_on, alpha=1.0 - decay)
        if (step + 1) % 100 == 0:
            print(
                f"  update {step + 1}: pred_loss={pred_loss.item():.5f} "
                f"r_loss={r_loss.item():.5f} v_loss={v_loss.item():.5f} "
                f"latent_std={z_final.std(dim=0).mean().item():.4f}"
            )


def reinit(module: torch.nn.Module) -> None:
    """Re-initialize every parametrized submodule (Linear/GRUCell/Embedding/Conv)."""
    for sub in module.modules():
        if hasattr(sub, "reset_parameters"):
            sub.reset_parameters()


def shrink_perturb(module: torch.nn.Module, alpha: float = 0.8) -> None:
    """Soft reset toward fresh init: p <- (1-alpha)*p + alpha*p_fresh.

    Qiao et al. (arXiv:2310.15017) world-model resetting recipe for Dreamer-style
    transition predictors (their alpha=0.8). Unlike reinit, retains a fraction of
    learned structure for faster recovery.
    """
    fresh = copy.deepcopy(module)
    reinit(fresh)
    with torch.no_grad():
        for p, p_fresh in zip(module.parameters(), fresh.parameters(), strict=True):
            p.mul_(1.0 - alpha).add_(p_fresh, alpha=alpha)


def evaluate(
    encoder, dynamics, reward_head, value_head, env_id: str, episodes: int, device: str
) -> float:
    """Greedy (eps=0) success rate of the current model; fresh env, fixed eval seeds."""
    env = make_minigrid_env(env_id, fully_observable=False)
    agent = RecurrentMPCAgent(
        encoder,
        dynamics,
        reward_head,
        int(env.action_space.n),
        device,
        value_head=value_head,
        horizon=20,
        candidates=512,
        iters=2,
        seed=123,
    )
    successes = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        agent.reset()
        done, reward = False, 0.0
        while not done:
            obs, reward, term, trunc, _ = env.step(agent.act(obs))
            done = term or trunc
        successes += int(reward > 0)
    # leave modules in train mode for the next round (agent set them to eval)
    for m in (encoder, dynamics, reward_head, value_head):
        m.train()
    return successes / episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-5x5-v0")
    parser.add_argument("--steps-per-round", type=int, default=60000)
    parser.add_argument(
        "--round0-steps",
        type=int,
        default=None,
        help="random-collection budget for round 0 (default: steps-per-round); "
        "exp 0007 lesson: long random round 0 burns budget before the flywheel spins",
    )
    parser.add_argument(
        "--mpc-steps",
        type=int,
        default=None,
        help="steps per MPC collection round (default: max(10000, steps-per-round/3))",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--updates-per-round", type=int, default=3000)
    parser.add_argument(
        "--round0-updates",
        type=int,
        default=None,
        help="override updates for round 0 only (exp 0013 warm-utd arm: high UTD "
        "later but light round-0 training to dodge the primacy cost)",
    )
    parser.add_argument("--seq-batch", type=int, default=32)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--burn-in", type=int, default=8)
    parser.add_argument("--sigreg-weight", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.3, help="exploration in MPC rounds")
    parser.add_argument(
        "--success-frac",
        type=float,
        default=0.25,
        help="fraction of each training batch drawn from windows ending in a reward "
        "(exp 0009: success episodes must not drown in replay); 0 disables",
    )
    parser.add_argument(
        "--ignition-events",
        type=int,
        default=5,
        help="round 0 keeps collecting (up to 2x round0-steps) until this many "
        "reward events exist — the value head can't ignite from one example",
    )
    parser.add_argument(
        "--reset",
        choices=["none", "heads", "deep", "qiao", "surgical"],
        default="none",
        help="retention mechanics: heads/deep = Nikishin-style reinit (exp 0011, "
        "failed); qiao = shrink-perturb alpha=0.8 of the next-latent predictor only "
        "(arXiv:2310.15017 world-model reset); surgical = reinit reward+value heads "
        "only, dynamics untouched (exp 0012)",
    )
    parser.add_argument(
        "--freeze",
        choices=["none", "encoder", "trunk"],
        default="none",
        help="exp 0013: freeze representation after --freeze-round (encoder = conv "
        "encoder only; trunk = encoder + GRU cell + action embed; heads keep "
        "training). The opposite bet to resets: protect, don't forget.",
    )
    parser.add_argument(
        "--freeze-round",
        type=int,
        default=2,
        help="apply --freeze at the start of this round's training",
    )
    parser.add_argument(
        "--freeze2",
        choices=["none", "gru"],
        default="none",
        help="exp 0015 staged pension: second freeze event — pin GRU cell + action "
        "embed at --freeze2-round (encoder typically already frozen earlier)",
    )
    parser.add_argument(
        "--freeze2-round",
        type=int,
        default=4,
        help="apply --freeze2 at the start of this round's training",
    )
    parser.add_argument(
        "--reset-round",
        type=int,
        default=-1,
        help="apply the reset only at this round (-1 = every round >= 1)",
    )
    parser.add_argument(
        "--post-reset-updates",
        type=int,
        default=None,
        help="updates for rounds where a reset fired (default: updates-per-round); "
        "exp 0011 lesson: reset heads need a generous relearn budget",
    )
    parser.add_argument(
        "--later-lr-scale",
        type=float,
        default=1.0,
        help="multiply lr by this after round 0 (exp 0010 retention: gentler updates "
        "once competence exists); 1.0 disables",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="EMA decay for a shadow copy of all modules; acting/eval use the EMA "
        "(exp 0010 retention: the agent shouldn't act with freshly-perturbed "
        "weights); 0 disables",
    )
    parser.add_argument("--gamma", type=float, default=0.98, help="return discount")
    parser.add_argument("--eval-episodes", type=int, default=10, help="per-round eval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="bf16 autocast on the training forward (~1.2x; profile_run.py). OFF by "
        "default to keep fp32 results comparable — use for long actor/Crafter runs",
    )
    parser.add_argument("--save", type=str, required=True)
    args = parser.parse_args()

    # TF32 for any fp32 matmuls — free, harmless (our train step is launch-bound so
    # the gain is ~nil today, but good hygiene as models grow). See compute-strategy.
    torch.set_float32_matmul_precision("high")

    if args.reset != "none" and args.ema_decay > 0:
        # reviewer finding (2026-06-12): in-place reinit never reaches the EMA
        # shadow that acting/eval/checkpoints use — the reset would be a silent
        # no-op on behavior. Combine only after designing that interaction.
        raise SystemExit("--reset and --ema-decay are mutually exclusive (see exp 0011 review)")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_minigrid_env(args.env_id, fully_observable=False)
    num_actions = int(env.action_space.n)

    encoder = ConvEncoder().to(device)
    dynamics = RecurrentDynamics(num_actions=num_actions).to(device)
    reward_head = RewardHead(latent_dim=dynamics.state_dim, num_actions=num_actions).to(device)
    value_head = ValueHead(state_dim=dynamics.state_dim).to(device)
    sigreg = SIGReg().to(device)
    modules = [encoder, dynamics, reward_head, value_head]
    opt = torch.optim.Adam([p for m in modules for p in m.parameters()], lr=3e-4)

    # retention (exp 0010): the ACTING agent (collection/eval/checkpoints) can use an
    # EMA shadow of the weights so fresh gradient noise never drives behavior directly
    ema_modules = None
    if args.ema_decay > 0:
        ema_modules = [copy.deepcopy(m) for m in modules]
    act_enc, act_dyn, act_rew, act_val = ema_modules or modules

    # MPC rounds collect fewer steps (planning per env step is ~100x slower than
    # random) with a deliberately cheap planner — the data only needs goal bias.
    round0_steps = args.round0_steps or args.steps_per_round
    mpc_steps = args.mpc_steps or max(10_000, args.steps_per_round // 3)
    # round 0 may extend to 2x for ignition; reserve capacity for the worst case
    total_capacity = 2 * round0_steps + (args.rounds - 1) * mpc_steps
    buffer = ReplayBuffer(total_capacity, env.observation_space.shape, seed=args.seed)
    best: dict = {"rate": None, "round": -1, "state": {}}
    total_env_steps = 0

    for rnd in range(args.rounds):
        if rnd == 0:
            agent = RandomAgent(env.action_space, seed=args.seed)
            kind, steps = "random", round0_steps
        else:
            agent = RecurrentMPCAgent(
                act_enc,
                act_dyn,
                act_rew,
                num_actions,
                device,
                value_head=act_val,
                epsilon=args.epsilon,
                horizon=12,
                candidates=128,
                iters=1,
                seed=args.seed + rnd,
            )
            for m in modules:
                m.train()  # agent construction sets eval; collection precedes training
            kind, steps = f"mpc(eps={args.epsilon})", mpc_steps
        print(f"round {rnd}: collecting {steps} steps with {kind} agent ...")
        rew, eps = collect_round(buffer, env, agent, steps, seed=args.seed + rnd)
        total_env_steps += steps
        # ignition: a value head can't learn success from 0-1 examples (exp 0008)
        while rnd == 0 and rew < args.ignition_events and total_env_steps < 2 * round0_steps:
            chunk = min(5000, 2 * round0_steps - total_env_steps)
            print(f"  ignition: only {rew}/{args.ignition_events} reward events, +{chunk} steps")
            r2_, e2_ = collect_round(buffer, env, agent, chunk, seed=args.seed + 100)
            rew, eps = rew + r2_, eps + e2_
            total_env_steps += chunk
        print(f"  -> {eps} episodes, {rew} reward events (total_env_steps={total_env_steps})")
        buffer.compute_returns(args.gamma)
        did_reset = False
        if rnd >= 1 and args.reset != "none" and args.reset_round in (-1, rnd):
            # forget weights, keep replay; oversampling speeds the relearn. Acting
            # agents hold refs to these modules (EMA combo excluded by the startup
            # guard), so mutate in place; optimizer rebuilt (fresh Adam moments).
            if args.reset == "qiao":
                shrink_perturb(dynamics.next_latent, alpha=0.8)
            elif args.reset == "surgical":
                reinit(reward_head)
                reinit(value_head)
            else:
                targets_to_reset = [reward_head, value_head, dynamics.next_latent]
                if args.reset == "deep":
                    targets_to_reset += [dynamics.cell, dynamics.action_embed]
                for m in targets_to_reset:
                    reinit(m)
            # single param group assumed: group[0]'s lr already carries later-lr-scale
            opt = torch.optim.Adam(
                [p for m in modules for p in m.parameters()], lr=opt.param_groups[0]["lr"]
            )
            did_reset = True
            print(f"  reset ({args.reset}) applied; optimizer rebuilt")
        freeze_now = []
        if args.freeze != "none" and rnd == args.freeze_round:
            freeze_now += [encoder]
            if args.freeze == "trunk":
                freeze_now += [dynamics.cell, dynamics.action_embed]
        if args.freeze2 == "gru" and rnd == args.freeze2_round:
            freeze_now += [dynamics.cell, dynamics.action_embed]
        if freeze_now:
            for m in freeze_now:
                for p in m.parameters():
                    p.requires_grad_(False)
            # optimizer over remaining trainable params only (fresh moments for them)
            trainable = [p for m in modules for p in m.parameters() if p.requires_grad]
            opt = torch.optim.Adam(trainable, lr=opt.param_groups[0]["lr"])
            n_frozen = sum(p.numel() for m in freeze_now for p in m.parameters())
            print(f"  freeze applied at round {rnd}: {n_frozen} params newly frozen")
        if rnd == 0 and args.round0_updates is not None:
            round_updates = args.round0_updates
        elif did_reset and args.post_reset_updates:
            round_updates = args.post_reset_updates
        else:
            round_updates = args.updates_per_round
        print(f"round {rnd}: training {round_updates} updates ...")
        train(
            buffer,
            encoder,
            dynamics,
            reward_head,
            value_head,
            sigreg,
            opt,
            round_updates,
            args.seq_batch,
            args.window,
            args.burn_in,
            args.sigreg_weight,
            device,
            success_frac=args.success_frac,
            ema=(modules, ema_modules, args.ema_decay) if ema_modules else None,
            amp=args.amp,
        )
        if rnd == 0 and args.later_lr_scale != 1.0:
            for group in opt.param_groups:
                group["lr"] *= args.later_lr_scale
            print(f"  lr scaled to {opt.param_groups[0]['lr']:.1e} for later rounds")
        # lesson from exp 0005: evaluate and snapshot EVERY round — round N+1
        # training can degrade the model, and the best agent must not be lost
        rate = evaluate(
            act_enc,
            act_dyn,
            act_rew,
            act_val,
            args.env_id,
            args.eval_episodes,
            device,
        )
        # eval_success= form is scraped by scripts/sweep.py — keep it machine-readable
        print(f"round {rnd}: eval_success={rate:.2f} ({args.eval_episodes} episodes)")
        if best["rate"] is None or rate >= best["rate"]:
            best = {
                "rate": rate,
                "round": rnd,
                "state": {
                    "encoder": copy.deepcopy(act_enc.state_dict()),
                    "dynamics": copy.deepcopy(act_dyn.state_dict()),
                    "reward_head": copy.deepcopy(act_rew.state_dict()),
                    "value_head": copy.deepcopy(act_val.state_dict()),
                },
            }

    path = Path(args.save)
    path = path.with_name(f"{path.stem}_s{args.seed}{path.suffix}")  # multi-seed safe
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "recurrent": True,
            **best["state"],
            "config": {
                **vars(args),
                "num_actions": num_actions,
                "obs_shape": tuple(env.observation_space.shape),
            },
            "metrics": {"eval_success": best["rate"], "best_round": best["round"]},
        },
        path,
    )
    # scrapable summary for scripts/sweep.py
    print(
        f"best_eval={best['rate']:.2f} best_round={best['round']} total_env_steps={total_env_steps}"
    )
    print(f"best checkpoint saved to {path}")


if __name__ == "__main__":
    main()
