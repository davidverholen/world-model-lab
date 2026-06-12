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

import torch
import torch.nn.functional as F

from world_model.agents import RandomAgent
from world_model.agents.mpc import RecurrentMPCAgent
from world_model.envs import make_minigrid_env
from world_model.models import ConvEncoder, RecurrentDynamics, RewardHead, SIGReg
from world_model.models.reward import reward_loss
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
    sigreg: SIGReg,
    opt: torch.optim.Optimizer,
    updates: int,
    seq_batch: int,
    window: int,
    burn_in: int,
    lam: float,
    device: str,
) -> None:
    no_act = dynamics.no_action
    for step in range(updates):
        batch = buffer.sample_sequences(seq_batch, window)
        obs_seq = torch.as_tensor(batch["obs"], device=device)
        actions = torch.as_tensor(batch["action"], device=device)  # (B, W)
        rewards = torch.as_tensor(batch["reward"], device=device)
        final_obs = torch.as_tensor(batch["next_obs"], device=device)

        b, w = actions.shape
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
        s_open = s
        z_hat = None
        for k in range(burn_in, w):
            if z_hat is not None:  # open-loop: belief updated with imagined latent
                s_open = dynamics.update(z_hat, actions[:, k - 1], s_open)
            r_loss = r_loss + reward_loss(reward_head(s_open, actions[:, k]), rewards[:, k])
            z_hat = dynamics.predict_next(s_open, actions[:, k])
            pred_loss = pred_loss + F.mse_loss(z_hat, targets[:, k])
        n = w - burn_in
        pred_loss, r_loss = pred_loss / n, r_loss / n

        if lam > 0:
            loss = (1 - lam) * pred_loss + lam * sigreg(z_seq.flatten(0, 1)) + r_loss
        else:
            loss = pred_loss + r_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], max_norm=10.0
        )
        opt.step()
        if (step + 1) % 100 == 0:
            print(
                f"  update {step + 1}: pred_loss={pred_loss.item():.5f} "
                f"latent_std={z_final.std(dim=0).mean().item():.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-5x5-v0")
    parser.add_argument("--steps-per-round", type=int, default=60000)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--updates-per-round", type=int, default=3000)
    parser.add_argument("--seq-batch", type=int, default=32)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--burn-in", type=int, default=8)
    parser.add_argument("--sigreg-weight", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.3, help="exploration in MPC rounds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_minigrid_env(args.env_id, fully_observable=False)
    num_actions = int(env.action_space.n)

    encoder = ConvEncoder().to(device)
    dynamics = RecurrentDynamics(num_actions=num_actions).to(device)
    reward_head = RewardHead(latent_dim=dynamics.state_dim, num_actions=num_actions).to(device)
    sigreg = SIGReg().to(device)
    opt = torch.optim.Adam(
        [*encoder.parameters(), *dynamics.parameters(), *reward_head.parameters()], lr=3e-4
    )

    # MPC rounds collect fewer steps (planning per env step is ~100x slower than
    # random) with a deliberately cheap planner — the data only needs goal bias.
    mpc_steps = max(10_000, args.steps_per_round // 3)
    total_capacity = args.steps_per_round + (args.rounds - 1) * mpc_steps
    buffer = ReplayBuffer(total_capacity, env.observation_space.shape, seed=args.seed)

    for rnd in range(args.rounds):
        if rnd == 0:
            agent = RandomAgent(env.action_space, seed=args.seed)
            kind, steps = "random", args.steps_per_round
        else:
            agent = RecurrentMPCAgent(
                encoder,
                dynamics,
                reward_head,
                num_actions,
                device,
                epsilon=args.epsilon,
                horizon=12,
                candidates=128,
                iters=1,
                seed=args.seed + rnd,
            )
            kind, steps = f"mpc(eps={args.epsilon})", mpc_steps
        print(f"round {rnd}: collecting {steps} steps with {kind} agent ...")
        rew, eps = collect_round(buffer, env, agent, steps, seed=args.seed + rnd)
        print(f"  -> {eps} episodes, {rew} reward events")
        print(f"round {rnd}: training {args.updates_per_round} updates ...")
        train(
            buffer,
            encoder,
            dynamics,
            reward_head,
            sigreg,
            opt,
            args.updates_per_round,
            args.seq_batch,
            args.window,
            args.burn_in,
            args.sigreg_weight,
            device,
        )

    from pathlib import Path

    path = Path(args.save)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "recurrent": True,
            "encoder": encoder.state_dict(),
            "dynamics": dynamics.state_dict(),
            "reward_head": reward_head.state_dict(),
            "config": {**vars(args), "num_actions": num_actions},
        },
        path,
    )
    print(f"checkpoint saved to {path}")


if __name__ == "__main__":
    main()
