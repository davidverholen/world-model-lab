"""Collect random trajectories in MiniGrid and train a first latent predictor.

This is the end-to-end smoke path for the whole stack:
env -> random agent -> replay buffer -> encoder -> latent dynamics predictor -> loss.

Run with:
    uv run python -m world_model.collect

Note: this trains encoder and predictor jointly on a plain latent-regression loss,
which is the textbook collapse-prone setup (the encoder can cheat by mapping everything
to a constant). That is intentional — observing the collapse is experiment #1.
See knowledge/concepts/jepa.md.
"""

import argparse

import torch
import torch.nn.functional as F

from world_model.agents import RandomAgent
from world_model.envs import make_minigrid_env
from world_model.models import ConvEncoder, LatentDynamicsPredictor
from world_model.training import ReplayBuffer, Transition


def collect(buffer: ReplayBuffer, env, agent, steps: int) -> None:
    obs, _ = env.reset(seed=0)
    for _ in range(steps):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(Transition(obs, action, float(reward), next_obs, done))
        obs = env.reset()[0] if done else next_obs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="MiniGrid-Empty-5x5-v0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_minigrid_env(args.env_id)
    agent = RandomAgent(env.action_space, seed=0)
    buffer = ReplayBuffer(args.steps, env.observation_space.shape, seed=0)

    print(f"collecting {args.steps} random steps in {args.env_id} ...")
    collect(buffer, env, agent, args.steps)

    encoder = ConvEncoder().to(device)
    predictor = LatentDynamicsPredictor(num_actions=int(env.action_space.n)).to(device)
    opt = torch.optim.Adam([*encoder.parameters(), *predictor.parameters()], lr=3e-4)

    print(f"training latent predictor for {args.updates} updates on {device} ...")
    for step in range(args.updates):
        batch = buffer.sample(args.batch_size)
        obs = torch.as_tensor(batch["obs"], device=device)
        action = torch.as_tensor(batch["action"], device=device)
        next_obs = torch.as_tensor(batch["next_obs"], device=device)

        z = encoder(obs)
        z_next_pred = predictor(z, action)
        z_next = encoder(next_obs)
        loss = F.mse_loss(z_next_pred, z_next)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            # latent std is the collapse indicator: -> 0 means the encoder cheated
            print(
                f"  update {step + 1}: loss={loss.item():.5f} "
                f"latent_std={z_next.std(dim=0).mean().item():.4f}"
            )

    print("done. If latent_std collapsed toward 0, you just reproduced why JEPA needs")
    print("a target-encoder / variance regularization — see knowledge/concepts/jepa.md")


if __name__ == "__main__":
    main()
