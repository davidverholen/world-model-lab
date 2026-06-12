"""Collect random trajectories in MiniGrid and train a latent predictor.

End-to-end pipeline: env -> random agent -> replay buffer -> encoder ->
latent dynamics predictor -> loss. Two modes:

- ``--sigreg-weight 0`` (default): plain latent regression. Collapses — the encoder
  maps everything to a constant (experiment 0001, knowledge/experiments/).
- ``--sigreg-weight 0.05``: adds SIGReg (LeJEPA, arXiv:2511.08544), which constrains
  embeddings toward N(0, I) and removes the collapse incentive (experiment 0002).

Diagnostics printed at the end:
- latent_std: mean per-dimension std of embeddings (collapse indicator, want ~1 with SIGReg)
- probe R^2: ridge regression from frozen latents to the agent's (x, y) position on
  held-out data — measures whether the latents actually encode world state.

Run: uv run python -m world_model.collect [--sigreg-weight 0.05]
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from world_model.agents import RandomAgent
from world_model.envs import make_minigrid_env
from world_model.models import ConvEncoder, LatentDynamicsPredictor, SIGReg
from world_model.training import ReplayBuffer, Transition


def collect(buffer: ReplayBuffer, env, agent, steps: int) -> None:
    obs, _ = env.reset(seed=0)
    for _ in range(steps):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(Transition(obs, action, float(reward), next_obs, done))
        obs = env.reset()[0] if done else next_obs


@torch.no_grad()
def probe_agent_position(encoder, env, agent, steps: int, device: str, seed: int = 1) -> float:
    """Held-out R^2 of ridge regression latent -> agent (x, y). Higher = latents know state."""
    observations, positions = [], []
    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        observations.append(obs)
        positions.append(np.array(env.unwrapped.agent_pos, dtype=np.float32))
        obs, _, terminated, truncated, _ = env.step(agent.act(obs))
        if terminated or truncated:
            obs, _ = env.reset()

    # No standardization: a collapsed encoder's microscopic residual variation would be
    # rescaled to full amplitude and memorized in a small env (verified: R^2=1.0 for a
    # fully collapsed encoder). Raw scale + unit ridge lets amplitude matter, as it
    # does for any downstream consumer of the latents.
    z = encoder(torch.as_tensor(np.stack(observations), device=device)).cpu()
    z = torch.cat([z, torch.ones(len(z), 1)], dim=1)  # bias column
    y = torch.as_tensor(np.stack(positions))

    split = int(0.8 * len(z))
    ridge = 1.0 * torch.eye(z.shape[1])
    w = torch.linalg.solve(z[:split].T @ z[:split] + ridge, z[:split].T @ y[:split])
    pred = z[split:] @ w
    ss_res = ((y[split:] - pred) ** 2).sum(dim=0)
    ss_tot = ((y[split:] - y[split:].mean(dim=0)) ** 2).sum(dim=0) + 1e-8
    return float((1 - ss_res / ss_tot).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="MiniGrid-Empty-5x5-v0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sigreg-weight", type=float, default=0.0, help="lambda; 0 disables")
    parser.add_argument("--sigreg-slices", type=int, default=512)
    parser.add_argument("--probe-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_minigrid_env(args.env_id)
    agent = RandomAgent(env.action_space, seed=args.seed)
    buffer = ReplayBuffer(args.steps, env.observation_space.shape, seed=args.seed)

    print(f"collecting {args.steps} random steps in {args.env_id} ...")
    collect(buffer, env, agent, args.steps)

    encoder = ConvEncoder().to(device)
    predictor = LatentDynamicsPredictor(num_actions=int(env.action_space.n)).to(device)
    sigreg = SIGReg(num_slices=args.sigreg_slices).to(device)
    opt = torch.optim.Adam([*encoder.parameters(), *predictor.parameters()], lr=3e-4)
    lam = args.sigreg_weight

    print(f"training latent predictor for {args.updates} updates on {device} (lambda={lam}) ...")
    for step in range(args.updates):
        batch = buffer.sample(args.batch_size)
        obs = torch.as_tensor(batch["obs"], device=device)
        action = torch.as_tensor(batch["action"], device=device)
        next_obs = torch.as_tensor(batch["next_obs"], device=device)

        z = encoder(obs)
        z_next = encoder(next_obs)
        pred_loss = F.mse_loss(predictor(z, action), z_next)
        if lam > 0:
            loss = (1 - lam) * pred_loss + lam * sigreg(torch.cat([z, z_next], dim=0))
        else:
            loss = pred_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            print(
                f"  update {step + 1}: pred_loss={pred_loss.item():.5f} "
                f"latent_std={z_next.std(dim=0).mean().item():.4f}"
            )

    r2 = probe_agent_position(encoder, env, agent, args.probe_steps, device)
    print(f"probe R^2 (latent -> agent xy, held-out): {r2:.3f}")
    print("interpretation: latent_std ~0 = collapse; probe R^2 ~1 = latents encode position")


if __name__ == "__main__":
    main()
