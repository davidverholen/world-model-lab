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
from world_model.models import ConvEncoder, LatentDynamicsPredictor, RewardHead, SIGReg
from world_model.models.reward import reward_loss
from world_model.training import ReplayBuffer, Transition


def collect(buffer: ReplayBuffer, env, agent, steps: int) -> None:
    obs, _ = env.reset(seed=0)
    for _ in range(steps):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(Transition(obs, action, float(reward), next_obs, done))
        obs = env.reset()[0] if done else next_obs


def _r2(pred: torch.Tensor, y: torch.Tensor) -> float:
    ss_res = ((y - pred) ** 2).sum(dim=0)
    ss_tot = ((y - y.mean(dim=0)) ** 2).sum(dim=0) + 1e-8
    return float((1 - ss_res / ss_tot).mean())


@torch.no_grad()
def probe_agent_position(encoder, env, agent, steps: int, device: str, seed: int = 1) -> float:
    """Test R^2 of ridge regression latent -> agent (x, y), alpha cross-validated.

    Protocol (exp 0003): 60/20/20 train/val/test split, ridge alpha selected on val
    from a log grid, R^2 reported on test only. No standardization of latents — a
    collapsed encoder's microscopic residual variation would be rescaled to full
    amplitude and memorized in a small env (verified in exp 0002: R^2=1.0 for a fully
    collapsed encoder). Raw scale lets amplitude matter, as it does for any real
    downstream consumer.
    """
    observations, positions = [], []
    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        observations.append(obs)
        positions.append(np.array(env.unwrapped.agent_pos, dtype=np.float32))
        obs, _, terminated, truncated, _ = env.step(agent.act(obs))
        if terminated or truncated:
            obs, _ = env.reset()

    z = encoder(torch.as_tensor(np.stack(observations), device=device)).cpu()
    z = torch.cat([z, torch.ones(len(z), 1)], dim=1)  # bias column
    y = torch.as_tensor(np.stack(positions))

    n = len(z)
    i_train, i_val = int(0.6 * n), int(0.8 * n)
    eye = torch.eye(z.shape[1])
    best = (-torch.inf, None)
    for alpha in [1e-2, 1e-1, 1.0, 1e1, 1e2]:
        w = torch.linalg.solve(
            z[:i_train].T @ z[:i_train] + alpha * eye, z[:i_train].T @ y[:i_train]
        )
        val_r2 = _r2(z[i_train:i_val] @ w, y[i_train:i_val])
        if val_r2 > best[0]:
            best = (val_r2, w)
    return _r2(z[i_val:] @ best[1], y[i_val:])


@torch.no_grad()
def eval_dynamics(encoder, predictor, env, agent, steps: int, device: str, seed: int = 2) -> dict:
    """Held-out latent dynamics quality vs the copy baseline (predict z' = z).

    Returns MSEs normalized by latent variance. A predictor that hasn't learned
    action-conditioned dynamics cannot beat the copy baseline; ratio < 1 means it did.
    """
    obs_l, act_l, next_l = [], [], []
    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        action = agent.act(obs)
        next_obs, _, terminated, truncated, _ = env.step(action)
        obs_l.append(obs)
        act_l.append(action)
        next_l.append(next_obs)
        obs = env.reset()[0] if terminated or truncated else next_obs

    z = encoder(torch.as_tensor(np.stack(obs_l), device=device))
    z_next = encoder(torch.as_tensor(np.stack(next_l), device=device))
    z_pred = predictor(z, torch.as_tensor(np.array(act_l), device=device))
    var = z_next.var(dim=0).mean() + 1e-12
    model_mse = float(F.mse_loss(z_pred, z_next) / var)
    copy_mse = float(F.mse_loss(z, z_next) / var)
    return {"model": model_mse, "copy": copy_mse, "ratio": model_mse / (copy_mse + 1e-12)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="MiniGrid-Empty-5x5-v0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sigreg-weight", type=float, default=0.0, help="lambda; 0 disables")
    parser.add_argument("--sigreg-slices", type=int, default=512)
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=1,
        help="train predictor on H-step imagined rollouts (1 = single-step); "
        "multi-step is required for usable planning (exp 0004: compounding error)",
    )
    parser.add_argument("--probe-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None, help="path to save checkpoint (.pt)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_minigrid_env(args.env_id)
    agent = RandomAgent(env.action_space, seed=args.seed)
    buffer = ReplayBuffer(args.steps, env.observation_space.shape, seed=args.seed)

    print(f"collecting {args.steps} random steps in {args.env_id} ...")
    collect(buffer, env, agent, args.steps)

    num_actions = int(env.action_space.n)
    encoder = ConvEncoder().to(device)
    predictor = LatentDynamicsPredictor(num_actions=num_actions).to(device)
    reward_head = RewardHead(num_actions=num_actions).to(device)
    sigreg = SIGReg(num_slices=args.sigreg_slices).to(device)
    opt = torch.optim.Adam(
        [*encoder.parameters(), *predictor.parameters(), *reward_head.parameters()], lr=3e-4
    )
    lam = args.sigreg_weight

    horizon = args.rollout_length
    seq_batch = args.batch_size if horizon == 1 else max(16, args.batch_size // horizon * 2)
    print(
        f"training latent predictor for {args.updates} updates on {device} "
        f"(lambda={lam}, rollout H={horizon}, batch={seq_batch}) ..."
    )
    for step in range(args.updates):
        batch = buffer.sample_sequences(seq_batch, horizon)
        obs_seq = torch.as_tensor(batch["obs"], device=device)  # (B, H, C, h, w)
        actions = torch.as_tensor(batch["action"], device=device)  # (B, H)
        rewards = torch.as_tensor(batch["reward"], device=device)  # (B, H)
        final_obs = torch.as_tensor(batch["next_obs"], device=device)  # (B, C, h, w)

        b, h = actions.shape
        z_seq = encoder(obs_seq.flatten(0, 1)).unflatten(0, (b, h))  # (B, H, D)
        z_final = encoder(final_obs)  # (B, D)
        targets = torch.cat([z_seq[:, 1:], z_final.unsqueeze(1)], dim=1)  # (B, H, D)

        # open-loop rollout from the window's first latent; losses at every step
        pred_loss = torch.zeros((), device=device)
        r_loss = torch.zeros((), device=device)
        z_hat = z_seq[:, 0]
        for k in range(h):
            # reward on real AND imagined latents: the planner queries drifted latents
            r_loss = r_loss + reward_loss(reward_head(z_seq[:, k], actions[:, k]), rewards[:, k])
            r_loss = r_loss + reward_loss(reward_head(z_hat, actions[:, k]), rewards[:, k])
            z_hat = predictor(z_hat, actions[:, k])
            pred_loss = pred_loss + F.mse_loss(z_hat, targets[:, k])
        pred_loss, r_loss = pred_loss / h, r_loss / (2 * h)

        if lam > 0:
            all_z = torch.cat([z_seq.flatten(0, 1), z_final], dim=0)
            loss = (1 - lam) * pred_loss + lam * sigreg(all_z) + r_loss
        else:
            loss = pred_loss + r_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            print(
                f"  update {step + 1}: pred_loss={pred_loss.item():.5f} "
                f"latent_std={z_final.std(dim=0).mean().item():.4f}"
            )

    r2 = probe_agent_position(encoder, env, agent, args.probe_steps, device)
    dyn = eval_dynamics(encoder, predictor, env, agent, args.probe_steps, device)
    print(f"probe R^2 (latent -> agent xy, test split, CV alpha): {r2:.3f}")
    print(
        f"dynamics (held-out, variance-normalized): model={dyn['model']:.3f} "
        f"copy-baseline={dyn['copy']:.3f} ratio={dyn['ratio']:.3f} (<1 = learned dynamics)"
    )

    if args.save:
        from pathlib import Path

        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "reward_head": reward_head.state_dict(),
                "config": {
                    **vars(args),
                    "num_actions": num_actions,
                    "obs_shape": tuple(env.observation_space.shape),
                },
                "metrics": {"probe_r2": r2, **dyn},
            },
            path,
        )
        print(f"checkpoint saved to {path}")


if __name__ == "__main__":
    main()
