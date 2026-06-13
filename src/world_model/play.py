"""Watch an agent play MiniGrid — live window or GIF recording.

    uv run python -m world_model.play                          # random agent, live window
    uv run python -m world_model.play --checkpoint runs/wm.pt  # trained world-model agent
    uv run python -m world_model.play --record out.gif         # headless, saves a GIF

With --checkpoint, the agent is an MPC planner over the learned latent world model
(see world_model/agents/mpc.py): every step it imagines candidate futures in latent
space and executes the first action of the best one. Checkpoints come from
`python -m world_model.collect --save runs/wm.pt`.

Close the window or Ctrl-C to stop. Prints per-episode outcomes and a final
success-rate summary.
"""

import argparse
import time

import numpy as np
import torch

from world_model.agents import MPCAgent, RandomAgent, RecurrentMPCAgent
from world_model.agents.rssm_agent import RSSMActorAgent
from world_model.envs import make_minigrid_env


def _crafter_viewer(size: int = 384):
    """Open a live pygame window for Crafter (its env renders frames, not an auto window)."""
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption("Crafter — world-model agent")
    return pygame, screen, pygame.time.Clock(), size


def _crafter_show(viewer, frame_64: np.ndarray, fps: float) -> bool:
    """Blit an upscaled frame; return False if the window was closed."""
    pygame, screen, clock, size = viewer
    k = max(1, size // frame_64.shape[0])
    up = np.repeat(np.repeat(frame_64, k, axis=0), k, axis=1)
    screen.blit(pygame.surfarray.make_surface(up.swapaxes(0, 1)), (0, 0))
    pygame.display.flip()
    clock.tick(fps)
    return not any(e.type == pygame.QUIT for e in pygame.event.get())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--checkpoint", type=str, default=None, help="trained world model (.pt)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=20, help="MPC imagination horizon")
    parser.add_argument("--candidates", type=int, default=1024, help="MPC imagined futures/step")
    parser.add_argument("--iters", type=int, default=3, help="CEM refinement iterations")
    parser.add_argument("--record", type=str, default=None, help="save GIF here instead of window")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="random-action fraction (recurrent agent only); breaks planner oscillation loops",
    )
    args = parser.parse_args()

    render_mode = "rgb_array" if args.record else "human"
    crafter = False

    if args.checkpoint:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        crafter = ckpt.get("encoder") == "dino"  # rung-3 Crafter checkpoint (frozen DINO)
        if crafter:
            from world_model.envs import make_crafter_env

            env = make_crafter_env(length=500)
            agent = RSSMActorAgent.from_checkpoint(
                args.checkpoint, device, epsilon=args.epsilon, seed=args.seed
            )
            pool = ckpt["config"].get("pool", "cls")
            print(f"Crafter RSSM agent (pool={pool}) from {args.checkpoint}")
        else:
            rssm_agent = bool(ckpt.get("rssm_agent", False))
            recurrent = bool(ckpt.get("recurrent", False))
            partial = rssm_agent or recurrent  # both use the partial RGB view
            # tile_size must match training: it changes the agent's OBSERVATIONS (not the
            # GIF render). A 16-vs-8 mismatch here silently cost 60pp success in exp 0006.
            env = make_minigrid_env(
                args.env_id, render_mode=render_mode, fully_observable=not partial
            )
            expected = ckpt.get("config", {}).get("obs_shape")
            if expected is not None and tuple(env.observation_space.shape) != tuple(expected):
                raise SystemExit(
                    f"observation shape mismatch: env {env.observation_space.shape} vs "
                    f"checkpoint {tuple(expected)} — check env-id/tile_size against training"
                )
            if rssm_agent:
                agent = RSSMActorAgent.from_checkpoint(
                    args.checkpoint, device, epsilon=args.epsilon, seed=args.seed
                )
                print(f"world-model reactive RSSM agent from {args.checkpoint} (device={device})")
            else:
                cls = RecurrentMPCAgent if recurrent else MPCAgent
                extra = {"epsilon": args.epsilon} if recurrent else {}
                agent = cls.from_checkpoint(
                    args.checkpoint,
                    device,
                    horizon=args.horizon,
                    candidates=args.candidates,
                    iters=args.iters,
                    seed=args.seed,
                    **extra,
                )
                kind = "recurrent belief-state" if recurrent else "feedforward"
                print(f"world-model MPC agent ({kind}) from {args.checkpoint} (device={device})")
    else:
        env = make_minigrid_env(args.env_id, render_mode=render_mode)
        agent = RandomAgent(env.action_space, seed=args.seed)
        print("random agent (pass --checkpoint to use a trained world model)")

    frames: list[np.ndarray] = []
    successes = 0
    viewer = _crafter_viewer() if (crafter and not args.record) else None  # live Crafter window
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            if hasattr(agent, "reset"):
                agent.reset()
            total_reward, steps, done, info = 0.0, 0, False, {}
            while not done:
                if args.record:
                    frame = env.render()
                    if crafter:  # 64->256 nearest-neighbour so pixel-art is watchable
                        frame = np.repeat(np.repeat(frame, 4, axis=0), 4, axis=1)
                    frames.append(frame)
                elif viewer is not None:  # live Crafter window
                    if not _crafter_show(viewer, env.render(), args.fps):
                        raise KeyboardInterrupt
                else:
                    time.sleep(1.0 / args.fps)
                obs, reward, terminated, truncated, info = env.step(agent.act(obs))
                total_reward += float(reward)
                steps += 1
                done = terminated or truncated
            if crafter:
                ach = sum(1 for v in info.get("achievements", {}).values() if v > 0)
                print(
                    f"episode {episode + 1}: {ach} achievements, "
                    f"reward={total_reward:.2f}, {steps} steps"
                )
            else:
                success = total_reward > 0
                successes += int(success)
                outcome = "reached goal" if success else "timed out"
                print(
                    f"episode {episode + 1}: {outcome} in {steps} steps, reward={total_reward:.2f}"
                )
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        env.close()
        if viewer is not None:
            viewer[0].quit()

    if not crafter:
        print(f"success rate: {successes}/{args.episodes}")
    if args.record and frames:
        import imageio

        imageio.mimsave(args.record, frames, fps=args.fps, loop=0)
        print(f"saved {len(frames)} frames to {args.record}")


if __name__ == "__main__":
    main()
