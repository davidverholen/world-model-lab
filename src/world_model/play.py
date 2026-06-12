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
from world_model.envs import make_minigrid_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--checkpoint", type=str, default=None, help="trained world model (.pt)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=20, help="MPC imagination horizon")
    parser.add_argument("--candidates", type=int, default=1024, help="MPC imagined futures/step")
    parser.add_argument("--record", type=str, default=None, help="save GIF here instead of window")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="random-action fraction (recurrent agent only); breaks planner oscillation loops",
    )
    args = parser.parse_args()

    render_mode = "rgb_array" if args.record else "human"

    if args.checkpoint:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        recurrent = bool(ckpt.get("recurrent", False))
        env = make_minigrid_env(
            args.env_id, render_mode=render_mode, tile_size=16, fully_observable=not recurrent
        )
        cls = RecurrentMPCAgent if recurrent else MPCAgent
        extra = {"epsilon": args.epsilon} if recurrent else {}
        agent = cls.from_checkpoint(
            args.checkpoint,
            device,
            horizon=args.horizon,
            candidates=args.candidates,
            seed=args.seed,
            **extra,
        )
        kind = "recurrent belief-state" if recurrent else "feedforward"
        print(f"world-model MPC agent ({kind}) from {args.checkpoint} (device={device})")
    else:
        env = make_minigrid_env(args.env_id, render_mode=render_mode, tile_size=16)
        agent = RandomAgent(env.action_space, seed=args.seed)
        print("random agent (pass --checkpoint to use a trained world model)")

    frames: list[np.ndarray] = []
    successes = 0
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            if hasattr(agent, "reset"):
                agent.reset()
            total_reward, steps, done = 0.0, 0, False
            while not done:
                if args.record:
                    frames.append(env.render())
                else:
                    time.sleep(1.0 / args.fps)
                obs, reward, terminated, truncated, _ = env.step(agent.act(obs))
                total_reward += float(reward)
                steps += 1
                done = terminated or truncated
            success = total_reward > 0
            successes += int(success)
            outcome = "reached goal" if success else "timed out"
            print(f"episode {episode + 1}: {outcome} in {steps} steps, reward={total_reward:.2f}")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        env.close()

    print(f"success rate: {successes}/{args.episodes}")
    if args.record and frames:
        import imageio

        imageio.mimsave(args.record, frames, fps=args.fps, loop=0)
        print(f"saved {len(frames)} frames to {args.record}")


if __name__ == "__main__":
    main()
