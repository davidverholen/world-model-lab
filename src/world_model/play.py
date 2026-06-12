"""Watch an agent play MiniGrid — live window or GIF recording.

    uv run python -m world_model.play                        # random agent, live window
    uv run python -m world_model.play --env-id MiniGrid-DoorKey-6x6-v0
    uv run python -m world_model.play --record out.gif       # headless, saves a GIF

Agents:
    random       baseline (default)
    [future]     trained world-model agents plug in here via --checkpoint
                 (exp 0004: planner / actor-critic over the learned latent model)

Close the window or Ctrl-C to stop. Episode outcomes are printed per episode.
"""

import argparse
import time

import numpy as np

from world_model.agents import RandomAgent
from world_model.envs import make_minigrid_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record", type=str, default=None, help="save GIF here instead of window")
    args = parser.parse_args()

    render_mode = "rgb_array" if args.record else "human"
    env = make_minigrid_env(args.env_id, render_mode=render_mode, tile_size=16)
    agent = RandomAgent(env.action_space, seed=args.seed)
    frames: list[np.ndarray] = []

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
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
            outcome = "reached goal" if total_reward > 0 else "timed out"
            print(f"episode {episode + 1}: {outcome} in {steps} steps, reward={total_reward:.2f}")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        env.close()

    if args.record and frames:
        import imageio

        imageio.mimsave(args.record, frames, fps=args.fps, loop=0)
        print(f"saved {len(frames)} frames to {args.record}")


if __name__ == "__main__":
    main()
