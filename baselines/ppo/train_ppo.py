"""Model-free PPO baseline on partially observable MiniGrid (exp 0007).

Mirrors the world-model agent's setting exactly: egocentric RGB observations
(7x7 tiles x 8 px), same env, same total env-step budget. The env wrapper below is
a deliberate copy of world_model/envs/minigrid.py (this project resolves different
gymnasium/minigrid versions than the main package — see pyproject.toml).

Run:  uv run --project baselines/ppo python baselines/ppo/train_ppo.py \
          --env-id MiniGrid-DoorKey-5x5-v0 --total-steps 100000 --seed 0
"""

import argparse

import gymnasium as gym
import numpy as np
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class ChwUint8Obs(gym.ObservationWrapper):
    """HWC uint8 -> CHW uint8 (sb3's NatureCNN auto-normalizes uint8 images)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, shape=(c, h, w), dtype=np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.transpose(2, 0, 1)


def make_env(env_id: str, tile_size: int = 8):
    def thunk():
        env = gym.make(env_id)
        env = RGBImgPartialObsWrapper(env, tile_size=tile_size)
        env = ImgObsWrapper(env)
        return ChwUint8Obs(env)

    return thunk


def evaluate(model: PPO, env_id: str, episodes: int, seed_base: int) -> float:
    env = make_env(env_id)()
    successes = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done, reward = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(int(action))
            done = term or trunc
        successes += int(reward > 0)
    return successes / episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-5x5-v0")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vec = DummyVecEnv([make_env(args.env_id) for _ in range(args.n_envs)])
    vec.seed(args.seed)
    model = PPO("CnnPolicy", vec, seed=args.seed, verbose=0)

    done_steps = 0
    while done_steps < args.total_steps:
        chunk = min(args.eval_every, args.total_steps - done_steps)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        done_steps += chunk
        rate = evaluate(model, args.env_id, episodes=10, seed_base=10_000)
        print(f"steps {done_steps}: eval_success={rate:.2f} (10 episodes)", flush=True)

    final = evaluate(model, args.env_id, episodes=20, seed_base=0)
    print(f"final: eval_success={final:.2f} (20 episodes, seeds 0+)")
    print(f"gymnasium/minigrid/sb3 versions: see `uv tree --project baselines/ppo`")


if __name__ == "__main__":
    main()
