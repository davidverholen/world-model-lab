"""Profile where wall-time goes in a train_recurrent run — measure before optimizing.

Times the per-unit cost of each component, then projects the breakdown for a
standard UTD run (updates 6000 x 7 rounds; 90k MPC-collection + 20k random env
steps; 7 x 10 eval episodes). Run: uv run python scripts/profile_run.py
"""

import time

import torch

from world_model.agents.mpc import RecurrentMPCAgent
from world_model.envs import make_minigrid_env
from world_model.models import ConvEncoder, RecurrentDynamics, RewardHead, ValueHead
from world_model.models.reward import reward_loss
from world_model.models.value import value_loss


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_minigrid_env("MiniGrid-DoorKey-6x6-v0", fully_observable=False)
    n_act = int(env.action_space.n)
    enc = ConvEncoder().to(dev)
    dyn = RecurrentDynamics(num_actions=n_act).to(dev)
    rh = RewardHead(latent_dim=dyn.state_dim, num_actions=n_act).to(dev)
    vh = ValueHead(state_dim=dyn.state_dim).to(dev)
    mods = [enc, dyn, rh, vh]
    opt = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=3e-4)
    obs_shape = env.observation_space.shape

    def timeit(fn, n, warmup=3):
        for _ in range(warmup):
            fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    # --- training update (batch 32, window 24, burn-in 8) ---
    b, w, burn = 32, 24, 8
    obs_seq = torch.rand(b, w, *obs_shape, device=dev)
    actions = torch.randint(n_act, (b, w), device=dev)
    rewards = torch.rand(b, w, device=dev)
    returns = torch.rand(b, w, device=dev)
    final_obs = torch.rand(b, *obs_shape, device=dev)

    def train_step():
        z = enc(obs_seq.flatten(0, 1)).unflatten(0, (b, w))
        zf = enc(final_obs)
        tgt = torch.cat([z[:, 1:], zf.unsqueeze(1)], dim=1)
        s = dyn.initial_state(b, dev)
        pa = torch.full((b,), dyn.no_action, dtype=torch.long, device=dev)
        for k in range(burn):
            s = dyn.update(z[:, k], pa, s)
            pa = actions[:, k]
        pl = rl = vl = torch.zeros((), device=dev)
        so, zh = s, None
        for k in range(burn, w):
            if zh is not None:
                so = dyn.update(zh, actions[:, k - 1], so)
            rl = rl + reward_loss(rh(so, actions[:, k]), rewards[:, k])
            vl = vl + value_loss(vh(so), returns[:, k])
            zh = dyn.predict_next(so, actions[:, k])
            pl = pl + torch.nn.functional.mse_loss(zh, tgt[:, k])
        loss = (pl + rl + vl) / (w - burn)
        opt.zero_grad()
        loss.backward()
        opt.step()

    t_train = timeit(train_step, 50)

    # --- MPC act: collection config (128 cand, H12, iter1) and eval config (512, H20, iter2) ---
    obs0, _ = env.reset(seed=0)
    coll = RecurrentMPCAgent(enc, dyn, rh, n_act, dev, vh, horizon=12, candidates=128, iters=1)
    evala = RecurrentMPCAgent(enc, dyn, rh, n_act, dev, vh, horizon=20, candidates=512, iters=2)
    t_coll = timeit(lambda: coll.act(obs0), 100)
    t_eval_step = timeit(lambda: evala.act(obs0), 50)

    # --- raw env step ---
    def env_step():
        env.step(env.action_space.sample())

    t_env = timeit(env_step, 2000, warmup=50)

    # --- project a standard UTD run ---
    n_updates = 6000 * 7
    n_mpc = 15000 * 6
    n_rand = 20000
    n_eval_steps = 7 * 10 * 120  # ~120 avg steps/episode
    parts = {
        "training (42k updates)": t_train * n_updates,
        "MPC collection (90k steps)": (t_coll + t_env) * n_mpc,
        "random collection (20k steps)": t_env * n_rand,
        "eval (7x10 episodes, full planner)": t_eval_step * n_eval_steps,
    }
    total = sum(parts.values())
    print(f"\ndevice: {dev}")
    print(f"  train update     : {t_train * 1e3:7.2f} ms")
    print(f"  MPC act (collect): {t_coll * 1e3:7.2f} ms  + env {t_env * 1e3:.3f} ms")
    print(f"  MPC act (eval)   : {t_eval_step * 1e3:7.2f} ms")
    print(f"  env step (raw)   : {t_env * 1e3:7.3f} ms")
    print(f"\nprojected run breakdown (~{total / 60:.0f} min total):")
    for k, v in sorted(parts.items(), key=lambda x: -x[1]):
        print(f"  {100 * v / total:5.1f}%  {v / 60:6.1f} min  {k}")


if __name__ == "__main__":
    main()
