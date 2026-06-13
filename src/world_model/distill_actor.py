"""Distill a reactive Actor from a trained world model's CEM planner (exp 0016).

Teacher = RecurrentMPCAgent loaded from a checkpoint; collect (belief, action) pairs
by rolling it, train Actor by cross-entropy, eval the Actor alone vs the teacher.

    uv run python -m world_model.distill_actor --checkpoint runs/remote/dk6_utd_s0.pt
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from world_model.agents.actor_agent import ActorAgent
from world_model.agents.mpc import RecurrentMPCAgent
from world_model.envs import make_minigrid_env
from world_model.models.actor import Actor


@torch.no_grad()
def collect_labeled(teacher, env, steps: int, seed: int):
    """Roll the teacher; record (belief_state, planner_action) per step."""
    states, actions = [], []
    obs, _ = env.reset(seed=seed)
    teacher.reset()
    for _ in range(steps):
        a = teacher.act(obs)  # updates teacher._state to the post-obs belief
        states.append(teacher._state.squeeze(0).cpu().numpy())
        actions.append(a)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
            teacher.reset()
    return np.stack(states), np.array(actions, dtype=np.int64)


def success_rate(agent, env, episodes: int, seed_base: int) -> float:
    successes = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        agent.reset()
        done, reward = False, 0.0
        while not done:
            obs, reward, term, trunc, _ = env.step(agent.act(obs))
            done = term or trunc
        successes += int(reward > 0)
    return successes / episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-6x6-v0")
    parser.add_argument("--label-steps", type=int, default=8000)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_minigrid_env(args.env_id, fully_observable=False)
    n_act = int(env.action_space.n)

    teacher = RecurrentMPCAgent.from_checkpoint(
        args.checkpoint, device, horizon=20, candidates=512, iters=2, seed=args.seed
    )
    print(f"collecting {args.label_steps} planner-labeled states ...")
    states, actions = collect_labeled(teacher, env, args.label_steps, seed=args.seed)
    frac = np.bincount(actions, minlength=n_act) / len(actions)
    print(f"  teacher action distribution: {np.round(frac, 2)}")

    states_t = torch.as_tensor(states, device=device)
    actions_t = torch.as_tensor(actions, device=device)
    actor = Actor(state_dim=teacher.dynamics.state_dim, num_actions=n_act).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=3e-4)

    print(f"distilling actor for {args.updates} updates ...")
    rng = np.random.default_rng(args.seed)
    for step in range(args.updates):
        idx = rng.integers(0, len(states_t), size=args.batch_size)
        logits = actor(states_t[idx])
        loss = F.cross_entropy(logits, actions_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            acc = (logits.argmax(-1) == actions_t[idx]).float().mean().item()
            print(f"  update {step + 1}: ce={loss.item():.4f} train_acc={acc:.3f}")

    # rebuild plain encoder/dynamics refs for the actor agent (teacher holds eval copies)
    actor_agent = ActorAgent(teacher.encoder, teacher.dynamics, actor, n_act, device)
    actor_succ = success_rate(actor_agent, env, args.eval_episodes, seed_base=0)
    teacher_succ = success_rate(teacher, env, args.eval_episodes, seed_base=0)
    ratio = actor_succ / teacher_succ if teacher_succ > 0 else 0.0
    print(
        f"actor_success={actor_succ:.2f} teacher_success={teacher_succ:.2f} "
        f"ratio={ratio:.2f} (bar: ratio >= 0.70)"
    )


if __name__ == "__main__":
    main()
