"""Behavioral QA gate — play a trained Crafter agent and flag DEGENERATE policies that
eval-reward alone misses (stationary agent, action-collapse). Operationalizes the exp 0023
diagnostic (Dave: "the agent never moves; it spams mk_iron_sword") so every future run is
auto-checked instead of needing a human to watch a GIF. Telemetry > vision here: Crafter's
egocentric render hides movement in pixels, but info['player_pos'] / the action stream make it
exact and cheap.

    uv run python scripts/behavior_report.py runs/crafter_xxx_s0.pt [episodes]

Exit 0 = PASS, 1 = FLAGGED (so it can gate a pipeline). See knowledge/experiments/0023.
"""

import sys
from collections import Counter

import numpy as np
import torch

from world_model.agents.rssm_agent import RSSMActorAgent
from world_model.envs import make_crafter_env

ACTIONS = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
]
MOVE = {1, 2, 3, 4}  # the four movement actions


def main() -> int:
    path = sys.argv[1]
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = RSSMActorAgent.from_checkpoint(path, device, epsilon=0.0, seed=0)

    acts: Counter = Counter()
    moved_steps = total_steps = 0
    span_sum = 0.0
    achievements: Counter = Counter()
    rewards, lengths = [], []
    for ep in range(episodes):
        env = make_crafter_env(length=500, seed=ep)
        obs, info = env.reset(seed=ep)
        agent.reset()
        prev_pos, done, ep_r, ep_len, poss = None, False, 0.0, 0, []
        while not done:
            a = agent.act(obs)
            acts[a] += 1
            obs, r, term, trunc, info = env.step(a)
            pos = tuple(info["player_pos"])
            poss.append(pos)
            if prev_pos is not None and pos != prev_pos:
                moved_steps += 1
            prev_pos = pos
            ep_r += r
            ep_len += 1
            total_steps += 1
            done = term or trunc
        p = np.array(poss)
        span_sum += float((p.max(0) - p.min(0)).sum())
        rewards.append(ep_r)
        lengths.append(ep_len)
        for k, v in info["achievements"].items():
            if v > 0:
                achievements[k] += 1

    n = sum(acts.values())
    probs = np.array([c / n for c in acts.values()])
    entropy = float(-(probs * np.log(probs + 1e-9)).sum())
    top_a, top_n = acts.most_common(1)[0]
    top_frac = top_n / n
    move_frac = sum(acts[a] for a in MOVE) / n
    moved_frac = moved_steps / max(1, total_steps)
    mean_span = span_sum / episodes

    print(f"=== behavior report: {path} ({episodes} eps) ===")
    print(
        f"mean reward={np.mean(rewards):.2f}  mean len={np.mean(lengths):.0f}  "
        f"achievements(unique seen)={len(achievements)}"
    )
    print(f"position: moved_frac={moved_frac:.2f}  mean bbox span={mean_span:.1f} tiles")
    print(
        f"actions:  entropy={entropy:.2f}/{np.log(len(ACTIONS)):.2f}  "
        f"move_actions={move_frac:.2f}  top={ACTIONS[top_a]}:{top_frac:.2f}"
    )
    print("  histogram: " + ", ".join(f"{ACTIONS[k]}:{v / n:.2f}" for k, v in acts.most_common(6)))

    # red-flag thresholds (a degenerate policy a human would catch by watching)
    flags = []
    if moved_frac < 0.05:
        flags.append(f"STATIONARY (moves on {moved_frac:.0%} of steps; bbox {mean_span:.1f} tiles)")
    if top_frac > 0.40:
        flags.append(f"ACTION-COLLAPSE (one action = {top_frac:.0%}: {ACTIONS[top_a]})")
    if entropy < 1.0:
        flags.append(f"LOW ACTION ENTROPY ({entropy:.2f})")
    if move_frac < 0.05:
        flags.append(f"BARELY USES MOVEMENT ({move_frac:.0%} of actions)")

    if flags:
        print("\nFLAGGED:")
        for f in flags:
            print(f"  ⚠ {f}")
        return 1
    print("\nPASS — no degenerate-policy red flags")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
