"""exp 0067: reading-generalization probe — does the rung-4 agent read MEANING or surface tokens?

Rewords the displayed manual on OUR side (an env wrapper; crafter-rtfm untouched) at three levels
and reuses ``harness.swap_follow_rate``, which scores against the structured recipe in
``info["displayed_facts"]`` — independent of the manual TEXT, so rewording can't change the scored
gesture. If swap-following survives natural-language rewording, the agent reads meaning (on track
for real tutorials); if it collapses, it binds env tokens (a lookup table). See
knowledge/experiments/0067-rtfm-reading-generalization-probe.md.

    uv run python scripts/paraphrase_probe.py runs/exp0065b/s3_s3.pt --length 2 --n-eval-seeds 60
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import crafter_rtfm as C
import torch
from crafter_rtfm import ManualMode, harness

sys.path.insert(0, str(Path(__file__).parent))
from visualize_rtfm import build_agent  # noqa: E402

from world_model.train_rtfm import swap_follow_prefix  # noqa: E402

# token → natural-language phrase (the "real tutorial" surrogate). Hand-built, local (no LLM).
NL = {
    "place_stone": "place a stone block",
    "place_table": "set up a crafting table",
    "place_furnace": "build a furnace",
    "place_plant": "plant a sapling",
    "make_wood_pickaxe": "craft a wooden pickaxe",
    "make_stone_pickaxe": "craft a stone pickaxe",
    "make_iron_pickaxe": "craft an iron pickaxe",
    "make_wood_sword": "craft a wooden sword",
    "make_stone_sword": "craft a stone sword",
    "make_iron_sword": "forge an iron sword",
    "wood": "wood",
    "stone": "stone",
    "coal": "coal",
    "iron": "iron",
    "diamond": "a diamond",
    "sapling": "a sapling",
}


def reword_l1(m: str) -> str:
    """Surface reword: change preamble + connectives, KEEP the backtick tokens (tests robustness to
    sentence structure, which the env already varies a little)."""
    m = m.replace(
        "The smith's manual. Hold the listed inputs, then perform the steps in order.",
        "Field notes for the apprentice. First gather the items listed, "
        "then carry out each step in order.",
    )
    m = m.replace(
        "Forge-lore of this land. Each recipe is scrambled today — read it exactly.",
        "Apprentice notes. The order is shuffled today, so follow it exactly.",
    )
    m = re.sub(r"Recipe for offering:", "Your task:", m)
    m = re.sub(r"To prove the recipe for offering,", "Your task:", m)
    m = m.replace("then perform", "then carry out").replace("and perform", "then carry out")
    m = re.sub(r"hold (\d+)x", r"gather \1", m)
    return m


def reword_l2(m: str) -> str:
    """Full natural language: L1 framing + replace each backtick env-token with its NL phrase
    (backticks gone). This is what a REAL tutorial looks like — the north-star test."""
    m = reword_l1(m)
    return re.sub(r"`([^`]+)`", lambda mo: NL.get(mo.group(1), mo.group(1).replace("_", " ")), m)


class RewordEnv:
    """Thin proxy that rewrites ``obs['manual']`` before the agent reads it; everything else
    (action_names, info['displayed_facts'] for scoring, dynamics) delegates to the wrapped env."""

    def __init__(self, env, reword):
        self._env = env
        self._reword = reword

    def _fix(self, obs):
        if isinstance(obs, dict) and "manual" in obs:
            return {**obs, "manual": self._reword(obs["manual"])}
        return obs

    def reset(self, **kw):
        obs, info = self._env.reset(**kw)
        return self._fix(obs), info

    def step(self, a):
        obs, r, term, trunc, info = self._env.step(a)
        return self._fix(obs), r, term, trunc, info

    def __getattr__(self, k):
        return getattr(self._env, k)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=str)
    p.add_argument("--length", type=int, default=2)
    p.add_argument("--n-eval-seeds", type=int, default=60)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = build_agent(args.checkpoint, device)
    seeds = list(range(args.n_eval_seeds))
    levels = [
        ("L0 original", lambda m: m),
        ("L1 surface", reword_l1),
        ("L2 natural-lang", reword_l2),
    ]

    # show the actual paraphrases (seed 0) so the rewording quality is auditable
    base = C.make_recipe_env(ManualMode.SWAPPED, length=args.length, one_shot=True)
    o, _ = base.reset(seed=0)
    print("=== example manual (seed 0) at each level ===")
    for name, rw in levels:
        print(f"  [{name}] {rw(o['manual'])!r}")

    print(f"\n=== swap-following on {args.n_eval_seeds} SWAPPED seeds (len {args.length}) ===")
    print(f"{'level':<18}{'swap_follow':>12}{'step-1':>10}{'ratio_full':>12}{'ratio_s1':>10}")
    base_full = base_s1 = None
    for name, rw in levels:
        raw = C.make_recipe_env(ManualMode.SWAPPED, length=args.length, one_shot=True)
        env = RewordEnv(raw, rw)
        full = harness.swap_follow_rate(env, agent, seeds, args.length)
        prefix = swap_follow_prefix(env, agent, seeds, args.length)
        s1 = prefix[0] if prefix else 0.0
        if base_full is None:
            base_full, base_s1 = full or 1e-9, s1 or 1e-9
        print(f"{name:<18}{full:>12.3f}{s1:>10.3f}{full / base_full:>12.2f}{s1 / base_s1:>10.2f}")


if __name__ == "__main__":
    main()
