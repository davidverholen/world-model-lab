"""exp 0067/0068: paraphrase the crafter-rtfm manual on OUR side (the env is untouched).

Two uses share this:
- **exp0067 probe** (eval): reword a manual to test MEANING vs surface-token reading.
- **exp0068 augmentation** (train): randomly reword manuals during collection so the binding learns
  to read across surface variation — closing the ~20% natural-language penalty exp0067 measured.

Faithful, local rewording (no external LLM). The recipe SEMANTICS are preserved; only the surface
form changes. See knowledge/experiments/0067-rtfm-reading-generalization-probe.md.
"""

from __future__ import annotations

import re

# token → natural-language phrase (the "real tutorial" surrogate).
NL: dict[str, str] = {
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
    """Surface reword: change preamble + connectives, KEEP the backtick tokens (sentence-structure
    variation, which the env already varies a little)."""
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
    (backticks gone). This is what a REAL tutorial looks like."""
    m = reword_l1(m)
    return re.sub(r"`([^`]+)`", lambda mo: NL.get(mo.group(1), mo.group(1).replace("_", " ")), m)


# Held-out natural-language style (DIFFERENT phrasings + framing). Used ONLY at eval (exp0068), NOT
# in training augmentation — so "does aug generalise to UNSEEN natural language?" is a real held-out
# test, not memorisation of the train-time L2 surface forms.
NL_ALT: dict[str, str] = {
    "place_stone": "lay a stone tile",
    "place_table": "put down a workbench",
    "place_furnace": "construct a furnace",
    "place_plant": "sow a seedling",
    "make_wood_pickaxe": "fashion a wooden pick",
    "make_stone_pickaxe": "fashion a stone pick",
    "make_iron_pickaxe": "fashion an iron pick",
    "make_wood_sword": "shape a wooden blade",
    "make_stone_sword": "shape a stone blade",
    "make_iron_sword": "create an iron blade",
    "wood": "timber",
    "stone": "rock",
    "coal": "charcoal",
    "iron": "iron ore",
    "diamond": "a gem",
    "sapling": "a seedling",
}


def reword_l2_heldout(m: str) -> str:
    """Held-out natural language (eval only): a DIFFERENT framing + the ``NL_ALT`` phrasings, so it
    shares no surface form with the train-time L1/L2 augmentation."""
    m = m.replace(
        "The smith's manual. Hold the listed inputs, then perform the steps in order.",
        "Apprentice's checklist. Bring the materials, then do the steps in sequence.",
    )
    m = m.replace(
        "Forge-lore of this land. Each recipe is scrambled today — read it exactly.",
        "Workshop card. Today's order is jumbled, so match it precisely.",
    )
    m = re.sub(r"Recipe for offering:", "Goal:", m)
    m = re.sub(r"To prove the recipe for offering,", "Goal:", m)
    m = m.replace("then perform", "next").replace("and perform", "next")
    m = re.sub(r"hold (\d+)x", r"bring \1", m)
    sub = lambda mo: NL_ALT.get(mo.group(1), mo.group(1).replace("_", " "))  # noqa: E731
    return re.sub(r"`([^`]+)`", sub, m)


# the levels, addressable by name (probe iterates these; trainer samples among L0/L1/L2 only)
LEVELS: dict[str, callable] = {
    "L0": lambda m: m,
    "L1": reword_l1,
    "L2": reword_l2,
    "L2b-heldout": reword_l2_heldout,
}


def sample_paraphrase(manual: str, rng, p_l1: float = 0.34, p_l2: float = 0.33) -> str:
    """exp 0068 train-time augmentation: with prob ``p_l2`` return the full-NL rewording, with
    ``p_l1`` the surface rewording, otherwise the original. Mixing all three exposes the binding to
    surface variation so it reads meaning, not the exact tokens. ``rng`` is a numpy Generator."""
    u = float(rng.random())
    if u < p_l2:
        return reword_l2(manual)
    if u < p_l2 + p_l1:
        return reword_l1(manual)
    return manual
