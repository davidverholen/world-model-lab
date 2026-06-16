"""Render a watchable GIF of a trained rung-4 crafter-rtfm reading agent playing one episode.

Research interpretability tooling: the goal is to SEE the agent read a manual and (try to) execute
the recipe gesture, with the displayed manual + per-step telemetry overlaid on each frame, then
embed the clip in a knowledge/experiments/ report page. SWAPPED mode is the interesting one — it
shows whether the agent follows the displayed-but-wrong manual (swap-following).

The agent is reconstructed exactly as world_model.train_rtfm.main() builds it (frozen DINO encoder +
frozen MiniLM text encoder + ConditionedRSSM + Actor/ConditionedActor), driven greedily (epsilon=0).

``compose_frame`` and ``write_gif`` are factored out as clean, importable helpers — a later
imagination-viewer reuses them.

Usage::

    uv run python scripts/visualize_rtfm.py runs/exp0051c05/s0_s0.pt \\
        --mode SWAPPED --length 2 --one-shot --seed 0 --max-steps 48 \\
        --out runs/exp0051c05/play_swapped.gif --fps 4 [--episodes 1]

See knowledge/design/rung4-manual-conditioned-agent.md.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import crafter_rtfm as C
import numpy as np
import torch
from crafter_rtfm import ManualMode
from PIL import Image, ImageDraw, ImageFont

from world_model.models import Actor, ConditionedActor, FrozenDinoEncoder
from world_model.models.manual_conditioning import ConditionedRSSM
from world_model.models.text_encoder import FrozenTextEncoder
from world_model.train_rtfm import RTFMAgent

# index → name; the 17 crafter-rtfm actions (must match the env action space order).
ACTION_NAMES: list[str] = [
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

GAME_PX = 320  # upscaled game-frame side (from 64×64, nearest-neighbour).
PANEL_W = 360  # side telemetry panel width.
PAD = 10
LINE_H = 16
BG = (18, 18, 22)
FG = (225, 225, 230)
ACCENT = (120, 220, 140)  # tutorial-achievement / score highlight.
DIM = (150, 150, 160)


def _load_font(size: int = 13) -> ImageFont.ImageFont:
    """A monospaced bitmap-ish font if available, else PIL's default."""
    for name in ("DejaVuSansMono.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


_FONT = _load_font()


def compose_frame(game_rgb: np.ndarray, panel_lines: list[str]) -> np.ndarray:
    """Upscaled game frame + a dark side panel of wrapped telemetry text → one RGB uint8 image.

    ``game_rgb`` is the raw (64,64,3) uint8 env frame; it is nearest-neighbour upscaled to keep the
    blocky pixels crisp. ``panel_lines`` are pre-formatted lines: a line starting with ``"!"`` is
    rendered in the accent colour (tutorial achievement / score), one starting with ``"~"`` in a
    dimmed colour (section labels); the leading marker char is stripped. Reusable by other viewers.
    """
    game = Image.fromarray(np.ascontiguousarray(game_rgb)).resize((GAME_PX, GAME_PX), Image.NEAREST)
    h = GAME_PX
    canvas = Image.new("RGB", (GAME_PX + PANEL_W, h), BG)
    canvas.paste(game, (0, 0))
    draw = ImageDraw.Draw(canvas)
    x = GAME_PX + PAD
    y = PAD
    for line in panel_lines:
        color = FG
        if line.startswith("!"):
            color, line = ACCENT, line[1:]
        elif line.startswith("~"):
            color, line = DIM, line[1:]
        draw.text((x, y), line, fill=color, font=_FONT)
        y += LINE_H
    return np.asarray(canvas, dtype=np.uint8)


def write_gif(frames: list[np.ndarray], path: str | Path, fps: float) -> None:
    """Write RGB uint8 frames to a looping GIF at ``fps`` frames/sec.

    Uses PIL directly (not imageio.mimsave, whose ``duration`` kwarg silently failed to write the
    per-frame delay — GIFs then defaulted to ~10 fps regardless of ``fps``). PIL ``duration`` is
    milliseconds per frame; sub-1 fps (slow, readable) is supported."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
    ms = max(20, int(round(1000.0 / fps)))  # per-frame delay; floor avoids a 0 ms "fastest" delay
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=ms, loop=0, disposal=2)


def _wrap(text: str, width: int = 46) -> list[str]:
    """Wrap a (possibly multi-line) manual string to a panel-friendly width."""
    out: list[str] = []
    for raw in text.splitlines() or [""]:
        out.extend(textwrap.wrap(raw, width=width) or [""])
    return out


def build_agent(checkpoint: str | Path, device: str) -> RTFMAgent:
    """Reconstruct the rung-4 agent from a checkpoint, as train_rtfm.main() does. Greedy (eps=0)."""
    ckpt = torch.load(checkpoint, map_location=device)
    if not ckpt.get("rtfm_agent"):
        raise ValueError(f"{checkpoint} is not an rtfm_agent checkpoint")
    cfg = ckpt["config"]
    enc = FrozenDinoEncoder(pool=cfg["pool"]).to(device)
    text_enc = FrozenTextEncoder(device=device)
    ed, td, n_act = cfg["embed_dim"], cfg["text_dim"], cfg["num_actions"]
    rssm = ConditionedRSSM(embed_dim=ed, num_actions=n_act, text_dim=td).to(device)
    rssm.load_state_dict(ckpt["rssm"])
    if cfg["actor_cond"]:
        actor: torch.nn.Module = ConditionedActor(
            belief_dim=rssm.state_dim, text_dim=td, num_actions=n_act
        ).to(device)
    else:
        actor = Actor(state_dim=rssm.state_dim, num_actions=n_act).to(device)
    actor.load_state_dict(ckpt["actor"])
    rssm.eval()
    actor.eval()
    return RTFMAgent(enc, text_enc, rssm, actor, n_act, device, epsilon=0.0)


def render_episode(
    agent: RTFMAgent,
    mode: ManualMode,
    *,
    length: int,
    one_shot: bool,
    seed: int,
    max_steps: int,
) -> list[np.ndarray]:
    """Roll one greedy episode, composing a telemetry-overlaid frame per step; returns frames."""
    env = C.make_recipe_env(mode, length=length, one_shot=one_shot)
    obs, info = env.reset(seed=seed)
    agent.reset(obs, info)
    manual_lines = _wrap(obs["manual"])
    cum_score = int(info.get("tutorial_score", 0))
    possible = int(info.get("tutorial_possible", 0))
    frames: list[np.ndarray] = []

    def panel(step: int, action: int | None, newly: list, score: int) -> list[str]:
        lines = [f"mode: {mode.name}    step: {step}/{max_steps}", ""]
        lines.append("~MANUAL")
        lines.extend(manual_lines)
        lines.append("")
        if action is None:
            lines.append("action: (reset)")
        else:
            lines.append(f"action: {action:2d}  {ACTION_NAMES[action]}")
        lines.append(f"{'!' if score else '~'}tutorial_score: {score}/{possible}")
        if newly:
            lines.append("!✓ TUTORIAL ACHIEVEMENT")
            for ach in newly:
                lines.append(f"!  + {ach}")
        return lines

    # Initial frame: the manual the agent just read, before any action.
    frames.append(compose_frame(env.render(), panel(0, None, [], cum_score)))

    for t in range(max_steps):
        action = agent.act(obs, info)
        obs, _r, term, trunc, info = env.step(action)
        newly = list(info.get("tutorial_newly", []))
        cum_score = int(info.get("tutorial_score", cum_score))
        frames.append(compose_frame(env.render(), panel(t + 1, action, newly, cum_score)))
        if term or trunc:
            break
    return frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=str, help="path to an rtfm_agent checkpoint (.pt)")
    p.add_argument(
        "--mode",
        type=str,
        default="CORRECT",
        choices=["CORRECT", "NONE", "SWAPPED"],
        help="manual mode shown to the agent",
    )
    p.add_argument("--length", type=int, default=2, help="recipe gesture length")
    p.add_argument(
        "--one-shot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="one-shot tutorial (no within-episode retry); matches training",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=48, help="episode step cap (training horizon)")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="number of seeds (seed, seed+1, ...) concatenated into one GIF",
    )
    p.add_argument(
        "--out", type=str, default=None, help="output GIF path (default: <stem>_play.gif)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    checkpoint = Path(args.checkpoint)
    out = Path(args.out) if args.out else checkpoint.with_name(f"{checkpoint.stem}_play.gif")
    mode = ManualMode[args.mode]

    agent = build_agent(checkpoint, device)

    frames: list[np.ndarray] = []
    for i in range(args.episodes):
        seed = args.seed + i
        ep = render_episode(
            agent,
            mode,
            length=args.length,
            one_shot=args.one_shot,
            seed=seed,
            max_steps=args.max_steps,
        )
        print(f"episode seed={seed}: {len(ep)} frames")
        frames.extend(ep)

    write_gif(frames, out, args.fps)
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({len(frames)} frames, {size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
