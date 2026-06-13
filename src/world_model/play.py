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

_INV_KEYS = [
    "health",
    "food",
    "drink",
    "energy",
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    "sapling",
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
]


def _crafter_viewer(world_px: int = 600):
    """Live pygame window: world a square of world_px, an HUD panel appended on the right.
    Font + panel scale with the window so they stay readable at large --window-size."""
    import pygame

    pygame.init()
    pygame.font.init()
    panel_px = max(360, int(world_px * 0.55))  # wide enough for the full title at big font
    font = pygame.font.SysFont("monospace", max(16, world_px // 42))
    big = pygame.font.SysFont("monospace", max(22, world_px // 28), bold=True)
    screen = pygame.display.set_mode((world_px + panel_px, world_px))
    pygame.display.set_caption("Crafter — world-model agent")
    return {
        "pygame": pygame, "screen": screen, "clock": pygame.time.Clock(),
        "world_px": world_px, "panel_px": panel_px, "font": font, "big": big,
    }


def _crafter_show(v, frame: np.ndarray, info: dict, events: list, title: str, fps: float) -> bool:
    """Render world + HUD; return False if the window was closed."""
    pg, screen, wpx = v["pygame"], v["screen"], v["world_px"]
    surf = pg.surfarray.make_surface(frame.swapaxes(0, 1))
    if surf.get_width() != wpx:
        surf = pg.transform.smoothscale(surf, (wpx, wpx))
    screen.blit(surf, (0, 0))
    screen.fill((18, 18, 22), (wpx, 0, v["panel_px"], wpx))
    pad = max(14, wpx // 60)
    y = [pad]

    def line(text, fnt=v["font"], color=(220, 220, 220)):
        if y[0] > wpx - fnt.get_height():  # don't draw past the bottom of the panel
            return
        screen.blit(fnt.render(text, True, color), (wpx + pad, y[0]))
        y[0] += fnt.get_height() + 4

    line(title, v["big"], (255, 255, 255))
    y[0] += 10
    line("INVENTORY", color=(140, 190, 255))
    inv = info.get("inventory", {})
    for k in _INV_KEYS:
        if inv.get(k, 0):
            line(f"  {k:13} {inv[k]}")
    y[0] += 14
    line("EVENTS", color=(150, 255, 180))
    for ev in events[-40:]:
        line("  " + ev, color=(190, 255, 190) if ev[0] == "+" else (255, 190, 190))
    pg.display.flip()
    v["clock"].tick(fps)
    return not any(e.type == pg.QUIT for e in pg.event.get())


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
    parser.add_argument("--window-size", type=int, default=600, help="Crafter window px")
    parser.add_argument(
        "--resolution",
        type=int,
        default=0,
        help="Crafter render px (0 = window-size); "
        "render >window gives crisper anti-aliased viewing, render <window is blocky-upscaled",
    )
    parser.add_argument(
        "--world-view",
        type=int,
        default=0,
        help="Crafter FOV in tiles (0 = agent's 9x9 view); "
        "e.g. 15 shows more world around the player (viewing only; agent obs unchanged)",
    )
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

            env = make_crafter_env(length=500, world_view=args.world_view)
            env.render_size = args.resolution or args.window_size  # decoupled render px
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
    viewer = _crafter_viewer(args.window_size) if (crafter and not args.record) else None
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            if hasattr(agent, "reset"):
                agent.reset()
            total_reward, steps, done = 0.0, 0, False
            unlocked: set[str] = set()
            events: list[str] = []
            while not done:
                if args.record:
                    frames.append(env.render())  # Crafter renders high-res directly
                elif viewer is not None:  # live Crafter window + HUD
                    title = (
                        f"ep {episode + 1}/{args.episodes}  r={total_reward:.1f}  "
                        f"ach={len(unlocked)}  t={steps}"
                    )
                    if not _crafter_show(viewer, env.render(), info, events, title, args.fps):
                        raise KeyboardInterrupt
                else:
                    time.sleep(1.0 / args.fps)
                obs, reward, terminated, truncated, info = env.step(agent.act(obs))
                total_reward += float(reward)
                steps += 1
                done = terminated or truncated
                if crafter:  # log achievement unlocks (+ death) as events
                    new = {k for k, v in info.get("achievements", {}).items() if v > 0} - unlocked
                    events += [f"+ {k} (t{steps})" for k in sorted(new)]
                    unlocked |= new
                    if terminated:
                        events.append(f"x died (t{steps})")
            if crafter:
                names = ", ".join(sorted(unlocked)) or "none"
                print(
                    f"episode {episode + 1}: {len(unlocked)} achievements "
                    f"[{names}], reward={total_reward:.2f}, {steps} steps"
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
            viewer["pygame"].quit()

    if not crafter:
        print(f"success rate: {successes}/{args.episodes}")
    if args.record and frames:
        import imageio

        imageio.mimsave(args.record, frames, fps=args.fps, loop=0)
        print(f"saved {len(frames)} frames to {args.record}")


if __name__ == "__main__":
    main()
