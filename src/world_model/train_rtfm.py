"""Rung-4 flywheel: a manual-conditioned world-model agent on crafter-rtfm (read-to-learn-dynamics).

Reuses the rung-3 Crafter recipe (frozen DINO + RSSM imagination AC, two-hot reward/critic) but the
RSSM is a ConditionedRSSM whose deterministic state is conditioned on the episode's manual via
cross-attention over frozen MiniLM token embeddings. The manual conditions the DYNAMICS (and thus
the belief the actor sees) — NOT a direct text→action policy — the anti-baking design. Success is
SWAP-FOLLOWING on held-out r1 (does the learner perform the *swapped* manual's recipe?), not reward.

See knowledge/design/rung4-manual-conditioned-agent.md. Requires the `rtfm` extra (crafter-rtfm).
"""

import argparse
import copy
from pathlib import Path

import crafter_rtfm as C
import numpy as np
import torch
import torch.nn.functional as F
from crafter_rtfm import ManualMode, harness, splits

from world_model.models import Actor, FrozenDinoEncoder, ValueHead
from world_model.models.continue_head import ContinueHead
from world_model.models.manual_conditioning import ConditionedRSSM
from world_model.models.text_encoder import FrozenTextEncoder
from world_model.models.twohot import TwoHotRewardHead, TwoHotValueHead
from world_model.train_rssm import kl_balanced
from world_model.training import ReplayBuffer, Transition


def img_to_chw(img: np.ndarray) -> np.ndarray:
    """crafter-rtfm obs['image'] is HWC uint8 → CHW float [0,1] for the frozen DINO encoder."""
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0


class ManualRegistry:
    """Intern manual strings ↔ integer ids (replay tags); resolve a batch of ids → token tensors."""

    def __init__(self, text_enc: FrozenTextEncoder):
        self.text_enc = text_enc
        self.manuals: list[str] = []
        self._id: dict[str, int] = {}

    def intern(self, manual: str) -> int:
        if manual not in self._id:
            self._id[manual] = len(self.manuals)
            self.manuals.append(manual)
        return self._id[manual]

    def tokens(self, ids):
        """ids (B,) ints → padded token embeddings (B,Lmax,text_dim) + mask (B,Lmax)."""
        return self.text_enc.encode([self.manuals[int(i)] for i in ids])


class RTFMAgent:
    """crafter-rtfm Policy (reset/act) wrapping the manual-conditioned world model. Used for both
    collection and eval. epsilon mixes random actions (collection)."""

    def __init__(self, enc, text_enc, rssm, actor, n_act, device, epsilon=0.0, seed=0):
        self.enc, self.text_enc, self.rssm, self.actor = enc, text_enc, rssm, actor
        self.n_act, self.device, self.epsilon = n_act, device, epsilon
        self.rng = np.random.default_rng(seed)
        self._tok = self._mask = self._state = self._prev_a = None

    @torch.no_grad()
    def reset(self, obs, info):
        self._tok, self._mask = self.text_enc.encode([obs["manual"]])
        self._state = self.rssm.initial(1, self.device)
        self._prev_a = torch.full((1,), self.rssm.no_action, device=self.device)

    @torch.no_grad()
    def act(self, obs, info) -> int:
        e = torch.as_tensor(img_to_chw(obs["image"]), device=self.device).unsqueeze(0)
        embed = self.enc(e)
        self._state, _, _ = self.rssm.obs_step(
            self._state, self._prev_a, embed, self._tok, self._mask
        )
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            a = int(self.rng.integers(self.n_act))
        else:
            a = int(self.actor(self.rssm.belief(self._state)).argmax(-1))
        self._prev_a = torch.tensor([a], device=self.device)
        return a


@torch.no_grad()
def collect_rtfm(
    buffer, registry, agent, n_episodes, length, seeds, mode, device, use_agent, max_steps,
    one_shot
):
    """Roll crafter-rtfm episodes (capped at max_steps = the task horizon); store DINO embeds +
    per-transition manual id (tag). one_shot kills within-episode search (HO-0006) → reading is the
    only path to reward (sparser, but the only honest grounding signal)."""
    rng = np.random.default_rng(0)
    events = 0

    def embed_of(obs):
        return agent.enc(torch.as_tensor(img_to_chw(obs["image"]), device=device).unsqueeze(0))

    for ep in range(n_episodes):
        env = C.make_recipe_env(mode, length=length, one_shot=one_shot)
        seed = int(seeds[ep % len(seeds)])
        obs, info = env.reset(seed=seed)
        mid = registry.intern(obs["manual"])
        if use_agent:
            agent.reset(obs, info)
        emb = embed_of(obs)
        for t in range(max_steps):
            a = agent.act(obs, info) if use_agent else int(rng.integers(env.action_space.n))
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc or (t == max_steps - 1)  # cap at the task horizon
            nemb = embed_of(obs)
            buffer.add(
                Transition(emb[0].cpu().numpy(), a, float(r), nemb[0].cpu().numpy(), done), tag=mid
            )
            events += int(r > 0)
            emb = nemb
            if done:
                break
    return events


def wm_train_rtfm(
    buffer,
    registry,
    rssm,
    recon_head,
    rew,
    cont,
    opt,
    updates,
    seq_batch,
    window,
    free_bits,
    device,
    success_frac=0.0,  # WM/reward head trains on the UNBIASED distribution (else it over-predicts
    # reward; success-oversampling is for the ACTOR's imagination start states only, in imagine_ac)
):
    recon = kl = torch.zeros(())
    w = window
    for _step in range(updates):
        batch = buffer.sample_sequences(seq_batch, window, success_frac=success_frac)
        embed = torch.as_tensor(batch["obs"], device=device)
        actions = torch.as_tensor(batch["action"], device=device)
        rewards = torch.as_tensor(batch["reward"], device=device)
        dones = torch.as_tensor(batch["done"], device=device, dtype=torch.float32)
        tok, mask = registry.tokens(batch["tag"])  # (B,L,td) manual tokens, same all window
        b, w = actions.shape
        state = rssm.initial(b, device)
        prev_a = torch.full((b,), rssm.no_action, dtype=torch.long, device=device)
        recon = kl = r_l = c_l = torch.zeros((), device=device)
        for k in range(w):
            state, prior, post = rssm.obs_step(state, prev_a, embed[:, k], tok, mask)
            belief = rssm.belief(state)
            kl = kl + kl_balanced(post, prior, free_bits)
            recon = recon + F.mse_loss(recon_head(belief), embed[:, k].detach())
            r_l = r_l + rew.twohot_loss(belief, actions[:, k], rewards[:, k])
            c_l = c_l + F.binary_cross_entropy_with_logits(cont(belief), 1.0 - dones[:, k])
            prev_a = actions[:, k]
        loss = (recon + kl + r_l + c_l) / w
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 100.0)
        opt.step()
    return float(recon.item() / w), float(kl.item() / w)


def imagine_ac_rtfm(
    buffer,
    registry,
    rssm,
    rew,
    cont,
    actor,
    critic,
    target_critic,
    opt,
    updates,
    seq_batch,
    window,
    burn_in,
    horizon,
    gamma,
    lam,
    ent_coef,
    device,
    beta_repval=0.3,
    success_frac=0.5,
):
    from torch.distributions import Categorical

    stats = (0.0, 0.0, 0.0)
    for _ in range(updates):
        batch = buffer.sample_sequences(seq_batch, window, success_frac=success_frac)
        embed = torch.as_tensor(batch["obs"], device=device)
        actions = torch.as_tensor(batch["action"], device=device)
        ret_real = torch.as_tensor(batch["return"], device=device)  # real MC returns
        tok, mask = registry.tokens(batch["tag"])
        b = actions.shape[0]
        with torch.no_grad():
            state = rssm.initial(b, device)
            prev_a = torch.full((b,), rssm.no_action, dtype=torch.long, device=device)
            real_bels = []  # beliefs over REAL burn-in states (critic-on-replay, exp 0021)
            for k in range(burn_in):
                state, _, _ = rssm.obs_step(state, prev_a, embed[:, k], tok, mask)
                real_bels.append(rssm.belief(state))
                prev_a = actions[:, k]
            beliefs, acts, rews, conts = [], [], [], []
            s = state
            for _t in range(horizon):
                bel = rssm.belief(s)
                a = Categorical(logits=actor(bel)).sample()
                beliefs.append(bel)
                acts.append(a)
                rews.append(rew(bel, a))
                conts.append(torch.sigmoid(cont(bel)))
                s, _ = rssm.img_step(s, a, tok, mask)
            bel_s = torch.stack(beliefs)
            a_s = torch.stack(acts)
            r_s = torch.stack(rews)
            c_s = torch.stack(conts)
            v_tgt = target_critic(bel_s.flatten(0, 1)).unflatten(0, (horizon, b))
            v_H = target_critic(rssm.belief(s))
            returns = torch.empty(horizon, b, device=device)
            g = v_H
            for t in reversed(range(horizon)):
                nextv = v_tgt[t + 1] if t + 1 < horizon else v_H
                g = r_s[t] + gamma * c_s[t] * ((1 - lam) * nextv + lam * g)
                returns[t] = g
        flat = bel_s.flatten(0, 1)
        adv = (returns - v_tgt).flatten().detach()
        dist = Categorical(logits=actor(flat))
        actor_loss = -(dist.log_prob(a_s.flatten()) * adv).mean() - ent_coef * dist.entropy().mean()
        critic_loss = critic.twohot_loss(flat, returns.flatten().detach())
        if beta_repval > 0:
            # critic-on-replay (exp 0021): anchor the critic to REAL returns so it can't drift with
            # the inflating imagined value (the rung-3 exploitation fix).
            rb = torch.stack(real_bels).flatten(0, 1)
            rt = ret_real[:, :burn_in].t().reshape(-1).detach()
            critic_loss = critic_loss + beta_repval * critic.twohot_loss(rb, rt)
        opt.zero_grad()
        (actor_loss + critic_loss).backward()
        opt.step()
        with torch.no_grad():
            for pt, pc in zip(target_critic.parameters(), critic.parameters(), strict=True):
                pt.mul_(0.98).add_(pc, alpha=0.02)
        stats = (actor_loss.item(), critic_loss.item(), returns.mean().item())
    return stats


def evaluate_rtfm(agent, eval_seeds, length, max_steps, one_shot):
    """Score in each of the four modes + swap-follow rate. Grounding = correct − none."""
    out = {}
    for mode in [ManualMode.CORRECT, ManualMode.NONE, ManualMode.SWAPPED]:
        env = C.make_recipe_env(mode, length=length, one_shot=one_shot)
        scores = [harness.run_episode(env, agent, seed=s, max_steps=max_steps) for s in eval_seeds]
        out[mode.name] = float(np.mean(scores))
    swap_env = C.make_recipe_env(ManualMode.SWAPPED, length=length, one_shot=one_shot)
    out["swap_follow"] = float(harness.swap_follow_rate(swap_env, agent, list(eval_seeds), length))
    out["grounding"] = out["CORRECT"] - out["NONE"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--episodes-per-round", type=int, default=60)
    p.add_argument("--updates-per-round", type=int, default=400)
    p.add_argument("--ac-updates-per-round", type=int, default=400)
    p.add_argument("--seq-batch", type=int, default=16)
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--burn-in", type=int, default=4)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--length", type=int, default=2)  # recipe gesture length
    p.add_argument("--max-steps", type=int, default=48)
    # one_shot (HO-0006): tutorial forfeit after the first gesture-length window → kills
    # within-episode search, so reading is the ONLY path to reward (the honest grounding test).
    p.add_argument("--one-shot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-train-seeds", type=int, default=400)
    p.add_argument("--n-eval-seeds", type=int, default=60)
    p.add_argument("--free-bits", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=3e-3)
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--pool", default="cls+patch")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", default="runs/rtfm.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    enc = FrozenDinoEncoder(pool=args.pool).to(device)
    text_enc = FrozenTextEncoder(device=device)
    ed, td = enc.latent_dim, text_enc.text_dim
    n_act = 17
    rssm = ConditionedRSSM(embed_dim=ed, num_actions=n_act, text_dim=td).to(device)
    sd = rssm.state_dim
    recon_head = torch.nn.Linear(sd, ed).to(device)
    rew = TwoHotRewardHead(state_dim=sd, num_actions=n_act).to(device)
    val = ValueHead(state_dim=sd).to(device)  # noqa: F841  (kept for parity / future repval)
    cont = ContinueHead(state_dim=sd).to(device)
    actor = Actor(state_dim=sd, num_actions=n_act).to(device)
    critic = TwoHotValueHead(state_dim=sd).to(device)
    target_critic = copy.deepcopy(critic)

    wm_mods = [rssm, recon_head, rew, cont]
    opt_wm = torch.optim.Adam([q for m in wm_mods for q in m.parameters()], lr=3e-4)
    opt_ac = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=3e-4)

    registry = ManualRegistry(text_enc)
    cap = args.rounds * args.episodes_per_round * args.max_steps + 1000
    buffer = ReplayBuffer(cap, (ed,), seed=args.seed, obs_dtype=np.float32)
    train_seeds = splits.split_seeds("r1", "train", args.n_train_seeds)
    eval_seeds = splits.split_seeds("r1", "eval", args.n_eval_seeds)
    agent = RTFMAgent(
        enc, text_enc, rssm, actor, n_act, device, epsilon=args.epsilon, seed=args.seed
    )

    for rnd in range(args.rounds):
        for m in wm_mods + [actor, critic]:
            m.train()
        use_agent = rnd > 0
        print(f"round {rnd}: collecting {args.episodes_per_round} episodes ...", flush=True)
        ev = collect_rtfm(
            buffer,
            registry,
            agent,
            args.episodes_per_round,
            args.length,
            train_seeds,
            ManualMode.CORRECT,
            device,
            use_agent,
            args.max_steps,
            args.one_shot,
        )
        buffer.compute_returns(args.gamma)
        print(
            f"  -> {ev} reward events; {len(registry.manuals)} manuals; buffer {buffer.size}"
        )
        recon, kl = wm_train_rtfm(
            buffer,
            registry,
            rssm,
            recon_head,
            rew,
            cont,
            opt_wm,
            args.updates_per_round,
            args.seq_batch,
            args.window,
            args.free_bits,
            device,
        )
        al, cl, ir = imagine_ac_rtfm(
            buffer,
            registry,
            rssm,
            rew,
            cont,
            actor,
            critic,
            target_critic,
            opt_ac,
            args.ac_updates_per_round,
            args.seq_batch,
            args.window,
            args.burn_in,
            args.horizon,
            args.gamma,
            args.lam,
            args.ent_coef,
            device,
        )
        for m in [rssm, actor]:
            m.eval()
        eval_agent = RTFMAgent(enc, text_enc, rssm, actor, n_act, device, epsilon=0.0)
        r = evaluate_rtfm(
            eval_agent, eval_seeds[: args.n_eval_seeds], args.length, args.max_steps, args.one_shot
        )
        print(
            f"  recon={recon:.3f} kl={kl:.3f} actor_loss={al:.3f} critic_loss={cl:.3f} "
            f"imagined_return={ir:.3f}",
            flush=True,
        )
        print(
            f"  correct={r['CORRECT']:.2f} none={r['NONE']:.2f} swapped={r['SWAPPED']:.2f} "
            f"grounding={r['grounding']:.2f} swap_follow={r['swap_follow']:.2f}",
            flush=True,
        )

    path = Path(args.save)
    path = path.with_name(f"{path.stem}_s{args.seed}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rtfm_agent": True,
            "rssm": rssm.state_dict(),
            "actor": actor.state_dict(),
            "config": {"pool": args.pool, "num_actions": n_act, "text_dim": td, "embed_dim": ed},
        },
        path,
    )
    print(f"saved -> {path}", flush=True)


if __name__ == "__main__":
    main()
