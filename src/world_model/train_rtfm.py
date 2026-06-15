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

from world_model.agents.rtfm_mpc import RTFMMPCAgent
from world_model.models import Actor, ConditionedActor, FrozenDinoEncoder, ValueHead
from world_model.models.continue_head import ContinueHead
from world_model.models.manual_aux import (
    MaskedManualHead,
    manual_aux_loss,
    manual_invariance_diagnostic,
)
from world_model.models.manual_conditioning import ConditionedRSSM
from world_model.models.text_encoder import FrozenTextEncoder
from world_model.models.twohot import TwoHotRewardHead, TwoHotValueHead
from world_model.train_rssm import kl_balanced
from world_model.training import ReplayBuffer, Transition


def img_to_chw(img: np.ndarray) -> np.ndarray:
    """crafter-rtfm obs['image'] is HWC uint8 → CHW float [0,1] for the frozen DINO encoder."""
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0


def actor_logits(actor, belief, tok, mask):
    """Single dispatch point for every actor call site (collection, eval, imagination).

    Off path (base ``Actor``): ``actor(belief)`` — byte-for-byte identical to the original code; the
    manual tokens are ignored, so a flag-off run is unchanged. On path (``ConditionedActor``, the
    OPTIONAL rung-4 §6.2 fallback): ``actor(belief, tok, mask)`` — the actor cross-attends over the
    SAME frozen manual token embeddings the world model sees (no privileged channel; see
    ``models.actor.ConditionedActor``).
    """
    if isinstance(actor, ConditionedActor):
        return actor(belief, tok, mask)
    return actor(belief)


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
            bel = self.rssm.belief(self._state)
            a = int(actor_logits(self.actor, bel, self._tok, self._mask).argmax(-1))
        self._prev_a = torch.tensor([a], device=self.device)
        return a


@torch.no_grad()
def collect_rtfm(
    buffer,
    registry,
    agent,
    n_episodes,
    length,
    seeds,
    mode,
    device,
    use_agent,
    max_steps,
    one_shot,
    shaping_coef=0.0,
):
    """Roll crafter-rtfm episodes (capped at max_steps = the task horizon); store DINO embeds +
    per-transition manual id (tag). one_shot kills within-episode search (HO-0006) → reading is the
    only path to the achievement (the honest grounding signal). shaping_coef>0 adds the env's dense
    reading-gated shaping (HO-0007) as the ignition gradient — annealed → 0 by the caller."""
    rng = np.random.default_rng(0)
    events = 0

    def embed_of(obs):
        return agent.enc(torch.as_tensor(img_to_chw(obs["image"]), device=device).unsqueeze(0))

    for ep in range(n_episodes):
        env = C.make_recipe_env(
            mode, length=length, one_shot=one_shot, reading_shaping_coef=shaping_coef
        )
        seed = int(seeds[ep % len(seeds)])
        obs, info = env.reset(seed=seed)
        mid = registry.intern(obs["manual"])
        if use_agent:
            agent.reset(obs, info)
        emb = embed_of(obs)
        for t in range(max_steps):
            a = agent.act(obs, info) if use_agent else int(rng.integers(env.action_space.n))
            obs, r, term, trunc, info = env.step(a)
            # HONEST grounding reward (exp 0035): the env step reward `r` MIXES base-Crafter
            # achievements (wood/food/drink — farmable WITHOUT reading) with the sparse tutorial
            # bonus, both magnitude 1.0. Training on `r` lets the agent farm the readingless base
            # reward and never read the manual (the exp 0034 all-zeros failure). Train on ONLY the
            # tutorial component (newly-earned reading achievements this step) PLUS the env's
            # reading-gated shaping (HO-0007, info["reading_shaping"]) — both require reading the
            # displayed manual; the readingless base reward `r` is deliberately discarded. The
            # shaping is the dense ignition gradient and is annealed → 0, so the final agent stands
            # on the sparse honest reward alone (and eval always runs at shaping_coef=0).
            r_read = float(len(info["tutorial_newly"])) + float(info.get("reading_shaping", 0.0))
            done = term or trunc or (t == max_steps - 1)  # cap at the task horizon
            nemb = embed_of(obs)
            buffer.add(
                Transition(emb[0].cpu().numpy(), a, r_read, nemb[0].cpu().numpy(), done), tag=mid
            )
            # Count honest tutorial achievements only (NOT shaping steps, which are dense once
            # shaping is on) — this stays the ignition diagnostic: are we EARNING the one_shot
            # achievement, not merely collecting the reading-shaping gradient.
            events += len(info["tutorial_newly"])
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
    manual_aux_head=None,
    manual_aux_coef=0.0,
    manual_aux_mask_frac=0.4,
):
    """Train the WM (recon/kl/reward/continue). When manual_aux_coef>0, ADD the dense
    masked-manual-reconstruction reading gradient ([[dynalang-2023]] option A, see
    world_model.models.manual_aux): at each step, reconstruct masked manual token embeddings from
    the anti-baking belief (cross-attention blocked) so the belief must internalise the manual. The
    aux term fires at every window step a manual is present — the dense signal Dynalang relies on.
    coef=0.0 leaves all existing behaviour byte-for-byte unchanged (head not even invoked)."""
    recon = kl = torch.zeros(())
    aux_last = 0.0
    w = window
    use_aux = manual_aux_coef > 0.0 and manual_aux_head is not None
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
        recon = kl = r_l = c_l = aux = torch.zeros((), device=device)
        for k in range(w):
            if use_aux:
                # Aux belief uses the PRE-step state + prev_a → deter_noctx (anti-baking). Compute
                # before obs_step advances the state so the no-context carry is well-defined.
                aux_k, _ = manual_aux_loss(
                    rssm, manual_aux_head, state, prev_a, tok, mask, manual_aux_mask_frac
                )
                aux = aux + aux_k
            state, prior, post = rssm.obs_step(state, prev_a, embed[:, k], tok, mask)
            belief = rssm.belief(state)
            kl = kl + kl_balanced(post, prior, free_bits)
            recon = recon + F.mse_loss(recon_head(belief), embed[:, k].detach())
            r_l = r_l + rew.twohot_loss(belief, actions[:, k], rewards[:, k])
            c_l = c_l + F.binary_cross_entropy_with_logits(cont(belief), 1.0 - dones[:, k])
            prev_a = actions[:, k]
        loss = (recon + kl + r_l + c_l + manual_aux_coef * aux) / w
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 100.0)
        opt.step()
        aux_last = float(aux.item() / w)
    if use_aux:
        return float(recon.item() / w), float(kl.item() / w), aux_last
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
                # Actor conditioned on THIS episode's manual (tok/mask are (B,...), matching bel's
                # batch B exactly — same per-row manual the WM uses at this imagined step).
                a = Categorical(logits=actor_logits(actor, bel, tok, mask)).sample()
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
        # bel_s is (horizon, B, state_dim) → flatten(0,1) is row-major (horizon, B). Broadcast the
        # per-episode manual the SAME way: tile tok/mask (B,...) over the horizon axis FIRST, then
        # flatten(0,1), so flat_tok[h*B + i] lines up with flat[h*B + i]'s episode i at every step.
        flat_tok = tok.unsqueeze(0).expand(horizon, *tok.shape).flatten(0, 1)
        flat_mask = mask.unsqueeze(0).expand(horizon, *mask.shape).flatten(0, 1)
        dist = Categorical(logits=actor_logits(actor, flat, flat_tok, flat_mask))
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


@torch.no_grad()
def oracle_gesture_probe(
    rssm,
    rew,
    enc,
    text_enc,
    eval_seeds,
    length,
    one_shot,
    device,
    n_random=200,
    gamma=0.99,
    n_rollout_samples=1,
):
    """Localize the exp-0044 wall: does the reward head RANK the TRUE gesture above random ones?

    exp 0044 found MPC ~= reactive at length-2 (neither cracks it) -> the wall relocated from the
    policy to the world model's multi-step rollout/reward FIDELITY. This probe forks that: per eval
    episode build the initial belief, then compare the predicted discounted return of the ORACLE
    gesture (privileged ``info["manual_facts"]``, eval-only, never fed to the agent) against
    ``n_random`` random action sequences of the same length, both scored by the SAME planner
    objective (``agents.rtfm_mpc._rollout_returns``). Headline = the oracle's percentile among the
    random ones: ~1.0 = the WM/reward-head DISCRIMINATES the gesture, so the gap is MPC SEARCH
    (tunable); ~0.5 = it does not, so WM rollout/reward FIDELITY is the real wall. See
    knowledge/experiments/0044-rtfm-mpc-execution.md.
    """
    from world_model.agents.rtfm_mpc import _rollout_returns

    pcts, o_rets, r_rets = [], [], []
    for s in eval_seeds:
        env = C.make_recipe_env(ManualMode.CORRECT, length=length, one_shot=one_shot)
        obs, info = env.reset(seed=int(s))
        rituals = info["manual_facts"].get("rituals", {})
        if not rituals:
            continue
        gesture = next(iter(rituals.values()))["gesture"]
        names = env.action_names
        try:
            oracle = [names.index(g) for g in gesture]
        except ValueError:
            continue
        h = len(oracle)
        tok, mask = text_enc.encode([obs["manual"]])
        state = rssm.initial(1, device)
        prev_a = torch.full((1,), rssm.no_action, device=device)
        embed = enc(torch.as_tensor(img_to_chw(obs["image"]), device=device).unsqueeze(0))
        state, _, _ = rssm.obs_step(state, prev_a, embed, tok, mask)
        oa = torch.tensor(oracle, device=device).unsqueeze(0)  # (1, h)
        o_ret = float(
            _rollout_returns(rssm, rew, state, tok, mask, oa, gamma, n_rollout_samples)[0]
        )
        ra = torch.randint(0, len(names), (n_random, h), device=device)
        r_ret = _rollout_returns(rssm, rew, state, tok, mask, ra, gamma, n_rollout_samples)
        pcts.append(float((r_ret < o_ret).float().mean()))
        o_rets.append(o_ret)
        r_rets.append(float(r_ret.mean()))
    n = len(pcts)
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
    return {"pct": mean(pcts), "oracle": mean(o_rets), "random": mean(r_rets), "n": n}


@torch.no_grad()
def _manual_invariance_eval(buffer, registry, rssm, head, args, device) -> dict[str, float]:
    """Burn a belief in over a replay sequence (real manual + obs), then run the manual-invariance
    diagnostic (correct vs shuffled-manual reconstruction). See world_model.models.manual_aux."""
    batch = buffer.sample_sequences(args.seq_batch, args.window)
    embed = torch.as_tensor(batch["obs"], device=device)
    actions = torch.as_tensor(batch["action"], device=device)
    tok, mask = registry.tokens(batch["tag"])
    b = actions.shape[0]
    state = rssm.initial(b, device)
    prev_a = torch.full((b,), rssm.no_action, dtype=torch.long, device=device)
    burn = min(args.burn_in, args.window - 1)
    for k in range(burn):
        state, _, _ = rssm.obs_step(state, prev_a, embed[:, k], tok, mask)
        prev_a = actions[:, k]
    return manual_invariance_diagnostic(
        rssm, head, state, prev_a, tok, mask, args.manual_aux_mask_frac
    )


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
    p.add_argument("--length", type=int, default=2)  # recipe gesture length (the TARGET length)
    # Length curriculum (RTFM/Messenger lineage + Dreamer-4 "deep tree needs staging"): for the
    # first N rounds, train+eval at length 1 (where reading IGNITES, exp 0036) so the agent carries
    # the reading skill into the harder --length target. 0 = OFF (fixed --length throughout, the
    # exp-0041 cold-start that did NOT ignite at length-2).
    p.add_argument("--curriculum-rounds", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=48)
    # one_shot (HO-0006): tutorial forfeit after the first gesture-length window → kills
    # within-episode search, so reading is the ONLY path to reward (the honest grounding test).
    p.add_argument("--one-shot", action=argparse.BooleanOptionalAction, default=True)
    # HO-0007: initial reading-shaping coefficient (dense reading-gated ignition gradient);
    # linearly annealed to 0 over the run. 0 disables shaping (the exp-0035 honest-but-sparse run).
    p.add_argument("--reading-shaping-coef", type=float, default=1.0)
    # Dynalang option A: dense masked-manual-reconstruction reading gradient (world_model.models.
    # manual_aux). 0.0 = OFF (default; existing runs unaffected). >0 adds coef*aux to the WM loss.
    p.add_argument("--manual-aux-coef", type=float, default=0.0)
    p.add_argument("--manual-aux-mask-frac", type=float, default=0.4)
    # OPTIONAL actor-conditioning (rung-4 §6.2 fallback) — flag-gated OFF by default. When ON the
    # ACTOR also cross-attends over the frozen manual tokens (concat(belief, ctx) → base Actor).
    # This is the baking-RISKIER policy-conditioning the design avoids by default; the EXISTING swap
    # test (swap_follow + swapped≪correct on held-out manuals) stays the unchanged acceptance gate.
    # OFF = byte-for-byte identical to the WM-only path. See models.actor.ConditionedActor.
    p.add_argument("--actor-cond", action=argparse.BooleanOptionalAction, default=False)
    # exp 0044 execution test (knowledge/design/hierarchical-imagination-agent.md §5,§7): after the
    # reactive eval, ALSO score the SAME trained rssm+rew with a CEM-MPC planner (search action
    # sequences in imagination instead of an amortized reactive policy). OFF = byte-for-byte
    # unchanged (no second eval, no extra prints). See agents.rtfm_mpc.
    p.add_argument("--mpc-eval", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--mpc-horizon", type=int, default=5)
    p.add_argument("--mpc-samples", type=int, default=200)
    p.add_argument("--mpc-iters", type=int, default=3)
    # exp 0046: K sampled rollouts per candidate, scored by mean return (K=1 = prior-mean, the
    # exp-0044/0045 behaviour). >1 penalises reward-head OOD false positives via sample averaging.
    p.add_argument("--mpc-rollout-samples", type=int, default=1)
    # exp 0044 localizing diagnostic (default off): does the reward head rank the TRUE gesture above
    # random ones? Forks the exp-0044 wall — MPC-search (oracle pct≈1) vs WM-fidelity (pct≈0.5).
    p.add_argument("--oracle-probe", action=argparse.BooleanOptionalAction, default=False)
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
    # OFF (default): the WM-only base Actor — unchanged. ON (--actor-cond): the OPTIONAL §6.2
    # fallback ConditionedActor (belief cross-attends over the frozen manual tokens). Same frozen
    # token embeddings the WM sees; no privileged channel. Swap test remains the acceptance gate.
    if args.actor_cond:
        actor = ConditionedActor(belief_dim=sd, text_dim=td, num_actions=n_act).to(device)
    else:
        actor = Actor(state_dim=sd, num_actions=n_act).to(device)
    critic = TwoHotValueHead(state_dim=sd).to(device)
    target_critic = copy.deepcopy(critic)

    wm_mods = [rssm, recon_head, rew, cont]
    # Dynalang option A aux head (only trained / added to the optimizer when the loss is enabled, so
    # coef=0 runs are byte-for-byte unchanged). The head regresses masked manual token embeddings.
    manual_aux_head = None
    if args.manual_aux_coef > 0.0:
        manual_aux_head = MaskedManualHead(belief_dim=rssm.state_dim, text_dim=td).to(device)
        wm_mods = wm_mods + [manual_aux_head]
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
        # HO-0007 reading-shaping: dense reading-gated ignition gradient, linearly annealed to 0 so
        # the agent ends up standing on the sparse honest reward alone (eval always runs at coef=0).
        denom = max(1, args.rounds - 1)
        shaping_coef = args.reading_shaping_coef * max(0.0, 1.0 - rnd / denom)
        # Length curriculum: length-1 warm-up for the first --curriculum-rounds, then the --length
        # target. cur_len drives BOTH collection and eval, so the round line shows the regime.
        cur_len = 1 if rnd < args.curriculum_rounds else args.length
        print(
            f"round {rnd}: collecting {args.episodes_per_round} episodes "
            f"(len={cur_len} shaping_coef={shaping_coef:.3f}) ...",
            flush=True,
        )
        ev = collect_rtfm(
            buffer,
            registry,
            agent,
            args.episodes_per_round,
            cur_len,
            train_seeds,
            ManualMode.CORRECT,
            device,
            use_agent,
            args.max_steps,
            args.one_shot,
            shaping_coef,
        )
        buffer.compute_returns(args.gamma)
        print(f"  -> {ev} tutorial events; {len(registry.manuals)} manuals; buffer {buffer.size}")
        wm_out = wm_train_rtfm(
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
            manual_aux_head=manual_aux_head,
            manual_aux_coef=args.manual_aux_coef,
            manual_aux_mask_frac=args.manual_aux_mask_frac,
        )
        aux_l = wm_out[2] if len(wm_out) == 3 else 0.0
        recon, kl = wm_out[0], wm_out[1]
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
            eval_agent, eval_seeds[: args.n_eval_seeds], cur_len, args.max_steps, args.one_shot
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
        if args.mpc_eval and cur_len == args.length:
            # exp 0044: same rssm+rew, same eval seeds, MPC search instead of the reactive actor.
            # The A/B for the execution-pillar hypothesis (does planning crack length-2?). Gated to
            # the target-length phase (the question lives at length-2; skip the length-1 warm-up,
            # where MPC eval is expensive and uninformative).
            mpc_agent = RTFMMPCAgent(
                enc,
                text_enc,
                rssm,
                rew,
                n_act,
                device,
                horizon=args.mpc_horizon,
                n_samples=args.mpc_samples,
                n_iters=args.mpc_iters,
                gamma=args.gamma,
                seed=args.seed,
                n_rollout_samples=args.mpc_rollout_samples,
            )
            rm = evaluate_rtfm(
                mpc_agent, eval_seeds[: args.n_eval_seeds], cur_len, args.max_steps, args.one_shot
            )
            print(
                f"  MPC correct={rm['CORRECT']:.2f} none={rm['NONE']:.2f} "
                f"swapped={rm['SWAPPED']:.2f} grounding={rm['grounding']:.2f} "
                f"swap_follow={rm['swap_follow']:.2f}",
                flush=True,
            )
        if args.oracle_probe and cur_len == args.length:
            # exp 0044 fork: does the reward head rank the TRUE gesture above random ones?
            op = oracle_gesture_probe(
                rssm,
                rew,
                enc,
                text_enc,
                eval_seeds[: args.n_eval_seeds],
                cur_len,
                args.one_shot,
                device,
                n_rollout_samples=args.mpc_rollout_samples,
            )
            print(
                f"  ORACLE pct={op['pct']:.2f} oracle_ret={op['oracle']:.3f} "
                f"rand_ret={op['random']:.3f} n={op['n']} "
                f"(pct~1 = ranks true gesture = MPC-search; ~0.5 = WM-fidelity wall)",
                flush=True,
            )
        if manual_aux_head is not None:
            # Manual-invariance diagnostic ([[rung4-manual-conditioned-agent]] §4c): wrong/correct
            # reconstruction-loss ratio. ratio≈1 ⇒ TRIVIAL (belief ignores the manual); ratio≫1 ⇒
            # genuine reading. Computed on a fresh replay batch with the burned-in belief.
            inv = _manual_invariance_eval(buffer, registry, rssm, manual_aux_head, args, device)
            print(
                f"  manual_aux={aux_l:.4f} inv_correct={inv['correct']:.4f} "
                f"inv_wrong={inv['wrong']:.4f} inv_ratio={inv['ratio']:.2f} "
                f"(ratio>>1 = genuine reading; ~1 = trivial copy-through)",
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
            "config": {
                "pool": args.pool,
                "num_actions": n_act,
                "text_dim": td,
                "embed_dim": ed,
                "actor_cond": args.actor_cond,
            },
        },
        path,
    )
    print(f"saved -> {path}", flush=True)


if __name__ == "__main__":
    main()
