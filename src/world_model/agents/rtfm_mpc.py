"""CEM-MPC planner + agent over the rung-4 manual-conditioned world model (exp 0044).

The execution pillar of knowledge/design/hierarchical-imagination-agent.md (§5, §7). exp 0043
([[0043-rtfm-execution-wall]]) localised the rung-4 wall to *execution*: the WM provably READS a
length-2 gesture (inv_ratio>1) but the reactive imagination-actor cannot EXECUTE it (correct≈0) —
a sequential credit-assignment failure under a sparse one-shot reward. The hypothesis here: a
planner that SEARCHES action sequences inside imagination — rather than learning them by
policy-gradient through that sparse reward — cracks length-2 where the reactive actor flatlined.

Clean A/B: the SAME trained ``ConditionedRSSM`` + ``TwoHotRewardHead`` are scored two ways — the
reactive actor (System 1) vs this CEM-MPC search (System 2). Discrete actions; CEM over a
per-step categorical, refit to the elite frequencies (structurally the rung-2 latent CEM-MPC of
exp 0004, ``agents.mpc``, here threading the episode manual through every imagined step).
"""

import torch

from world_model.models.manual_conditioning import ConditionedRSSM
from world_model.models.twohot import TwoHotRewardHead


@torch.no_grad()
def _rollout_returns(
    rssm: ConditionedRSSM,
    rew: TwoHotRewardHead,
    state: tuple[torch.Tensor, torch.Tensor],
    tokens: torch.Tensor,
    mask: torch.Tensor | None,
    actions: torch.Tensor,
    gamma: float,
    n_rollout_samples: int = 1,
) -> torch.Tensor:
    """Discounted predicted return of each action sequence, rolled forward in imagination.

    ``state`` is a single belief state (batch 1) broadcast to ``actions.shape[0]`` candidates;
    ``tokens``/``mask`` (the episode manual) are likewise broadcast and threaded into EVERY
    imagined step so the conditioned prior depends on the manual at each rollout step — the same
    per-step conditioning the WM was trained under.

    ``n_rollout_samples`` (K) controls the rollout estimator (exp 0046). **K=1** (default) uses the
    prior **mean** — a single deterministic, optimistic point estimate per candidate (the
    exp-0044/0045 behaviour). **K>1** rolls each candidate forward K times sampling z ~ prior
    (``.rsample``) and scores by the **mean** return across the K samples: averaging regresses
    high-variance/unreliable OOD predictions toward the baseline, so the reward head's OOD *false
    positives* (exp 0045: ~12% of random sequences overrated above the true gesture) are penalised
    relative to the gesture the WM values *consistently*. See
    knowledge/experiments/0046-rtfm-robust-planning.md.

    Reward at step ``t`` is ``rew(belief_t, a_t)`` — the predicted reward of taking ``a_t`` from the
    current belief — matching the imagination AC's ``rew(bel, a)``-then-step ordering in
    ``train_rtfm.imagine_ac_rtfm``. No value tail: the horizon is meant to cover the short gesture.
    """
    n = actions.shape[0]
    k = max(1, n_rollout_samples)
    h, z = state
    h = h.expand(n * k, -1).contiguous()
    z = z.expand(n * k, -1).contiguous()
    tok = tokens.expand(n * k, *tokens.shape[1:]).contiguous()
    msk = mask.expand(n * k, *mask.shape[1:]).contiguous() if mask is not None else None
    # Each candidate occupies a contiguous block of k rows (repeat_interleave), so a later
    # ``.view(n, k)`` groups the k samples of one candidate together.
    act = actions.repeat_interleave(k, dim=0) if k > 1 else actions
    returns = torch.zeros(n * k, device=actions.device)
    for t in range(act.shape[1]):
        a_t = act[:, t]
        belief = torch.cat([h, z], dim=-1)
        returns = returns + (gamma**t) * rew(belief, a_t)
        # One imagination step: fold the manual into h (the conditioned deterministic state), then
        # advance the stochastic latent — prior MEAN for K=1, a prior SAMPLE per rollout for K>1.
        h = rssm._deter_cond(h, z, a_t, tok, msk)
        dist = rssm._dist(rssm.prior_net(h))
        z = dist.rsample() if k > 1 else dist.mean
    if k > 1:
        returns = returns.view(n, k).mean(dim=1)
    return returns


@torch.no_grad()
def plan_action(
    rssm: ConditionedRSSM,
    rew: TwoHotRewardHead,
    state: tuple[torch.Tensor, torch.Tensor],
    tokens: torch.Tensor,
    mask: torch.Tensor | None,
    horizon: int,
    n_samples: int,
    n_iters: int,
    n_elites: int,
    n_actions: int,
    device: str,
    gamma: float,
    rng: torch.Generator | None = None,
    n_rollout_samples: int = 1,
) -> int:
    """CEM-MPC: search action sequences in imagination, return the best first action.

    Maintains a per-step categorical action distribution (``horizon × n_actions``). Each CEM
    iteration: sample ``n_samples`` length-``horizon`` action sequences; roll each forward in the
    manual-conditioned WM accumulating discounted predicted reward (``_rollout_returns``); keep the
    top ``n_elites`` by return; refit the per-step categorical to the elite action frequencies
    (50/50 blend with the previous, clamped away from 0). Repeat ``n_iters`` times; return the
    argmax first action of the final refit distribution. All under ``torch.no_grad``.

    The execution-pillar planner for exp 0044 (knowledge/design/hierarchical-imagination-agent.md
    §5, §7) — it SEARCHES the gesture the reactive actor could not learn.
    """
    if rng is None:
        rng = torch.Generator(device="cpu").manual_seed(0)
    probs = torch.full((horizon, n_actions), 1.0 / n_actions)
    for _ in range(n_iters):
        # (horizon, n_actions) → sample n_samples per step → (n_samples, horizon) action sequences.
        actions = torch.multinomial(probs, n_samples, replacement=True, generator=rng).T.to(device)
        returns = _rollout_returns(
            rssm, rew, state, tokens, mask, actions, gamma, n_rollout_samples
        )
        elites = actions[returns.topk(min(n_elites, n_samples)).indices].cpu()
        counts = torch.zeros(horizon, n_actions)
        for t in range(horizon):
            counts[t] = torch.bincount(elites[:, t], minlength=n_actions).float()
        elite_probs = counts / counts.sum(dim=1, keepdim=True)
        probs = 0.5 * probs + 0.5 * elite_probs
        probs = probs.clamp_min(0.02)
        probs = probs / probs.sum(dim=1, keepdim=True)
    return int(probs[0].argmax())


class RTFMMPCAgent:
    """crafter-rtfm Policy (reset/act) that plans with CEM-MPC over the manual-conditioned WM.

    Drop-in for ``train_rtfm.RTFMAgent`` in ``harness.run_episode`` / ``evaluate_rtfm`` — same SAME
    trained ``rssm`` + ``rew``, so the only difference vs the reactive eval is HOW the action is
    chosen (search vs amortized policy). ``reset`` encodes the episode manual once and zeroes the
    belief; ``act`` does one closed-loop ``obs_step`` (real observation) to update the belief, then
    plans an open-loop action sequence in imagination and executes its first action.

    The execution-pillar agent of knowledge/design/hierarchical-imagination-agent.md §7 (exp 0044).
    """

    def __init__(
        self,
        enc,
        text_enc,
        rssm: ConditionedRSSM,
        rew: TwoHotRewardHead,
        n_act: int,
        device: str,
        horizon: int = 5,
        n_samples: int = 200,
        n_iters: int = 3,
        n_elites: int = 20,
        gamma: float = 0.99,
        seed: int = 0,
        n_rollout_samples: int = 1,
    ):
        self.enc, self.text_enc, self.rssm, self.rew = enc, text_enc, rssm, rew
        self.n_act, self.device = n_act, device
        self.horizon, self.n_samples, self.n_iters = horizon, n_samples, n_iters
        self.n_elites, self.gamma = n_elites, gamma
        self.n_rollout_samples = n_rollout_samples
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self._tok = self._mask = self._state = self._prev_a = None

    @torch.no_grad()
    def reset(self, obs, info) -> None:
        self._tok, self._mask = self.text_enc.encode([obs["manual"]])
        self._state = self.rssm.initial(1, self.device)
        self._prev_a = torch.full((1,), self.rssm.no_action, device=self.device)

    @torch.no_grad()
    def act(self, obs, info) -> int:
        from world_model.train_rtfm import img_to_chw

        e = torch.as_tensor(img_to_chw(obs["image"]), device=self.device).unsqueeze(0)
        embed = self.enc(e)
        self._state, _, _ = self.rssm.obs_step(
            self._state, self._prev_a, embed, self._tok, self._mask
        )
        a = plan_action(
            self.rssm,
            self.rew,
            self._state,
            self._tok,
            self._mask,
            self.horizon,
            self.n_samples,
            self.n_iters,
            self.n_elites,
            self.n_act,
            self.device,
            self.gamma,
            self.rng,
            self.n_rollout_samples,
        )
        self._prev_a = torch.tensor([a], device=self.device)
        return a
