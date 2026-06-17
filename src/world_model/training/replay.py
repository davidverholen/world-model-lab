"""Trajectory storage for world-model training.

A world model is trained on (obs, action, next_obs, reward, done) transitions —
for multi-step latent rollouts we will later need whole trajectory segments, so the
buffer stores episodes contiguously and can sample both single transitions and segments.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    # exp 0052: optional decoupled validated-reading reward channel (raw, unscaled). Kept SEPARATE
    # from `reward` (the task channel) so a dedicated VR head learns it without blurring the task
    # reward, and the imagination actor optimises task + λ·VR. Defaults to 0 → all non-VR callers
    # (and any run with --vr-head-coef 0) are byte-for-byte unchanged.
    vr: float = 0.0
    # exp 0066 (backward curriculum): in-order gesture-prefix progress AT this transition — the
    # number of leading displayed-gesture steps the agent has completed in order as of this step
    # (0 = none / pointer reset by a miss; k = first k steps done, the agent is at the frontier
    # between step k and step k+1). -1 = NOT TRACKED (no displayed gesture, or a non-rtfm caller).
    # The backward curriculum mines this to oversample imagination start states whose prefix is
    # already complete. Default -1 keeps every non-rtfm caller byte-for-byte unchanged.
    gesture_pos: int = -1


class ReplayBuffer:
    """Flat ring buffer of transitions with uniform sampling."""

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        seed: int | None = None,
        obs_dtype: np.dtype = np.uint8,
    ):
        # Pixel obs are stored as uint8 (originate from uint8 pixels scaled to [0,1], so
        # round-trip via round(x*255) is exact) — 4x less RAM than float32 (exp 0010 ops
        # lesson: 6 parallel float32 buffers paged the desktop to death). obs_dtype=float32
        # stores embeddings verbatim (no [0,1] assumption) — for the frozen-encoder cache
        # (train_crafter: store DINO embeddings, not pixels; skip the encoder in WM updates).
        self._u8 = np.dtype(obs_dtype) == np.uint8
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        # exp 0052: per-transition validated-reading reward (raw, unscaled). Parallel to `rewards`.
        self.vr = np.zeros(capacity, dtype=np.float32)
        # exp 0066: per-transition in-order gesture-prefix progress (see Transition.gesture_pos).
        # -1 = not tracked. Parallel to `rewards`; mined by the backward curriculum start sampler.
        self.gesture_pos = np.full(capacity, -1, dtype=np.int64)
        self.dones = np.zeros(capacity, dtype=bool)
        # optional per-transition integer tag (rung-4: the episode's manual id). Default 0.
        self.tags = np.zeros(capacity, dtype=np.int64)
        self.size = 0
        self.pos = 0
        self.total_adds = 0
        self.rng = np.random.default_rng(seed)
        # Curious-Replay state (exp 0028; arxiv:2306.15934). Priority is per window-START
        # index: p_i = c*beta^visits_i + (|loss_i|+eps)^alpha. New starts get p_max until
        # first scored. Unused unless the prioritized methods are called → the uniform path
        # (sample_sequences) is untouched.
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.visits = np.zeros(capacity, dtype=np.int64)
        self._valid_cache: np.ndarray | None = None

    def add(self, t: Transition, tag: int = 0) -> None:
        i = self.pos
        self.obs[i] = np.round(t.obs * 255.0) if self._u8 else t.obs
        self.next_obs[i] = np.round(t.next_obs * 255.0) if self._u8 else t.next_obs
        self.actions[i] = t.action
        self.rewards[i] = t.reward
        self.vr[i] = t.vr
        self.gesture_pos[i] = t.gesture_pos  # exp 0066: in-order gesture-prefix progress (-1=n/a)
        self.dones[i] = t.done
        self.tags[i] = tag
        self.visits[i] = 0
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_adds += 1

    def _to_float(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32) / 255.0 if self._u8 else obs.astype(np.float32, copy=False)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self._to_float(self.obs[idx]),
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_obs": self._to_float(self.next_obs[idx]),
            "done": self.dones[idx],
        }

    def compute_returns(self, gamma: float) -> None:
        """Fill `returns` with discounted return-to-go, reset at episode ends.

        Walks the (unwrapped) buffer backwards: G_i = r_i + gamma * G_{i+1}, restarting
        at dones. The trailing partial episode of a collection round is not
        done-terminated, so the next round's first-episode return leaks backward into
        it (mild overestimate, one partial episode per round boundary) — acceptable
        as a value target.
        """
        g = 0.0
        for i in range(self.size - 1, -1, -1):
            if self.dones[i]:
                g = 0.0
            g = float(self.rewards[i]) + gamma * g
            self.returns[i] = g

    def _valid_starts(self, cand: np.ndarray, length: int) -> np.ndarray:
        """Filter window starts: in range, no episode boundary before the final step."""
        cand = cand[(cand >= 0) & (cand <= self.size - length)]
        if len(cand) == 0:
            return cand
        return cand[~self.dones[cand[:, None] + np.arange(length - 1)].any(axis=1)]

    def success_starts(self, length: int) -> np.ndarray:
        """Valid window starts whose final transition carries a nonzero reward.

        Terminal rewards end episodes, so a reward transition can only sit at a
        window's last position — one canonical window per reward event.
        """
        reward_idx = np.flatnonzero(np.abs(self.rewards[: self.size]) > 1e-6)
        return self._valid_starts(reward_idx - length + 1, length)

    def prefix_done_starts(
        self, length: int, burn_in: int, min_pos: int = 1, max_pos: int | None = None
    ) -> np.ndarray:
        """exp 0066 (backward curriculum): valid window starts whose IMAGINATION-START belief has an
        in-order gesture prefix of at least ``min_pos`` (and, if ``max_pos`` given, strictly fewer
        than ``max_pos``) steps complete.

        ``imagine_ac_rtfm`` burns in ``burn_in`` real steps then imagines from the posterior ``s``.
        Index trace (reviewer-confirmed): the last burn-in iteration is
        ``obs_step(state, prev_a=a[start+burn_in-2], embed=obs[start+burn_in-1])`` and the *next*
        real action ``a[start+burn_in-1]`` is OVERRIDDEN by the actor in imagination — so ``s``
        embodies real progress only through ``a[start+burn_in-2]``. The gesture progress true at the
        imagination start is therefore ``gesture_pos[start + burn_in - 2]`` (NOT burn_in-1: that
        would tag by the first imagined action's discarded real counterpart, selecting states one
        step BEFORE the frontier — the exp0066 bug the reviewer caught).

        ``max_pos`` (exclusive, = the gesture length) excludes already-COMPLETED chains
        (``gesture_pos == length``): for length-2, ``min_pos=1, max_pos=2`` selects exactly the
        strict frontier (step-1 done, step-2 still to do), per §3a's one-level curriculum. Only REAL
        buffered latents are returned (ROMI: never seed from imagined states). Possibly empty — the
        caller falls back to the normal distribution and logs the supply shortfall."""
        off = max(0, burn_in - 2)  # window position whose gesture_pos the start belief embodies
        all_cand = np.arange(0, self.size - length + 1)
        valid = self._valid_starts(all_cand, length)
        if len(valid) == 0:
            return valid
        gp = self.gesture_pos[valid + off]
        sel = gp >= min_pos
        if max_pos is not None:
            sel &= gp < max_pos
        return valid[sel]

    def sample_sequences(
        self, batch_size: int, length: int, success_frac: float = 0.0
    ) -> dict[str, np.ndarray]:
        """Sample time-contiguous windows for multi-step rollout training.

        Windows never cross an episode boundary (no `done` inside, except possibly at
        the final transition). Assumes the buffer hasn't wrapped (our usage: capacity
        == collected steps); rejection-samples valid start indices.

        success_frac > 0 oversamples windows that END in a reward transition (exp
        0009 ignition fix: success episodes must not drown in the replay) — that
        fraction of the batch is drawn from `success_starts`, the rest uniformly.

        Returns obs (B, L, ...), action/reward/return (B, L), next_obs (B, ...).
        """
        if self.total_adds > self.capacity:
            raise NotImplementedError("sequence sampling assumes an unwrapped buffer")
        starts = np.empty(batch_size, dtype=np.int64)
        found = 0
        if success_frac > 0:
            pool = self.success_starts(length)
            if len(pool) > 0:
                want = int(batch_size * success_frac)
                starts[:want] = pool[self.rng.integers(0, len(pool), size=want)]
                found = want
        while found < batch_size:
            cand = self.rng.integers(0, self.size - length, size=2 * batch_size)
            valid = self._valid_starts(cand, length)
            take = min(len(valid), batch_size - found)
            starts[found : found + take] = valid[:take]
            found += take
        return self._window_batch(starts, length)

    def sample_sequences_prefix_curriculum(
        self,
        batch_size: int,
        length: int,
        burn_in: int,
        bc_frac: float,
        min_pos: int = 1,
        max_pos: int | None = None,
    ) -> tuple[dict[str, np.ndarray], float]:
        """exp 0066 (backward curriculum): like ``sample_sequences`` but OVERSAMPLE windows whose
        imagination-start belief has a complete gesture prefix (``prefix_done_starts`` carries
        the reviewer-confirmed ``burn_in-2`` offset and the optional ``max_pos`` frontier bound).
        A ``bc_frac`` fraction of the batch is drawn (with replacement) from the prefix-done
        pool, the rest uniformly from all valid starts. Returns (batch, realized_prefix_frac) where
        the second value is the ACTUAL fraction of batch starts that are prefix-done — the caller
        logs it as the buffer-supply telemetry.

        Buffer-supply fallback (the §3a 'buffer supply' risk): if the prefix-done pool is empty the
        whole batch falls back to the uniform distribution, so the AC update never stalls for lack
        of frontier seeds. Uses ONLY real buffered latents (start indices are real buffer rows)."""
        if self.total_adds > self.capacity:
            raise NotImplementedError("sequence sampling assumes an unwrapped buffer")
        bc_frac = float(np.clip(bc_frac, 0.0, 1.0))  # guard: bc_frac>1 would overflow starts[:want]
        off = max(0, burn_in - 2)
        pool = self.prefix_done_starts(length, burn_in, min_pos, max_pos)
        starts = np.empty(batch_size, dtype=np.int64)
        found = 0
        if len(pool) > 0 and bc_frac > 0:
            want = int(round(batch_size * bc_frac))
            starts[:want] = pool[self.rng.integers(0, len(pool), size=want)]
            found = want
        while found < batch_size:
            cand = self.rng.integers(0, self.size - length, size=2 * batch_size)
            valid = self._valid_starts(cand, length)
            take = min(len(valid), batch_size - found)
            starts[found : found + take] = valid[:take]
            found += take
        gp = self.gesture_pos[starts + off]
        sel = gp >= min_pos
        if max_pos is not None:
            sel &= gp < max_pos
        realized = float(np.mean(sel))
        return self._window_batch(starts, length), realized

    def _window_batch(self, starts: np.ndarray, length: int) -> dict[str, np.ndarray]:
        idx = starts[:, None] + np.arange(length)
        return {
            "obs": self._to_float(self.obs[idx]),
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "vr": self.vr[idx],  # exp 0052: decoupled validated-reading channel (raw)
            "return": self.returns[idx],
            "done": self.dones[idx],  # terminations (windows may end in one)
            "next_obs": self._to_float(self.next_obs[starts + length - 1]),
            "tag": self.tags[starts],  # per-window tag (rung-4 manual id; constant within episode)
        }

    # --- Curious Replay (exp 0028; arxiv:2306.15934) -----------------------------------
    def prepare_prioritized(self, length: int, p_max: float = 1e5) -> int:
        """Cache the valid window starts for this length and seed never-scored starts to
        p_max (highest priority → sampled first, à la PER new-transition init). Call once
        per WM-train round; the buffer is fixed during training so the valid set is stable.
        Returns the number of valid starts.
        """
        if self.total_adds > self.capacity:
            raise NotImplementedError("prioritized sampling assumes an unwrapped buffer")
        all_cand = np.arange(0, self.size - length + 1)
        self._valid_cache = self._valid_starts(all_cand, length)
        fresh = self._valid_cache[self.visits[self._valid_cache] == 0]
        self.priorities[fresh] = p_max
        return len(self._valid_cache)

    def sample_sequences_prioritized(
        self, batch_size: int, length: int
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Sample windows with probability ∝ start priority (Curious Replay). Returns the
        batch dict AND the chosen start indices (the caller scores them via
        update_priorities). Requires prepare_prioritized() first."""
        if self._valid_cache is None:
            raise RuntimeError("call prepare_prioritized(length) before prioritized sampling")
        v = self._valid_cache
        p = self.priorities[v]
        probs = p / p.sum()
        starts = v[self.rng.choice(len(v), size=batch_size, p=probs)]
        return self._window_batch(starts, length), starts

    def update_priorities(
        self,
        starts: np.ndarray,
        losses: np.ndarray,
        alpha: float = 0.7,
        beta: float = 0.7,
        c: float = 1e4,
        eps: float = 0.01,
    ) -> None:
        """p_i = c*beta^visits_i + (|loss_i|+eps)^alpha after a gradient step (CR Eq. 1).
        Increments the visit count (count term decays repeatedly-sampled starts) and folds
        in the fresh model loss (loss term up-weights surprising/high-error windows)."""
        self.visits[starts] += 1
        self.priorities[starts] = c * beta ** self.visits[starts] + (np.abs(losses) + eps) ** alpha

    def __len__(self) -> int:
        return self.size
