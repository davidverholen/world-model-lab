"""Masked-manual-reconstruction auxiliary loss (Dynalang option A) — a dense self-supervised
reading gradient that strengthens grounding without leaning on sparse task reward.

The rung-4 agent reads but only weakly (swap_follow ~0.15-0.40). Dynalang's insight
([[dynalang-2023]] §2,§5) is that a self-supervised text-reconstruction loss gives a gradient on
language at *every* timestep a token is present, decoupled from reward. Their language is streaming
so "predict the token" is causal and non-trivial; our manual is STATIC and already handed to the
model as cross-attention key/values, so a naive reconstruction is solved by copy-through (route the
manual back to the output — zero grounding). [[dynalang-2023]] "static-manual transfer" option A is
the adaptation: reconstruct MASKED manual token embeddings from the BELIEF, with the manual
cross-attention pathway BLOCKED at decode, so the belief must have *internalised* the content.

Anti-baking crux (knowledge/design/rung4-manual-conditioned-agent.md §2-3): ConditionedRSSM folds
the manual into the deterministic state ``h`` every step (``_deter_cond``). So we do NOT reconstruct
from the ordinary belief ``concat(h, z)`` (that h literally contains the cross-attention context —
trivially copy-through). We reconstruct from ``concat(deter_noctx, z)``: the GRU carry WITHOUT the
current-step attention injection. That carry only knows the manual through what the recurrent
dynamics internalised over prior steps — exactly the grounded quantity. We additionally stop-grad
the conditioner so the aux loss can never tune cross-attention into a better shortcut.

Manual-invariance diagnostic ([[rung4-manual-conditioned-agent]] §4c, [[dynalang-2023]] option A
triviality test): if reconstruction is just as good when the belief was formed under a WRONG
(shuffled) manual, the loss is trivially satisfied (the belief ignores the manual). We log the
wrong/correct loss ratio every eval; ratio ≈ 1 ⇒ triviality, ratio ≫ 1 ⇒ genuine grounding.

Reconstruction target = the FROZEN MiniLM token embeddings, by MSE regression. Justification: the
frozen encoder (text_encoder.FrozenTextEncoder) only ever exposes continuous per-token embeddings,
not token ids — there is no vocabulary head to do Dynalang's categorical-cross-entropy over. MSE on
the frozen embedding is the simplest faithful analog (regress the masked target vector). It keeps
the encoder frozen ([[frozen-encoder-lean]]) and adds no new dependency/vocab.
"""

import torch
import torch.nn.functional as F
from torch import nn

from world_model.models.manual_conditioning import ConditionedRSSM


def random_span_mask(
    valid: torch.Tensor,
    frac: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """MLM-style random-token mask over the real (non-padding) manual tokens.

    ``valid`` (B, L) bool, True = real token. Returns (B, L) bool, True = token is MASKED (held out
    for reconstruction). We mask a ``frac`` fraction of each manual's real tokens, chosen uniformly
    at random per-row — NOT a hardcoded "gesture" span (baking env structure would violate
    [[no-hardcoded-env]]; the loss must stay general). At least one token is masked per row that has
    any real tokens, so the gradient never silently vanishes.
    """
    b, length = valid.shape
    noise = torch.rand(b, length, generator=generator, device=valid.device)
    noise = noise.masked_fill(~valid, 2.0)  # padding sorts last → never selected
    n_valid = valid.sum(dim=1)  # (B,)
    k = torch.clamp((n_valid.float() * frac).round().long(), min=1)
    k = torch.minimum(k, n_valid)  # rows with 0 real tokens get k=0 below
    order = noise.argsort(dim=1)  # ascending; smallest noise = masked
    ranks = order.argsort(dim=1)  # ranks[b, j] = position of token j in the sort
    masked = (ranks < k.unsqueeze(1)) & valid
    return masked


class MaskedManualHead(nn.Module):
    """Regress the frozen embeddings of MASKED manual tokens from the (anti-baking) belief.

    Per masked token, the predictor sees ``[belief, query_embedding]`` where query_embedding is the
    masked token's own frozen embedding with its content zeroed (we keep only a positional/"this
    slot is masked" signal — the constant-zero query) so it cannot read the answer off the input.
    It must recover the held-out vector from the belief. Predicts the frozen embedding by MSE.

    See module docstring for the masking/target justification and the anti-baking construction.
    """

    def __init__(self, belief_dim: int, text_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.text_dim = text_dim
        self.net = nn.Sequential(
            nn.Linear(belief_dim + text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, text_dim),
        )

    def forward(self, belief: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """belief (B, belief_dim); tokens (B, L, text_dim) frozen embeddings (ALREADY masked: the
        masked slots zeroed). Returns predictions (B, L, text_dim)."""
        b, length, _ = tokens.shape
        bel = belief.unsqueeze(1).expand(b, length, belief.shape[-1])
        x = torch.cat([bel, tokens], dim=-1)
        return self.net(x)


def _anti_baking_belief(
    rssm: ConditionedRSSM,
    state: tuple[torch.Tensor, torch.Tensor],
    prev_action: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Belief that EXCLUDES the current-step manual cross-attention injection (anti-baking).

    ``concat(deter_noctx, z)``: the GRU carry without ``+ ctx_to_h(conditioner(...))``, paired with
    the current stochastic code z. The conditioner is run under ``no_grad`` everywhere it touched
    this carry (it didn't, by construction of deter_noctx) — and the caller additionally freezes the
    conditioner params for the aux backward (see ``manual_aux_loss``). This makes the manual content
    reachable only through what the recurrent dynamics internalised, not the attention shortcut.
    """
    _h, z = state
    h_noctx = rssm.deter_noctx(state, prev_action)
    return torch.cat([h_noctx, z], dim=-1)


def manual_aux_loss(
    rssm: ConditionedRSSM,
    head: MaskedManualHead,
    state: tuple[torch.Tensor, torch.Tensor],
    prev_action: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    mask_frac: float = 0.4,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked-manual-reconstruction aux loss for ONE step ([[dynalang-2023]] option A).

    Args:
        rssm: the ConditionedRSSM (provides the anti-baking ``deter_noctx`` belief path).
        head: the MaskedManualHead regressor.
        state: the CURRENT RSSM state (h, z) — z carries the internalised manual.
        prev_action: (B,) long, the action that produced this step.
        tokens: (B, L, text_dim) frozen manual token embeddings.
        mask: (B, L) bool, True = real token.
        mask_frac: fraction of real tokens to hold out (random-span MLM masking).
        generator: optional RNG for reproducible masking.

    Returns:
        (loss, masked) where loss is the mean MSE over masked tokens (scalar) and masked is the
        (B, L) bool mask of which tokens were held out (handy for the invariance diagnostic).

    The conditioner pathway is frozen for this loss: ``deter_noctx`` never calls the conditioner;
    so the loss gets NO gradient to cross-attention (it cannot tune it into a copy-through).
    """
    masked = random_span_mask(mask, mask_frac, generator)
    # Block the cross-attention shortcut: build the belief from the no-context carry. (The
    # conditioner is not even invoked here; deter_noctx is the GRU output pre-injection.)
    belief = _anti_baking_belief(rssm, state, prev_action, tokens, mask)
    # Input tokens have the masked slots zeroed so the head can't read the answer off its own input.
    masked_input = tokens.masked_fill(masked.unsqueeze(-1), 0.0)
    pred = head(belief, masked_input)
    target = tokens.detach()  # frozen embeddings — regression target, no grad to the encoder
    per_tok = F.mse_loss(pred, target, reduction="none").mean(dim=-1)  # (B, L)
    denom = masked.sum().clamp(min=1)
    loss = (per_tok * masked).sum() / denom
    return loss, masked


@torch.no_grad()
def manual_invariance_diagnostic(
    rssm: ConditionedRSSM,
    head: MaskedManualHead,
    state: tuple[torch.Tensor, torch.Tensor],
    prev_action: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    mask_frac: float = 0.4,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Triviality check ([[rung4-manual-conditioned-agent]] §4c): reconstruction loss under the
    CORRECT manual vs a row-SHUFFLED (wrong) manual, holding the belief fixed.

    The belief was formed under the correct manual. We reconstruct (a) the correct manual's masked
    tokens and (b) a wrong manual's masked tokens (manuals shuffled across the batch, so each belief
    is asked to reconstruct someone else's manual). If the belief genuinely internalised THIS
    episode's manual, (b) should be much WORSE than (a). The ratio wrong/correct is the diagnostic:

    - ratio ≈ 1  ⇒ TRIVIAL: reconstruction ignores which manual formed the belief (copy-through /
      the head learned the manual prior, not grounding). The aux loss is not buying grounding.
    - ratio ≫ 1  ⇒ the belief is manual-specific — genuine reading.

    Returns {correct, wrong, ratio}. Caller prints it every eval.
    """
    correct, masked = manual_aux_loss(
        rssm, head, state, prev_action, tokens, mask, mask_frac, generator
    )
    # Shuffle manuals across the batch (a derangement when B>1: roll by 1).
    b = tokens.shape[0]
    perm = torch.roll(torch.arange(b, device=tokens.device), shifts=1)
    wrong_tokens, wrong_mask = tokens[perm], mask[perm]
    belief = _anti_baking_belief(rssm, state, prev_action, tokens, mask)  # belief: correct manual
    wrong_masked = random_span_mask(wrong_mask, mask_frac, generator)
    wrong_input = wrong_tokens.masked_fill(wrong_masked.unsqueeze(-1), 0.0)
    wrong_pred = head(belief, wrong_input)
    per_tok = F.mse_loss(wrong_pred, wrong_tokens, reduction="none").mean(dim=-1)
    wrong = (per_tok * wrong_masked).sum() / wrong_masked.sum().clamp(min=1)
    c, w = float(correct), float(wrong)
    return {"correct": c, "wrong": w, "ratio": (w / c if c > 1e-9 else float("nan"))}
