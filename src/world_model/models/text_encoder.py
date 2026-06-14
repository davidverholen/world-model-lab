"""Frozen token-level text encoder for the rung-4 manual (reading-to-learn-dynamics).

Wraps a frozen sentence-transformer (MiniLM-L6 by default) and returns per-TOKEN embeddings + a
padding mask — not a pooled sentence vector (the pre-check `world_model.text_probe` showed pooling
loses the ordered recipe). Mirrors the frozen-DINO visual side ([[frozen-encoder-lean]]): the
manual is encoded once per episode (cached) and the small learned conditioner attends over the
tokens. See knowledge/design/rung4-manual-conditioned-agent.md.
"""

import torch
from torch import nn


class FrozenTextEncoder(nn.Module):
    """manuals (list[str]) → token embeddings (B, Lmax, text_dim) + mask (B, Lmax). Frozen; the
    per-manual token tensor is cached (manuals are fixed within an episode, so encode-once)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        dim_fn = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        self.text_dim = dim_fn()
        self.device = device
        self._cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _tokens(self, manual: str) -> torch.Tensor:
        """(n_tok, text_dim) token embeddings for one manual, cached on CPU."""
        if manual not in self._cache:
            t = self.model.encode(
                [manual], output_value="token_embeddings", convert_to_tensor=True
            )[0]
            self._cache[manual] = t.float().cpu()
        return self._cache[manual]

    @torch.no_grad()
    def encode(self, manuals: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad manuals to (B, Lmax, text_dim) + bool mask (B, Lmax); mask True = real token."""
        toks = [self._tokens(m) for m in manuals]
        lmax = max(int(t.shape[0]) for t in toks)
        b = len(toks)
        out = torch.zeros(b, lmax, self.text_dim)
        mask = torch.zeros(b, lmax, dtype=torch.bool)
        for i, t in enumerate(toks):
            n = t.shape[0]
            out[i, :n] = t
            mask[i, :n] = True
        return out.to(self.device), mask.to(self.device)
