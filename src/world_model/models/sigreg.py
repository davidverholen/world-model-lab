"""SIGReg — Sketched Isotropic Gaussian Regularization (LeJEPA).

Anti-collapse regularizer from Balestriero & LeCun, "LeJEPA: Provable and Scalable
Self-Supervised Learning Without the Heuristics" (arXiv:2511.08544). Pushes the batch
embedding distribution toward an isotropic Gaussian N(0, I) — the provably optimal
embedding distribution for downstream prediction risk. A constant (collapsed) embedding
is maximally far from N(0, I), so the JEPA prediction loss can no longer be minimized
by cheating. See knowledge/concepts/latent-collapse.md.

Mechanics (paper §SIGReg, reference impl github.com/rbalestr-lab/lejepa):
- sketch: project embeddings onto random unit directions, resampled every step
  (1024 slices recommended, 512 still competitive);
- test: per direction, the Epps-Pulley statistic — weighted L2 distance between the
  empirical characteristic function of the projected samples and the CF of N(0, 1),
  integrated by trapezoid over t in [-5, 5] with 17 points:

      EP = N * integral |phi_hat(t) - exp(-t^2/2)|^2 w(t) dt,   w(t) = exp(-t^2)

  (the N factor keeps the statistic O(1) under H0 since |phi_hat - phi| ~ 1/sqrt(N));
- SIGReg = mean of EP over slices. Gradients are bounded (paper Thm 4), so training
  is stable regardless of how degenerate the embeddings start.
"""

import torch
from torch import nn


class SIGReg(nn.Module):
    """Epps-Pulley sliced test of embeddings against N(0, I), as a loss."""

    def __init__(self, num_slices: int = 512, num_points: int = 17, t_max: float = 5.0):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(-t_max, t_max, num_points)
        self.register_buffer("t", t)
        self.register_buffer("target_cf", torch.exp(-0.5 * t**2))  # CF of N(0,1)
        self.register_buffer("weight", torch.exp(-(t**2)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, D) embeddings -> scalar SIGReg loss."""
        n, d = z.shape
        directions = torch.randn(d, self.num_slices, device=z.device, dtype=z.dtype)
        directions = directions / directions.norm(dim=0, keepdim=True)
        x = z @ directions  # (B, S) sliced 1D samples

        tx = self.t.view(-1, 1, 1) * x.unsqueeze(0)  # (T, B, S)
        ecf_real = tx.cos().mean(dim=1)  # (T, S)
        ecf_imag = tx.sin().mean(dim=1)
        sq_err = (ecf_real - self.target_cf.unsqueeze(1)) ** 2 + ecf_imag**2
        ep = n * torch.trapezoid(sq_err * self.weight.unsqueeze(1), self.t, dim=0)  # (S,)
        return ep.mean()
