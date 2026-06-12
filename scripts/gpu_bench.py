"""Tiny GPU throughput benchmark — comparable across machines.

Two workloads bracketing our use cases:
- matmul: raw sustained throughput (what long training runs see after warm-up)
- train-step: our actual world-model training step shape (encoder + GRU rollout),
  closer to real updates/sec

Run on any machine: uv run python scripts/gpu_bench.py
Results belong in knowledge/concepts/compute-strategy.md.
"""

import time

import torch

from world_model.models import ConvEncoder, RecurrentDynamics, RewardHead


def bench_matmul(seconds: float = 20.0) -> float:
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    # warm-up
    for _ in range(5):
        a = (a @ b).tanh()
    torch.cuda.synchronize()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = (a @ b).tanh()
        n += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    flops = n * 2 * 4096**3
    return flops / elapsed / 1e12  # TFLOPS


def bench_train_step(seconds: float = 20.0) -> float:
    """Our recurrent training step: encode window, GRU rollout, backward."""
    torch.manual_seed(0)
    enc = ConvEncoder().cuda()
    dyn = RecurrentDynamics().cuda()
    rh = RewardHead(latent_dim=dyn.state_dim).cuda()
    opt = torch.optim.Adam([*enc.parameters(), *dyn.parameters(), *rh.parameters()], lr=3e-4)
    b, w = 32, 24
    obs = torch.randn(b * w, 3, 56, 56, device="cuda")
    actions = torch.randint(7, (b, w), device="cuda")

    def step():
        z = enc(obs).unflatten(0, (b, w))
        s = dyn.initial_state(b, "cuda")
        loss = torch.zeros((), device="cuda")
        for k in range(w - 1):
            s = dyn.update(z[:, k], actions[:, k], s)
            loss = loss + (dyn.predict_next(s, actions[:, k]) - z[:, k + 1]).pow(2).mean()
            loss = loss + rh(s, actions[:, k]).pow(2).mean() * 0.1
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        step()
        n += 1
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)  # updates/sec


if __name__ == "__main__":
    name = torch.cuda.get_device_name(0)
    tflops = bench_matmul()
    ups = bench_train_step()
    print(f"{name}: matmul {tflops:.1f} TFLOPS fp32 | train-step {ups:.2f} updates/s")
