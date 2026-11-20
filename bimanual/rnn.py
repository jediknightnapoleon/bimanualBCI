"""GRU decoder for two-cursor control, with checkpoint/resume (Fig 4; Supp Fig 3).

Reproduces
----------
The RNN decoder of the paper (Methods "RNN decoder"; Supp Fig 3), built like
the handwriting decoder of Willett et al. 2021:

    x_t (192 z-scored rates)
      -> day-specific affine transform  W_day x_t + b_day  (handles day-to-day drift)
      -> single-layer GRU, 512 units
      -> linear heads: y_t (4 cursor velocities) and z_t (4 context logits:
         none / uni-R / uni-L / bimanual)

Loss per snippet = MSE(velocity) + CE(context) + l2 * ||weights||^2.

Training-time noise (as in Willett 2021; the paper lists "input noise" and
"input mean drift" among its random-search hyperparameters):
* white noise: x_t += N(0, noise_sd), independent per bin and channel;
* mean drift:  x_t += m, with m ~ N(0, drift_sd) per channel, CONSTANT over
  a snippet. This simulates slow baseline shifts.

DEVIATION (register #28): PyTorch instead of TensorFlow v1; the optimizer
is Adam with gradient clipping; there is no "inference step" detail (20-ms
steps here).

Checkpoint / resume
-------------------
``train`` saves model, optimizer, torch and numpy RNG states and the step
count to ``ckpt_path`` every ``ckpt_every`` steps (atomic write). If the file
exists when ``train`` starts, it resumes from it. An interrupted run
continues exactly where it stopped: the batch order is drawn from the saved
numpy RNG.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from bimanual.decoders import DEVICE, pearson_per_dim


@dataclass
class HParams:
    """Hyperparameters (the random-search space is in scripts/fig4_rnn.py)."""

    lr: float = 1e-3
    batch: int = 32
    steps: int = 3000
    l2: float = 1e-5
    noise_sd: float = 0.3
    drift_sd: float = 0.1
    ce_weight: float = 1.0
    hidden: int = 512
    grad_clip: float = 1.0


class GRUDecoder(nn.Module):
    """Day-affine -> GRU(hidden) -> velocity (4) + context logits (4)."""

    def __init__(self, n_in: int = 192, n_days: int = 1, hidden: int = 512, n_vel: int = 4, n_ctx: int = 4):
        super().__init__()
        # one affine input layer per day, initialized to identity
        self.day_W = nn.Parameter(torch.eye(n_in).repeat(n_days, 1, 1))
        self.day_b = nn.Parameter(torch.zeros(n_days, n_in))
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.vel = nn.Linear(hidden, n_vel)
        self.ctx = nn.Linear(hidden, n_ctx)

    def forward(self, x, day_idx, h0=None):
        """x: (B, T, n_in); day_idx: (B,) long. Returns vel (B, T, 4), ctx logits (B, T, 4), h."""
        W, b = self.day_W[day_idx], self.day_b[day_idx]  # (B, n, n), (B, n)
        x = torch.einsum("btn,bmn->btm", x, W) + b[:, None, :]
        h, hT = self.gru(x, h0)
        return self.vel(h), self.ctx(h), hT

    def l2_penalty(self):
        """Sum of squared weights (biases and the identity-initialized day transform excluded)."""
        return sum((p**2).sum() for n, p in self.named_parameters() if "weight" in n)


def _atomic_torch_save(obj, path):
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(
    model: GRUDecoder,
    batch_fn,
    hp: HParams,
    ckpt_path: str | None = None,
    ckpt_every: int = 250,
    seed: int = 0,
    log_every: int = 250,
    log=print,
):
    """Train with per-batch data from ``batch_fn(rng, batch_size)``.

    Args:
        model: a GRUDecoder (moved to DEVICE here).
        batch_fn: callable(np.random.Generator, B) -> (X (B,T,N), vel (B,T,4),
            ctx (B,T) int, day (B,) int), all numpy.
        hp: hyperparameters.
        ckpt_path: checkpoint file; resumes from it if it exists.
        seed: seeds both the numpy batch RNG and the torch noise RNG.

    Returns:
        dict with the final step, the elapsed seconds of this call and the last
        logged losses.
    """
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    step = 0
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["np_rng"]; gen.set_state(ck["torch_rng"].to(gen.device) if DEVICE.type == "cpu" else ck["torch_rng"])
        step = ck["step"]
        log(f"  resumed from {ckpt_path} at step {step}")
    t0, last = time.time(), {}
    while step < hp.steps:
        X, V, Cx, day = batch_fn(rng, hp.batch)
        X = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        V = torch.as_tensor(V, dtype=torch.float32, device=DEVICE)
        Cx = torch.as_tensor(Cx, dtype=torch.long, device=DEVICE)
        day = torch.as_tensor(day, dtype=torch.long, device=DEVICE)
        # training noise: white + per-snippet constant mean drift
        X = X + hp.noise_sd * torch.randn(X.shape, generator=gen, device=DEVICE)
        X = X + hp.drift_sd * torch.randn((X.shape[0], 1, X.shape[2]), generator=gen, device=DEVICE)
        model.train()
        v_hat, c_hat, _ = model(X, day)
        mse = nn.functional.mse_loss(v_hat, V)
        ce = nn.functional.cross_entropy(c_hat.reshape(-1, c_hat.shape[-1]), Cx.reshape(-1))
        loss = mse + hp.ce_weight * ce + hp.l2 * model.l2_penalty()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
        opt.step()
        step += 1
        if step % log_every == 0 or step == hp.steps:
            last = dict(step=step, loss=loss.item(), mse=mse.item(), ce=ce.item())
            log(f"  step {step}/{hp.steps}  loss {loss.item():.4f}  mse {mse.item():.4f}  ce {ce.item():.3f}")
        if ckpt_path and (step % ckpt_every == 0 or step == hp.steps):
            _atomic_torch_save(dict(model=model.state_dict(), opt=opt.state_dict(), np_rng=rng.bit_generator.state,
                                    torch_rng=gen.get_state(), step=step, hp=asdict(hp)), ckpt_path)
    return {**last, "step": step, "seconds": time.time() - t0}


@torch.no_grad()
def decode(model: GRUDecoder, X: np.ndarray, day: int = 0, chunk: int = 5000):
    """Run the RNN causally over one continuous recording X (T, N) from a zero state.

    Long recordings are processed in chunks, carrying the hidden state over,
    so it is the same as running all of it at once.

    Returns:
        (vel (T, 4), context probabilities (T, 4)).
    """
    model.eval()
    h, vs, cs = None, [], []
    d = torch.tensor([day], device=DEVICE)
    for i in range(0, len(X), chunk):
        x = torch.as_tensor(X[None, i : i + chunk], dtype=torch.float32, device=DEVICE)
        v, c, h = model(x, d, h)
        vs.append(v[0].cpu().numpy()); cs.append(torch.softmax(c[0], -1).cpu().numpy())
    return np.concatenate(vs), np.concatenate(cs)


def evaluate(model, X, V, mask=None, day=0):
    """Pearson r per velocity dim on a continuous recording (optionally only bins in ``mask``)."""
    v_hat, c_hat = decode(model, X, day)
    m = slice(None) if mask is None else mask
    return pearson_per_dim(V[m], v_hat[m]), v_hat, c_hat
