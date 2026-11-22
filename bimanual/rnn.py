"""GRU (paper) and LSTM (extension) decoders for two-cursor control, with checkpoint/resume (Fig 4; Supp Fig 3).

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

Extension (Phase 8): ``RNNDecoder(cell="lstm")`` swaps the GRU core for an LSTM and keeps
everything else the same. ``GRUDecoder(...)`` still builds the paper's model, and old
checkpoints load unchanged.

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
    # Phase 10 (register #39): random channel dropout. Each training sample gets a rate
    # p ~ U(0, chan_dropout), and each channel is set to 0 (its mean in z-units) with probability p
    # for the whole snippet, imitating channels muted for bad signal quality. 0 = off (no extra
    # random numbers are drawn, so older runs reproduce exactly).
    chan_dropout: float = 0.0


CELLS = ("gru", "lstm")


class RNNDecoder(nn.Module):
    """Day-affine -> recurrent core (GRU or LSTM, ``hidden`` units) -> velocity (4) + context logits (4).

    ``cell="gru"`` is the paper's decoder. ``cell="lstm"`` is the Phase 8 extension: the same
    architecture with an LSTM core. The core is stored under the attribute named after the cell
    (``self.gru`` / ``self.lstm``), so the state-dict keys of GRU checkpoints saved before the
    extension (``gru.weight_ih_l0`` ...) are unchanged and still load.

    DECISION (register #32): LSTM forget-gate biases start at 1 (PyTorch's default is 0), the
    usual choice for LSTMs (Gers et al. 2000; Jozefowicz et al. 2015). It helps the cell keep its
    memory early in training. PyTorch keeps two bias vectors (b_ih, b_hh), each ordered as
    gates [i, f, g, o]; we set f = 0.5 in each, so they sum to 1.
    """

    def __init__(self, n_in: int = 192, n_days: int = 1, hidden: int = 512, n_vel: int = 4, n_ctx: int = 4,
                 cell: str = "gru"):
        super().__init__()
        if cell not in CELLS:
            raise ValueError(f"cell must be one of {CELLS}, got {cell!r}")
        self.cell, self.hidden = cell, hidden
        # one affine input layer per day, initialized to identity
        self.day_W = nn.Parameter(torch.eye(n_in).repeat(n_days, 1, 1))
        self.day_b = nn.Parameter(torch.zeros(n_days, n_in))
        if cell == "gru":
            self.gru = nn.GRU(n_in, hidden, batch_first=True)
        else:
            self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
            with torch.no_grad():
                for name, p in self.lstm.named_parameters():
                    if name.startswith("bias"):
                        p[hidden:2 * hidden].fill_(0.5)  # forget gate; b_ih + b_hh = 1
        self.vel = nn.Linear(hidden, n_vel)
        self.ctx = nn.Linear(hidden, n_ctx)

    @property
    def core(self) -> nn.Module:
        return self.gru if self.cell == "gru" else self.lstm

    def forward(self, x, day_idx, h0=None):
        """x: (B, T, n_in); day_idx: (B,) long.

        Returns vel (B, T, 4), ctx logits (B, T, 4), and the final state: a tensor for the GRU, an
        (h, c) tuple for the LSTM. Either can be passed back as ``h0``.
        """
        W, b = self.day_W[day_idx], self.day_b[day_idx]  # (B, n, n), (B, n)
        x = torch.einsum("btn,bmn->btm", x, W) + b[:, None, :]
        h, hT = self.core(x, h0)
        return self.vel(h), self.ctx(h), hT

    def l2_penalty(self):
        """Sum of squared weights (biases and the identity-initialized day transform excluded)."""
        return sum((p**2).sum() for n, p in self.named_parameters() if "weight" in n)

    def n_params(self, recurrent_only: bool = False) -> int:
        """Number of trainable parameters (``recurrent_only``: the GRU/LSTM core alone)."""
        mod = self.core if recurrent_only else self
        return int(sum(p.numel() for p in mod.parameters()))


def GRUDecoder(n_in: int = 192, n_days: int = 1, hidden: int = 512, n_vel: int = 4, n_ctx: int = 4) -> RNNDecoder:
    """Backward-compatible constructor: the paper's GRU decoder."""
    return RNNDecoder(n_in, n_days, hidden, n_vel, n_ctx, cell="gru")


def recurrent_params(cell: str, n_in: int, hidden: int) -> int:
    """Closed-form parameter count of a PyTorch GRU/LSTM core (weights + both bias vectors)."""
    gates = 3 if cell == "gru" else 4
    return gates * (n_in * hidden + hidden * hidden + 2 * hidden)


def matched_hidden(n_in: int = 192, ref_cell: str = "gru", ref_hidden: int = 512, cell: str = "lstm") -> int:
    """Width of ``cell`` whose recurrent parameter count is closest to ``ref_cell`` at ``ref_hidden``."""
    target = recurrent_params(ref_cell, n_in, ref_hidden)
    return min(range(16, 2 * ref_hidden), key=lambda h: abs(recurrent_params(cell, n_in, h) - target))


def build_from_record(rec: dict, n_in: int) -> RNNDecoder:
    """Rebuild the model of one ``trials.jsonl`` record (records written before Phase 8 have no
    ``cell`` field; they are GRUs)."""
    return RNNDecoder(n_in=n_in, hidden=rec["hp"]["hidden"], cell=rec.get("cell", "gru"),
                      n_vel=rec.get("n_vel", 4), n_ctx=rec.get("n_ctx", 4))


def load_from_record(rec: dict, n_in: int, ckpt_dir) -> RNNDecoder:
    """Build the model of a record and load its trained weights from ``ckpt_dir/<run_id>.pt``."""
    m = build_from_record(rec, n_in).to(DEVICE)
    ck = torch.load(os.path.join(str(ckpt_dir), f"{rec['run_id']}.pt"), map_location=DEVICE, weights_only=False)
    m.load_state_dict(ck["model"])
    return m


def _atomic_torch_save(obj, path):
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(
    model: RNNDecoder,
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
        model: an RNNDecoder (moved to DEVICE here).
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
        if hp.chan_dropout > 0:
            # ASSUMPTION (#39): a muted channel reads exactly 0 after the noise (a detected bad
            # channel is zeroed, not left noisy); no rescaling of the kept channels, because at
            # test time the surviving channels are not rescaled either.
            p = hp.chan_dropout * torch.rand((X.shape[0], 1, 1), generator=gen, device=DEVICE)
            keep = torch.rand((X.shape[0], 1, X.shape[2]), generator=gen, device=DEVICE) >= p
            X = X * keep
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
def decode(model: RNNDecoder, X: np.ndarray, day: int = 0, chunk: int = 5000):
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
