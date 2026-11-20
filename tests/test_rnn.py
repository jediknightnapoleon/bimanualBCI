"""Tests for bimanual.rnn: shapes, learning a toy task, and exact checkpoint/resume."""

import numpy as np
import torch

from bimanual import rnn as R


def _toy_batch_fn(N=8, T=40):
    """Velocity target = a fixed linear readout of the input; context = sign of channel 0."""
    W = np.random.default_rng(123).normal(size=(N, 4))

    def fn(rng, B):
        X = rng.normal(size=(B, T, N)).astype(np.float32)
        V = (X @ W * 0.3).astype(np.float32)
        C = (X[..., 0] > 0).astype(int)
        return X, V, C, np.zeros(B, int)

    return fn


def test_forward_shapes_and_day_affine():
    m = R.GRUDecoder(n_in=8, n_days=3, hidden=16)
    v, c, h = m(torch.randn(2, 5, 8), torch.tensor([0, 2]))
    assert v.shape == (2, 5, 4) and c.shape == (2, 5, 4) and h.shape == (1, 2, 16)
    assert torch.allclose(m.day_W[1], torch.eye(8))  # identity init


def test_learns_toy_task_and_decode_matches():
    torch.manual_seed(0)
    m = R.GRUDecoder(n_in=8, hidden=32)
    hp = R.HParams(lr=3e-3, batch=16, steps=300, noise_sd=0.0, drift_sd=0.0, hidden=32)
    R.train(m, _toy_batch_fn(), hp, log=lambda *a: None)
    X, V, _, _ = _toy_batch_fn(T=400)(np.random.default_rng(9), 1)
    r, _, _ = R.evaluate(m, X[0], V[0])
    assert r.mean() > 0.8
    # chunked decoding must equal one-shot decoding (the hidden state is carried over)
    v1, _ = R.decode(m, X[0], chunk=37)
    v2, _ = R.decode(m, X[0], chunk=10_000)
    np.testing.assert_allclose(v1, v2, atol=1e-5)


def test_checkpoint_resume_is_exact(tmp_path):
    """10 steps straight == 5 steps, then a resume for 5 more (same seed); on CPU for determinism."""
    dev = R.DEVICE
    R.DEVICE = torch.device("cpu")
    try:
        hp = R.HParams(lr=1e-3, batch=4, steps=10, hidden=16)
        torch.manual_seed(0); a = R.GRUDecoder(n_in=8, hidden=16)
        torch.manual_seed(0); b = R.GRUDecoder(n_in=8, hidden=16)
        R.train(a, _toy_batch_fn(), hp, seed=5, log=lambda *x: None)
        ck = str(tmp_path / "ck.pt")
        R.train(b, _toy_batch_fn(), R.HParams(**{**hp.__dict__, "steps": 5}), ckpt_path=ck, ckpt_every=5,
                seed=5, log=lambda *x: None)
        torch.manual_seed(0); b2 = R.GRUDecoder(n_in=8, hidden=16)  # a fresh object, as after a crash
        R.train(b2, _toy_batch_fn(), hp, ckpt_path=ck, ckpt_every=5, seed=5, log=lambda *x: None)
        for (n, pa), (_, pb) in zip(a.named_parameters(), b2.named_parameters()):
            assert torch.allclose(pa, pb, atol=1e-6), n
    finally:
        R.DEVICE = dev
