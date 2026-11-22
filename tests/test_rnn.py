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


# ----------------------------------------------------------------------------- LSTM extension (Phase 8)
def test_lstm_shapes_state_and_forget_bias():
    m = R.RNNDecoder(n_in=8, n_days=2, hidden=16, cell="lstm")
    v, c, (h, cs) = m(torch.randn(3, 5, 8), torch.tensor([0, 1, 1]))
    assert v.shape == (3, 5, 4) and c.shape == (3, 5, 4) and h.shape == (1, 3, 16) and cs.shape == (1, 3, 16)
    # forget-gate biases sum to 1 (b_ih + b_hh)
    f = m.lstm.bias_ih_l0[16:32] + m.lstm.bias_hh_l0[16:32]
    assert torch.allclose(f, torch.ones(16))
    assert m.n_params(recurrent_only=True) == R.recurrent_params("lstm", 8, 16)


def test_lstm_learns_and_chunked_decode_matches():
    torch.manual_seed(0)
    m = R.RNNDecoder(n_in=8, hidden=32, cell="lstm")
    hp = R.HParams(lr=3e-3, batch=16, steps=300, noise_sd=0.0, drift_sd=0.0, hidden=32)
    R.train(m, _toy_batch_fn(), hp, log=lambda *a: None)
    X, V, _, _ = _toy_batch_fn(T=400)(np.random.default_rng(9), 1)
    r, _, _ = R.evaluate(m, X[0], V[0])
    assert r.mean() > 0.8
    v1, _ = R.decode(m, X[0], chunk=37)  # the (h, c) tuple must be carried across chunks
    v2, _ = R.decode(m, X[0], chunk=10_000)
    np.testing.assert_allclose(v1, v2, atol=1e-5)


def test_lstm_checkpoint_resume_is_exact(tmp_path):
    dev = R.DEVICE
    R.DEVICE = torch.device("cpu")
    try:
        hp = R.HParams(lr=1e-3, batch=4, steps=10, hidden=16)
        mk = lambda: R.RNNDecoder(n_in=8, hidden=16, cell="lstm")  # noqa: E731
        torch.manual_seed(0); a = mk()
        R.train(a, _toy_batch_fn(), hp, seed=5, log=lambda *x: None)
        ck = str(tmp_path / "ck.pt")
        torch.manual_seed(0); b = mk()
        R.train(b, _toy_batch_fn(), R.HParams(**{**hp.__dict__, "steps": 5}), ckpt_path=ck, ckpt_every=5,
                seed=5, log=lambda *x: None)
        torch.manual_seed(0); b2 = mk()
        R.train(b2, _toy_batch_fn(), hp, ckpt_path=ck, ckpt_every=5, seed=5, log=lambda *x: None)
        for (n, pa), (_, pb) in zip(a.named_parameters(), b2.named_parameters()):
            assert torch.allclose(pa, pb, atol=1e-6), n
    finally:
        R.DEVICE = dev


def test_old_gru_state_dict_and_records_load():
    """GRU checkpoints written before Phase 8 (keys 'gru.*', records without 'cell') still load."""
    old = R.GRUDecoder(n_in=8, hidden=16).state_dict()
    assert any(k.startswith("gru.") for k in old)
    m = R.build_from_record({"run_id": "x", "hp": {"hidden": 16}}, n_in=8)
    assert m.cell == "gru"
    m.load_state_dict(old)  # strict: raises if any key differs


def test_matched_hidden_is_parameter_matched():
    h = R.matched_hidden(192, "gru", 512, "lstm")
    g, l_ = R.recurrent_params("gru", 192, 512), R.recurrent_params("lstm", 192, h)
    assert abs(l_ - g) / g < 0.005 and 400 < h < 460


# ----------------------------------------------------------------------------- Phase 10: channel dropout
def test_chan_dropout_zero_is_bit_identical_to_default():
    """chan_dropout=0 draws no extra random numbers: weights equal a run without the field set."""
    fn = _toy_batch_fn()
    out = []
    for hp in (R.HParams(batch=8, steps=15, hidden=16), R.HParams(batch=8, steps=15, hidden=16, chan_dropout=0.0)):
        torch.manual_seed(0)
        m = R.RNNDecoder(n_in=8, hidden=16, cell="lstm")
        R.train(m, fn, hp, seed=3, log=lambda *a: None)
        out.append(torch.cat([p.detach().flatten().cpu() for p in m.parameters()]))
    assert torch.equal(out[0], out[1])


def test_chan_dropout_zeroes_whole_channels(monkeypatch):
    """With dropout on, the model sees inputs whose muted channels are exactly 0 for the whole snippet,
    and the muted fraction averages chan_dropout / 2 (p ~ U(0, chan_dropout))."""
    seen = []
    orig = R.RNNDecoder.forward

    def spy(self, x, day_idx, h0=None):
        if self.training:
            seen.append(x.detach().cpu())
        return orig(self, x, day_idx, h0)

    monkeypatch.setattr(R.RNNDecoder, "forward", spy)
    # identity day transform at init: zero inputs stay zero through the first forward pass
    m = R.RNNDecoder(n_in=64, hidden=8, cell="lstm")
    hp = R.HParams(batch=64, steps=1, hidden=8, noise_sd=0.1, drift_sd=0.1, chan_dropout=0.6)
    R.train(m, _toy_batch_fn(N=64, T=20), hp, seed=0, log=lambda *a: None)
    x = seen[0]  # (B, T, N) after noise + dropout (day transform not yet applied in the spy)
    muted = (x == 0).all(dim=1)  # (B, N): a channel is 0 at every time step
    assert muted.any(dim=1).float().mean() > 0.5  # most samples have some muted channels
    frac = muted.float().mean().item()
    assert 0.2 < frac < 0.4, frac  # E = 0.6 / 2
    assert not ((x == 0) & ~muted[:, None, :]).any()  # zeros only on fully muted channels
