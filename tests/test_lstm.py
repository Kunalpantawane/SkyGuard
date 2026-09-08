"""Layer 2 tests.

The gradient gate comes first: hand-rolled BPTT is trusted only if central
differences agree with it. The rest proves the autoencoder learns normal
structure (training reduces loss), reacts to faults (a spike reconstructs
worse), and round-trips through the shared `.npz` weight format.
"""

import os
import tempfile

import numpy as np

from skyguard.config import ModelConfig
from skyguard.model.gradcheck import check_gradients, checked_window_data, gate_summary
from skyguard.model.lstm import LstmAutoencoder, make_model_input, time_features


def tiny_config(**kw) -> ModelConfig:
    base = {"window": 4, "hidden_size": 4, "learning_rate": 0.01,
            "batch_size": 4, "max_epochs": 30, "patience": 6,
            "grad_clip": 5.0, "seed": 7}
    base.update(kw)
    return ModelConfig(**base)


def smooth_batch(n: int, window: int, seed: int):
    """Clean diurnal-like windows: the learnable 'normal' for these tests."""
    rng = np.random.default_rng(seed)
    hours = np.tile(np.arange(window, dtype=np.float64), (n, 1))
    doys = np.full((n, window), 180.0)
    phys = np.concatenate([
        np.sin(2 * np.pi * hours / 24.0)[..., None] + 0.05 * rng.standard_normal((n, window, 1)),
        0.5 * np.cos(2 * np.pi * hours / 24.0)[..., None] + 0.05 * rng.standard_normal((n, window, 1)),
        -0.7 * np.sin(2 * np.pi * hours / 24.0)[..., None] + 0.05 * rng.standard_normal((n, window, 1)),
    ], axis=-1)
    xb = make_model_input(phys, hours, doys)
    return xb, xb[:, :, :3].copy()


# --------------------------------------------------------------------------
# Gradient gate — the load-bearing test of this layer
# --------------------------------------------------------------------------

def test_bptt_matches_finite_differences():
    model = LstmAutoencoder(config=tiny_config())
    xb, yb = checked_window_data(window=4, batch=2, seed=0)
    report = check_gradients(model, xb, yb, seed=0)
    assert report.passed, gate_summary(report)


# --------------------------------------------------------------------------
# Forward / loss sanity
# --------------------------------------------------------------------------

def test_forward_shapes_and_finite_loss():
    model = LstmAutoencoder(config=tiny_config())
    xb, yb = smooth_batch(3, 4, seed=1)
    loss, y_hat, _ = model.forward(xb, yb)
    assert y_hat.shape == (3, 4, 3)
    assert np.isfinite(loss) and loss >= 0.0


def test_time_features_are_cyclic():
    assert np.allclose(time_features(0.0, 80.0), time_features(24.0, 80.0), atol=1e-12)
    feats = time_features(np.array([0.0, 6.0]), np.array([80.0, 80.0]))
    assert feats.shape == (2, 4)


# --------------------------------------------------------------------------
# Learning: loss must fall on clean structure
# --------------------------------------------------------------------------

def test_training_reduces_validation_loss():
    cfg = tiny_config(window=6, hidden_size=8, max_epochs=30, patience=8)
    model = LstmAutoencoder(config=cfg)
    train_x, train_y = smooth_batch(48, 6, seed=2)
    val_x, val_y = smooth_batch(16, 6, seed=3)
    before = model.loss(val_x, val_y)
    history = model.fit(train_x, train_y, val_x, val_y)
    after = model.loss(val_x, val_y)
    assert after < before, f"val loss did not fall: {before:.4f} -> {after:.4f}"
    assert history["epochs"] >= 1


def test_spike_window_reconstructs_worse_than_clean():
    cfg = tiny_config(window=6, hidden_size=8, max_epochs=30, patience=8)
    model = LstmAutoencoder(config=cfg)
    train_x, train_y = smooth_batch(48, 6, seed=4)
    val_x, val_y = smooth_batch(16, 6, seed=5)
    model.fit(train_x, train_y, val_x, val_y)
    clean, _ = smooth_batch(8, 6, seed=6)
    spiked = clean.copy()
    spiked[:, 3, 0] += 6.0  # temperature spike mid-window
    _, clean_err, _ = model.reconstruct_batch(clean)
    _, spike_err, per_var = model.reconstruct_batch(spiked)
    assert float(np.mean(spike_err)) > float(np.mean(clean_err))
    # The native explanation must blame temperature most.
    assert float(np.mean(per_var[:, 0])) > float(np.mean(per_var[:, 1]))


def test_reconstruct_window_contract():
    from skyguard.config import VARIABLES

    model = LstmAutoencoder(config=tiny_config(window=6, hidden_size=8))
    xb, _ = smooth_batch(1, 6, seed=7)
    per_variable, totals, reconstruction = model.reconstruct_window(xb[0])
    assert set(per_variable) == set(VARIABLES)
    assert set(reconstruction) == set(VARIABLES)
    assert totals["error_total"] >= 0.0


# --------------------------------------------------------------------------
# Persistence: the `.npz` format the torch trainer must also honour
# --------------------------------------------------------------------------

def test_save_load_roundtrip_preserves_reconstruction():
    model = LstmAutoencoder(config=tiny_config(window=6, hidden_size=8))
    xb, _ = smooth_batch(4, 6, seed=8)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "lstm_ae.npz")
        model.save(path)
        loaded = LstmAutoencoder.load(path, config=model.config)
    first, _, _ = model.reconstruct_batch(xb)
    second, _, _ = loaded.reconstruct_batch(xb)
    assert np.allclose(first, second, rtol=1e-9, atol=1e-12)
