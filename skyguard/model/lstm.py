"""Layer 2 — LSTM autoencoder, the temporal intelligence core.

Why a sequence model instead of a point model: a spike and the start of a
legitimate heatwave can share the same instantaneous value; only the trajectory
distinguishes them. The autoencoder is trained on *trusted* normal windows only,
so unfamiliar behaviour reconstructs badly and the per-variable reconstruction
errors say *which sensor* looks wrong with no post-hoc method.

Topology (see `context/ml.md`): encoder LSTM (W x F -> h), latent = final
hidden state, latent repeated W times as the decoder LSTM input (-> h), dense
(h -> 3) reconstructing only the three physical variables. Cyclic time features
are encoder inputs, never reconstruction targets — predicting the clock is
trivial and would dilute the error signal.

Numpy only, including BPTT by hand. Correctness is proven by
`model/gradcheck.py`, not by trusting a framework; PyTorch exists only as an
optional faster trainer exporting the same `.npz` layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

import numpy as np

from ..config import DEFAULT_CONFIG, ModelConfig

# Parameter keys for the `.npz` weight format. The optional PyTorch trainer
# must export exactly these names and shapes.
PARAM_KEYS: Tuple[str, ...] = ("enc_W", "enc_b", "dec_W", "dec_b", "dense_W", "dense_b")


# --------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _dsigmoid_from_output(s: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given the already-computed output."""
    return s * (1.0 - s)


# --------------------------------------------------------------------------
# Feature helpers
# --------------------------------------------------------------------------

def time_features(hour: np.ndarray | float, day_of_year: np.ndarray | float) -> np.ndarray:
    """Cyclic time context so the model knows 38 C at 14:00 differs from 04:00.

    Returns the last four columns of the model input vector:
    [sin(2 pi hour/24), cos(...), sin(2 pi doy/365), cos(...)].
    """
    h = np.asarray(hour, dtype=np.float64)
    d = np.asarray(day_of_year, dtype=np.float64)
    s_h = np.sin(2.0 * np.pi * h / 24.0)
    c_h = np.cos(2.0 * np.pi * h / 24.0)
    s_d = np.sin(2.0 * np.pi * d / 365.0)
    c_d = np.cos(2.0 * np.pi * d / 365.0)
    return np.stack([s_h, c_h, s_d, c_d], axis=-1)


def make_model_input(
    phys_z: np.ndarray, hour: np.ndarray, day_of_year: np.ndarray
) -> np.ndarray:
    """Stack standardized physical variables with cyclic time features.

    `phys_z` is (..., 3), `hour`/`day_of_year` broadcast against it; returns
    (..., 7). Standardisation itself lives in `model/scaler.py` — this only
    assembles the columns in the canonical order.
    """
    phys = np.asarray(phys_z, dtype=np.float64)
    feats = time_features(np.asarray(hour), np.asarray(day_of_year))
    return np.concatenate([phys, feats], axis=-1)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class _StepCache:
    """Everything one LSTM step needs for its backward pass."""

    x: np.ndarray          # (B, I) step input
    h_prev: np.ndarray     # (B, H)
    c_prev: np.ndarray     # (B, H)
    f: np.ndarray          # (B, H) forget gate
    i: np.ndarray          # (B, H) input gate
    o: np.ndarray          # (B, H) output gate
    g: np.ndarray          # (B, H) candidate
    tanh_c: np.ndarray     # (B, H) tanh(cell)
    cell: np.ndarray       # (B, H)


@dataclass
class _ForwardCache:
    """Full unrolled forward pass, consumed by `backward`."""

    enc_steps: List[_StepCache] = field(default_factory=list)
    dec_steps: List[_StepCache] = field(default_factory=list)
    dec_hidden: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    latent: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    y_hat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))


class LstmAutoencoder:
    """Encoder-decoder LSTM reconstructing the three physical channels.

    Encoder input width is F=7 (3 standardized variables + 4 time features);
    decoder input width is H (latent repeated); dense maps H -> 3.
    """

    def __init__(self, config: ModelConfig | None = None, seed: int | None = None) -> None:
        self.config: ModelConfig = config or DEFAULT_CONFIG.model
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        hidden = self.config.hidden_size
        n_in = self.config.n_features
        self.params: Dict[str, np.ndarray] = {
            "enc_W": _xavier(rng, (n_in + hidden, 4 * hidden)),
            "enc_b": _forget_bias(hidden),
            "dec_W": _xavier(rng, (2 * hidden, 4 * hidden)),
            "dec_b": _forget_bias(hidden),
            "dense_W": _xavier(rng, (hidden, self.config.n_physical)),
            "dense_b": np.zeros((self.config.n_physical,), dtype=np.float64),
        }
        # Adam state, lazily zero-initialised to the parameter shapes.
        self._adam_m: Dict[str, np.ndarray] = {}
        self._adam_v: Dict[str, np.ndarray] = {}
        self._adam_t: int = 0

    # -- properties ------------------------------------------------------

    @property
    def window(self) -> int:
        """Observations per window."""
        return self.config.window

    @property
    def hidden_size(self) -> int:
        """LSTM hidden units, also the latent width."""
        return self.config.hidden_size

    # -- forward ----------------------------------------------------------

    def forward(
        self, inputs: np.ndarray, targets: np.ndarray
    ) -> Tuple[float, np.ndarray, _ForwardCache]:
        """Reconstruct a batch and score it.

        Returns (loss, y_hat, cache). Loss is MSE over the three physical
        channels only, averaged over batch, time and variable.
        """
        xb = np.asarray(inputs, dtype=np.float64)
        yb = np.asarray(targets, dtype=np.float64)
        batch, width, _ = xb.shape
        hidden = self.hidden_size

        enc_steps: List[_StepCache] = []
        h = np.zeros((batch, hidden), dtype=np.float64)
        c = np.zeros((batch, hidden), dtype=np.float64)
        for t in range(width):
            h, c, step = _lstm_forward(xb[:, t, :], h, c, self.params["enc_W"], self.params["enc_b"])
            enc_steps.append(step)
        latent = h
        cell_final = c

        dec_steps: List[_StepCache] = []
        h_d = latent.copy()
        c_d = cell_final.copy()
        dec_hidden = np.empty((batch, width, hidden), dtype=np.float64)
        for t in range(width):
            h_d, c_d, step = _lstm_forward(
                latent, h_d, c_d, self.params["dec_W"], self.params["dec_b"]
            )
            dec_steps.append(step)
            dec_hidden[:, t, :] = h_d

        y_hat = dec_hidden @ self.params["dense_W"] + self.params["dense_b"]
        loss = float(np.mean((y_hat - yb) ** 2))
        cache = _ForwardCache(
            enc_steps=enc_steps,
            dec_steps=dec_steps,
            dec_hidden=dec_hidden,
            latent=latent,
            y_hat=y_hat,
        )
        return loss, y_hat, cache

    def loss(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Forward-only MSE, for validation and the gradient checker."""
        value, _, _ = self.forward(np.asarray(inputs), np.asarray(targets))
        return value

    # -- backward (BPTT) ----------------------------------------------------

    def backward(
        self, inputs: np.ndarray, targets: np.ndarray, cache: _ForwardCache
    ) -> Dict[str, np.ndarray]:
        """Backpropagate through decoder, dense head and encoder.

        Returns gradients keyed like `self.params`. Shapes match parameters;
        the caller clips and applies them.
        """
        xb = np.asarray(inputs, dtype=np.float64)
        yb = np.asarray(targets, dtype=np.float64)
        batch, width, _ = xb.shape
        hidden = self.hidden_size
        normaliser = float(batch * width * self.config.n_physical)

        grad_y_hat = 2.0 * (cache.y_hat - yb) / normaliser  # (B, W, 3)

        flat_h = cache.dec_hidden.reshape(batch * width, hidden)
        flat_dy = grad_y_hat.reshape(batch * width, self.config.n_physical)
        grad_dense_w: np.ndarray = flat_h.T @ flat_dy
        grad_dense_b: np.ndarray = flat_dy.sum(axis=0)
        # (B, W, H) gradient into decoder hidden states.
        grad_dec_h = grad_y_hat @ self.params["dense_W"].T

        # Decoder pass, backwards in time. The decoder input at every step is
        # the shared latent vector, so input-gradients accumulate into it.
        grad_dec_w = np.zeros_like(self.params["dec_W"])
        grad_dec_b = np.zeros_like(self.params["dec_b"])
        grad_latent = np.zeros_like(cache.latent)
        dh_next = np.zeros((batch, hidden), dtype=np.float64)
        dc_next = np.zeros((batch, hidden), dtype=np.float64)
        for t in range(width - 1, -1, -1):
            dh = grad_dec_h[:, t, :] + dh_next
            dx, dh_next, dc_next, dW, db = _lstm_backward(
                dh, dc_next, cache.dec_steps[t], self.params["dec_W"]
            )
            grad_latent += dx
            grad_dec_w += dW
            grad_dec_b += db
        # Decoder initial (h, c) are the encoder final (h, c): gradients flow on.
        grad_latent += dh_next
        grad_cell_final = dc_next

        # Encoder pass, backwards in time. Only the final step receives the
        # latent/cell gradients; earlier steps receive recurrent flow.
        grad_enc_w = np.zeros_like(self.params["enc_W"])
        grad_enc_b = np.zeros_like(self.params["enc_b"])
        dh_next = np.zeros((batch, hidden), dtype=np.float64)
        dc_next = grad_cell_final
        for t in range(width - 1, -1, -1):
            dh = dh_next + (grad_latent if t == width - 1 else 0.0)
            _, dh_next, dc_next, dW, db = _lstm_backward(
                dh, dc_next, cache.enc_steps[t], self.params["enc_W"]
            )
            grad_enc_w += dW
            grad_enc_b += db

        return {
            "enc_W": grad_enc_w,
            "enc_b": grad_enc_b,
            "dec_W": grad_dec_w,
            "dec_b": grad_dec_b,
            "dense_W": grad_dense_w,
            "dense_b": grad_dense_b,
        }

    def clipped_gradients(
        self, grads: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Rescale gradients to the configured global-norm ceiling.

        LSTMs routinely produce exploding gradients; without clipping a single
        bad batch can undo epochs of training.
        """
        max_norm = self.config.grad_clip
        total = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())))
        if total <= max_norm or total == 0.0:
            return grads
        scale = max_norm / total
        return {k: g * scale for k, g in grads.items()}

    # -- optimisation -------------------------------------------------------

    def _adam_step(self, grads: Dict[str, np.ndarray]) -> None:
        """Apply one Adam update with bias correction."""
        if not self._adam_m:
            self._adam_m = {k: np.zeros_like(v) for k, v in self.params.items()}
            self._adam_v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._adam_t += 1
        b1, b2, eps = self.config.beta1, self.config.beta2, self.config.epsilon
        lr = self.config.learning_rate
        for key in self.params:
            g = grads[key]
            self._adam_m[key] = b1 * self._adam_m[key] + (1.0 - b1) * g
            self._adam_v[key] = b2 * self._adam_v[key] + (1.0 - b2) * (g * g)
            m_hat = self._adam_m[key] / (1.0 - b1 ** self._adam_t)
            v_hat = self._adam_v[key] / (1.0 - b2 ** self._adam_t)
            self.params[key] = self.params[key] - lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(
        self,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
        val_inputs: np.ndarray,
        val_targets: np.ndarray,
        verbose: bool = False,
    ) -> Dict[str, object]:
        """Train with minibatching and early stopping on clean validation loss.

        Both splits must be trusted-normal data (see `context/data.md`): an
        autoencoder trained on dirty data learns faults as normal. Returns a
        history dict; the best-validation parameters are restored.
        """
        cfg = self.config
        train_x = np.asarray(train_inputs, dtype=np.float64)
        train_y = np.asarray(train_targets, dtype=np.float64)
        val_x = np.asarray(val_inputs, dtype=np.float64)
        val_y = np.asarray(val_targets, dtype=np.float64)
        rng = np.random.default_rng(cfg.seed)
        n = train_x.shape[0]

        best_val = float("inf")
        best_params = {k: v.copy() for k, v in self.params.items()}
        train_history: List[float] = []
        val_history: List[float] = []
        stalled = 0

        for epoch in range(cfg.max_epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, cfg.batch_size):
                idx = order[start:start + cfg.batch_size]
                loss, _, cache = self.forward(train_x[idx], train_y[idx])
                grads = self.backward(train_x[idx], train_y[idx], cache)
                self._adam_step(self.clipped_gradients(grads))
                epoch_loss += loss
                n_batches += 1
            train_history.append(epoch_loss / max(n_batches, 1))
            val_loss = self.loss(val_x, val_y)
            val_history.append(val_loss)
            if verbose:
                print(f"epoch {epoch + 1}: train={train_history[-1]:.6f} val={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                best_params = {k: v.copy() for k, v in self.params.items()}
                stalled = 0
            else:
                stalled += 1
                if stalled >= cfg.patience:
                    break

        self.params = best_params
        return {
            "epochs": len(train_history),
            "best_val_loss": best_val,
            "train_history": train_history,
            "val_history": val_history,
        }

    # -- inference ----------------------------------------------------------

    def reconstruct_batch(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct a batch of windows.

        Returns (y_hat (B, W, 3), error_total (B,), error_per_var (B, 3)).
        Per-variable errors are the AE's native explanation of *which* sensor
        looks wrong.
        """
        xb = np.asarray(inputs, dtype=np.float64)
        batch, width, _ = xb.shape
        h = np.zeros((batch, self.hidden_size), dtype=np.float64)
        c = np.zeros((batch, self.hidden_size), dtype=np.float64)
        for t in range(width):
            h, c, _ = _lstm_forward_no_cache(
                xb[:, t, :], h, c, self.params["enc_W"], self.params["enc_b"]
            )
        latent = h
        h_d, c_d = latent.copy(), c.copy()
        dec_hidden = np.empty((batch, width, self.hidden_size), dtype=np.float64)
        for t in range(width):
            h_d, c_d, _ = _lstm_forward_no_cache(
                latent, h_d, c_d, self.params["dec_W"], self.params["dec_b"]
            )
            dec_hidden[:, t, :] = h_d
        y_hat = dec_hidden @ self.params["dense_W"] + self.params["dense_b"]
        targets = xb[:, :, : self.config.n_physical]
        sq = (y_hat - targets) ** 2
        error_total = sq.mean(axis=(1, 2))
        error_per_var = sq.mean(axis=1)
        return y_hat, error_total, error_per_var

    def reconstruct_window(
        self, window: np.ndarray
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Score one window; returns (per_variable, totals, reconstruction).

        `per_variable` maps temp_c/pressure_hpa/rh_pct to MSE; `totals` holds
        `error_total`; `reconstruction` holds the last-step estimate per
        variable for the correction stage.
        """
        from ..config import VARIABLES

        xb = np.asarray(window, dtype=np.float64).reshape(1, self.window, -1)
        y_hat, error_total, error_per_var = self.reconstruct_batch(xb)
        per_variable = {var: float(error_per_var[0, k]) for k, var in enumerate(VARIABLES)}
        totals = {"error_total": float(error_total[0])}
        reconstruction = {var: float(y_hat[0, -1, k]) for k, var in enumerate(VARIABLES)}
        return per_variable, totals, reconstruction

    # -- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        """Store weights in the shared `.npz` format (also the torch target)."""
        np.savez(
            path,
            **{k: self.params[k] for k in PARAM_KEYS},
            window=np.array(self.window),
            hidden_size=np.array(self.hidden_size),
        )

    @classmethod
    def load(cls, path: str, config: ModelConfig | None = None) -> "LstmAutoencoder":
        """Load weights saved by `save` (or the optional torch trainer).

        The artifact's own `window`/`hidden_size` win over the config defaults.
        A model exported with a non-default geometry must reload with that
        geometry or inference silently reshapes windows wrongly — `pipeline.py`
        treats the artifact as authoritative, so this loader must too. An
        explicit `config` still supplies everything else (learning rate, seed),
        but cannot contradict the saved shapes.
        """
        data = np.load(path, allow_pickle=False)
        cfg = config or DEFAULT_CONFIG.model

        overrides: Dict[str, int] = {}
        if "window" in data.files:
            overrides["window"] = int(np.asarray(data["window"]).reshape(()))
        if "hidden_size" in data.files:
            overrides["hidden_size"] = int(np.asarray(data["hidden_size"]).reshape(()))
        if overrides:
            cfg = replace(cfg, **overrides)

        model = cls(config=cfg)
        for key in PARAM_KEYS:
            weights = np.asarray(data[key], dtype=np.float64)
            expected = model.params[key].shape
            if weights.shape != expected:
                raise ValueError(
                    f"{path}: parameter {key} has shape {weights.shape}, but the "
                    f"artifact's geometry (window={model.window}, "
                    f"hidden_size={model.hidden_size}) implies {expected}"
                )
            model.params[key] = weights
        return model


# --------------------------------------------------------------------------
# LSTM step maths
# --------------------------------------------------------------------------

def _xavier(rng: np.random.Generator, shape: Tuple[int, int]) -> np.ndarray:
    """Xavier-uniform initialisation, keeping gate activations in range."""
    fan_in, fan_out = shape
    limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
    return rng.uniform(-limit, limit, size=shape).astype(np.float64)


def _forget_bias(hidden: int) -> np.ndarray:
    """Zero biases except forget gates at 1.0 so early training remembers."""
    bias = np.zeros((4 * hidden,), dtype=np.float64)
    bias[:hidden] = 1.0
    return bias


def _lstm_forward(
    x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray,
    weights: np.ndarray, bias: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, _StepCache]:
    """One LSTM step with cache for BPTT."""
    concat = np.concatenate([x, h_prev], axis=1)
    pre = concat @ weights + bias
    hidden = h_prev.shape[1]
    zf, zi, zo, zg = np.split(pre, 4, axis=1)
    f = _sigmoid(zf)
    i = _sigmoid(zi)
    o = _sigmoid(zo)
    g = np.tanh(zg)
    cell = f * c_prev + i * g
    tanh_c = np.tanh(cell)
    h_new = o * tanh_c
    cache = _StepCache(x=x, h_prev=h_prev, c_prev=c_prev, f=f, i=i, o=o, g=g,
                       tanh_c=tanh_c, cell=cell)
    return h_new, cell, cache


def _lstm_forward_no_cache(
    x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray,
    weights: np.ndarray, bias: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, None]:
    """Cache-free step for inference, where no backward pass follows."""
    h_new, cell, _ = _lstm_forward(x, h_prev, c_prev, weights, bias)
    return h_new, cell, None


def _lstm_backward(
    grad_h: np.ndarray, grad_c_next: np.ndarray,
    cache: _StepCache, weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One LSTM step backwards.

    Returns (grad_x, grad_h_prev, grad_c_prev, grad_W, grad_b).
    """
    dh = np.asarray(grad_h, dtype=np.float64)
    dc_next = np.asarray(grad_c_next, dtype=np.float64)

    do_raw = dh * cache.tanh_c * _dsigmoid_from_output(cache.o)
    dtanh = dh * cache.o
    dc = dtanh * (1.0 - cache.tanh_c ** 2) + dc_next
    df_raw = dc * cache.c_prev * _dsigmoid_from_output(cache.f)
    di_raw = dc * cache.g * _dsigmoid_from_output(cache.i)
    dg_raw = dc * cache.i * (1.0 - cache.g ** 2)

    dz = np.concatenate([df_raw, di_raw, do_raw, dg_raw], axis=1)
    xh = np.concatenate([cache.x, cache.h_prev], axis=1)
    grad_W = xh.T @ dz
    grad_b = dz.sum(axis=0)
    dxh = dz @ weights.T
    input_width = cache.x.shape[1]
    grad_x = dxh[:, :input_width]
    grad_h_prev = dxh[:, input_width:]
    grad_c_prev = dc * cache.f
    return grad_x, grad_h_prev, grad_c_prev, grad_W, grad_b
