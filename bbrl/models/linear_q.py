"""Lineer fonksiyon yaklasimli Q-learning.

Sahibi: IRMAK

Tablosal modelin bir ust basamagi. Her aksiyon icin bir agirlik vektoru:

    Q(s, a) = w_a . s + b_a

Tablodan farki: benzer durumlar arasinda GENELLEME yapar. Tablo, daha once
gormedigi bir durumda hicbir sey bilmez; lineer model feature'lara bakarak
tahmin uretir. Bu, 17x17 tahtada tablonun patlamasini onler.

Yine "dersten teknik" kapsaminda -- semi-gradient TD(0) / Q-learning.
Numpy'den baska bagimliligi yok, CPU'da mikrosaniye mertebesinde calisir.
"""

from __future__ import annotations

import numpy as np

from . import register, _pick


@register("linear_q")
class LinearQ:
    def __init__(self, feature_shape, n_actions: int, cfg):
        if len(feature_shape) != 1:
            raise ValueError(
                f"linear_q duz feature vektoru bekler, {feature_shape} aldi.")
        self.n_features = int(feature_shape[0])
        self.n_actions = int(n_actions)
        self.lr = float(cfg.hp("lr", 0.01))
        self.gamma = float(cfg.gamma)
        self.double = bool(cfg.hp("double", False))
        self.grad_clip = float(cfg.hp("grad_clip", 10.0))

        rng = np.random.default_rng(cfg.seed)
        scale = float(cfg.hp("init_scale", 0.0))
        self.W = (rng.normal(0, scale, (self.n_actions, self.n_features))
                  if scale > 0 else
                  np.zeros((self.n_actions, self.n_features))).astype(np.float32)
        self.b = np.zeros(self.n_actions, dtype=np.float32)
        self._rng = rng

    # -- sozlesme ----------------------------------------------------------
    def q_values(self, states) -> np.ndarray:
        """(B, F) -> (B, A)"""
        x = np.atleast_2d(np.asarray(states, dtype=np.float32))
        return x @ self.W.T + self.b

    def act(self, features, eps: float = 0.0, mask=None) -> int:
        qs = self.q_values(features)[0]
        return _pick(self._rng, qs, mask, self.n_actions, eps)

    def update(self, batch) -> dict:
        states, actions, rewards, next_states, dones, _idx, weights = batch[:7]
        x = np.asarray(states, dtype=np.float32)
        xn = np.asarray(next_states, dtype=np.float32)
        a = np.asarray(actions, dtype=np.int64)
        r = np.asarray(rewards, dtype=np.float32)
        d = np.asarray(dones, dtype=bool)
        w = np.asarray(weights, dtype=np.float32)

        q_next = self.q_values(xn)
        boot = q_next.max(axis=1)
        boot[d] = 0.0                                  # terminalde bootstrap yok
        target = r + self.gamma * boot

        q_cur = self.q_values(x)[np.arange(len(a)), a]
        td = (target - q_cur).astype(np.float32)
        td_clipped = np.clip(td, -self.grad_clip, self.grad_clip)

        # semi-gradient: yalnizca SECILEN aksiyonun agirliklari guncellenir
        coef = (self.lr * w * td_clipped).astype(np.float32)
        np.add.at(self.W, a, coef[:, None] * x)
        np.add.at(self.b, a, coef)

        return {"td_abs_mean": float(np.abs(td).mean()),
                "w_norm": float(np.linalg.norm(self.W)),
                "td_errors": td}

    # -- kalicilik ---------------------------------------------------------
    def save(self, path) -> None:
        # Dosya tanitici uzerinden yaz: np.savez(path) uzantiyi '.npz' diye
        # DEGISTIRIR ve load() ayni yolda dosya bulamaz.
        with open(path, "wb") as fh:
            np.savez(fh, W=self.W, b=self.b)

    def load(self, path) -> None:
        with open(path, "rb") as fh:
            blob = np.load(fh)
            self.W = blob["W"].astype(np.float32)
            self.b = blob["b"].astype(np.float32)
