from __future__ import annotations

import numpy as np

from . import register


@register("umut_linear_q")
class UmutLinearQ:
    def __init__(self, feature_shape, n_actions: int, cfg):
        if len(feature_shape) != 1:
            raise ValueError(
                f"umut_linear_q expects a flat feature vector, got {feature_shape}.")
        self.n_features = int(feature_shape[0])
        self.n_actions = int(n_actions)
        self.lr = float(cfg.hp("lr", 0.002))       # matches original LEARNING_RATE
        self.gamma = float(cfg.gamma)                # matches original GAMMA=0.95
        self.grad_clip = float(cfg.hp("grad_clip", 5.0))

        rng = np.random.default_rng(cfg.seed)
        self.W = np.zeros((self.n_actions, self.n_features), dtype=np.float32)
        self._rng = rng

    # -- contract ------------------------------------------------------
    def q_values(self, states) -> np.ndarray:
        x = np.atleast_2d(np.asarray(states, dtype=np.float32))
        return x @ self.W.T

    def act(self, features, eps: float = 0.0) -> int:
        if eps > 0 and self._rng.random() < eps:
            return int(self._rng.integers(self.n_actions))
        qs = self.q_values(features)[0]
        best = np.flatnonzero(qs == qs.max())
        return int(self._rng.choice(best))

    def update(self, batch) -> dict:
        states, actions, rewards, next_states, dones, _idx, weights = batch[:7]
        x = np.asarray(states, dtype=np.float32)
        xn = np.asarray(next_states, dtype=np.float32)
        a = np.asarray(actions, dtype=np.int64)
        r = np.asarray(rewards, dtype=np.float32)
        d = np.asarray(dones, dtype=bool)
        w = np.asarray(weights, dtype=np.float32)

        q_next_all = self.q_values(xn)
        boot = q_next_all.max(axis=1)
        boot[d] = 0.0
        target = r + self.gamma * boot

        td_all = np.zeros(len(a), dtype=np.float32)
        for a_idx in range(self.n_actions):
            mask = a == a_idx
            if not mask.any():
                continue
            S = x[mask]
            targets = target[mask]
            pred = S @ self.W[a_idx]
            err = pred - targets                       # note: pred - target,
                                                         # matches original sign
            td_all[mask] = -err                         # report td in the usual
                                                         # (target - pred) sign
            grad = S.T @ (err * w[mask]) / max(1, mask.sum())
            grad = np.clip(grad, -self.grad_clip, self.grad_clip)
            self.W[a_idx] -= self.lr * grad

        return {"td_abs_mean": float(np.abs(td_all).mean()),
                "w_norm": float(np.linalg.norm(self.W)),
                "td_errors": td_all}

    # -- persistence -----------------------------------------------------
    def save(self, path) -> None:
        with open(path, "wb") as fh:
            np.savez(fh, W=self.W)

    def load(self, path) -> None:
        with open(path, "rb") as fh:
            blob = np.load(fh)
            self.W = blob["W"].astype(np.float32)