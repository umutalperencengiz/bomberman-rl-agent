"""
bbrl/models/umut_dueling_q.py

Sahibi: UMUT

Dueling Double DQN with vectorized pure NumPy execution.
Decouples state value V(s) from action advantages A(s, a):
    Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))
"""

from __future__ import annotations
import numpy as np
from . import register, _pick


def _he_init(rng, fan_in, fan_out):
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float32)


class _AdamState:
    def __init__(self, shape):
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.t = 0

    def step(self, param, grad, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        self.m = beta1 * self.m + (1 - beta1) * grad
        self.v = beta2 * self.v + (1 - beta2) * (grad * grad)
        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return param


class _DuelingMLP:
    """Shared trunk + separate Value and Advantage heads."""

    def __init__(self, n_features, h1, h2, n_actions, rng):
        # Shared trunk
        self.W1 = _he_init(rng, n_features, h1)
        self.b1 = np.zeros(h1, dtype=np.float32)

        # Value stream: h1 -> h2 -> 1
        self.W_v1 = _he_init(rng, h1, h2)
        self.b_v1 = np.zeros(h2, dtype=np.float32)
        self.W_v2 = _he_init(rng, h2, 1)
        self.b_v2 = np.zeros(1, dtype=np.float32)

        # Advantage stream: h1 -> h2 -> n_actions
        self.W_a1 = _he_init(rng, h1, h2)
        self.b_a1 = np.zeros(h2, dtype=np.float32)
        self.W_a2 = _he_init(rng, h2, n_actions)
        self.b_a2 = np.zeros(n_actions, dtype=np.float32)

    def params(self):
        return [
            self.W1, self.b1,
            self.W_v1, self.b_v1, self.W_v2, self.b_v2,
            self.W_a1, self.b_a1, self.W_a2, self.b_a2,
        ]

    def set_params(self, params):
        (self.W1, self.b1,
         self.W_v1, self.b_v1, self.W_v2, self.b_v2,
         self.W_a1, self.b_a1, self.W_a2, self.b_a2) = params

    def clone(self):
        other = _DuelingMLP.__new__(_DuelingMLP)
        other.set_params([p.copy() for p in self.params()])
        return other

    def forward(self, x, cache: dict | None = None):
        # Shared trunk
        z1 = x @ self.W1 + self.b1
        h = np.maximum(z1, 0.0)

        # Value stream
        zv1 = h @ self.W_v1 + self.b_v1
        hv = np.maximum(zv1, 0.0)
        v = hv @ self.W_v2 + self.b_v2  # (batch, 1)

        # Advantage stream
        za1 = h @ self.W_a1 + self.b_a1
        ha = np.maximum(za1, 0.0)
        adv = ha @ self.W_a2 + self.b_a2  # (batch, n_actions)

        # Dueling aggregation: Q = V + (A - mean(A))
        adv_mean = adv.mean(axis=1, keepdims=True)
        q = v + (adv - adv_mean)

        if cache is not None:
            cache.update(
                x=x, z1=z1, h=h,
                zv1=zv1, hv=hv, v=v,
                za1=za1, ha=ha, adv=adv, adv_mean=adv_mean, q=q
            )
        return q

    def backward(self, cache: dict, dq: np.ndarray):
        x, z1, h = cache["x"], cache["z1"], cache["h"]
        zv1, hv = cache["zv1"], cache["hv"]
        za1, ha = cache["za1"], cache["ha"]
        batch_size = x.shape[0]
        n_actions = dq.shape[1]

        # Gradients through dueling aggregation
        # dQ/dV = 1, dQ/dA = 1 - 1/|A|
        dv = dq.sum(axis=1, keepdims=True)
        dadv = dq - (dq.sum(axis=1, keepdims=True) / n_actions)

        # Value stream backward
        dW_v2 = hv.T @ dv / batch_size
        db_v2 = dv.mean(axis=0)
        dhv = dv @ self.W_v2.T
        dzv1 = dhv * (zv1 > 0)
        dW_v1 = h.T @ dzv1 / batch_size
        db_v1 = dzv1.mean(axis=0)
        dh_v = dzv1 @ self.W_v1.T

        # Advantage stream backward
        dW_a2 = ha.T @ dadv / batch_size
        db_a2 = dadv.mean(axis=0)
        dha = dadv @ self.W_a2.T
        dza1 = dha * (za1 > 0)
        dW_a1 = h.T @ dza1 / batch_size
        db_a1 = dza1.mean(axis=0)
        dh_a = dza1 @ self.W_a1.T

        # Shared trunk backward
        dh = dh_v + dh_a
        dz1 = dh * (z1 > 0)
        dW1 = x.T @ dz1 / batch_size
        db1 = dz1.mean(axis=0)

        return [
            dW1, db1,
            dW_v1, db_v1, dW_v2, db_v2,
            dW_a1, db_a1, dW_a2, db_a2,
        ]


@register("umut_dueling_q")
class UmutDuelingQ:
    def __init__(self, feature_shape, n_actions: int, cfg):
        self.n_features = int(feature_shape[0])
        self.n_actions = int(n_actions)

        self.lr = float(cfg.hp("lr", 0.001))
        self.gamma = float(cfg.gamma)
        self.n_steps = int(cfg.hp("n_steps", 3))
        self.grad_clip = float(cfg.hp("grad_clip", 5.0))
        self.h1 = int(cfg.hp("hidden1", 64))
        self.h2 = int(cfg.hp("hidden2", 32))
        self.target_update_every = int(cfg.hp("target_update_every", 500))

        rng = np.random.default_rng(cfg.seed)
        self._rng = rng
        self.online = _DuelingMLP(self.n_features, self.h1, self.h2, self.n_actions, rng)
        self.target = self.online.clone()
        self._adam = [_AdamState(p.shape) for p in self.online.params()]
        self._learn_steps = 0

    def q_values(self, states) -> np.ndarray:
        x = np.atleast_2d(np.asarray(states, dtype=np.float32))
        return self.online.forward(x)

    def act(self, features, eps: float = 0.0, mask=None) -> int:
        # Diger uc modelle ayni arayuz. mask=None iken secim, orijinal
        # kodla BIREBIR ayni RNG cekilislerini yapar
        # (tests/test_action_mask.py::test_mask_none_is_unchanged).
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
        n = len(a)

        # Double-DQN target with n-step discount factor: gamma^n
        discount = self.gamma ** self.n_steps
        q_next_online = self.online.forward(xn)
        best_a = np.argmax(q_next_online, axis=1)
        q_next_target = self.target.forward(xn)
        boot = q_next_target[np.arange(n), best_a]
        boot[d] = 0.0
        target = r + discount * boot

        cache: dict = {}
        q_pred_all = self.online.forward(x, cache=cache)
        pred = q_pred_all[np.arange(n), a]
        err = pred - target

        dq = np.zeros_like(q_pred_all)
        dq[np.arange(n), a] = err * w

        grads = self.online.backward(cache, dq)
        params = self.online.params()
        for i, (p, g) in enumerate(zip(params, grads)):
            gn = np.linalg.norm(g)
            if gn > self.grad_clip:
                g = g * (self.grad_clip / (gn + 1e-8))
            params[i] = self._adam[i].step(p, g, self.lr)
        self.online.set_params(params)

        self._learn_steps += 1
        if self._learn_steps % max(1, self.target_update_every) == 0:
            self.target = self.online.clone()

        return {
            "td_abs_mean": float(np.abs(err).mean()),
            "w_norm": float(sum(np.linalg.norm(p) for p in params)),
        }

    def save(self, path) -> None:
        with open(path, "wb") as fh:
            np.savez(fh, **{f"p_{i}": p for i, p in enumerate(self.online.params())})

    def load(self, path) -> None:
        with open(path, "rb") as fh:
            blob = np.load(fh)
            params = [blob[f"p_{i}"].astype(np.float32) for i in range(len(self.online.params()))]
            self.online.set_params(params)
            self.target = self.online.clone()
            self._adam = [_AdamState(p.shape) for p in self.online.params()]
            self._learn_steps = 0
