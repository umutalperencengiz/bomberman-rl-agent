"""bbrl/models/umut_nonlinear_q.py

Umut's nonlinear Q-learning model -- a small MLP replacing the linear
Q(s,a) = w_a . s of umut_linear_q.py with Q(s,a) = MLP(s)[a].

Registered under "umut_nonlinear_q" -- again deliberately NOT colliding with
Irmak's or anyone else's names. Keep umut_linear_q.py alongside this one;
this is a genuinely separate architecture, not a rename.

WHY THIS EXISTS
  The linear model can only ever draw a hyperplane per action in feature
  space. v1_handcrafted already gives directional/danger signals as
  *information*, not decisions (per the project's "no deterministic best
  action" rule) -- which means the model still has to combine e.g.
  "safe move UP" AND "danger RIGHT" AND "crate direction diagonal" into a
  single action preference. Those interactions are exactly what a linear
  combination of 25 independent features cannot represent well; a small
  MLP can.

ARCHITECTURE
  Plain numpy, no torch dependency (callbacks.py only *tries* to import
  torch for thread-limiting; it must not be a hard requirement).
    input (n_features)
      -> Dense(h1) + ReLU
      -> Dense(h2) + ReLU
      -> Dense(n_actions)              (linear output = Q-values)
  He initialization for ReLU layers, Adam optimizer, global-norm gradient
  clipping per layer (mirrors umut_linear_q's per-layer grad_clip, just
  applied to a deeper stack).

DIFFERENCES from umut_linear_q.py (document in the report):
  * Q(s,a) is no longer linear in the features -- two hidden ReLU layers.
  * Adam instead of a raw SGD step (raw SGD on a multi-layer net with
    this feature scale was unstable in early testing -- gradients
    through two ReLU layers have much higher variance than the single
    linear layer's).
  * Adds a target network (hard-updated every `target_update_every`
    learning steps) to stop the bootstrapped target from chasing the
    same weights being updated -- classic DQN instability that a single
    linear layer mostly gets away without.
  * Optionally Double-DQN style target (`double_q: true` in hparams):
    action is argmax'd with the ONLINE network, evaluated with the
    TARGET network, which reduces the max-operator overestimation bias
    that gets worse as capacity increases.
  * grad_clip now clips each layer's gradient by global L2 norm rather
    than elementwise clipping at +/-5 -- elementwise clipping on a deep
    net tends to just flatten every gradient to the clip value.
"""

from __future__ import annotations

import numpy as np

from . import register


def _he_init(rng, fan_in, fan_out):
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float32)


class _AdamState:
    """Per-parameter Adam moment buffers."""

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


class _MLP:
    """Two-hidden-layer ReLU MLP: n_features -> h1 -> h2 -> n_actions."""

    def __init__(self, n_features, h1, h2, n_actions, rng):
        self.W1 = _he_init(rng, n_features, h1)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = _he_init(rng, h1, h2)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.W3 = _he_init(rng, h2, n_actions)
        self.b3 = np.zeros(n_actions, dtype=np.float32)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = params

    def clone(self):
        other = _MLP.__new__(_MLP)
        other.set_params([p.copy() for p in self.params()])
        return other

    def forward(self, x, cache: dict | None = None):
        """x: (batch, n_features) -> q: (batch, n_actions).

        If `cache` is passed, intermediate activations are stashed in it
        for use by `backward`.
        """
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0.0)
        q = a2 @ self.W3 + self.b3
        if cache is not None:
            cache.update(x=x, z1=z1, a1=a1, z2=z2, a2=a2, q=q)
        return q

    def backward(self, cache: dict, dq: np.ndarray):
        """dq: (batch, n_actions) gradient of loss w.r.t. output.

        Returns grads in the same order as `params()`.
        """
        x, a1, a2 = cache["x"], cache["a1"], cache["a2"]
        z1, z2 = cache["z1"], cache["z2"]
        n = x.shape[0]

        dW3 = a2.T @ dq / n
        db3 = dq.mean(axis=0)

        da2 = dq @ self.W3.T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2 / n
        db2 = dz2.mean(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = x.T @ dz1 / n
        db1 = dz1.mean(axis=0)

        return [dW1, db1, dW2, db2, dW3, db3]


@register("umut_nonlinear_q")
class UmutNonlinearQ:
    def __init__(self, feature_shape, n_actions: int, cfg):
        if len(feature_shape) != 1:
            raise ValueError(
                f"umut_nonlinear_q expects a flat feature vector, "
                f"got {feature_shape}.")
        self.n_features = int(feature_shape[0])
        self.n_actions = int(n_actions)

        self.lr = float(cfg.hp("lr", 0.001))
        self.gamma = float(cfg.gamma)
        self.grad_clip = float(cfg.hp("grad_clip", 5.0))
        self.h1 = int(cfg.hp("hidden1", 64))
        self.h2 = int(cfg.hp("hidden2", 64))
        self.target_update_every = int(cfg.hp("target_update_every", 500))
        self.double_q = bool(cfg.hp("double_q", True))

        rng = np.random.default_rng(cfg.seed)
        self._rng = rng
        self.online = _MLP(self.n_features, self.h1, self.h2,
                           self.n_actions, rng)
        self.target = self.online.clone()
        self._adam = [_AdamState(p.shape) for p in self.online.params()]
        self._learn_steps = 0

    # -- contract ------------------------------------------------------
    def q_values(self, states) -> np.ndarray:
        x = np.atleast_2d(np.asarray(states, dtype=np.float32))
        return self.online.forward(x)

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
        n = len(a)

        # -- bootstrapped target, from the TARGET network -------------
        q_next_target = self.target.forward(xn)
        if self.double_q:
            # online net picks the action, target net evaluates it
            q_next_online = self.online.forward(xn)
            best_a = np.argmax(q_next_online, axis=1)
            boot = q_next_target[np.arange(n), best_a]
        else:
            boot = q_next_target.max(axis=1)
        boot[d] = 0.0
        target = r + self.gamma * boot

        # -- forward pass on current states, cache for backward -------
        cache: dict = {}
        q_pred_all = self.online.forward(x, cache=cache)
        pred = q_pred_all[np.arange(n), a]

        err = pred - target                     # matches umut_linear_q's sign
        td_all = -err                            # report in (target - pred) sign

        # dLoss/dQ is nonzero only at the taken action's output slot;
        # MSE-style: dL/dpred = 2*err (folded into 1.0 via lr elsewhere,
        # keep it simple/consistent with the linear model's plain err use)
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

        return {"td_abs_mean": float(np.abs(td_all).mean()),
                "w_norm": float(sum(np.linalg.norm(p) for p in params)),
                "td_errors": td_all}

    # -- persistence -----------------------------------------------------
    def save(self, path) -> None:
        with open(path, "wb") as fh:
            np.savez(fh, W1=self.online.W1, b1=self.online.b1,
                     W2=self.online.W2, b2=self.online.b2,
                     W3=self.online.W3, b3=self.online.b3)

    def load(self, path) -> None:
        with open(path, "rb") as fh:
            blob = np.load(fh)
            self.online.set_params([
                blob["W1"].astype(np.float32), blob["b1"].astype(np.float32),
                blob["W2"].astype(np.float32), blob["b2"].astype(np.float32),
                blob["W3"].astype(np.float32), blob["b3"].astype(np.float32),
            ])
            self.target = self.online.clone()
            self._adam = [_AdamState(p.shape) for p in self.online.params()]
            self._learn_steps = 0