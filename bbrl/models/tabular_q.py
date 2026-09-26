"""Tablosal Q-learning -- en basit beyin.

Sahibi: IRMAK

Fikir: her (durum, aksiyon) cifti icin bir sayi tut. "Bu durumda bu hamlenin
degeri ne kadar?" Ogrenme kurali klasik Q-learning:

    Q(s,a) <- Q(s,a) + lr * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Kosede duran ama COK degerli bir model:
  * dersten teknik  -> proje "en az bir model dersten olsun" sartini karsilar
  * laptopta saniyeler icinde egitilir
  * CPU'da sozluk aramasi kadar hizli -> 0.5 s limiti hic sorun degil
  * DQN yakinsamazsa turnuvaya giden GUVENLI TABAN budur

Sinir: durum uzayi ayrik ve kucuk olmali. v0_minimal gibi ikili/one-hot
feature'larla calisir; surekli degerlerde patlar. Bu yuzden `discretize`
esikleme yapar ve tablo boyutu loglanir (patlamayi erken gorelim diye).
"""

from __future__ import annotations

import pickle
from collections import defaultdict

import numpy as np

from . import register, _pick


@register("tabular_q")
class TabularQ:
    def __init__(self, feature_shape, n_actions: int, cfg):
        if len(feature_shape) != 1:
            raise ValueError(
                f"tabular_q duz feature vektoru bekler, {feature_shape} aldi. "
                "CNN duzlemleri (v2_planes) icin 'dqn' kullan.")
        self.n_actions = int(n_actions)
        self.lr = float(cfg.hp("lr", 0.1))
        self.gamma = float(cfg.gamma)
        self.optimistic = float(cfg.hp("optimistic_init", 0.0))
        self.q: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.full(self.n_actions, self.optimistic, dtype=np.float32))
        self._rng = np.random.default_rng(cfg.seed)

    # -- durum anahtari ----------------------------------------------------
    @staticmethod
    def key(features) -> tuple:
        """Feature vektorunu sozluk anahtarina cevir.

        v0_minimal zaten 0/1 degerli; yine de guvenlik icin esikliyoruz ki
        surekli bir feature eklendiginde tablo sessizce patlamasin.
        """
        return tuple(np.asarray(features, dtype=np.float32).round(2).tolist())

    # -- sozlesme ----------------------------------------------------------
    def act(self, features, eps: float = 0.0, mask=None) -> int:
        qs = self.q[self.key(features)]
        # esitlikte rastgele -> bias yok
        return _pick(self._rng, qs, mask, self.n_actions, eps)

    def update(self, batch) -> dict:
        states, actions, rewards, next_states, dones, _idx, weights = batch[:7]
        td_errors = np.zeros(len(actions), dtype=np.float32)
        for i in range(len(actions)):
            k = self.key(states[i])
            a = int(actions[i])
            target = float(rewards[i])
            if not dones[i]:
                target += self.gamma * float(self.q[self.key(next_states[i])].max())
            td = target - float(self.q[k][a])
            self.q[k][a] += self.lr * float(weights[i]) * td
            td_errors[i] = td
        return {"td_abs_mean": float(np.abs(td_errors).mean()),
                "table_size": float(len(self.q)),
                "td_errors": td_errors}

    # -- kalicilik ---------------------------------------------------------
    def save(self, path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"q": dict(self.q), "n_actions": self.n_actions,
                         "optimistic": self.optimistic}, fh)

    def load(self, path) -> None:
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        self.n_actions = blob["n_actions"]
        self.optimistic = blob.get("optimistic", 0.0)
        self.q = defaultdict(
            lambda: np.full(self.n_actions, self.optimistic, dtype=np.float32))
        self.q.update(blob["q"])
