"""Replay buffer -- uniform ve prioritized.

Sahibi: IRMAK (infra)

Transition sozlesmesi (docs/INTERFACE.md 3) -- DEGISMEZ:
    Transition(state, action, next_state, reward, done)
    state/next_state : np.ndarray | None   (None = terminal)
    action           : int   (gamelogic.ACTIONS indeksi)
    reward           : float
    done             : bool

!!! OLUM TRANSITION'I: agents.py, ajan olduğu adimda `game_events_occurred`
CAGIRMAZ -- o transition yalnizca `end_of_round`'a gelir. Olum cezasini
tasiyan TEK transition odur; train.py'de mutlaka buraya yazilmali.
Bkz. docs/GAME_MECHANICS.md 6 (tuzak 3) ve EXP-005.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

Transition = namedtuple("Transition",
                        ["state", "action", "next_state", "reward", "done"])


class ReplayBuffer:
    """Sabit kapasiteli uniform replay buffer (numpy ring buffer).

    State'ler onceden ayrilmis dizilerde tutulur -> python listesi tutmaya
    gore hem hizli hem bellek-ongorulebilir. `state_shape` FeatureSet.shape.
    """

    def __init__(self, capacity: int, state_shape: tuple[int, ...],
                 dtype=np.float32, seed: int = 0):
        self.capacity = int(capacity)
        self.state_shape = tuple(state_shape)
        self.states = np.zeros((self.capacity, *self.state_shape), dtype=dtype)
        self.next_states = np.zeros_like(self.states)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=bool)
        # Her transition'in KAC ADIMLIK oldugu. 1 = klasik. n-step'te
        # hedefte gamma**n kullanilmali; bolum sonunda n kisalabilir.
        self.nsteps = np.ones(self.capacity, dtype=np.int8)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def push(self, state, action: int, next_state, reward: float,
             done: bool, n_step: int = 1) -> None:
        if state is None:
            return                       # tur basi -- ogrenilecek bir sey yok
        i = self.pos
        self.nsteps[i] = max(1, min(127, int(n_step)))
        self.states[i] = state
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.dones[i] = bool(done)
        # terminalde next_state anlamsiz; sifirla (maskeleme dones ile yapilir)
        self.next_states[i] = 0.0 if next_state is None else next_state
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_transition(self, t: Transition) -> None:
        self.push(t.state, t.action, t.next_state, t.reward, t.done)

    def sample(self, batch_size: int):
        """(states, actions, rewards, next_states, dones, idx, weights)."""
        if self.size == 0:
            raise ValueError("bos buffer'dan ornekleme")
        n = min(batch_size, self.size)
        idx = self.rng.integers(0, self.size, size=n)
        w = np.ones(n, dtype=np.float32)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx], idx, w,
                self.nsteps[idx])

    def update_priorities(self, idx, td_errors) -> None:
        """Uniform buffer'da no-op -- arayuz PER ile ayni kalsin diye var."""

    # -- kalicilik ---------------------------------------------------------
    #
    # NEDEN GEREKLI: mufredat asamalari AYRI SURECLER. Buffer diske
    # yazilmazsa her asamada bosalir ve ajan biriktirdigi deneyimi kaybeder.
    # (Bu, cok-asamali egitimi bozan bes seyden biriydi.)

    def save(self, path) -> None:
        with open(path, "wb") as fh:
            np.savez_compressed(
                fh, states=self.states[:self.size],
                next_states=self.next_states[:self.size],
                actions=self.actions[:self.size],
                rewards=self.rewards[:self.size],
                dones=self.dones[:self.size], nsteps=self.nsteps[:self.size],
                pos=np.int64(self.pos), size=np.int64(self.size))

    def load(self, path) -> bool:
        """Diskteki buffer'i geri yukler. Uyusmazlikta False doner, patlamaz."""
        try:
            with open(path, "rb") as fh:
                blob = np.load(fh)
                n = int(blob["size"])
                if n == 0:
                    return False
                if blob["states"].shape[1:] != self.state_shape:
                    return False          # feature seti degismis -> yok say
                n = min(n, self.capacity)
                self.states[:n] = blob["states"][:n]
                self.next_states[:n] = blob["next_states"][:n]
                self.actions[:n] = blob["actions"][:n]
                self.rewards[:n] = blob["rewards"][:n]
                self.dones[:n] = blob["dones"][:n]
                if "nsteps" in blob:
                    self.nsteps[:n] = blob["nsteps"][:n]
                self.size = n
                self.pos = n % self.capacity
        except (OSError, KeyError, ValueError):
            return False
        self._after_load()
        return True

    def _after_load(self) -> None:
        """Alt siniflar icin kanca (PER sum-tree'yi yeniden kurar)."""


class SumTree:
    """Prioritized replay icin toplam agaci. O(log n) ornekleme/guncelleme."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def set(self, i: int, value: float) -> None:
        j = i + self.capacity
        self.tree[j] = value
        j //= 2
        while j >= 1:
            self.tree[j] = self.tree[2 * j] + self.tree[2 * j + 1]
            j //= 2

    def get(self, i: int) -> float:
        return float(self.tree[i + self.capacity])

    def find(self, prefix: float) -> int:
        """`prefix` toplamina karsilik gelen yaprak indeksi."""
        j = 1
        while j < self.capacity:
            left = 2 * j
            if prefix <= self.tree[left]:
                j = left
            else:
                prefix -= self.tree[left]
                j = left + 1
        return j - self.capacity


class PrioritizedReplayBuffer(ReplayBuffer):
    """Schaul et al. 2016 -- proportional prioritization.

    p_i = (|TD_i| + eps)^alpha ,  ornekleme olasiligi p_i / sum(p)
    Onem agirligi w_i = (N * P_i)^(-beta) / max(w)  -> yanliligi duzeltir.

    beta egitim boyunca beta0 -> 1.0 arttirilir (`anneal_beta`).
    """

    def __init__(self, capacity: int, state_shape, dtype=np.float32,
                 seed: int = 0, alpha: float = 0.6, beta0: float = 0.4,
                 eps: float = 1e-3):
        super().__init__(capacity, state_shape, dtype, seed)
        self.alpha = float(alpha)
        self.beta = float(beta0)
        self.beta0 = float(beta0)
        self.eps = float(eps)
        self.tree = SumTree(self.capacity)
        self.max_p = 1.0

    def push(self, state, action, next_state, reward, done,
             n_step: int = 1) -> None:
        if state is None:
            return
        i = self.pos
        super().push(state, action, next_state, reward, done, n_step)
        # yeni ornekler en yuksek oncelikle girer -> en az bir kez gorulur
        self.tree.set(i, self.max_p ** self.alpha)

    def sample(self, batch_size: int):
        if self.size == 0:
            raise ValueError("bos buffer'dan ornekleme")
        n = min(batch_size, self.size)
        total = self.tree.total
        if total <= 0:
            return super().sample(batch_size)
        # stratified: araligi n dilime bol, her dilimden bir ornek
        edges = np.linspace(0.0, total, n + 1)
        u = self.rng.uniform(edges[:-1], edges[1:])
        idx = np.array([self.tree.find(x) for x in u], dtype=np.int64)
        idx = np.clip(idx, 0, self.size - 1)

        p = np.array([self.tree.get(int(i)) for i in idx], dtype=np.float64)
        probs = np.maximum(p, 1e-12) / total
        w = (self.size * probs) ** (-self.beta)
        w = (w / w.max()).astype(np.float32)

        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx], idx, w,
                self.nsteps[idx])

    def update_priorities(self, idx, td_errors) -> None:
        td = np.abs(np.asarray(td_errors, dtype=np.float64)) + self.eps
        for i, t in zip(np.asarray(idx).ravel(), td.ravel()):
            self.tree.set(int(i), float(t) ** self.alpha)
        self.max_p = max(self.max_p, float(td.max()))

    def anneal_beta(self, frac: float) -> None:
        """frac in [0,1] -- egitimin ne kadari bitti."""
        self.beta = self.beta0 + (1.0 - self.beta0) * float(np.clip(frac, 0, 1))

    def _after_load(self) -> None:
        """Diskten yuklendikten sonra sum-tree'yi yeniden kur.

        Oncelikler kaydedilmiyor; hepsi max_p ile baslatilir, boylece her
        ornek en az bir kez gorulur ve TD hatasi tazelenir.
        """
        self.tree = SumTree(self.capacity)
        p = self.max_p ** self.alpha
        for i in range(self.size):
            self.tree.set(i, p)


def make_buffer(kind: str, capacity: int, state_shape, **kw) -> ReplayBuffer:
    if kind in ("uniform", "replay", None):
        return ReplayBuffer(capacity, state_shape,
                            seed=kw.get("seed", 0))
    if kind in ("prioritized", "per"):
        return PrioritizedReplayBuffer(
            capacity, state_shape, seed=kw.get("seed", 0),
            alpha=kw.get("alpha", 0.6), beta0=kw.get("beta0", 0.4))
    raise KeyError(f"bilinmeyen buffer turu {kind!r}")
