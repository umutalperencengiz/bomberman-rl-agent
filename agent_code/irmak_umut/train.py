"""Yalnizca egitim modunda yuklenen kisim.

Takim: Irmak & Umut

!!! EN KRITIK NOKTA -- OLUM TRANSITION'I !!!
environment.send_game_events yalnizca `if a.train and not a.dead` icin
`game_events_occurred` cagirir. Yani ajanin OLDUGU adimin transition'i
buraya HIC gelmez; `end_of_round`'a gelir. Olum cezasini tasiyan TEK
deneyim odur -- orada buffer'a yazilmazsa ajan olmemeyi ASLA ogrenemez.
(docs/GAME_MECHANICS.md 6, tuzak 3; EXP-005 bunu olcuyor.)

Bu dosyanin zaman limiti YOK -- ogrenme adimini rahatca burada yapabiliriz.
"""

import json
import os
import time
from collections import defaultdict, deque

import numpy as np

from bbrl import features as F
from bbrl import rewards as R
from bbrl.buffer import make_buffer
from bbrl.gamelogic import ACTIONS

ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}


def setup_training(self):
    """callbacks.setup'tan SONRA cagrilir."""
    cfg = self.cfg
    self.buffer = make_buffer(cfg.buffer, cfg.buffer_size,
                              self.featureset.shape, seed=cfg.seed)
    # Buffer'i asamalar arasi tasi -- yoksa her asamada bosalir.
    from .callbacks import _buffer_path
    if self.buffer.load(_buffer_path(cfg)):
        self.logger.info(f"Buffer devralindi: {len(self.buffer)} transition")
    self.train_step = 0
    # Kumulatif devam: onceki asamalarin turlari sayilmis olarak baslar.
    self.round_idx = int(getattr(self, "rounds_done", 0))
    self._round_idx_at_start = self.round_idx
    self.t0 = time.time()
    # Duvar saati laptop UYKUYA girince askida gecen sureyi de sayiyor
    # (bir kosuda 3.58 saatlik hayali "egitim suresi" olustu). process_time
    # yalnizca CPU'da gecen sureyi olcer -> rapordaki egitim suresi iddialari
    # icin dogru olan bu.
    self.cpu0 = time.process_time()

    # Tur bazli olcumler -- egitim ilerleme grafiginin ham verisi
    self.stat_log = []
    self._round = defaultdict(float)

    # Son islenen adim no -- cift sayimi engellemek icin (bkz. end_of_round).
    self._last_processed_step = None
    # Adimlar arasi hafiza (SAFE_AFTER_BOMB, BOMB_IDLE) -- tur basi sifirlanir.
    self._mem = {}

    # n-step return biriktirici. n>1 iken krediyi n adim geriye tasir --
    # bizde rakip oldurme 5 puan ve COK seyrek, tek adimlik bootstrap onu
    # yavas yayiyor.
    self.n_step = max(1, int(getattr(cfg, "n_step", 1)))
    self._nq = deque(maxlen=self.n_step)

    # Simetri augmentation: duzlem (C,W,H) VEYA dir_slots tanimlayan duz
    # vektor. EXP-012'de simetri CNN'i 3x'lemisti ama en iyi modelimiz olan
    # MLP duz vektor kullandigi icin o kazanctan faydalanamiyordu.
    self.use_symmetry = bool(cfg.symmetry) and F.supports_symmetry(self.featureset)
    if cfg.symmetry and not self.use_symmetry:
        self.logger.warning(
            f"symmetry=true ama {self.featureset.name} donusumu desteklemiyor "
            "(duzlem degil ve dir_slots tanimli degil) -> augmentation kapali.")

    self.logger.info(
        f"Egitim baslıyor | config={cfg.name} model={cfg.model} "
        f"features={cfg.features} rewards={cfg.rewards} "
        f"buffer={cfg.buffer}({cfg.buffer_size}) symmetry={self.use_symmetry}")


def _push_raw(self, s, a_idx, s_next, reward, done, n_step=1):
    """Tek transition'i (ve istenirse 8 simetrik kopyasini) buffer'a yaz."""
    if s is None:
        return
    self.buffer.push(s, a_idx, s_next, reward, done, n_step)
    if not self.use_symmetry:
        return
    for k in range(4):
        for flip in (False, True):
            if k == 0 and not flip:
                continue                       # orijinali iki kez yazma
            self.buffer.push(
                F.transform_features(s, k, flip, self.featureset),
                F.transform_action(a_idx, k, flip),
                None if s_next is None else
                F.transform_features(s_next, k, flip, self.featureset),
                reward, done, n_step)


def _push(self, s, a_idx, s_next, reward, done):
    """n-step biriktirici uzerinden yaz.

    n=1 iken davranis eskisiyle AYNI. n>1 iken en eski transition,
    biriken indirimli odul ve n adim sonraki state ile yazilir:
        R = r_t + g*r_{t+1} + ... + g^(n-1)*r_{t+n-1}
    Bolum bittiginde kuyrukta kalan her sey KISALMIS n ile bosaltilir --
    aksi halde olum transition'lari (tek olum cezasi tasiyanlar) kaybolur.
    """
    if s is None:
        return
    if self.n_step <= 1:
        _push_raw(self, s, a_idx, s_next, reward, done, 1)
        return

    self._nq.append((s, a_idx, reward))
    g = self.cfg.gamma
    if len(self._nq) == self.n_step:
        s0, a0, _ = self._nq[0]
        R = sum((g ** i) * tr[2] for i, tr in enumerate(self._nq))
        _push_raw(self, s0, a0, s_next, R, done, self.n_step)
        self._nq.popleft()

    if done:                       # kuyrugu kisalmis n ile bosalt
        while self._nq:
            s0, a0, _ = self._nq[0]
            R = sum((g ** i) * tr[2] for i, tr in enumerate(self._nq))
            _push_raw(self, s0, a0, None, R, True, len(self._nq))
            self._nq.popleft()


def _learn(self):
    """Bir guncelleme adimi. PER kullaniliyorsa oncelikleri de tazeler."""
    cfg = self.cfg
    if len(self.buffer) < max(cfg.batch_size, 100):
        return
    batch = self.buffer.sample(cfg.batch_size)
    out = self.model.update(batch)
    td = out.pop("td_errors", None)
    if td is not None:
        self.buffer.update_priorities(batch[5], td)
    for k, v in out.items():
        self._round[f"m_{k}"] = float(v)


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events):
    """Her adimdan sonra (SON adim ve olum adimi HARIC) cagrilir."""
    if old_game_state is None or self_action is None:
        return

    extra = R.custom_events(old_game_state, self_action, new_game_state,
                            events, self._mem)
    all_events = list(events) + extra

    reward = R.reward_from_events(all_events, self.cfg.rewards)
    # Potansiyel-tabanli shaping: F = gamma*Phi(s') - Phi(s). Ileri-geri
    # gidiste teleskopik sifirlanir -> salinim kazanc saglamaz.
    reward += R.shaping_reward(old_game_state, new_game_state,
                               self.cfg.shaping, self.cfg.shaping_gamma)

    s = self.featureset(old_game_state)
    s_next = self.featureset(new_game_state)
    _push(self, s, ACTION_INDEX[self_action], s_next, reward, False)

    # tur istatistikleri
    self._round["steps"] += 1
    self._round["reward"] += reward
    self._round["game_score"] += R.game_score_from_events(all_events)
    for ev in all_events:
        self._round[f"e_{ev}"] += 1

    self._last_processed_step = int(old_game_state.get("step", -1))

    self.train_step += 1
    if self.train_step % max(1, self.cfg.train_every) == 0:
        _learn(self)


def end_of_round(self, last_game_state: dict, last_action: str, events):
    """Tur sonunda -- VE ajan oldugunde -- bir kez cagrilir.

    Olum transition'i BURADA yakalanir. done=True, next_state=None.

    !!! CIFT SAYIM TUZAGI !!!
    Ajan turu HAYATTA bitirirse (400 adim / yapacak is kalmadi), o son adim
    icin `game_events_occurred` ZATEN cagrilmistir; framework `a.events`'i
    arada sifirlamadigi icin ayni olaylar buraya bir kez daha gelir.
    Onlem almazsak:
      * skor/odul istatistikleri son adimda iki kez sayilir
        (coin-heaven'da tavan 50 iken 51 gorulmesinin sebebi buydu),
      * ayni transition buffer'a hem done=False hem done=True olarak
        yazilir -> ogrenmeye CELISKILI iki hedef gider.

    Ayirt etme: ajan OLDUYSE o adim icin `game_events_occurred` hic
    cagrilmamistir, dolayisiyla adim numarasi eslesmez.
      eslesmiyor -> ajan oldu      -> terminal transition YAZ (done=True)
      eslesiyor   -> ajan hayatta  -> zaten yazildi (done=False, truncation
                                      semantigi dogrusu da budur) -> tekrar yazma
    """
    step_no = (int(last_game_state.get("step", -1))
               if last_game_state is not None else -1)
    already = (self._last_processed_step is not None
               and step_no == self._last_processed_step)

    if already:
        # Bu adim zaten islendi. Yalnizca SADECE burada uretilen olaylari
        # (SURVIVED_ROUND) istatistige ekle; transition TEKRAR YAZILMAZ.
        self._round["e_SURVIVED_ROUND"] += sum(
            1 for ev in events if ev == "SURVIVED_ROUND")
    elif last_game_state is not None and last_action is not None:
        extra = R.custom_events(last_game_state, last_action, None,
                                events, self._mem)
        all_events = list(events) + extra
        reward = R.reward_from_events(all_events, self.cfg.rewards)
        # Terminalde Phi(s') = 0 (Ng et al.'in sarti) -- shaping_reward
        # new_state=None icin bunu zaten uyguluyor.
        reward += R.shaping_reward(last_game_state, None,
                                   self.cfg.shaping, self.cfg.shaping_gamma)

        s = self.featureset(last_game_state)
        _push(self, s, ACTION_INDEX[last_action], None, reward, True)

        self._round["steps"] += 1
        self._round["reward"] += reward
        self._round["game_score"] += R.game_score_from_events(all_events)
        for ev in all_events:
            self._round[f"e_{ev}"] += 1

    # tur sonunda birkac ekstra guncelleme -- zaman limiti yok
    for _ in range(int(self.cfg.hp("updates_per_round", 4))):
        _learn(self)

    self.round_idx += 1
    if hasattr(self.buffer, "anneal_beta"):
        self.buffer.anneal_beta(self.round_idx / max(1, self.cfg.total_rounds))

    rec = {"round": self.round_idx,
           "eps": round(self.cfg.epsilon(self.round_idx - 1), 4),
           "buffer": len(self.buffer),
           "elapsed_s": round(time.time() - self.t0, 1),      # duvar saati
           "cpu_s": round(time.process_time() - self.cpu0, 1),  # uykudan etkilenmez
           **{k: round(v, 4) for k, v in self._round.items()}}
    self.stat_log.append(rec)
    self._round = defaultdict(float)
    self._last_processed_step = None        # yeni tur -> sayaci sifirla
    self._mem = {}                          # tur ici hafizayi temizle
    self._nq.clear()                        # n-step kuyrugu tur icinde kalir

    save_every = int(self.cfg.hp("save_every_rounds", 200))
    if self.round_idx % save_every == 0 or self.round_idx >= self.cfg.total_rounds:
        _save(self)

    if self.round_idx % max(1, save_every // 4) == 0:
        last = self.stat_log[-1]
        self.logger.info(
            f"tur {self.round_idx} eps={last['eps']} "
            f"skor={last.get('game_score', 0)} odul={last.get('reward', 0):.1f} "
            f"buffer={last['buffer']}")


def _save(self):
    """Model + ilerleme + buffer + egitim gunlugu. GORELI yollar."""
    from .callbacks import _buffer_path, _model_path, _progress_path
    path = _model_path(self.cfg)
    self.model.save(path)

    # Kumulatif tur sayaci -- bir sonraki asama buradan devam eder.
    with open(_progress_path(self.cfg), "w", encoding="utf-8") as fh:
        json.dump({"rounds_done": self.round_idx}, fh)

    try:
        self.buffer.save(_buffer_path(self.cfg))
    except Exception as ex:            # buffer buyukse disk dolabilir
        self.logger.warning(f"Buffer kaydedilemedi: {ex}")

    # Gunlugu EKLE, uzerine yazma: aksi halde yalnizca son asama kalir.
    log_path = f"trainlog_{self.cfg.name}.json"
    old = []
    try:
        with open(log_path, encoding="utf-8") as fh:
            old = json.load(fh).get("rounds", [])
    except (OSError, ValueError):
        pass
    seen = {r.get("round") for r in old}
    merged = old + [r for r in self.stat_log if r.get("round") not in seen]
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump({"config": self.cfg.name,
                   "fingerprint": self.cfg.fingerprint(),
                   "rounds": merged}, fh)
    self.logger.info(f"Kaydedildi: {path} (kumulatif {self.round_idx} tur, "
                     f"buffer {len(self.buffer)})")
