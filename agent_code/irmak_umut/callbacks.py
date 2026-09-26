"""Turnuvaya gidecek ajan -- her zaman yuklenen kisim.

Takim: Irmak & Umut

DIKKAT -- framework davranislari (docs/GAME_MECHANICS.md):
  * Callback'ler calisirken framework cwd'yi BU klasore cevirir (agents.py
    os.chdir). Yani model dosyasi GORELI yolla acilir ve dogru yerde bulunur.
    Mutlak yol KULLANMA -- Docker submission test'inde patlayan klasik hata.
  * `setup(self)` ve `act(self, game_state)` imzalari BIREBIR bu olmali;
    agents.py argüman SAYISINI kontrol edip hata firlatiyor.
  * Turnuvada tek CPU thread'i ve adim basina 0.5 s var. Zaman asimi
    KUMULATIF ceza getiriyor -> ilk cagrida yavaslamamak icin setup()'ta
    isinma cagrisi yapiyoruz.

TESLIMDEN ONCE: bu dosya `bbrl`'i import ediyor. Turnuvada yalnizca bu klasor
kopyalanacagi icin gerekli bbrl kodunun buraya vendor'lanmasi gerekiyor
(bkz. docs/INTERFACE.md 0). `python -m bbrl.vendor` scripti Persembe
entegrasyon gununde yazilacak.
"""

import os

import numpy as np

from bbrl import features as F
from bbrl import models as Mo
from bbrl.config import Config
from bbrl.gamelogic import ACTIONS, free_mask

#: Aksiyon sirasi DONMUS -- kaydedilmis modeller bu indekslemeye bagli.
#: (bbrl.gamelogic.ACTIONS ile ayni olmak ZORUNDA)
ACTION_NAMES = list(ACTIONS)

DEFAULT_CONFIG = "config.yaml"


def _config_path():
    """Config yolu: BBRL_CONFIG env degiskeni, yoksa klasordeki config.yaml.

    !!! BBRL_CONFIG TUM SURECE AITTIR, TEK AJANA DEGIL !!!
    Ortam degiskeni masadaki HER ajana miras kalir. Egitilen ajan icin set
    edilen config'i bir rakip de okursa, o isimde bir model dosyasini KENDI
    klasorunde arar, bulamaz ve RASTGELE AGIRLIKLARLA oynar -- sadece bir
    WARNING satiri birakarak.

    Bu iki deneyi sessizce gecersiz kildi:
      * EXP-019 e19c_league: havuz ajani `model_e19c_league_s1.bin` aradi
      * Dalga 1 e19d_selfplay: havuz ajani `model_e19d_selfplay_s3.bin` aradi
    Ikisinde de aranan isim egitim KOSUSUNUN adiydi, havuz ajaninin degil.

    Cozum: BBRL_CONFIG_AGENT hangi ajan klasoru icin oldugunu soyler. Set
    edilmisse ve bu klasor degilse, yok say ve kendi config'ini kullan.
    (Set edilmemisse eski davranis -- tek ajanli manuel kosular icin.)
    """
    cfg = os.environ.get("BBRL_CONFIG")
    if cfg:
        target = os.environ.get("BBRL_CONFIG_AGENT")
        me = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        if target is None or target == me:
            return cfg
    return DEFAULT_CONFIG


def _model_path(cfg):
    """Egitilmis parametrelerin yolu -- GORELI, ajan klasorune gore."""
    return f"model_{cfg.name}.bin"


def _progress_path(cfg):
    """Kumulatif egitim ilerlemesi -- asamalar arasi TASINIR."""
    return f"progress_{cfg.name}.json"


def _buffer_path(cfg):
    return f"buffer_{cfg.name}.npz"


def load_progress(cfg) -> int:
    """Bu config ile SIMDIYE KADAR kac tur egitildi.

    !!! NEDEN VAR !!!
    Mufredat asamalari ayri `main.py` surecleridir ve `game_state["round"]`
    her surecte 1'den baslar. Epsilon yalnizca ona bakarsa HER ASAMADA
    kesif 1.0'a sifirlanir. `e18b_mixed_opp` (18 asama, 800-2400 turluk
    bloklar, eps_decay=7200) bu yuzden 24000 turun tamamini %68'in altina
    hic inmeyen kesifle gecirdi -- ajan neredeyse hep rastgele oynadi.
    """
    import json
    try:
        with open(_progress_path(cfg), encoding="utf-8") as fh:
            return int(json.load(fh).get("rounds_done", 0))
    except (OSError, ValueError, KeyError):
        return 0


def setup(self):
    """Ilk turdan once bir kez cagrilir."""
    cfg = Config.load(_config_path())
    cfg.validate()
    self.cfg = cfg

    self.featureset = F.get(cfg.features)
    self.model = Mo.build(cfg.model, self.featureset.shape,
                          len(ACTION_NAMES), cfg)

    path = _model_path(cfg)
    if os.path.isfile(path):
        try:
            self.model.load(path)
            self.logger.info(f"Model yuklendi: {path}")
        except Exception as ex:
            self.logger.warning(f"Model yuklenemedi ({ex}) -- sifirdan.")
    elif not self.train:
        # Degerlendirme/turnuva modunda egitilmis dosya YOKSA sessizce
        # rastgele oynamak en kotu senaryo -- gorunur olsun.
        self.logger.warning(
            f"{path} bulunamadi ve egitim modunda degiliz: "
            "egitilmemis model ile oynanacak!")
    else:
        self.logger.info("Egitim modu: model sifirdan olusturuldu.")

    # torch varsa tek thread'e sabitle (turnuva tek thread, multiprocessing yasak)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    # Isinma: ilk gercek act() cagrisinda gecikme olmasin.
    try:
        warm = np.zeros(self.featureset.shape, dtype=self.featureset.dtype)
        self.model.act(warm, 0.0)
    except Exception as ex:
        self.logger.debug(f"Isinma cagrisi atlandi: {ex}")

    self.steps_seen = 0
    # Onceki asamalarda tamamlanan tur sayisi (epsilon icin).
    self.rounds_done = load_progress(cfg)
    if self.rounds_done:
        self.logger.info(f"Onceki asamalardan {self.rounds_done} tur "
                         f"devralindi (eps={cfg.epsilon(self.rounds_done):.3f}).")


def _legal_mask(game_state: dict) -> np.ndarray:
    """Hangi aksiyonlar cevrede GERCEKTEN bir sey yapar.

    environment.GenericWorld.perform_agent_action ile birebir (satir 132-150):
      - yon: hedef kare tile_is_free ise gecerli
      - BOMB: yalnizca bombs_left dogruysa
      - WAIT: HER ZAMAN gecerli
    Bunlarin disindaki her sey INVALID_ACTION -> adim harcanir, ajan yerinde
    kalir ve -0.10 ceza yer.

    ACTIONS sirasi: UP, RIGHT, DOWN, LEFT, WAIT, BOMB
    """
    free = free_mask(game_state)
    _n, _s, bombs_left, (x, y) = game_state["self"]
    m = np.zeros(len(ACTION_NAMES), dtype=bool)
    for i, (dx, dy) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
        nx, ny = x + dx, y + dy
        if 0 <= nx < free.shape[0] and 0 <= ny < free.shape[1]:
            m[i] = bool(free[nx, ny])
    m[4] = True                      # WAIT her zaman gecerli
    m[5] = bool(bombs_left)
    return m


def act(self, game_state: dict):
    """Her adimda bir kez cagrilir. Turnuvada limit 0.5 s."""
    features = self.featureset(game_state)
    if features is None:
        return "WAIT"

    # Kesif YALNIZCA egitimde. Degerlendirmede eps=0 (greedy) --
    # aksi halde olctugumuz sey politika degil, gurultu olur.
    eps = 0.0
    if getattr(self, "train", False):
        # KUMULATIF tur: onceki asamalar + bu surecteki tur.
        eps = self.cfg.epsilon(
            self.rounds_done + int(game_state.get("round", 1)) - 1)

    mask = _legal_mask(game_state) if self.cfg.action_mask else None
    action_idx = self.model.act(features, eps, mask=mask)
    self.steps_seen += 1
    return ACTION_NAMES[int(action_idx)]
