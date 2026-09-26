"""Umut'un Dueling Double DQN ajani (saf NumPy) -- her zaman yuklenen kisim.

Sahibi: UMUT
"""

import os
from collections import deque
import numpy as np

from bbrl import features as F
from bbrl import models as Mo
from bbrl.config import Config
from bbrl.gamelogic import ACTIONS


ACTION_NAMES = list(ACTIONS)

# GORELI olmak ZORUNDA: framework callback'leri calistirirken cwd'yi bu
# klasore cevirir. Eskiden repo kokundeki configs/umut_dueling.yaml'a mutlak
# yolla bakiyordu; turnuvada yalnizca bu klasor kopyalandigi icin setup()
# cokerdi (PDF s.11: "A common error is the use of absolute paths").
# vendor.py bu satiri "config.json" olarak yeniden yazar -- birebir bu metin
# olmali.
DEFAULT_CONFIG = "config.yaml"


def _config_path():
    # BBRL_CONFIG surecteki TUM ajanlara miras kalir; yalnizca bu klasor
    # hedeflendiyse kullan. Bkz. irmak_umut/callbacks.py::_config_path.
    cfg = os.environ.get("BBRL_CONFIG")
    if cfg:
        target = os.environ.get("BBRL_CONFIG_AGENT")
        me = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        if target is None or target == me:
            return cfg
    return DEFAULT_CONFIG


def _model_path(cfg):
    return f"model_{cfg.name}.bin"


def _progress_path(cfg):
    return f"progress_{cfg.name}.json"


def load_progress(cfg) -> int:
    import json
    try:
        with open(_progress_path(cfg), encoding="utf-8") as fh:
            return int(json.load(fh).get("rounds_done", 0))
    except (OSError, ValueError, KeyError):
        return 0


def setup(self):
    """Ilk turdan once bir kez cagrilir."""
    cfg_file = _config_path()
    cfg = Config.load(cfg_file)
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
        self.logger.warning(
            f"{path} bulunamadi ve egitim modunda degiliz: "
            "egitilmemis model ile oynanacak!")
    else:
        self.logger.info("Egitim modu: model sifirdan olusturuldu.")

    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    try:
        warm = np.zeros(self.featureset.shape, dtype=self.featureset.dtype)
        self.model.act(warm, 0.0)
    except Exception as ex:
        self.logger.debug(f"Isinma cagrisi atlandi: {ex}")

    self.steps_seen = 0
    self.rounds_done = load_progress(cfg)
    if self.rounds_done:
        self.logger.info(f"Onceki asamalardan {self.rounds_done} tur devralindi.")

    self.epsilon = 1.0 if self.train else 0.0
    self.recent_positions = deque(maxlen=4)


def act(self, game_state: dict):
    # Transition cache synchronization: guarantees s_next in buffer is identical to s in act()
    if getattr(self, "_next_features", None) is not None:
        features = self._next_features
        self._next_features = None
    else:
        features = self.featureset(game_state)
        if features is None:
            self.last_features = None
            return "WAIT"
        _n, _sc, _b, (ax, ay) = game_state["self"]
        features = features.copy()
        features[10] = 1.0 if (ax, ay) in self.recent_positions else 0.0

    _n, _sc, _b, (ax, ay) = game_state["self"]
    self.recent_positions.append((ax, ay))
    self.last_features = features          # Cache for train.py to reuse

    eps = self.epsilon if getattr(self, "train", False) else 0.0
    action_idx = self.model.act(features, eps)
    self.steps_seen += 1
    return ACTION_NAMES[int(action_idx)]
