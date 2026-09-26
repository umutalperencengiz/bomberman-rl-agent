"""Turnuvaya gidecek ajan (nonlinear variant) -- her zaman yuklenen kisim.

Takim: Irmak & Umut

Bu dosya callbacks.py (linear ajan) ile MANTIKSAL OLARAK AYNI -- Mo.build()
zaten cfg.model uzerinden hangi modelin insa edilecegine karar veriyor, yani
bu dosyanin tek gercek farki DEFAULT_CONFIG'in umut_nonlinear.yaml'a isaret
etmesi (agent_code/umut_nonlinear_q/ klasorunde durur, agent_code/
umut_linear_q/callbacks.py ile KARISTIRILMAMALI).
"""

import os
from collections import deque
from pathlib import Path
import numpy as np

from bbrl import features as F
from bbrl import models as Mo
from bbrl.config import Config
from bbrl.gamelogic import ACTIONS


ACTION_NAMES = list(ACTIONS)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = str(REPO_ROOT / "configs" / "umut_nonlinear.yaml")


def _config_path():
    return os.environ.get("BBRL_CONFIG", DEFAULT_CONFIG)


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

    # --- EXACT PORT (see callbacks.py / umut_linear_q): bypasses
    # cfg.epsilon()'un lineer semasi, orijinal ustel semaya sadik kalir.
    self.epsilon = 1.0 if self.train else 0.0

    # --- EXACT PORT of old "recently visited tile" feature memory.
    # v1_handcrafted's feature #10 is a hardcoded 0.0 placeholder (pure
    # FeatureSet.fn can't hold cross-step state) -- patched back in here
    # at the agent level, same as the linear agent.
    self.recent_positions = deque(maxlen=4)


def act(self, game_state: dict):
    features = self.featureset(game_state)
    if features is None:
        self.last_features = None
        return "WAIT"

    _n, _sc, _b, (ax, ay) = game_state["self"]
    features = features.copy()
    features[10] = 1.0 if (ax, ay) in self.recent_positions else 0.0
    self.recent_positions.append((ax, ay))

    self.last_features = features          # <-- cache for train.py to reuse

    eps = self.epsilon if getattr(self, "train", False) else 0.0
    action_idx = self.model.act(features, eps)
    self.steps_seen += 1
    return ACTION_NAMES[int(action_idx)]