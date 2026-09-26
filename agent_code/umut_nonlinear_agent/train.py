"""train.py (nonlinear variant).

Logically identical to the linear agent's train.py -- _learn() just calls
self.model.update(batch), and self.model here is whatever Mo.build(cfg.model,
...) constructed (umut_nonlinear_q, per umut_nonlinear.yaml). The only
concrete difference from the linear file is the _save() import pointing at
this folder's callbacks_nonlinear.py instead of callbacks.py.
"""

from collections import defaultdict, deque
import json
import time
import numpy as np

from bbrl import features as F
from bbrl import rewards as R
from bbrl.buffer import make_buffer
from bbrl.gamelogic import ACTIONS

ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}

# --- EXACT PORT of old epsilon schedule constants ---
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.999


def setup_training(self):
    cfg = self.cfg
    self.buffer = make_buffer(cfg.buffer, cfg.buffer_size, self.featureset.shape, seed=cfg.seed)
    self.train_step = 0
    self.round_idx = int(getattr(self, "rounds_done", 0))
    self._last_processed_step = None
    self._mem = {}

    # --- EXACT PORT: reset epsilon to 1.0 at the start of every training
    # run, exactly like the linear agent's setup_training().
    self.epsilon = 1.0
    self.epsilon_min = EPSILON_MIN
    self.epsilon_decay = EPSILON_DECAY



def _learn(self):
    cfg = self.cfg
    if len(self.buffer) < cfg.batch_size:
        return
    batch = self.buffer.sample(cfg.batch_size)
    self.model.update(batch)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events):
    if old_game_state is None or self_action is None:
        return

    extra = R.custom_events(old_game_state, self_action, new_game_state, events, self._mem)
    extra += R.umut_custom_events(old_game_state, self_action, new_game_state, events, self._mem)
    all_events = list(events) + extra

    reward = R.reward_from_events(all_events, self.cfg.rewards)

    s = self.last_features                 # <-- reuse instead of recomputing
    s_next = self.featureset(new_game_state)
    if s_next is not None and new_game_state is not None:
        s_next = s_next.copy()
        _n2, _sc2, _b2, (nx, ny) = new_game_state["self"]
        s_next[10] = 1.0 if (nx, ny) in self.recent_positions else 0.0

    self.buffer.push(s, ACTION_INDEX[self_action], s_next, reward, False, 1)
    self._last_processed_step = int(old_game_state.get("step", -1))

    self.train_step += 1
    if self.train_step % max(1, self.cfg.train_every) == 0:
        _learn(self)


def end_of_round(self, last_game_state: dict, last_action: str, events):
    step_no = int(last_game_state.get("step", -1)) if last_game_state is not None else -1
    already = (self._last_processed_step is not None and step_no == self._last_processed_step)

    if not already and last_game_state is not None and last_action is not None:
        extra = R.custom_events(last_game_state, last_action, None, events, self._mem)
        extra += R.umut_custom_events(last_game_state, last_action, None, events, self._mem)
        all_events = list(events) + extra
        reward = R.reward_from_events(all_events, self.cfg.rewards)
        s = self.last_features             # <-- reuse
        self.buffer.push(s, ACTION_INDEX[last_action], None, reward, True, 1)

    _learn(self)
    self.round_idx += 1
    self._last_processed_step = None
    self._mem.clear()
    self.recent_positions.clear()          # EXACT PORT: cleared every round

    # --- EXACT PORT of old epsilon decay: max(min, eps * decay) per round.
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    save_every = int(self.cfg.hp("save_every_rounds", 1000))
    if self.round_idx % save_every == 0 or self.round_idx >= self.cfg.total_rounds:
        _save(self)


def _save(self):
    from .callbacks import _model_path, _progress_path
    self.model.save(_model_path(self.cfg))
    with open(_progress_path(self.cfg), "w", encoding="utf-8") as fh:
        json.dump({"rounds_done": self.round_idx}, fh)