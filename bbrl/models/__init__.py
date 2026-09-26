"""Model zoo -- registry + ortak sozlesme.

Sahibi: IRMAK

Sozlesme (docs/INTERFACE.md 4). Her model:

    m = Model(feature_shape, n_actions, cfg)
    a = m.act(features, eps)          -> int   (eps=0 -> greedy, degerlendirme)
    d = m.update(batch)               -> dict  (loglanacak skalerler + td_errors)
    m.save(path) / m.load(path)

`batch` = ReplayBuffer.sample() ciktisi:
    (states, actions, rewards, next_states, dones, idx, weights)

`update` DAIMA dict doner (bos olabilir) -> logger tek kod yolunda calisir.
PER kullaniliyorsa dict'te "td_errors" bulunmali.
"""

from __future__ import annotations

REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        if name in REGISTRY:
            raise KeyError(f"model {name!r} zaten kayitli")
        REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return deco


def get(name: str) -> type:
    if name not in REGISTRY:
        # dqn torch gerektiriyor -> ancak istenirse import et
        if name in ("dqn", "cnn_dqn"):
            try:
                from . import dqn as _dqn  # noqa: F401
            except Exception as ex:  # noqa: BLE001
                raise KeyError(
                    f"model {name!r} yuklenemedi ({type(ex).__name__}: {ex}). "
                    "torch kurulu mu?") from ex
    if name not in REGISTRY:
        raise KeyError(f"bilinmeyen model {name!r}. Mevcut: {sorted(REGISTRY)}")
    return REGISTRY[name]


def build(name: str, feature_shape, n_actions: int, cfg):
    return get(name)(feature_shape, n_actions, cfg)


def _pick(rng, qs, mask, n_actions, eps):
    """eps-greedy secim, istege bagli GECERLILIK MASKESI ile.

    `mask` bool dizisi (n_actions,): True = cevrede gecerli aksiyon.
    Maske verilirse hem rastgele kol hem argmax YALNIZCA gecerliler
    uzerinden secer. Boylece ajan duvara carpip adim harcamaz.

    Maske tumuyle False olamaz (WAIT her zaman gecerlidir), ama savunma
    amacli yine de kontrol ediliyor -- bos maskede maskesiz davraniriz.
    """
    import numpy as _np
    if mask is not None:
        legal = _np.flatnonzero(_np.asarray(mask, dtype=bool))
        if legal.size:
            if eps > 0 and rng.random() < eps:
                return int(rng.choice(legal))
            sub = _np.asarray(qs)[legal]
            best = legal[_np.flatnonzero(sub == sub.max())]
            return int(rng.choice(best))
    if eps > 0 and rng.random() < eps:
        return int(rng.integers(n_actions))
    qs = _np.asarray(qs)
    best = _np.flatnonzero(qs == qs.max())
    return int(rng.choice(best))


# kayit icin import (torch gerektirmeyenler)
from . import linear_q, tabular_q, umut_linear_q , umut_nonlinear_q, umut_dueling_q  # noqa: E402,F401
