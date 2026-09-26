"""Gecerlilik maskesi testleri.

Sahibi: IRMAK

NEDEN VAR: gonderilen ajanin olculen `invalid_action_rate` degerleri
EK-1 0.121, EK-5 0.124, EK-3 **0.297**. Yani coin-heaven'da 400 adimin
~120'si duvara carpmakla geciyor -- ajan yerinde kaliyor, adim yaniyor ve
-0.10 ceza aliyor.

Maske, environment.GenericWorld.perform_agent_action (satir 132-150) ile
BIREBIR ayni gecerlilik tanimini kullanmali. Sapma olursa maske ya gecerli
bir hamleyi yasaklar (politikayi bozar) ya da gecersizi gecirir (ise
yaramaz). Bu yuzden test taniмi kopyalamiyor, GERCEK ORTAMA karsi dogruluyor.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import settings as s                                   # noqa: E402
from environment import BombeRLeWorld, WorldArgs        # noqa: E402

sys.path.insert(0, str(REPO / "agent_code" / "irmak_umut"))
from callbacks import _legal_mask, ACTION_NAMES         # noqa: E402
from bbrl.models import _pick                           # noqa: E402


_LOGDIR = None


def _logdir():
    global _LOGDIR
    if _LOGDIR is None:
        _LOGDIR = tempfile.mkdtemp(prefix="bbrl_mask_")
    return _LOGDIR


def _make_world(scenario="classic", seed=0, agents=("random_agent",) * 3):
    args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False,
        continue_without_training=True, log_dir=_logdir(), save_stats=False,
        match_name=None, seed=seed, silence_errors=False, scenario=scenario,
    )
    return BombeRLeWorld(args, [(a, False) for a in agents])


def _states(scenario, seed, steps):
    """Gercek oyundan (dunya, ajan, state) uclusu URET -- tembel.

    !!! GENERATOR OLMAK ZORUNDA !!!
    Liste dondurursek cagiran taraf tuketirken `do_step` coktan calismis
    olur; `agent.x/y` ve `world.tile_is_free` ILERLEMIS dunyayi gosterir,
    elimizdeki `st` ise eski adima aittir. Ilk yazimda oyleydi ve 1430
    aksiyonun 625'i "uyusmuyor" cikti -- maskede degil, testte hata.
    """
    world = _make_world(scenario=scenario, seed=seed)
    world.user_input = None
    world.new_round()
    for _ in range(steps):
        if not world.running:
            break
        for a in list(world.active_agents):
            st = world.get_state_for_agent(a)
            if st is not None:
                yield world, a, st
        world.do_step("WAIT")


def test_mask_matches_environment(scenarios=("classic", "coin-heaven"),
                                  seeds=(0, 1, 2), steps=60):
    """Maske, ortamin gecerlilik tanimiyla BIREBIR ayni olmali."""
    DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
    checked = mism = 0
    for scen in scenarios:
        for sd in seeds:
            for world, agent, st in _states(scen, sd, steps):
                m = _legal_mask(st)
                x, y = agent.x, agent.y

                for i, (dx, dy) in enumerate(DELTAS):
                    want = bool(world.tile_is_free(x + dx, y + dy))
                    if bool(m[i]) != want:
                        mism += 1
                    checked += 1

                # BOMB yalnizca bombs_left dogruysa
                if bool(m[5]) != bool(agent.bombs_left):
                    mism += 1
                checked += 1

                # WAIT her zaman gecerli (environment satir 147)
                assert m[4], "WAIT maskede kapali -- ortamda her zaman gecerli"
                assert m.any(), "maske tumuyle False -- imkansiz"

    assert mism == 0, f"{mism}/{checked} aksiyonda ortamla uyusmazlik"
    print(f"  ok  ortam paritesi     {checked} aksiyon, 0 uyusmazlik")


def test_masked_choice_never_illegal(seed=0, steps=120):
    """Maskeli secim ASLA gecersiz aksiyon dondurmemeli (eps>0 dahil).

    eps kolu da maskeden gecmeli: egitimde rastgele aksiyon secilirken
    duvara girilirse adim yine yanar.
    """
    rng = np.random.default_rng(seed)
    n = len(ACTION_NAMES)
    bad_greedy = bad_eps = total = 0
    for world, agent, st in _states("classic", seed, steps):
        m = _legal_mask(st)
        q = rng.normal(size=n)
        for eps in (0.0, 1.0):           # 1.0 = daima rastgele kol
            a = _pick(rng, q, m, n, eps)
            if not m[a]:
                if eps:
                    bad_eps += 1
                else:
                    bad_greedy += 1
            total += 1
    assert bad_greedy == 0, f"{bad_greedy} gecersiz greedy secim"
    assert bad_eps == 0, f"{bad_eps} gecersiz eps-rastgele secim"
    print(f"  ok  secim hep gecerli  {total} secim, greedy+eps kollari")


def test_mask_none_is_unchanged(seed=7, trials=4000):
    """mask=None ile davranis ESKISIYLE birebir ayni olmali.

    Bu bir regresyon kapisi: maske varsayilan KAPALI, yani `_pick`'in
    maskesiz yolu eski act() kodunun tipatip aynisini yapmali. Aksi halde
    action_mask=false olan butun mevcut configler sessizce degisir ve
    kampanyanin kontrolu bozulur.
    """
    n = len(ACTION_NAMES)
    a = np.random.default_rng(seed)
    b = np.random.default_rng(seed)
    qs = np.random.default_rng(123).normal(size=(trials, n))
    diff = 0
    for i in range(trials):
        q = qs[i]
        got = _pick(a, q, None, n, 0.0)
        # eski kod birebir:
        best = np.flatnonzero(q == q.max())
        want = int(b.choice(best))
        if got != want:
            diff += 1
    assert diff == 0, f"{diff}/{trials} secimde maskesiz yol degismis"
    print(f"  ok  maskesiz regresyon {trials} secim, sapma yok")


def main() -> int:
    print("gecerlilik maskesi testleri")
    print("-" * 60)
    try:
        test_mask_matches_environment()
        test_masked_choice_never_illegal()
        test_mask_none_is_unchanged()
    except AssertionError as ex:
        print(f"  FAIL {ex}")
        print("-" * 60); print("BASARISIZ"); return 1
    except Exception as ex:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print(f"  ERR  {type(ex).__name__}: {ex}")
        print("-" * 60); print("BASARISIZ"); return 1
    print("-" * 60); print("TUMU GECTI")
    if _LOGDIR:
        shutil.rmtree(_LOGDIR, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
