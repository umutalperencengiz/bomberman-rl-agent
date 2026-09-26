"""bbrl.gamelogic'i GERCEK environment'a karsi dogrular.

Sahibi: IRMAK (infra)

Kosum (pytest gerekmez):
    ml_homework/Scripts/python.exe tests/test_gamelogic.py

En kritik test `test_danger_map_parity`: gercek bir oyun kosturulur ve her
adimda `danger_map(state) == 0` karelerinin, o adimin sonunda GERCEKTEN
oldurucu olan karelerle BIREBIR ayni oldugu dogrulanir.

Bu test yesil degilse hicbir tehlike feature'ina guvenilemez.
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
os.chdir(REPO)  # framework goreli yol kullaniyor

import settings as s                      # noqa: E402
from environment import BombeRLeWorld, WorldArgs   # noqa: E402
from items import Bomb                    # noqa: E402

from bbrl.gamelogic import (              # noqa: E402
    SAFE, blast_coords, danger_map, free_mask, lethal_at,
)


def _make_world(scenario="classic", seed=0, agents=("random_agent",) * 3,
                log_dir=None):
    args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False,
        continue_without_training=True, log_dir=log_dir, save_stats=False,
        match_name=None, seed=seed, silence_errors=False, scenario=scenario,
    )
    return BombeRLeWorld(args, [(a, False) for a in agents])


def _first_state(world):
    """Aktif bir ajanin state'i (yoksa None).

    `get_state_for_agent` `self.user_input`'a bakiyor ama o alan yalnizca
    `do_step` icinde atanıyor -> ilk adimdan once elle set etmek gerekiyor.
    """
    if not hasattr(world, "user_input"):
        world.user_input = None
    for a in world.active_agents:
        st = world.get_state_for_agent(a)
        if st is not None:
            return a, st
    return None, None


# --------------------------------------------------------------------------

def test_blast_coords_parity(n_fields=200, seed=0):
    """blast_coords == items.Bomb.get_blast_coords (rastgele alanlarda)."""
    rng = np.random.default_rng(seed)
    checked = 0
    for _ in range(n_fields):
        field = np.zeros((s.COLS, s.ROWS), int)
        field[rng.random((s.COLS, s.ROWS)) < 0.5] = 1
        field[:1, :] = field[-1:, :] = field[:, :1] = field[:, -1:] = -1
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    field[x, y] = -1
        free = [(x, y) for x in range(s.COLS) for y in range(s.ROWS)
                if field[x, y] != -1]
        for (x, y) in free[:: max(1, len(free) // 8)]:
            bomb = Bomb((x, y), None, s.BOMB_TIMER, s.BOMB_POWER, None)
            expected = bomb.get_blast_coords(field)
            got = blast_coords(field, x, y)
            assert sorted(got) == sorted(expected), (
                f"blast_coords farkli @({x},{y})\n"
                f"  beklenen={sorted(expected)}\n  bulunan  ={sorted(got)}")
            checked += 1
    print(f"  ok  blast_coords parity        ({checked} konum)")


def test_crates_do_not_block_blast():
    """Sandiklar patlamayi durdurmaz -- yalnizca duvarlar durdurur."""
    field = np.zeros((s.COLS, s.ROWS), int)
    field[:1, :] = field[-1:, :] = field[:, :1] = field[:, -1:] = -1
    for x in range(s.COLS):
        for y in range(s.ROWS):
            if (x + 1) * (y + 1) % 2 == 1:
                field[x, y] = -1
    # (1,1)'in sagina sandik koy; blast ondan GECMELI
    field[2, 1] = 1
    coords = set(blast_coords(field, 1, 1))
    assert (2, 1) in coords, "sandik karesi blast'ta olmali"
    assert (3, 1) in coords, "blast sandiktan GECMELI (sadece duvar durdurur)"
    assert (4, 1) in coords, "menzil 3 -> (4,1) dahil"
    assert (5, 1) not in coords, "menzil 3'u asmamali"
    print("  ok  sandiklar blast'i durdurmuyor")


def test_danger_map_parity(n_rounds=25, seed=0, verbose=False):
    """EN KRITIK TEST.

    Her adimda, adimin BASINDA gorunen state'ten hesaplanan
    `danger_map == 0` kumesi, adimin SONUNDA gercekten oldurucu olan
    kare kumesine BIREBIR esit olmali.
    """
    tmp = tempfile.mkdtemp(prefix="bbrl_test_")
    world = _make_world(seed=seed, log_dir=tmp)
    total_steps = 0
    mismatches = []
    try:
        for rnd in range(n_rounds):
            world.new_round()
            while world.running:
                _agent, state = _first_state(world)
                if state is None:
                    break

                d = danger_map(state)
                predicted = {(int(x), int(y))
                             for x, y in zip(*np.where(d == 0))}

                world.do_step()

                # do_step sonrasi stage==0 olan patlamalar, TAM OLARAK bu
                # adimin sonunda evaluate_explosions'da olduren patlamalardir.
                actual = set()
                for exp in world.explosions:
                    if exp.is_dangerous():
                        actual |= {(int(x), int(y)) for x, y in exp.blast_coords}

                total_steps += 1
                if predicted != actual:
                    mismatches.append({
                        "round": rnd, "step": world.step,
                        "eksik": sorted(actual - predicted),
                        "fazla": sorted(predicted - actual),
                    })
                    if verbose and len(mismatches) <= 3:
                        print("   ", mismatches[-1])
    finally:
        try:
            world.end()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)

    assert not mismatches, (
        f"danger_map {len(mismatches)}/{total_steps} adimda uyusmadi. "
        f"Ilk 3: {mismatches[:3]}\n"
        "-> 'eksik' = oldurdu ama tahmin edilmedi (OLUMCUL HATA)\n"
        "-> 'fazla' = tahmin edildi ama oldurmedi (asiri temkinli)")
    print(f"  ok  danger_map parity          ({total_steps} adim, "
          f"{n_rounds} tur)")


def test_explosion_map_alone_is_insufficient(n_rounds=15, seed=1):
    """explosion_map TEK BASINA yetersiz -- bunu SAYIYLA kanitla.

    Bu, docs/GAME_MECHANICS.md 2'deki iddianin testi ve EXP-003'un
    gerekcesi. Sadece explosion_map kullanan naif bir tehlike haritasinin
    kacirdigi oldurucu kareleri sayar; sayi 0'dan buyuk OLMALI.
    """
    tmp = tempfile.mkdtemp(prefix="bbrl_test_")
    world = _make_world(seed=seed, log_dir=tmp)
    missed = 0
    caught = 0
    try:
        for _ in range(n_rounds):
            world.new_round()
            while world.running:
                _agent, state = _first_state(world)
                if state is None:
                    break
                naive = {(int(x), int(y)) for x, y in
                         zip(*np.where(np.asarray(state["explosion_map"]) >= 1))}
                world.do_step()
                actual = set()
                for exp in world.explosions:
                    if exp.is_dangerous():
                        actual |= {(int(x), int(y)) for x, y in exp.blast_coords}
                missed += len(actual - naive)
                caught += len(actual & naive)
    finally:
        try:
            world.end()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)

    assert missed > 0, "beklenen kor nokta gozlenmedi -- testi gozden gecir"
    pct = 100.0 * missed / max(1, missed + caught)
    print(f"  ok  explosion_map kor noktasi  (naif harita {missed} oldurucu "
          f"kareyi kacirdi = tumunun %{pct:.0f}'i)")
    return missed, caught


def test_free_mask_parity(n_rounds=10, seed=2):
    """free_mask == environment.tile_is_free (tum kareler, her adim)."""
    tmp = tempfile.mkdtemp(prefix="bbrl_test_")
    world = _make_world(seed=seed, log_dir=tmp)
    checked = 0
    try:
        for _ in range(n_rounds):
            world.new_round()
            while world.running:
                agent, state = _first_state(world)
                if state is None:
                    break
                m = free_mask(state)
                for x in range(s.COLS):
                    for y in range(s.ROWS):
                        expected = world.tile_is_free(x, y)
                        # tile_is_free ajanin KENDISINI de engel sayar;
                        # free_mask 'others' kullandigi icin kendi karesi
                        # serbest gorunur -> o kareyi haric tut.
                        if (x, y) == (agent.x, agent.y):
                            continue
                        assert bool(m[x, y]) == bool(expected), (
                            f"free_mask({x},{y}) != tile_is_free")
                        checked += 1
                world.do_step()
    finally:
        try:
            world.end()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"  ok  free_mask parity           ({checked} kare kontrolu)")


def test_lethal_at_window():
    """d degerindeki kare, S+d ve S+d+1 adimlarinda oldurucu; digerinde degil."""
    d = np.full((3, 3), SAFE, dtype=np.int16)
    d[1, 1] = 2
    assert not lethal_at(d, 1, 1, 0)
    assert not lethal_at(d, 1, 1, 1)
    assert lethal_at(d, 1, 1, 2)
    assert lethal_at(d, 1, 1, 3)
    assert not lethal_at(d, 1, 1, 4)
    print("  ok  lethal_at penceresi")


# --------------------------------------------------------------------------

def main() -> int:
    tests = [
        test_lethal_at_window,
        test_crates_do_not_block_blast,
        test_blast_coords_parity,
        test_free_mask_parity,
        test_danger_map_parity,
        test_explosion_map_alone_is_insufficient,
    ]
    print("bbrl.gamelogic testleri")
    print("-" * 60)
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as ex:
            failed += 1
            print(f"  FAIL {t.__name__}\n       {ex}")
        except Exception as ex:  # noqa: BLE001
            failed += 1
            print(f"  ERR  {t.__name__}: {type(ex).__name__}: {ex}")
    print("-" * 60)
    print("TUMU GECTI" if not failed else f"{failed} TEST BASARISIZ")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
