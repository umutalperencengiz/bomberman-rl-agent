"""features / rewards / buffer testleri.

Sahibi: IRMAK (infra)

Kosum:
    ml_homework/Scripts/python.exe tests/test_features.py

EN KRITIK: `test_symmetry_action_mapping`. Simetri augmentation'da aksiyon
etiketi yanlis eslenirse veri seti SESSIZCE zehirlenir -- model kotu ogrenir
ama hicbir yerde hata patlamaz. Bu test o eslemeyi dogrudan dogrular.
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

import settings as s                                        # noqa: E402
from environment import BombeRLeWorld, WorldArgs            # noqa: E402

from bbrl import features as F                              # noqa: E402
from bbrl import rewards as R                               # noqa: E402
from bbrl.buffer import PrioritizedReplayBuffer, ReplayBuffer  # noqa: E402
from bbrl.gamelogic import ACTIONS, DELTAS, MOVES           # noqa: E402


def _states(n_states=60, seed=3):
    """Gercek oyundan state ornekleri topla."""
    tmp = tempfile.mkdtemp(prefix="bbrl_feat_")
    args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                     save_replay=False, replay=None, make_video=False,
                     continue_without_training=True, log_dir=tmp,
                     save_stats=False, match_name=None, seed=seed,
                     silence_errors=False, scenario="classic")
    world = BombeRLeWorld(args, [("random_agent", False)] * 3)
    out = []
    try:
        while len(out) < n_states:
            world.new_round()
            while world.running and len(out) < n_states:
                if not hasattr(world, "user_input"):
                    world.user_input = None
                for a in world.active_agents:
                    st = world.get_state_for_agent(a)
                    if st is not None:
                        out.append(st)
                        break
                world.do_step()
    finally:
        try:
            world.end()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)
    return out


# --------------------------------------------------------------------------

def test_feature_contracts(states):
    """Her kayitli feature seti shape/dtype sozlesmesine uymali, NaN uretmemeli."""
    for name, fs in sorted(F.FEATURE_SETS.items()):
        assert fs(None) is None, f"{name}: None state -> None dondurmeli"
        for st in states[:25]:
            v = fs(st)
            assert v.shape == fs.shape, f"{name}: shape {v.shape} != {fs.shape}"
            assert v.dtype == fs.dtype, f"{name}: dtype {v.dtype}"
            assert np.isfinite(v).all(), f"{name}: NaN/inf uretti"
    print(f"  ok  feature sozlesmeleri       ({len(F.FEATURE_SETS)} set: "
          f"{', '.join(sorted(F.FEATURE_SETS))})")


def test_v2_danger_channel(states):
    """v2'nin danger kanali, tehlike varken sifirdan farkli olmali."""
    fs = F.get("v2_planes")
    ch = F.V2_CHANNEL_NAMES.index("danger")
    nonzero = 0
    for st in states:
        if st["bombs"] or np.asarray(st["explosion_map"]).max() >= 1:
            v = fs(st)
            if v[ch].max() > 0:
                nonzero += 1
    assert nonzero > 0, "bomba varken danger kanali hep sifir -- bagli degil?"
    print(f"  ok  v2 danger kanali           ({nonzero} state'te aktif)")


def test_symmetry_action_mapping():
    """EN KRITIK: duzlem donusumu ile aksiyon donusumu TUTARLI olmali.

    Yontem: iki isaretci koy -- biri ajanin yeri p, digeri `a` hamlesinden
    sonraki yer q = p + delta(a). Duzlemleri (k, flip) ile donustur, p' ve q'
    yeni yerleri oku. transform_action(a) ile hesaplanan yon p''den q''ye
    goturmeli.
    """
    W, H = s.COLS, s.ROWS
    bad = []
    for a_name in MOVES:
        ai = ACTIONS.index(a_name)
        dx, dy = DELTAS[a_name]
        px, py = 8, 6                       # merkeze yakin, tasma yok
        qx, qy = px + dx, py + dy
        for k in range(4):
            for flip in (False, True):
                P = np.zeros((2, W, H), dtype=np.float32)
                P[0, px, py] = 1.0
                P[1, qx, qy] = 1.0
                Q = F.transform_planes(P, k, flip)
                (nx,), (ny,) = np.where(Q[0])[0], np.where(Q[0])[1]
                (mx,), (my,) = np.where(Q[1])[0], np.where(Q[1])[1]
                a2 = F.transform_action(ai, k, flip)
                edx, edy = DELTAS[ACTIONS[a2]]
                if (nx + edx, ny + edy) != (mx, my):
                    bad.append((a_name, k, flip, ACTIONS[a2],
                                (int(nx), int(ny)), (int(mx), int(my))))
    assert not bad, (
        "simetri aksiyon eslemesi HATALI (veri setini sessizce zehirler):\n"
        + "\n".join(f"    {a} k={k} flip={f} -> {a2}: "
                    f"p'={p} beklenen q'={q}" for a, k, f, a2, p, q in bad[:8]))
    print("  ok  simetri aksiyon eslemesi   (4 hamle x 4 donme x 2 aynalama)")


def test_symmetry_invariants():
    """WAIT ve BOMB donusum altinda degismez."""
    for a in ("WAIT", "BOMB"):
        ai = ACTIONS.index(a)
        for k in range(4):
            for flip in (False, True):
                assert F.transform_action(ai, k, flip) == ai, f"{a} degisti"
    print("  ok  WAIT/BOMB donusum altinda sabit")


def test_reward_schemes(states):
    """Semalar cagrilabilir olmali; intiharda cifte ceza kontrolu."""
    import events as e
    for name in sorted(R.REWARD_SCHEMES):
        assert isinstance(R.reward_from_events([e.COIN_COLLECTED], name), float)

    # r00 saf oyun odulu: shaping event'i etkisiz olmali
    assert R.reward_from_events([e.INVALID_ACTION], "r00_pure_game") == 0.0
    assert R.reward_from_events([e.COIN_COLLECTED], "r00_pure_game") == 1.0
    assert R.reward_from_events([e.KILLED_OPPONENT], "r00_pure_game") == 5.0

    # Intihar GERCEKTE iki event birden uretir -> semanin toplami kontrol edilir
    suicide = [e.KILLED_SELF, e.GOT_KILLED]
    r = R.reward_from_events(suicide, "r01_minimal")
    assert r == -5.0, (
        f"r01_minimal intihar cezasi {r}, -5.0 bekleniyordu. "
        "KILLED_SELF + GOT_KILLED birlikte tetiklenir; sema bunu hesaba katmali "
        "(docs/GAME_MECHANICS.md 6).")
    print("  ok  odul semalari              (intihar cifte-ceza kontrolu dahil)")


def test_custom_events(states):
    """custom_events `events`'i mutate etmemeli ve gecerli adlar uretmeli."""
    import events as e
    valid = set(R.CUSTOM_EVENTS)
    seen = set()
    for i in range(len(states) - 1):
        ev = [e.WAITED]
        before = list(ev)
        extra = R.custom_events(states[i], "WAIT", states[i + 1], ev)
        assert ev == before, "custom_events girdi listesini MUTATE etti"
        assert set(extra) <= valid, f"bilinmeyen event: {set(extra) - valid}"
        seen |= set(extra)
    # bomba senaryosu
    for st in states:
        extra = R.custom_events(st, "BOMB", None, [e.BOMB_DROPPED])
        assert set(extra) <= valid
        seen |= set(extra)
    print(f"  ok  custom_events              (uretilenler: "
          f"{', '.join(sorted(seen)) or '-'})")


def test_flat_symmetry_matches_action(states):
    """EN KRITIK (EXP-013): duz vektor donusumu aksiyon eslemesiyle TUTARLI olmali.

    Yontem: feature'da "coin YUKARIDA" diyen one-hot'u (k, flip) ile donustur.
    Sonuc, transform_action(UP, k, flip) hangi yonu veriyorsa ORAYI
    isaretlemeli. Tutmazsa augment edilen ornekler yanlis etiketlenir --
    duzlem versiyonunda ayni hatayi bir kez yapmistim (UP -> LEFT).
    """
    fs = F.get("v0b_task2")
    assert fs.dir_slots, "v0b_task2 dir_slots tanimlamali"
    bad = []
    for start, _n in fs.dir_slots:
        for i, mv in enumerate(MOVES):
            x = np.zeros(fs.shape, dtype=np.float32)
            x[start + i] = 1.0
            for k in range(4):
                for flip in (False, True):
                    y = F.transform_flat(x, k, flip, fs.dir_slots)
                    got = int(np.argmax(y[start:start + 4]))
                    want_action = F.transform_action(ACTIONS.index(mv), k, flip)
                    want = MOVES.index(ACTIONS[want_action])
                    if got != want:
                        bad.append((start, mv, k, flip, MOVES[got],
                                    ACTIONS[want_action]))
    assert not bad, (
        "duz vektor simetrisi aksiyon eslemesiyle UYUSMUYOR:\n"
        + "\n".join(f"    slot{s} {mv} k={k} flip={f}: feature->{g} "
                    f"ama aksiyon->{w}" for s, mv, k, f, g, w in bad[:8]))

    # Donusum bilgi KAYBETMEMELI: one-hot toplami korunmali, yon disi
    # alanlar aynen kalmali.
    for st in states[:20]:
        x = fs(st)
        for k in range(4):
            for flip in (False, True):
                y = F.transform_flat(x, k, flip, fs.dir_slots)
                assert abs(float(x.sum()) - float(y.sum())) < 1e-5, \
                    "donusum toplami degistirdi -- bilgi kaybi"
                for idx in (14, 15, 21, 22, 23):     # yon disi alanlar
                    assert x[idx] == y[idx], f"indeks {idx} degismemeliydi"
    print("  ok  duz vektor simetrisi      (4 dilim x 4 hamle x 8 donusum)")


def test_shaping_telescopes(states):
    """EN KRITIK (EXP-010): A -> B -> A gidisi TAM OLARAK sifir getirmeli.

    Salinimin ilaci bu ozellik. gamma_shaping=1 iken
        F(A->B) + F(B->A) = (Phi(B)-Phi(A)) + (Phi(A)-Phi(B)) = 0
    Bozulursa ajan yine ileri-geri giderek odul farm'lar.
    """
    for scheme in ("p01_crate", "p02_crate_coin", "p03_crate_coin_safe"):
        worst = 0.0
        for i in range(len(states) - 1):
            a, b = states[i], states[i + 1]
            f_ab = R.shaping_reward(a, b, scheme, 1.0)
            f_ba = R.shaping_reward(b, a, scheme, 1.0)
            worst = max(worst, abs(f_ab + f_ba))
        assert worst < 1e-6, (
            f"{scheme}: ileri-geri toplami {worst:.3g} != 0 -- teleskopik "
            "sifirlanma bozuk, salinim yine odul kazandirir")
    print("  ok  shaping teleskopik sifirlaniyor (3 sema)")


def test_shaping_stationary_is_zero(states):
    """Ayni state'te kalmak TAM sifir odul vermeli.

    gamma<1 ile Phi=-mesafe kullanilsaydi F = Phi*(1-gamma) > 0 olur,
    yani "uzakta kimildamadan durmak" odullendirilirdi -- kirmaya
    calistigimiz davranisin ta kendisi. gamma_shaping=1 bunu engeller.
    """
    for st in states[:30]:
        for scheme in ("p01_crate", "p02_crate_coin", "p03_crate_coin_safe"):
            assert abs(R.shaping_reward(st, st, scheme, 1.0)) < 1e-9, \
                f"{scheme}: yerinde durmak sifirdan farkli odul verdi"
    # gamma<1 ile gercekten pozitif oluyor mu -- iddianin kaniti
    pos = sum(1 for st in states[:60]
              if R.shaping_reward(st, st, "p01_crate", 0.95) > 1e-9)
    assert pos > 0, "gamma<1 artefakti gozlenmedi -- testi gozden gecir"
    print(f"  ok  yerinde durmak = 0        (gamma=0.95 olsaydi {pos}/60 "
          f"state'te POZITIF odul verirdi)")


def test_new_bomb_events(states):
    """GOOD_BOMB / SAFE_AFTER_BOMB / BOMB_IDLE uretilebiliyor mu."""
    import events as e
    seen = set()
    for st in states:
        seen |= set(R.custom_events(st, "BOMB", None, [e.BOMB_DROPPED], {}))
    mem = {}
    for i in range(len(states) - 1):
        seen |= set(R.custom_events(states[i], "WAIT", states[i + 1],
                                    [e.WAITED], mem))
    # GOOD_BOMB ve SUICIDAL_BOMB ayni anda OLAMAZ (biri kacis var, digeri yok)
    for st in states:
        ev = set(R.custom_events(st, "BOMB", None, [e.BOMB_DROPPED], {}))
        assert not (R.GOOD_BOMB in ev and R.SUICIDAL_BOMB in ev), \
            "GOOD_BOMB ve SUICIDAL_BOMB ayni adimda uretildi -- mantik hatasi"
    assert R.GOOD_BOMB in seen or R.SUICIDAL_BOMB in seen
    print(f"  ok  yeni bomba event'leri     ({', '.join(sorted(seen)) or '-'})")


def test_reward_ratio_sanity():
    """r04'un oranlari: calismak > bos durmak > olmek siralamasi."""
    import events as e
    crate = R.REWARD_SCHEMES["r04_ratio"][e.CRATE_DESTROYED]
    death = R.REWARD_SCHEMES["r04_ratio"][e.KILLED_SELF]
    living = R.REWARD_SCHEMES["r04_ratio"][e.MOVED_UP]

    iyi_tur = 117 * crate + 400 * living          # rule_based kadar sandik
    bos_tur = 400 * living                        # 400 adim hicbir sey yapma
    olum = death
    assert iyi_tur > bos_tur > olum, (
        f"siralama bozuk: iyi={iyi_tur:.1f} bos={bos_tur:.1f} olum={olum:.1f}")
    oran = abs(death) / crate
    assert 5 <= oran <= 20, (
        f"olum:sandik orani {oran:.0f}:1 -- literaturdeki 10:1'den cok uzak "
        "(ICAART 2018: -300 olum / +30 duvar)")
    print(f"  ok  r04 odul oranlari         (iyi tur +{iyi_tur:.0f} > "
          f"bos {bos_tur:.0f} > olum {olum:.0f}; olum:sandik = {oran:.0f}:1)")


def test_nstep_field():
    """n-step alani buffer'da tasiniyor ve save/load'dan sagliyor mi."""
    from bbrl.buffer import make_buffer
    for kind in ("uniform", "prioritized"):
        b = make_buffer(kind, 200, (5,), seed=0)
        for i in range(150):
            b.push(np.full(5, i, np.float32), i % 6,
                   np.full(5, i + 1, np.float32), 1.0, False,
                   n_step=3 if i % 2 else 1)
        out = b.sample(64)
        assert len(out) == 8, f"{kind}: batch {len(out)} alan, 8 bekleniyordu"
        assert set(out[7].tolist()) <= {1, 3}, f"{kind}: beklenmedik n degeri"
        # save/load n-step'i korumali
        b.save("runs/_nstep_test.npz")
        b2 = make_buffer(kind, 200, (5,), seed=0)
        assert b2.load("runs/_nstep_test.npz")
        assert set(b2.sample(64)[7].tolist()) <= {1, 3}, f"{kind}: n kayboldu"
        os.remove("runs/_nstep_test.npz")
    print("  ok  n-step alani              (uniform + prioritized, save/load dahil)")


def test_buffers():
    """Uniform ve prioritized buffer: push/sample/oncelik guncelleme."""
    shape = (7,)
    for kind, buf in (("uniform", ReplayBuffer(64, shape, seed=0)),
                      ("per", PrioritizedReplayBuffer(64, shape, seed=0))):
        for i in range(200):                       # kapasiteyi tasir -> ring
            term = (i % 17 == 0)
            buf.push(np.full(shape, i, np.float32), i % 6,
                     None if term else np.full(shape, i + 1, np.float32),
                     float(i), term)
        assert len(buf) == 64, f"{kind}: size {len(buf)} != 64"
        st, ac, rw, ns, dn, idx, w, nst = buf.sample(32)
        assert st.shape == (32, *shape) and ns.shape == (32, *shape)
        assert ac.shape == rw.shape == dn.shape == (32,)
        assert w.shape == (32,) and np.isfinite(w).all() and (w > 0).all()
        assert nst.shape == (32,) and (nst >= 1).all()   # n-step alani
        assert dn.dtype == bool
        buf.update_priorities(idx, np.abs(np.random.randn(32)))
        # terminal ornekte next_state sifirlanmis olmali
        assert (ns[dn] == 0).all(), f"{kind}: terminal next_state sifirlanmadi"
    print("  ok  replay buffer              (uniform + prioritized)")


def test_per_prioritizes():
    """PER gercekten yuksek TD-hatali ornekleri daha sik cekmeli."""
    buf = PrioritizedReplayBuffer(100, (2,), seed=0)
    for i in range(100):
        buf.push(np.array([i, 0], np.float32), 0,
                 np.array([i, 1], np.float32), 0.0, False)
    # yalnizca 0..4 arasi indekslere yuksek oncelik ver
    buf.update_priorities(np.arange(100), np.r_[np.full(5, 100.0),
                                                np.full(95, 0.001)])
    _, _, _, _, _, idx, _, _ = buf.sample(500)
    frac = float(np.mean(idx < 5))
    assert frac > 0.5, (f"PER onceligi calismiyor: yuksek-oncelikli ornek "
                        f"orani {frac:.2f}, >0.5 bekleniyordu")
    print(f"  ok  PER onceliklendirmesi      (yuksek-TD orani {frac:.0%})")


# --------------------------------------------------------------------------

def main() -> int:
    print("bbrl features / rewards / buffer testleri")
    print("-" * 60)
    print("  .. state ornekleri toplaniyor")
    states = _states()
    tests = [
        (test_feature_contracts, (states,)),
        (test_v2_danger_channel, (states,)),
        (test_symmetry_action_mapping, ()),
        (test_symmetry_invariants, ()),
        (test_reward_schemes, (states,)),
        (test_custom_events, (states,)),
        (test_flat_symmetry_matches_action, (states,)),
        (test_shaping_telescopes, (states,)),
        (test_shaping_stationary_is_zero, (states,)),
        (test_new_bomb_events, (states,)),
        (test_reward_ratio_sanity, ()),
        (test_nstep_field, ()),
        (test_buffers, ()),
        (test_per_prioritizes, ()),
    ]
    failed = 0
    for fn, a in tests:
        try:
            fn(*a)
        except AssertionError as ex:
            failed += 1
            print(f"  FAIL {fn.__name__}\n       {ex}")
        except Exception as ex:  # noqa: BLE001
            failed += 1
            print(f"  ERR  {fn.__name__}: {type(ex).__name__}: {ex}")
    print("-" * 60)
    print("TUMU GECTI" if not failed else f"{failed} TEST BASARISIZ")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
