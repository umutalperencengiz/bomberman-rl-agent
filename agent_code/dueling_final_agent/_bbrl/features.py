"""Feature setleri -- registry + implementasyonlar.

SAHIPLIK
  * Registry, sozlesme, dogrulama, `v0_minimal`, `v2_planes`   -> IRMAK (infra + DL)
  * `v1_handcrafted` (elle tasarlanmis 25 feature)             -> UMUT
  * Feature ablation deneyleri (EXP-003 vb.)                   -> UMUT

Kural (proje PDF s.9): deterministik olarak "en iyi aksiyonu" donduren bir
feature YASAK. Yon/mesafe/tehlike BILGISI vermek serbest (PDF bunlari acikca
guclu feature ornegi olarak sayiyor); nihai karari model verir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

import settings as s

from .gamelogic import (
    CRATE, FREE, MOVES, SAFE, WALL, bomb_impact, danger_map,
    direction_to_nearest, escape_directions, free_mask,
)


@dataclass(frozen=True)
class FeatureSet:
    """Bir feature seti. `shape` ve `dtype` sozlesmesi ZORUNLU."""

    name: str
    shape: tuple[int, ...]
    fn: Callable[[dict], np.ndarray]
    dtype: np.dtype = np.dtype(np.float32)
    doc: str = ""
    #: DUZ vektorlerde simetri icin: hangi indeks araliklari MOVES sirasinda
    #: yon one-hot'u? Her eleman (baslangic, uzunluk=4). Bunlar verilmezse
    #: duz vektorde augmentation yapilamaz.
    #: (EXP-012: simetri CNN'i 3x'ledi ama MLP -- en iyi modelimiz -- o
    #: kazanctan faydalanamiyordu, cunku duz vektorde donusum tanimsizdi.)
    dir_slots: tuple[tuple[int, int], ...] = ()

    def __call__(self, game_state: dict | None) -> np.ndarray | None:
        if game_state is None:          # tur basi/sonu
            return None
        v = self.fn(game_state)
        v = np.asarray(v, dtype=self.dtype)
        if v.shape != self.shape:
            raise ValueError(
                f"FeatureSet {self.name!r}: shape {v.shape} != "
                f"sozlesme {self.shape}")
        return v


FEATURE_SETS: dict[str, FeatureSet] = {}


def register(fs: FeatureSet) -> FeatureSet:
    if fs.name in FEATURE_SETS:
        raise KeyError(f"feature set {fs.name!r} zaten kayitli")
    FEATURE_SETS[fs.name] = fs
    return fs


def get(name: str) -> FeatureSet:
    if name not in FEATURE_SETS:
        raise KeyError(f"bilinmeyen feature set {name!r}. "
                       f"Mevcut: {sorted(FEATURE_SETS)}")
    return FEATURE_SETS[name]


# ==========================================================================
# v0_minimal -- boru hattini test etmek icin en kucuk calisan set  [IRMAK]
# ==========================================================================

_V0_LEN = 4 + 5 + 5 + 1 + 1


def _v0(gs: dict) -> np.ndarray:
    _n, _sc, can_bomb, (ax, ay) = gs["self"]
    passable = free_mask(gs)
    d = danger_map(gs)

    out = np.zeros(_V0_LEN, dtype=np.float32)
    i = 0
    # 4x: komsu kare gecilebilir mi
    for mv in MOVES:
        from .gamelogic import DELTAS
        dx, dy = DELTAS[mv]
        out[i] = float(passable[ax + dx, ay + dy])
        i += 1
    # 5x: en yakin coin yonu (one-hot + "yok")
    out[i:i + 5] = direction_to_nearest(gs, gs["coins"], passable)
    i += 5
    # 5x: guvenli kacis yonleri
    out[i:i + 5] = escape_directions(gs)
    i += 5
    # 1x: su an tehlikede miyim (0 = bu adim olur)
    out[i] = 1.0 if d[ax, ay] < SAFE else 0.0
    i += 1
    # 1x: bomba atabilir miyim
    out[i] = float(bool(can_bomb))
    return out


register(FeatureSet(
    "v0_minimal", (_V0_LEN,), _v0,
    doc="Boru hattini uctan uca test etmek icin minimal set. "
        "Gercek bir aday degil -- baseline/smoke test."))


# ==========================================================================
# v0b_task2 -- Task 2 icin minimum ekleme                       [IRMAK]
# ==========================================================================
#
# EXP-008: `v0_minimal` ile egitilen ajan Task 2'de rastgeleden KOTU
# (9.5 adimda oluyor, rastgele 19.3). Eksik olan uc bilgi:
#   * sandik NEREDE (classic'te coin'ler sandik altinda -> coin feature'i
#     tur basinda hep "YOK" diyor, ajanin gidecek hedefi yok)
#   * buraya bomba atarsam KURTULABILIR miyim
#   * buraya bomba atmak ISE YARAR mi (sandik/rakip vurur mu)
#
# Bu, Umut'un `v1_handcrafted`'inin yerini TUTMAZ -- Task 2'yi calisir hale
# getirecek en kucuk eklemedir ve v1 icin somut bir karsilastirma tabani olur.
_V0B_LEN = _V0_LEN + 5 + 1 + 1 + 1


def _v0b(gs: dict) -> np.ndarray:
    out = np.zeros(_V0B_LEN, dtype=np.float32)
    out[:_V0_LEN] = _v0(gs)
    i = _V0_LEN

    field = gs["field"]
    _n, _sc, can_bomb, (ax, ay) = gs["self"]

    # 5x: en yakin sandik yonu (BFS). classic'te asil hedef budur.
    crates = list(zip(*np.where(field == CRATE)))
    out[i:i + 5] = direction_to_nearest(gs, crates)
    i += 5

    # 1x: buraya bomba atarsam guvenli kacis var mi  -> intiharin oncul sinyali
    out[i] = float(bool(can_bomb) and bool(escape_directions(
        gs, extra_bomb_at=(ax, ay)).any()))
    i += 1

    # 1x / 1x: bomba ise yarar mi
    n_crates, n_enemies = bomb_impact(gs, ax, ay)
    out[i] = float(n_crates > 0)
    out[i + 1] = float(n_enemies > 0)
    return out


#: v0b_task2 duzeni (simetri icin yon dilimleri):
#:   [0:4]   4 yon gecilebilir mi              -> yon dilimi
#:   [4:9]   en yakin coin yonu (4 yon + YOK)  -> yon dilimi [4:8], 8 = YOK
#:   [9:14]  guvenli kacis yonleri (4 + WAIT)  -> yon dilimi [9:13], 13 = WAIT
#:   [14]    tehlikede miyim                   -> degismez
#:   [15]    bomba atabilir miyim              -> degismez
#:   [16:21] en yakin sandik yonu (4 + YOK)    -> yon dilimi [16:20], 20 = YOK
#:   [21]    bomba kurtulabilir                -> degismez
#:   [22]    bomba sandik vurur                -> degismez
#:   [23]    bomba rakip vurur                 -> degismez
_V0B_DIR_SLOTS = ((0, 4), (4, 4), (9, 4), (16, 4))

register(FeatureSet(
    "v0b_task2", (_V0B_LEN,), _v0b, dir_slots=_V0B_DIR_SLOTS,
    doc="v0_minimal + sandik yonu + bomba kurtulabilirligi + bomba faydasi. "
        "EXP-008'in tespit ettigi uc eksigi kapatan minimum set (Irmak). "
        "dir_slots sayesinde D4 simetri augmentation'i destekler."))


# ==========================================================================
# v0c_task4 -- rakip farkindaligi                               [IRMAK]
# ==========================================================================
#
# EXP-015 olctu: rakipsiz Task 2'de intihar 0.000 idi, EK-2'de (3 rakip)
# 0.591'e firladi -- rule_based'in 0.528'inden BILE KOTU. Tek acik alanimiz bu.
#
# Sebep muhtemelen su: `escape_directions` rakipleri SU ANKI yerlerinde sabit
# varsayiyor. Ama rakipler HAREKET EDER; simdi acik olan kacis yolu bir adim
# sonra kapanabilir. Bomba birakirken "kacisim var" diyoruz, rakip yolu
# kesiyor, oluyoruz.
#
# Uc yeni bilgi:
#   1. Kac TANE ayri kacis yolu var (1 mi 3 mu -- bool degil sayi)
#   2. Rakipler komsuluklariyla birlikte bloklanirsa kacis KALIYOR mu
#      (temkinli kacis -- rakip hareketine karsi saglam mi)
#   3. En yakin rakip nerede / ne kadar yakin
#           rakip yonu(5) + kacis sayisi(1) + temkinli kacis(1)
#           + bitisik/yakin(2) + rakip sayisi(1)  =  10
_V0C_LEN = _V0B_LEN + 5 + 1 + 1 + 2 + 1
_V0C_DIR_SLOTS = _V0B_DIR_SLOTS + ((_V0B_LEN, 4),)   # rakip yonu dilimi


def _v0c(gs: dict) -> np.ndarray:
    out = np.zeros(_V0C_LEN, dtype=np.float32)
    out[:_V0B_LEN] = _v0b(gs)
    i = _V0B_LEN

    others = [(ox, oy) for _n, _s, _b, (ox, oy) in gs["others"]]
    _n, _sc, can_bomb, (ax, ay) = gs["self"]

    # 5x: en yakin rakip yonu
    out[i:i + 5] = direction_to_nearest(gs, others)
    i += 5

    # 1x: kac ayri kacis yolu var (0..4 -> 0..1). Tek yol varsa kirilgan.
    esc = escape_directions(gs)
    out[i] = float(esc[:4].sum()) / 4.0
    i += 1

    # 1x: TEMKINLI kacis -- rakipleri komsuluklariyla birlikte bloklu say.
    #     Rakip hareketine karsi saglam mi?
    if others and can_bomb:
        gs2 = dict(gs)
        f2 = np.array(gs["field"], copy=True)
        for (ox, oy) in others:
            for dx, dy in ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)):
                nx, ny = ox + dx, oy + dy
                if 0 <= nx < f2.shape[0] and 0 <= ny < f2.shape[1] \
                        and f2[nx, ny] == FREE and (nx, ny) != (ax, ay):
                    f2[nx, ny] = CRATE          # gecilemez say
        gs2["field"] = f2
        out[i] = float(bool(escape_directions(
            gs2, extra_bomb_at=(ax, ay)).any()))
    i += 1

    # 2x: en yakin rakip bitisik mi / 3 kare icinde mi
    if others:
        d = min(abs(ox - ax) + abs(oy - ay) for (ox, oy) in others)
        out[i] = float(d <= 1)
        out[i + 1] = float(d <= 3)
    i += 2

    # 1x: hayatta kalan rakip sayisi (0..3 -> 0..1)
    out[i] = len(others) / 3.0
    return out


register(FeatureSet(
    "v0c_task4", (_V0C_LEN,), _v0c, dir_slots=_V0C_DIR_SLOTS,
    doc="v0b_task2 + rakip farkindaligi: rakip yonu, kacis yolu SAYISI, "
        "rakip hareketine karsi TEMKINLI kacis, rakip yakinligi. "
        "EXP-015'te olculen intihar acigini (0.591 vs rule_based 0.528) hedefler."))


# ==========================================================================
# v1_handcrafted -- ASIL elle tasarlanmis set
# ==========================================================================





_V1_LEN = 25


def _v1(gs: dict) -> np.ndarray:
    field = gs["field"]
    _n, _sc, can_bomb, (ax, ay) = gs["self"]
    coins = gs["coins"]
    others = [o[3] for o in gs.get("others", [])]

    out = np.zeros(_V1_LEN, dtype=np.float32)
    i = 0

    # 1-4: guvenli ve gecilebilir ilk hamleler (UP, RIGHT, DOWN, LEFT)
    safe_moves = escape_directions(gs)
    out[i:i + 4] = safe_moves[:4]
    i += 4

    # 5: su an tehlikede miyim
    d = danger_map(gs)
    out[i] = 1.0 if d[ax, ay] < SAFE else 0.0
    i += 1

    # 6: bomba atabilir miyim
    out[i] = float(bool(can_bomb))
    i += 1

    # 7-8: en yakin coin yonu (normalize dx, dy)
    """"""
    if coins:
        nearest = min(coins, key=lambda c: abs(c[0] - ax) + abs(c[1] - ay))
        dx, dy = nearest[0] - ax, nearest[1] - ay
        norm = max(abs(dx), abs(dy), 1)
        out[i], out[i + 1] = dx / norm, dy / norm
    i += 2

    # 9-10: en yakin sandik yonu (normalize dx, dy)
    crate_positions = np.argwhere(field == CRATE)
    if len(crate_positions) > 0:
        dists = np.abs(crate_positions[:, 0] - ax) + np.abs(crate_positions[:, 1] - ay)
        nearest_crate = crate_positions[np.argmin(dists)]
        dx, dy = nearest_crate[0] - ax, nearest_crate[1] - ay
        norm = max(abs(dx), abs(dy), 1)
        out[i], out[i + 1] = dx / norm, dy / norm
    i += 2

    # 11: dynamic recency placeholder (stateless FeatureSet sozlesmesi geregi 0.0)
    out[i] = 0.0
    i += 1

    # 12: buraya bomba atarsam kesin kacis yolu var mi
    escape_with_bomb = escape_directions(gs, extra_bomb_at=(ax, ay))
    out[i] = 1.0 if escape_with_bomb.any() else 0.0
    i += 1

    # 13-16: komsu kareler su an tehlikeli mi (UP, RIGHT, DOWN, LEFT)
    from .gamelogic import DELTAS
    for mv in MOVES:
        dx, dy = DELTAS[mv]
        nx, ny = ax + dx, ay + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            out[i] = 1.0 if d[nx, ny] < SAFE else 0.0
        else:
            out[i] = 1.0
        i += 1

    # 17: bu karenin bomba degeri (sandik + 2*rakip)
    crates_hit, enemies_hit = bomb_impact(gs, ax, ay)
    potential = crates_hit + enemies_hit * 2
    out[i] = potential / 5.0
    i += 1

    # 18: bosa bomba bayragi (0 hedef)
    out[i] = 1.0 if potential == 0 else 0.0
    i += 1

    # 19: bias terimi
    out[i] = 1.0
    i += 1

    # ======================================================================
    # YENI EKLENEN FEATURE'LAR (19 -> 25)
    # ======================================================================

    # 20-23: SAF GECILEBILIRLIK (Walkability: UP, RIGHT, DOWN, LEFT)
    # Tehlikeden bagimsiz salt harita ve engel durumu (duvar toslamayi engeller)
    passable = free_mask(gs)
    for mv in MOVES:
        dx, dy = DELTAS[mv]
        nx, ny = ax + dx, ay + dy
        walkable = bool(passable[nx, ny]) if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] else False
        out[i] = 1.0 if walkable else 0.0
        i += 1

    # 24-25: EN YAKIN RAKIP YONU (Normalized dx, dy)
    # Modelin rakipleri gormesini ve pasif kalmamasini saglar
    if others:
        nearest_enemy = min(others, key=lambda o: abs(o[0] - ax) + abs(o[1] - ay))
        edx, edy = nearest_enemy[0] - ax, nearest_enemy[1] - ay
        enorm = max(abs(edx), abs(edy), 1)
        out[i], out[i + 1] = edx / enorm, edy / enorm
    i += 2

    return out


#: dir_slots: D4 simetri donusumunde 4 yonlu one-hot/dilim alanlar.
#: [0:4]   -> safe moves
#: [12:16] -> adjacent danger
#: [19:23] -> plain walkability
_V1_DIR_SLOTS = ((0, 4), (12, 4), (19, 4))

register(FeatureSet(
    "v1_handcrafted", (_V1_LEN,), _v1, dir_slots=_V1_DIR_SLOTS,
    doc="Umut'un 25 sinyalli gelismis feature seti. "
        "gamelogic uzerine kurulu; saf walkability ve rakip yonu icerir."))



# ==========================================================================
# v2_planes -- CNN/DQN icin kanal yigini                        [IRMAK]
# ==========================================================================

V2_CHANNELS = 9
V2_SHAPE = (V2_CHANNELS, s.COLS, s.ROWS)
V2_CHANNEL_NAMES = [
    "wall", "crate", "coin", "self", "others",
    "bomb_urgency", "danger", "explosion_now", "can_bomb",
]


def _v2(gs: dict) -> np.ndarray:
    field = gs["field"]
    p = np.zeros(V2_SHAPE, dtype=np.float32)

    p[0] = (field == WALL)
    p[1] = (field == CRATE)
    for (cx, cy) in gs["coins"]:
        p[2, cx, cy] = 1.0

    _n, _sc, can_bomb, (ax, ay) = gs["self"]
    p[3, ax, ay] = 1.0
    for _nm, _s2, _b, (ox, oy) in gs["others"]:
        p[4, ox, oy] = 1.0

    # bomba aciliyeti: t=0 -> 1.0 (bu adim patlar), t=BOMB_TIMER -> ~0
    for (bx, by), t in gs["bombs"]:
        p[5, bx, by] = max(p[5, bx, by], 1.0 - t / float(s.BOMB_TIMER))

    # tehlike: d=0 -> 1.0, d buyudukce azalir, guvenli -> 0
    d = danger_map(gs).astype(np.float32)
    p[6] = np.where(d < SAFE, 1.0 / (1.0 + d), 0.0)

    p[7] = np.asarray(gs["explosion_map"], dtype=np.float32) >= 1
    p[8] = float(bool(can_bomb))       # sabit duzlem
    return p


register(FeatureSet(
    "v2_planes", V2_SHAPE, _v2,
    doc="9 kanalli (C,17,17) yigin -- CNN/DQN icin. Kanal 6 (danger) "
        "gamelogic.danger_map'ten gelir; explosion_map tek basina kullanilmaz "
        "(oldurucu karelerin ~%60'ini kacirir, bkz. tests/test_gamelogic.py)."))


# ==========================================================================
# Simetri -- D4 grubu (4 donme x 2 aynalama). Ornek verimliligi icin.
# ==========================================================================

#: MOVES sirasi [UP, RIGHT, DOWN, LEFT] -> 90 derece saat yonu donusu
#: bu listede tek adim kaydirmadir. WAIT ve BOMB degismez.
def transform_planes(x: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """(C, W, H) duzlem yigininı k*90 derece dondur, istege bagli aynala."""
    y = np.rot90(x, k=k, axes=(1, 2))
    if flip:
        y = y[:, ::-1, :]
    return np.ascontiguousarray(y)


def transform_flat(x: np.ndarray, k: int, flip: bool,
                   dir_slots) -> np.ndarray:
    """Duz feature vektorunu k*90 derece dondur, istege bagli aynala.

    Yalnizca `dir_slots`'ta belirtilen 4'lu yon one-hot dilimleri permute
    edilir; geri kalan her sey (tehlikede miyim, bomba atabilir miyim, ...)
    donusum altinda DEGISMEZ.

    Permutasyon `transform_action` ile AYNI olmak zorunda -- aksi halde
    feature ile etiket birbirini tutmaz ve veri sessizce zehirlenir.
    tests/test_features.py::test_flat_symmetry_matches_action bunu dogrular.
    """
    y = np.array(x, dtype=x.dtype, copy=True)
    for start, n in dir_slots:
        if n != 4:
            raise ValueError(f"yon dilimi 4 uzunlugunda olmali, {n} verildi")
        src = x[start:start + 4]
        out = np.empty_like(src)
        for i in range(4):
            j = (i + k) % 4
            if flip:
                j = {0: 0, 1: 3, 2: 2, 3: 1}[j]
            out[j] = src[i]           # i yonundeki deger j yonune tasinir
        y[start:start + 4] = out
    return y


def transform_features(x: np.ndarray, k: int, flip: bool,
                       fs: "FeatureSet") -> np.ndarray:
    """Feature setine gore dogru donusumu uygular (duzlem veya duz vektor)."""
    if len(fs.shape) == 3:
        return transform_planes(x, k, flip)
    if len(fs.shape) == 1 and fs.dir_slots:
        return transform_flat(x, k, flip, fs.dir_slots)
    raise ValueError(
        f"{fs.name}: simetri donusumu tanimsiz "
        f"(shape={fs.shape}, dir_slots={fs.dir_slots})")


def supports_symmetry(fs: "FeatureSet") -> bool:
    return len(fs.shape) == 3 or bool(fs.dir_slots)


def transform_action(action_idx: int, k: int, flip: bool) -> int:
    """Aksiyon indeksini ayni donusumle esle.

    UYARI: yanlis esleme veri setini SESSIZCE zehirler -- once
    tests/test_features.py'daki simetri testini yesillendir.
    """
    from .gamelogic import ACTIONS
    a = ACTIONS[action_idx]
    if a in ("WAIT", "BOMB"):
        return action_idx
    i = MOVES.index(a)
    # transform_planes: ONCE rot90(k, axes=(1,2)), SONRA flip (x ekseni).
    # rot90(k=1) altinda yer degistirme (0,-1)[UP] -> (+1,0)[RIGHT], yani
    # MOVES=[UP,RIGHT,DOWN,LEFT] icinde indeks ILERI kayar.
    # (tests/test_features.py::test_symmetry_action_mapping ile dogrulandi)
    i = (i + k) % 4
    if flip:
        i = {0: 0, 1: 3, 2: 2, 3: 1}[i]   # x aynalanir -> RIGHT <-> LEFT
    return ACTIONS.index(MOVES[i])
