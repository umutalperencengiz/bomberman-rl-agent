"""Oyun kurallarinin saf-fonksiyon hali. Her feature bunun ustune kurulur.

Sahibi: IRMAK (infra)

NEDEN AYRI BIR MODUL:
Tehlike hesabi bu projenin en kolay yanlis yapilan parcasi ve HER model ona
bagimli. Tek yerde, test edilmis olarak durmali; feature dosyalarinda tekrar
tekrar yeniden yazilmamali.

Referans: docs/GAME_MECHANICS.md
"""

from __future__ import annotations

import numpy as np

import settings as s

WALL, FREE, CRATE = -1, 0, 1
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}

#: (dx, dy) -- image koordinati: y ARTARSA asagi gidilir (GUI'de asagi).
DELTAS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0),
          "WAIT": (0, 0), "BOMB": (0, 0)}
MOVES = ["UP", "RIGHT", "DOWN", "LEFT"]

#: "guvenli" isaretleyicisi. Gercek tehlike degerleri her zaman < BOMB_TIMER+2.
SAFE = 99


# --------------------------------------------------------------------------
# Patlama geometrisi
# --------------------------------------------------------------------------

def blast_coords(field: np.ndarray, x: int, y: int,
                 power: int = s.BOMB_POWER) -> list[tuple[int, int]]:
    """items.Bomb.get_blast_coords ile BIREBIR ayni.

    DIKKAT: dongü yalnizca DUVAR'da (-1) kirilir -- SANDIKLAR PATLAMAYI
    DURDURMAZ. Sandiklar zamanla yikilsa da duvarlar sabit oldugu icin bu
    fonksiyon gelecekteki bir patlama icin de KESIN sonuc verir; guncel
    field ile hesaplamak guvenlidir.
    """
    out = [(x, y)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, power + 1):
            nx, ny = x + i * dx, y + i * dy
            if field[nx, ny] == WALL:
                break
            out.append((nx, ny))
    return out


# --------------------------------------------------------------------------
# Tehlike haritasi  --  bu projenin en kritik fonksiyonu
# --------------------------------------------------------------------------

def danger_map(game_state: dict) -> np.ndarray:
    """Her kare icin "kac adim sonra olumcul olacak" (SAFE = tehlike yok).

    Semantik (docs/GAME_MECHANICS.md 2):
      d = danger_map[x, y]
      -> kare, `S+d` ve `S+d+1` adimlarinin SONUNDA olumculdur (S = su anki adim)
      -> d == 0  =>  BU ADIMIN SONUNDA oldurur

    Iki kaynak birlestirilir:
      1. `bombs`      : timer t olan bomba, adim S+t sonunda patlar -> d = t
      2. `explosion_map == 1` : zaten yaniyor, bu adim oldurur      -> d = 0

    !!! `explosion_map` TEK BASINA YETMEZ. Patlamak uzere olan bombayi (t=0)
    hicbir zaman gostermez; o kare icin explosion_map 0'dir ama ajan o adimin
    sonunda olur. KILLED_SELF vakalarinin cogunun sebebi budur.
    """
    field = game_state["field"]
    d = np.full(field.shape, SAFE, dtype=np.int16)

    for (bx, by), t in game_state["bombs"]:
        for (x, y) in blast_coords(field, bx, by):
            if t < d[x, y]:
                d[x, y] = t

    em = game_state.get("explosion_map")
    if em is not None:
        d[np.asarray(em) >= 1] = 0

    return d


def lethal_at(danger: np.ndarray, x: int, y: int, j: int) -> bool:
    """`S+j` adiminin sonunda (x, y) olumcul mu?

    d(u) degerindeki kare `S+d` ve `S+d+1` adimlarinda olumcul oldugundan,
    `S+j`'de olumcul olmasi icin  d(u) in {j-1, j}  olmalidir.
    """
    dv = int(danger[x, y])
    return dv == j or dv == j - 1


# --------------------------------------------------------------------------
# Gecilebilirlik ve mesafeler
# --------------------------------------------------------------------------

def free_mask(game_state: dict, block_agents: bool = True,
              block_bombs: bool = True) -> np.ndarray:
    """environment.GenericWorld.tile_is_free ile ayni: bos + bomba yok + ajan yok.

    Coin'ler ENGEL DEGILDIR (altlarindaki field zaten 0).
    """
    field = game_state["field"]
    m = field == FREE
    if block_bombs:
        for (bx, by), _t in game_state["bombs"]:
            m[bx, by] = False
    if block_agents:
        for _n, _sc, _b, (ox, oy) in game_state["others"]:
            m[ox, oy] = False
    return m


def bfs_distances(passable: np.ndarray, starts) -> np.ndarray:
    """`starts`'tan cok-kaynakli BFS. Ulasilamayan = -1.

    Duvarlari/sandiklari DOLASAN gercek mesafe verir -- Manhattan mesafesi
    bu oyunda yanilticidir (labirent yapisi yuzunden).
    """
    w, h = passable.shape
    dist = np.full((w, h), -1, dtype=np.int32)
    frontier = []
    for (x, y) in starts:
        if 0 <= x < w and 0 <= y < h and dist[x, y] < 0:
            dist[x, y] = 0
            frontier.append((x, y))
    while frontier:
        nxt = []
        for (x, y) in frontier:
            nd = dist[x, y] + 1
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and dist[nx, ny] < 0 \
                        and passable[nx, ny]:
                    dist[nx, ny] = nd
                    nxt.append((nx, ny))
        frontier = nxt
    return dist


def direction_to_nearest(game_state: dict, targets, passable=None):
    """En yakin hedefe goturen ilk hamle(ler) -- one-hot uzunluk 5.

    Sira: [UP, RIGHT, DOWN, LEFT, YOK]. Birden fazla yon esit derecede iyiyse
    HEPSI 1 olur (deterministik tek aksiyon dondurmemek icin bilincli tercih,
    bkz. asagidaki not).

    KURAL NOTU: proje "deterministik olarak en iyi aksiyonu donduren feature"i
    yasakliyor. Bu fonksiyon bir YON BILGISI uretir (proje metninde acikca
    onerilen "pathfinding features, e.g. the direction to move which brings you
    closest to the nearest coin" ornegi) -- nihai aksiyonu secmez, bomba/bekle
    kararini vermez ve tehlikeyi hesaba katmaz. Karar modele aittir.
    """
    targets = list(targets)
    out = np.zeros(5, dtype=np.float32)
    if not targets:
        out[4] = 1.0
        return out
    if passable is None:
        passable = free_mask(game_state)

    _n, _sc, _b, (ax, ay) = game_state["self"]
    # Hedeflerden geriye dogru BFS -> her kareden hedefe mesafe
    p = passable.copy()
    for (tx, ty) in targets:
        p[tx, ty] = True          # hedef karesi (or. coin) gecilebilir sayilir
    dist = bfs_distances(p, targets)

    best = None
    for i, mv in enumerate(MOVES):
        dx, dy = DELTAS[mv]
        nx, ny = ax + dx, ay + dy
        if not passable[nx, ny] and (nx, ny) not in targets:
            continue
        dv = dist[nx, ny]
        if dv < 0:
            continue
        if best is None or dv < best:
            best = dv
    if best is None:
        out[4] = 1.0
        return out
    for i, mv in enumerate(MOVES):
        dx, dy = DELTAS[mv]
        nx, ny = ax + dx, ay + dy
        if (passable[nx, ny] or (nx, ny) in targets) and dist[nx, ny] == best:
            out[i] = 1.0
    return out


# --------------------------------------------------------------------------
# Kacis analizi  --  hayat kurtarici feature'larin temeli
# --------------------------------------------------------------------------

def escape_directions(game_state: dict, horizon: int = s.BOMB_TIMER + 4,
                      extra_bomb_at=None) -> np.ndarray:
    """Her ilk hamle icin: o hamleyle baslayan GUVENLI bir kacis var mi?

    Donus: uzunluk 5 float dizi -- [UP, RIGHT, DOWN, LEFT, WAIT]

    Zaman-genisletilmis BFS: (kare, j) durumu, j = kacinci adimdayiz.
    `S+j` adiminin sonunda bulundugumuz karede `lethal_at(...)` ise oluruz.
    Tum tehlikeler gecene kadar (j > max danger + 1) hayatta kalabiliyorsak
    o ilk hamle guvenlidir.

    `extra_bomb_at=(x, y)` verilirse, oraya SIMDI bomba birakilmis gibi
    hesaplanir -> "buraya bomba atarsam kacabilir miyim?" feature'i.
    Intihari dogrudan engelleyen en degerli sinyal budur.
    """
    field = game_state["field"]
    danger = danger_map(game_state)

    if extra_bomb_at is not None:
        bx, by = extra_bomb_at
        for (x, y) in blast_coords(field, bx, by):
            if s.BOMB_TIMER < danger[x, y]:
                danger[x, y] = s.BOMB_TIMER

    passable = free_mask(game_state)
    if extra_bomb_at is not None:
        # Kendi biraktigimiz bombanin uzerinde duruyoruz; uzerinden inebiliriz
        # ama geri cikamayiz -> hedef kare olarak gecilemez isaretle.
        passable[extra_bomb_at[0], extra_bomb_at[1]] = False

    _n, _sc, _b, (ax, ay) = game_state["self"]
    max_d = int(danger[danger < SAFE].max()) if (danger < SAFE).any() else -1
    if max_d < 0:
        return np.ones(5, dtype=np.float32)      # hic tehlike yok
    limit = min(horizon, max_d + 2)

    out = np.zeros(5, dtype=np.float32)
    first_moves = MOVES + ["WAIT"]

    for i, mv in enumerate(first_moves):
        dx, dy = DELTAS[mv]
        sx, sy = ax + dx, ay + dy
        if mv != "WAIT" and not passable[sx, sy]:
            continue
        if lethal_at(danger, sx, sy, 0):
            continue
        # (kare, j) uzerinde BFS
        seen = {(sx, sy)}
        frontier = [(sx, sy)]
        j = 1
        survived = False
        while frontier and j <= limit:
            nxt = []
            for (x, y) in frontier:
                for mv2 in MOVES + ["WAIT"]:
                    ddx, ddy = DELTAS[mv2]
                    nx, ny = x + ddx, y + ddy
                    if mv2 != "WAIT" and not passable[nx, ny]:
                        continue
                    if lethal_at(danger, nx, ny, j):
                        continue
                    if (nx, ny, j) in seen:
                        continue
                    seen.add((nx, ny, j))
                    nxt.append((nx, ny))
            if not nxt:
                break
            frontier = nxt
            if j >= limit:
                survived = True
                break
            j += 1
        else:
            survived = bool(frontier)
        if survived or (frontier and j > limit):
            out[i] = 1.0
    return out


def bomb_impact(game_state: dict, x: int, y: int) -> tuple[int, int]:
    """(x, y)'ye simdi bomba birakilirsa: (yikilacak sandik, vurulacak rakip).

    Rakip sayisi ust sinirdir -- rakipler kacabilir. Yine de "bu bomba ise
    yarar mi?" sorusuna dogru buyuklukte bir sinyal verir.
    """
    field = game_state["field"]
    coords = set(blast_coords(field, x, y))
    crates = sum(1 for (cx, cy) in coords if field[cx, cy] == CRATE)
    enemies = sum(1 for _n, _s, _b, (ox, oy) in game_state["others"]
                  if (ox, oy) in coords)
    return crates, enemies
