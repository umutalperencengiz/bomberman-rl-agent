"""Odul semalari ve custom event'ler.

SAHIPLIK
  * Registry, sozlesme, `r00_pure_game`, ornek custom event'ler -> IRMAK (infra)
  * Odul semalari r02+, sema ablation'lari (EXP-004)            -> UMUT

UYARILAR (docs/GAME_MECHANICS.md 6)
  1. Intihar HEM `KILLED_SELF` HEM `GOT_KILLED` uretir. Ikisine de tam ceza
     verirsen cezayi farkinda olmadan 2x yaparsin -> asiri korkak, bomba
     atmayan politika. Bilincli sec, EXP-004'te olc.
  2. Yardimci oduller turnuvada YOK. Performans daima gercek skorla olculur
     (bbrl/eval.py). Shaped odul yalnizca ogrenmeyi hizlandirmak icindir.
  3. Teori: yardimci odul mumkun oldugunca STATE'e bagli olsun, oraya goturen
     AKSIYONA degil (potential-based shaping, Ng et al. 1999). Aksi halde
     ileri-geri gidip odul farm'layan politikalar dogar.
"""

from __future__ import annotations

from typing import Callable

import events as e
import settings as s

import numpy as np

from .gamelogic import (
    CRATE, SAFE, bfs_distances, blast_coords, bomb_impact, danger_map,
    escape_directions, free_mask,
)

# --------------------------------------------------------------------------
# Custom event adlari
# --------------------------------------------------------------------------

SUICIDAL_BOMB = "SUICIDAL_BOMB"      # kacisi olmayan bomba birakildi
USELESS_BOMB = "USELESS_BOMB"        # hicbir sandiga/rakibe degmeyen bomba
ESCAPED_DANGER = "ESCAPED_DANGER"    # tehlikeliden guvenli kareye gecildi
ENTERED_DANGER = "ENTERED_DANGER"    # guvenliden tehlikeli kareye girildi
STAYED_IN_DANGER = "STAYED_IN_DANGER"  # tehlikede kalindi (kacis varken)

# EXP-010 ile eklenenler
GOOD_BOMB = "GOOD_BOMB"              # kacis VAR ve en az bir sandik/rakip vurur
SAFE_AFTER_BOMB = "SAFE_AFTER_BOMB"  # kendi bombasinin menzilinden CIKTI
BOMB_IDLE = "BOMB_IDLE"              # bomba hazirken uzun sure atmadi

CUSTOM_EVENTS = [SUICIDAL_BOMB, USELESS_BOMB, ESCAPED_DANGER,
                 ENTERED_DANGER, STAYED_IN_DANGER,
                 GOOD_BOMB, SAFE_AFTER_BOMB, BOMB_IDLE]

#: `BOMB_IDLE` esigi -- bomba hazirken bu kadar adim atmazsa tetiklenir.
BOMB_IDLE_STEPS = 12




# --------------------------------------------------------------------------
# Custom event uretimi
# --------------------------------------------------------------------------

def custom_events(old_state: dict | None, action: str,
                  new_state: dict | None, events: list[str],
                  memory: dict | None = None) -> list[str]:
    """Yalnizca EK event adlarini dondurur. `events`'i MUTATE ETMEZ.

    `old_state` = act() cagrildigi state, `action` = secilen aksiyon,
    `new_state` = adim sonrasi state (olduysa None olabilir).

    `memory`: cagiranin sahip oldugu, TUR boyunca yasayan sozluk. Adimlar
    arasi hafiza gerektiren event'ler (SAFE_AFTER_BOMB, BOMB_IDLE) icin.
    Verilmezse o event'ler uretilmez -- fonksiyon yine saf kalir.
    """
    extra: list[str] = []
    if old_state is None:
        return extra

    _n, _sc, can_bomb, (ax, ay) = old_state["self"]

    # --- bomba kalitesi (yalnizca bomba GERCEKTEN birakildiysa) -----------
    if e.BOMB_DROPPED in events:
        crates, enemies = bomb_impact(old_state, ax, ay)
        survivable = bool(escape_directions(
            old_state, extra_bomb_at=(ax, ay)).any())
        if crates == 0 and enemies == 0:
            extra.append(USELESS_BOMB)
        # "buraya bomba atarsam kacabilir miyim?" -- intiharin ONCUL sinyali
        if not survivable:
            extra.append(SUICIDAL_BOMB)
        elif crates > 0 or enemies > 0:
            # SUICIDAL_BOMB'un pozitif ikizi: kacisi olan VE ise yarayan bomba.
            # Yalnizca cezalandirmak bombayi tek yonlu korkutuyordu (EXP-009'da
            # ajan tamamen pasifist oldu, bomba orani %2'ye dustu).
            extra.append(GOOD_BOMB)

    # --- adimlar arasi hafiza gerektirenler --------------------------------
    if memory is not None:
        if e.BOMB_DROPPED in events:
            memory["own_bomb"] = (ax, ay)
            memory["idle"] = 0
        else:
            memory["idle"] = int(memory.get("idle", 0)) + 1

        # SAFE_AFTER_BOMB: kendi bombamizin menzilinden CIKMAK. Task 2'nin
        # asil becerisi bu -- "bombayi at, kac" dizisinin tamamlanmasi.
        own = memory.get("own_bomb")
        if own is not None and new_state is not None:
            still_ticking = any((bx, by) == own for (bx, by), _t
                                in new_state["bombs"])
            _n2, _s2, _b2, (nx, ny) = new_state["self"]
            if still_ticking:
                if (nx, ny) not in blast_coords(new_state["field"], *own):
                    extra.append(SAFE_AFTER_BOMB)
                    memory["own_bomb"] = None      # tur basina bir kez
            else:
                memory["own_bomb"] = None          # patladi/kayboldu

        # BOMB_IDLE: bomba hazirken uzun sure atmamak. UYARI -- bu event
        # bilgi tasimayan kaba bir baski; EXP-010'da bilerek AYRI bir
        # varyantta test ediliyor (zarar vermesini bekliyoruz).
        if can_bomb and memory.get("idle", 0) >= BOMB_IDLE_STEPS:
            extra.append(BOMB_IDLE)
            memory["idle"] = 0

    # --- tehlike gecisleri ------------------------------------------------
    if new_state is not None:
        d_old = danger_map(old_state)
        d_new = danger_map(new_state)
        _n2, _s2, _b2, (nx, ny) = new_state["self"]
        was = d_old[ax, ay] < s.BOMB_TIMER + 2
        now = d_new[nx, ny] < s.BOMB_TIMER + 2
        if was and not now:
            extra.append(ESCAPED_DANGER)
        elif not was and now:
            extra.append(ENTERED_DANGER)
        elif was and now and (nx, ny) == (ax, ay):
            if escape_directions(old_state)[:4].any():
                extra.append(STAYED_IN_DANGER)

    return extra


# --------------------------------------------------------------------------
# Odul semalari  --  KOD DEGIL, VERI. YAML'den secilir -> ablation bedava.
# --------------------------------------------------------------------------

REWARD_SCHEMES: dict[str, dict[str, float]] = {}


def register_scheme(name: str, table: dict[str, float], doc: str = "") -> None:
    if name in REWARD_SCHEMES:
        raise KeyError(f"odul semasi {name!r} zaten kayitli")
    REWARD_SCHEMES[name] = dict(table)
    SCHEME_DOCS[name] = doc


SCHEME_DOCS: dict[str, str] = {}


# --- r00: SADECE gercek oyun odulleri -- durust taban --------------------
register_scheme("r00_pure_game", {
    e.COIN_COLLECTED: float(s.REWARD_COIN),
    e.KILLED_OPPONENT: float(s.REWARD_KILL),
}, doc="Shaping YOK. Turnuvadaki gercek odul yapisi. Seyrek sinyal -> "
       "yavas ogrenir ama overfit riski sifir. Her shaping'in kiyaslanacagi taban.")

# --- r01: minimal shaping ------------------------------------------------
register_scheme("r01_minimal", {
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.CRATE_DESTROYED: 0.1,
    e.COIN_FOUND: 0.2,
    e.KILLED_SELF: -5.0,
    e.GOT_KILLED: 0.0,          # <-- intiharda cifte ceza OLMASIN diye 0
    e.INVALID_ACTION: -0.2,
    e.WAITED: -0.05,
    SUICIDAL_BOMB: -2.0,
    USELESS_BOMB: -0.3,
    ESCAPED_DANGER: 0.3,
    ENTERED_DANGER: -0.3,       # dengeli: ESCAPED ile ayni buyuklukte
    STAYED_IN_DANGER: -0.4,
}, doc="Hafif shaping. GOT_KILLED=0 cunku KILLED_SELF ile birlikte tetikleniyor; "
       "rakip bombasiyla olum bu semada ayrica cezalandirilmiyor (EXP-004'te olc).")

# --- r02: yasam cezasi -- EXP-002'deki DONMA'nin hedefli tedavisi -------
#
# Sorun: `coin-heaven`'da WAIT secilince dunyada hicbir sey degismez, yani
# ayni state -> ayni argmax -> sonsuz dongu. m01 adimlarinin %92'sini boyle
# harcadi. Q(WAIT) bu dongude  -c / (1 - gamma)  sabit noktasina oturuyor;
# c = 0 iken bu 0'dir ve hareket alternatiflerini yenmesi kolaydir.
#
# Tedavi: her adima kucuk bir maliyet koy, BEKLEMEYE daha buyugunu koy.
#   hareket  -0.02  ->  Q(dongu) = -0.4
#   bekleme  -0.15  ->  Q(dongu) = -3.0   <- artik derin bir cukur
# Bekleme hareketten KESIN olarak kotu; oyalanmak da pahali.
#
# Yan fayda: Task 1'in gercek metrigi ADIM SAYISI (EXP-000: rule_based 125.19).
# Yasam cezasi tam olarak onu optimize eder -- skor zaten tavana yapisiyor.
#
# Her adim en az bir hareket/bekleme/gecersiz olayi uretir, bu yuzden yasam
# cezasi icin ayri bir mekanizmaya gerek yok; olay tablosuna yazmak yeterli.
_LIVING = -0.02
register_scheme("r02_living_cost", {
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.CRATE_DESTROYED: 0.1,
    e.COIN_FOUND: 0.2,
    e.KILLED_SELF: -5.0,
    e.GOT_KILLED: 0.0,          # KILLED_SELF ile birlikte gelir -> cifte ceza yok
    e.MOVED_UP: _LIVING,
    e.MOVED_DOWN: _LIVING,
    e.MOVED_LEFT: _LIVING,
    e.MOVED_RIGHT: _LIVING,
    e.WAITED: -0.15,            # hareketten KESIN olarak kotu
    e.INVALID_ACTION: -0.30,
    SUICIDAL_BOMB: -2.0,
    USELESS_BOMB: -0.3,
    ESCAPED_DANGER: 0.3,
    ENTERED_DANGER: -0.3,
    STAYED_IN_DANGER: -0.4,
}, doc="r01 + adim basina yasam cezasi. Donmayi kirmayi ve adim sayisini "
       "azaltmayi hedefler. EXP-007'de r01'e karsi olculuyor.")

# --- r03: Task 2 (sandikli, rakipsiz) -----------------------------------
#
# !!! r02'yi Task 2'ye OLDUGU GIBI sokmak PERVERS bir tesvik yaratir:
#       400 adim hayatta kalmak :  400 * -0.02      = -8.0
#       ilk adimda intihar      :  KILLED_SELF -5.0 = -5.0
#     Yani "bos bos hayatta kalmak" intihardan PAHALI. Sandik kirmayi
#     cozemeyen bir ajan icin kendini oldurmek rasyonel hale gelir.
#     EXP-008'de ajan zaten %100 intihar ediyordu; bu sema onu ODULLENDIRIRDI.
#
# Duzeltme: olum cezasi, en kotu hayatta kalma senaryosundan BELIRGIN olarak
# daha aci olmali.  KILLED_SELF = -15  <  -8.
#
# Ust sinir kontrolu (iyi bir tur): 117 sandik * 0.15 + 9 coin * 1.0
#   = 17.6 + 9 = +26.6,  eksi 400 * 0.02 = 8  ->  net +18.6  >>  -15.
# Yani "calis ve yasa" acik ara en iyi secenek, "idle" ve "intihar" ikisi de
# kotu, ve intihar daha kotu. Istenen siralama bu.
register_scheme("r03_task2", {
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.CRATE_DESTROYED: 0.15,    # Task 2'nin asil isi -- coin'ler sandik altinda
    e.COIN_FOUND: 0.30,
    e.KILLED_SELF: -15.0,       # <-- 400 adimlik yasam cezasindan (-8) AGIR
    e.GOT_KILLED: 0.0,          # KILLED_SELF ile birlikte gelir
    e.MOVED_UP: _LIVING,
    e.MOVED_DOWN: _LIVING,
    e.MOVED_LEFT: _LIVING,
    e.MOVED_RIGHT: _LIVING,
    e.WAITED: -0.15,
    e.INVALID_ACTION: -0.30,
    e.SURVIVED_ROUND: 1.0,      # hayatta bitirmek acikca odullendirilir
    SUICIDAL_BOMB: -3.0,        # kacisi olmayan bombayi ONCEDEN cezalandir
    USELESS_BOMB: -0.3,
    ESCAPED_DANGER: 0.3,
    ENTERED_DANGER: -0.3,
    STAYED_IN_DANGER: -0.4,
}, doc="Task 2. r02'nin pervers tesvigini duzeltir: olum (-15) bos hayatta "
       "kalmaktan (-8) daha aci. Sandik kirma odulu artirildi cunku classic'te "
       "coin'ler sandik altinda.")

# --- r04: EXP-010, oranlari LITERATURE gore duzeltilmis --------------------
#
# EXP-009'da ajan pasifist oldu (bomba %2, sandik 15.7/116.9). Sebep sayisal:
#
#   ICAART 2018 (Groningen, Bomberman'de Q-learning) Tablo 1:
#       rakip oldurme +100 | duvar kirma +30 | hareket -1 | gecersiz -2 | olme -300
#   -> olum : duvar orani  =  -10 : 1   (bolum ~150 adim)
#   bizim r03_task2'de     =  -100 : 1        <-- 10 KAT fazla agir
#
# Oyunun kendi olcegine (oldurme = 5) sabitleyip makalenin oranlarini
# uygulayinca: sandik 5*30/100 = 1.5, olum 5*300/100 = -15, hareket -0.05,
# gecersiz -0.1. Yani OLUM CEZASI dogruydu, SANDIK ODULU 10 kat kucuktu.
#
# !!! BOLUM UZUNLUGU DUZELTMESI !!!
# Makalenin bolumleri ~150 adim, bizimki 400. `action -1` degerini yalnizca
# olcek katsayisiyla cevirince yasam cezasi -0.05 cikiyor ve
#     bos tur = 400 * -0.05 = -20   <   olum = -15
# yani OLMEK BOS DURMAKTAN UCUZ hale geliyor -- r03'te duzelttigimiz pervers
# tesvigin aynisi geri geliyor. (tests/test_features.py::test_reward_ratio_sanity
# bunu yakaladi.)
# Makalenin KENDI ic tutarliligi iki sarti birden istiyor:
#     olum : duvar = 10 : 1        ve      olum ~ 2 x (bos tur maliyeti)
# 400 adim icin:  living = olum / 800 = -15/800 ~ -0.02
_LIVING4 = -0.02
register_scheme("r04_ratio", {
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.CRATE_DESTROYED: 1.5,     # <-- r03'te 0.15 idi
    e.COIN_FOUND: 0.5,
    e.KILLED_SELF: -15.0,
    e.GOT_KILLED: 0.0,
    e.MOVED_UP: _LIVING4,
    e.MOVED_DOWN: _LIVING4,
    e.MOVED_LEFT: _LIVING4,
    e.MOVED_RIGHT: _LIVING4,
    e.WAITED: -0.15,
    e.INVALID_ACTION: -0.10,
    e.SURVIVED_ROUND: 1.0,
    SUICIDAL_BOMB: -3.0,
    USELESS_BOMB: -0.3,
    ESCAPED_DANGER: 0.3,
    ENTERED_DANGER: -0.3,
    STAYED_IN_DANGER: -0.4,
}, doc="EXP-010 tabani. ICAART 2018 oranlarina gore duzeltildi: sandik 1.5 "
       "(r03'te 0.15), olum -15 -> olum:sandik = -10:1.")

# --- r05: r04 + iyi bomba odulu (GOOD_BOMB, SAFE_AFTER_BOMB) -------------
register_scheme("r05_good_bomb", {
    **REWARD_SCHEMES["r04_ratio"],
    GOOD_BOMB: 0.5,             # kacisi olan VE ise yarayan bomba
    SAFE_AFTER_BOMB: 1.0,       # kendi bombasinin menzilinden cikmak
}, doc="r04 + bombanin POZITIF sinyalleri. Simdiye kadar bombayi yalnizca "
       "cezalandiriyorduk (SUICIDAL/USELESS); tek yonlu baski pasifizm uretti.")

# --- r06: r05 + BOMB_IDLE cezasi ----------------------------------------
register_scheme("r06_bomb_idle", {
    **REWARD_SCHEMES["r05_good_bomb"],
    BOMB_IDLE: -0.5,
}, doc="r05 + 'bomba hazirken atmazsan ceza'. Kaba ve bilgi tasimayan bir "
       "baski; EXP-010'da ZARAR VERMESINI BEKLEYEREK test ediliyor -- "
       "negatif sonuc da rapor §4 icin malzeme (EXP-007'deki iyimser "
       "baslatma gibi).")

# --- r07: Task 4 -- GERCEK HEDEFE hizalanmis ----------------------------
#
# !!! EXP-016'nin ana bulgusu !!!
# EK-2'de ajanin bir turda aldigi shaped odulun dagilimi olculdu:
#     sandiktan 50.3  |  coinden 1.72  |  kill'den 0.56     -> %96 SANDIK
# Gercek oyun skoru ise  coin*1 + kill*5;  sandigin skora katkisi SIFIR.
# Yani bir SANDIK CIFTLIGI BOTU egitmisiz. EXP-016'da intihari %38 dusuren
# feature'lar skoru degistirmedi -- cunku ajan zaten yanlis seyi optimize
# ediyordu.
#
# Duzeltme mantigi: sandik ARAC, amac degil. `classic`'te 9 coin sandik
# altinda oldugu icin kazmak gerekli -- ama odul, coini ORTAYA CIKARAN
# sandiga gitmeli, her sandiga degil.
#     CRATE_DESTROYED  1.5 -> 0.15   (yalnizca kazmayi baslatacak kadar)
#     COIN_FOUND       0.5 -> 2.0    (coini ACAN sandik degerli)
#     COIN_COLLECTED   1.0 -> 2.0    (gercek hedef)
#     KILLED_OPPONENT  5.0           (oyunun kendi degeri, 5 puan)
# Yeni dagilim (EK-2 sayilariyla): sandik ~5.0, coin_found ~3.4,
# coin ~3.4, kill ~0.55  -> sandik %40. Kazmak hala carpiyor ama artik
# skoru domine etmiyor.
register_scheme("r07_objective", {
    e.COIN_COLLECTED: 2.0,
    e.KILLED_OPPONENT: 5.0,
    e.CRATE_DESTROYED: 0.15,
    e.COIN_FOUND: 2.0,
    e.KILLED_SELF: -15.0,
    e.GOT_KILLED: 0.0,
    e.MOVED_UP: _LIVING4, e.MOVED_DOWN: _LIVING4,
    e.MOVED_LEFT: _LIVING4, e.MOVED_RIGHT: _LIVING4,
    e.WAITED: -0.15,
    e.INVALID_ACTION: -0.10,
    e.SURVIVED_ROUND: 1.0,
    SUICIDAL_BOMB: -3.0,
    USELESS_BOMB: -0.3,
    ESCAPED_DANGER: 0.3,
    ENTERED_DANGER: -0.3,
    STAYED_IN_DANGER: -0.4,
    GOOD_BOMB: 0.5,
    SAFE_AFTER_BOMB: 1.0,
}, doc="Task 4. Odul dagilimi gercek oyun skoruna hizalandi: sandik arac "
       "(0.15), coin ve coini acan sandik amac (2.0). EXP-016'da shaped "
       "odulun %96'si sandiktan geliyordu ve sandigin skora katkisi yok.")

# --- r09: r07 ile AYNI, yalnizca iki no-op'un FIYATI TAKAS EDILDI
#
# BULGU-01: ortamda `INVALID_ACTION` ile `WAITED` birebir ayni sey --
# ikisi de adimi yakar, ajani yerinde birakir, baska hicbir etkisi yoktur
# (environment.py 132-150). Ama r07'de tosla -0.10, bekle -0.15; yani
# "bu turu pas gec" demenin ucuz ve pahali iki yolu var ve ajan ucuz olani
# ogrenmis: trainlog_e19b_types_s2 son 2000 turda tur basina 20.00 gecersiz
# aksiyon, 2.61 bekleme -- 7.7 kat.
#
# BUYUKLUKLER AYNEN TAKAS EDILIYOR (-0.10 ve -0.15 yer degistiriyor),
# toplam ceza olcegi degismiyor. Boylece test edilen sey "cezayi artirmak"
# degil, YALNIZCA SIRALAMA. Tek degisken.
register_scheme("r09_noop_swap", dict(REWARD_SCHEMES["r07_objective"],
                                      **{e.WAITED: -0.10,
                                         e.INVALID_ACTION: -0.15}),
                doc="r07_objective + iki no-op'un fiyati takas edildi. "
                    "Bkz. BULGU-01: ortamda ayni olan iki aksiyonu farkli "
                    "fiyatlamak, ajana anlamsiz olani ogretiyor.")

# --- r08: neredeyse SEYREK -- shaping'in gercekten gerekli olup olmadigini test et
register_scheme("r08_sparse", {
    e.COIN_COLLECTED: 1.0,          # oyunun kendi degeri
    e.KILLED_OPPONENT: 5.0,         # oyunun kendi degeri
    e.KILLED_SELF: -5.0,
    e.GOT_KILLED: 0.0,
    e.SURVIVED_ROUND: 1.0,
    SUICIDAL_BOMB: -1.0,            # tek yardimci sinyal: olumcul bombayi onle
}, doc="Neredeyse saf oyun odulu. DQN ogrenebiliyorsa (EXP-011 gucunu gosterdi) "
       "yogun shaping'e gerek olmayabilir -- ve shaping yoksa ona overfit de yok.")


LOOPING = "LOOPING"                  # same tile visited >=3x in last 15 steps,
                                      # or 'BOMB' chosen 3x in last 10 actions
OSCILLATING = "OSCILLATING"          # position[-1] == position[-3] (A-B-A)
COIN_MOVEMENT = "COIN_MOVEMENT"      # moved strictly closer to nearest coin
RUNNED_FROM_BOMB = "RUNNED_FROM_BOMB"  # was in danger, now isn't
BOMB_DROPPED_INCORRECT = "BOMB_DROPPED_INCORRECT"
UMUT_CUSTOM_EVENTS = [LOOPING, OSCILLATING, COIN_MOVEMENT, RUNNED_FROM_BOMB,
                      STAYED_IN_DANGER, BOMB_DROPPED_INCORRECT]

def umut_custom_events(old_state: dict | None, action: str,
                       new_state: dict | None, events: list[str],
                       memory: dict | None = None) -> list[str]:
    extra: list[str] = []
    if old_state is None or new_state is None or memory is None:
        return extra

    _n, _sc, _b, new_pos = new_state["self"]

    pos_hist = memory.setdefault("position_history", [])
    pos_hist.append(new_pos)
    if len(pos_hist) > 15:
        pos_hist.pop(0)
    if len(pos_hist) >= 3 and pos_hist[-1] == pos_hist[-3]:
        extra.append(OSCILLATING)
    if len(pos_hist) == 15 and pos_hist.count(new_pos) >= 3:
        extra.append(LOOPING)

    if action == "BOMB" and old_state is not None:
        _n2, _s2, _b2, (bx, by) = old_state["self"]
        was_in_danger = danger_map(old_state)[bx, by] < SAFE
        crates_hit, enemies_hit = bomb_impact(old_state, bx, by)
        escape_with_bomb = escape_directions(old_state, extra_bomb_at=(bx, by))
        bad_bomb = (
            was_in_danger
            or (crates_hit + enemies_hit) == 0
            or not bool(escape_with_bomb.any())
        )
        if bad_bomb:
            extra.append(BOMB_DROPPED_INCORRECT)

    act_hist = memory.setdefault("recent_actions", [])
    act_hist.append(action)
    if len(act_hist) > 10:
        act_hist.pop(0)
    if act_hist.count("BOMB") >= 3:
        extra.append(LOOPING)

    coins = old_state["coins"]
    if coins:
        _n0, _s0, _b0, old_pos = old_state["self"]
        old_d = min(abs(old_pos[0] - c[0]) + abs(old_pos[1] - c[1]) for c in coins)
        new_d = min(abs(new_pos[0] - c[0]) + abs(new_pos[1] - c[1]) for c in coins)
        if new_d < old_d:
            extra.append(COIN_MOVEMENT)

    d_old = danger_map(old_state)
    d_new = danger_map(new_state)
    _n1, _s1, _b1, old_pos2 = old_state["self"]
    was_in_danger = d_old[old_pos2] < SAFE
    now_in_danger = d_new[new_pos] < SAFE
    if now_in_danger:
        extra.append(STAYED_IN_DANGER)
    if was_in_danger and not now_in_danger:
        extra.append(RUNNED_FROM_BOMB)

    return extra


register_scheme("r09_umut_shaped", {
    e.COIN_COLLECTED: 40.0,
    e.KILLED_OPPONENT: 40.0,
    e.CRATE_DESTROYED: 8.0,
    e.COIN_FOUND: 15.0,
    e.KILLED_SELF: -40.0,
    e.GOT_KILLED: -40.0,
    e.BOMB_DROPPED: 1.0,
    e.WAITED: -1.0,
    e.INVALID_ACTION: -1.0,
    e.SURVIVED_ROUND: 5.0,
    GOOD_BOMB: 10.0, #when 10 it thinks a lot 
    BOMB_DROPPED_INCORRECT: -20.0,   # exact port, replaces SUICIDAL_BOMB/USELESS_BOMB
    STAYED_IN_DANGER: -3.0,
    LOOPING: -8.0, # -5.0 previous value
    OSCILLATING: -8.0, # -5.0 previous value
    COIN_MOVEMENT: 3.0,
    RUNNED_FROM_BOMB: 1.0,
}, doc="Umut's scheme, EXACT bit-for-bit port of the standalone "
       "Qlearned_agent's reward table and epsilon schedule. "
       "BOMB_DROPPED_INCORRECT replicates the old flat OR-logic penalty "
       "(-20, never stacks) instead of Irmak's stackable "
       "SUICIDAL_BOMB/USELESS_BOMB pair.")



# --- r10: Umut'un dovus agirlikli semasi (EXP-047 / EXP-053)          [UMUT]
register_scheme("r10_umut_combat", {
    # 1. Combat payoff matching true game value
    e.KILLED_OPPONENT: 120.0,

    # 2. Suicide penalty preventing exploitation
    e.KILLED_SELF: -80.0,
    e.GOT_KILLED: -40.0,

    # 3. Balanced economic signals
    e.COIN_COLLECTED: 25.0,
    e.COIN_FOUND: 10.0,
    COIN_MOVEMENT: 2.0,
    e.CRATE_DESTROYED: 6.0,

    # 4. Movement, danger, and action modifiers
    e.BOMB_DROPPED: 1.0,
    e.WAITED: -1.0,
    e.INVALID_ACTION: -1.0,
    e.SURVIVED_ROUND: 5.0,
    GOOD_BOMB: 10.0,
    BOMB_DROPPED_INCORRECT: -20.0,
    STAYED_IN_DANGER: -3.0,
    LOOPING: -8.0,
    OSCILLATING: -8.0,
    RUNNED_FROM_BOMB: 1.0,
}, doc="Umut's combat-weighted scheme: realigns kill/coin payoff and penalizes suicidal bomb drops.")

def reward_from_events(events, scheme: str = "r01_minimal") -> float:
    """Event listesinin toplam odulu."""
    table = REWARD_SCHEMES.get(scheme)
    if table is None:
        raise KeyError(f"bilinmeyen odul semasi {scheme!r}. "
                       f"Mevcut: {sorted(REWARD_SCHEMES)}")
    return float(sum(table.get(ev, 0.0) for ev in events))


# --------------------------------------------------------------------------
# Potansiyel-tabanli shaping  (Ng et al. 1999)
# --------------------------------------------------------------------------
#
#   F(s, s') = gamma * Phi(s') - Phi(s)
#
# Bu bicimdeki shaping optimal politikayi DEGISTIRMEZ; yalnizca ogrenmeyi
# hizlandirir. Bizim icin asil cazibesi: ileri-geri giden bir ajanda
# TELESKOPIK olarak sifirlanir -> EXP-009'daki salinim (adimlarin %85'i)
# hicbir sey kazandirmaz.
#
# !!! gamma SECIMI -- bilincli bir sapma !!!
# Teorik bicim gamma<1 ister. Ama Phi = -mesafe ile gamma<1 kullanildiginda
#     ayni yerde durmak:  F = Phi*(1-gamma) = w*d*0.05  >  0
# yani "sandiktan uzakta KIMILDAMADAN durmak" pozitif odul kazanir --
# tam da kirmaya calistigimiz davranis. Bu yuzden shaping teriminde
# gamma_shaping = 1.0 kullaniyoruz:
#     yaklas -> +w   uzaklas -> -w   dur -> tam 0
# Bolumlu (episodic) gorevlerde yaygin pratik budur; Ng garantisi
# indirimli MDP'de tam degil, YAKLASIK olur. `shaping_gamma` config
# alanindan degistirilebilir, istenirse ayri bir ablation yapilir.

MAX_SHAPING_DIST = 40      # ulasilamayan hedef icin ust sinir


def _nearest_distance(gs: dict, targets) -> float:
    """Ajandan hedeflere labirenti dolasan en kisa mesafe (BFS)."""
    targets = list(targets)
    if not targets:
        return float(MAX_SHAPING_DIST)
    passable = free_mask(gs)
    for (tx, ty) in targets:
        passable[tx, ty] = True          # hedef karesi erisilebilir sayilir
    dist = bfs_distances(passable, targets)
    _n, _sc, _b, (ax, ay) = gs["self"]
    d = int(dist[ax, ay])
    return float(MAX_SHAPING_DIST) if d < 0 else float(min(d, MAX_SHAPING_DIST))


def _phi_crate_dist(gs: dict) -> float:
    """Phi = -(en yakin sandiga mesafe). Yaklasmak Phi'yi ARTIRIR."""
    field = gs["field"]
    crates = list(zip(*np.where(field == CRATE)))
    if not crates:
        return 0.0                        # sandik yok -> sabit -> shaping yok
    return -_nearest_distance(gs, crates)


def _phi_coin_dist(gs: dict) -> float:
    """Phi = -(en yakin GORUNUR coin'e mesafe)."""
    coins = list(gs["coins"])
    if not coins:
        return 0.0
    return -_nearest_distance(gs, coins)


def _phi_safety(gs: dict) -> float:
    """Phi = -(bulundugun karenin tehlikeliligi). Guvenli kare -> 0."""
    d = danger_map(gs)
    _n, _sc, _b, (ax, ay) = gs["self"]
    v = int(d[ax, ay])
    if v >= SAFE:
        return 0.0
    # d=0 en tehlikeli (bu adim olduruyor) -> en negatif
    return -float(SAFE_STEPS - min(v, SAFE_STEPS))


SAFE_STEPS = 5

def _phi_crate_count(gs: dict) -> float:
    """Phi = -(kalan sandik sayisi). Sandik YIKMAK Phi'yi ARTIRIR.

    !!! EXP-010'un ana dersi !!!
    Tek basina `crate_dist` (Phi = -mesafe) KENDI KENDINI BALTALIYOR: hedefimiz
    sandiklari YOK ETMEK, ama yok etmek "en yakin sandik" mesafesini buyutuyor
    ve Phi'yi DUSURUYOR. Olculdu: bitisikteki sandigi yikmak shaping'den
    -0.600 getiriyor (CRATE_DESTROYED odulunun +1.5'inin %40'i). Phi'yi
    maksimize eden politika "sandigin yaninda dur, yikma" oluyor -- izde
    355 adim salinim, 799 adim donma, 189 bos bomba olarak gorundu.
    Sandik 14.7 -> 3.8'e dustu.

    Genel ders: "en yakin X'e mesafe" potansiyeli, amac X'i ORTADAN KALDIRMAK
    oldugunda ters calisir. Potansiyel ILERLEMEYI de icermeli.
    """
    return -float((gs["field"] == CRATE).sum())


POTENTIALS = {
    "crate_dist": _phi_crate_dist,
    "crate_count": _phi_crate_count,
    "coin_dist": _phi_coin_dist,
    "safety": _phi_safety,
}

#: sema adi -> {potansiyel adi: agirlik}
POTENTIAL_SCHEMES: dict[str, dict[str, float]] = {
    "p00_none": {},
    # 0.15: bir adim yaklasmak +0.15 -- yasam cezasinin (0.05) 3 kati,
    # ama gercek sandik odulunun (1.5) onda biri. Yani yol gosterir,
    # hedefin yerini almaz.
    "p01_crate": {"crate_dist": 0.15},
    "p02_crate_coin": {"crate_dist": 0.15, "coin_dist": 0.15},
    "p03_crate_coin_safe": {"crate_dist": 0.15, "coin_dist": 0.15,
                            "safety": 0.10},
    # p04: EXP-010'un duzeltmesi. crate_count agirligi (1.0), sandik
    # yikildiginda +1.0 getirir ve mesafe siçramasinin getirdigi ~-0.6'yi
    # fazlasiyla karsilar -> ilerleme ARTIK cezalandirilmiyor.
    "p04_crate_progress": {"crate_dist": 0.15, "crate_count": 1.0},
    "p05_progress_only": {"crate_count": 1.0},
}


def shaping_reward(old_state: dict | None, new_state: dict | None,
                   scheme: str = "p00_none", gamma: float = 1.0) -> float:
    """F = gamma * Phi(s') - Phi(s), semadaki tum potansiyeller uzerinden."""
    weights = POTENTIAL_SCHEMES.get(scheme)
    if weights is None:
        raise KeyError(f"bilinmeyen potansiyel semasi {scheme!r}. "
                       f"Mevcut: {sorted(POTENTIAL_SCHEMES)}")
    if not weights or old_state is None:
        return 0.0
    total = 0.0
    for name, w in weights.items():
        phi = POTENTIALS[name]
        # Terminal state: Phi(terminal) = 0  (Ng et al.'in sarti)
        phi_next = 0.0 if new_state is None else phi(new_state)
        total += w * (gamma * phi_next - phi(old_state))
    return float(total)


def game_score_from_events(events) -> int:
    """GERCEK oyun skoru delta'si -- raporlama icin, ogrenme icin degil."""
    return (s.REWARD_COIN * sum(1 for ev in events if ev == e.COIN_COLLECTED)
            + s.REWARD_KILL * sum(1 for ev in events if ev == e.KILLED_OPPONENT))