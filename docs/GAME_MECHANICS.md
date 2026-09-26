# Oyun Mekaniği — Kaynak Koddan Çıkarılmış Kesin Referans

> Bu dosya `environment.py`, `items.py`, `agents.py`, `settings.py` okunarak yazıldı.
> Feature ve ödül tasarlarken **buradaki zamanlamaya** güvenin, sezginize değil.
> Bir madde yanlışsa düzeltin ve commit mesajında belirtin — bu dosya takımın ortak doğrusu.

---

## 1. Bir adımın (`do_step`) çalışma sırası

```
1. poll_and_run_agents()
     a) TÜM aktif ajanlar için state snapshot alınır  (get_state_for_agent)
     b) hepsi act() çağrılır
     c) aksiyonlar RASTGELE PERMÜTASYON sırasıyla uygulanır (perform_agent_action)
2. collect_coins()
3. update_explosions()      <-- patlamalar ÖNCE ilerler
4. update_bombs()           <-- bombalar SONRA sayar/patlar
5. evaluate_explosions()    <-- ölümler burada
6. send_game_events()       <-- game_events_occurred burada tetiklenir
```

Kritik sonuç: state, adımın **başında** alınır. Aynı adımda olacak her şey (bomba patlaması, ölüm)
state'te **henüz görünmez.**

---

## 2. Bomba zaman çizelgesi ⚠️ EN ÖNEMLİ BÖLÜM

`BOMB_TIMER = 4`, `EXPLOSION_TIMER = 2`, `BOMB_POWER = 3`.

Adım **T**'de bomba bırakıldı:

| Adım | `game_state['bombs']` timer | `explosion_map` | Adım sonunda ne olur |
|------|------|------|------|
| T    | görünmez | 0 | bırakıldı, timer 4→3 |
| T+1  | 3 | 0 | 3→2 |
| T+2  | 2 | 0 | 2→1 |
| T+3  | 1 | 0 | 1→0 |
| **T+4** | **0** | **0** ⚠️ | **PATLAR — blast karelerindeki ajanlar ÖLÜR** |
| T+5  | — | **1** | patlama hâlâ öldürücü, ajanlar ÖLÜR |
| T+6  | — | 0 | duman, güvenli. `bombs_left` geri verilir |
| T+7  | — | 0 | yeni bomba atılabilir |

### ⚠️ 1 NUMARALI TUZAK

> Adım T+4'te ajan `bomb timer == 0` görür ama blast karelerinde **`explosion_map` hâlâ 0'dır**
> — ve o adımın sonunda ölür.
>
> **`explosion_map` asla patlamak üzere olan bir bombayı haber vermez.**
> Tehlike haritanızı `bombs` listesinden hesaplamak ZORUNDASINIZ.

`KILLED_SELF` vakalarının büyük çoğunluğu bu yüzden olur. Danger feature'ı yanlış kurarsanız
ajan haftalarca "kaçamayan" bir politika öğrenir.

### `explosion_map` ikilidir

`EXPLOSION_TIMER = 2` olduğu için `max(explosion_map, exp.timer - 1)` ∈ **{0, 1}**.
- `1` → bu adım öldürücü
- `0` → güvenli (ama `bombs`'u ayrıca kontrol et!)

### Kaçış penceresi

T'de bırakıldı → hareket edilebilir adımlar: **T+1, T+2, T+3, T+4 = 4 hamle.**
Blast menzili 3. Yani:
- Düz kaçış: 4 hamlede 4 kare uzaklaş → **tam yeter, hata payı yok**
- Köşe kaçışı: 1–3 hamle ilerle + 1 hamle dik yöne sap → daha güvenli

`bombs_left` cooldown'u: T'de bırak → **T+7'de** yeni bomba.

---

## 3. Patlama geometrisi

`Bomb.get_blast_coords(arena)`:

```python
for i in range(1, power+1):
    if arena[x+i, y] == -1: break      # SADECE duvar durdurur
    blast_coords.append((x+i, y))
# ... 4 yön için aynı
```

- **Sandıklar patlamayı DURDURMAZ.** Blast sandıkların içinden geçer, 3 kare gider.
  (Yaygın yanlış varsayım — bu yüzden "sandığın arkasında güvendeyim" yanlıştır.)
- Sadece taş duvar (`-1`) durdurur.
- Köşeyi dönmez (sadece 4 eksen).
- Blast, **patlama anındaki arena** ile hesaplanır → daha önce yıkılmış sandık yolu açar.

---

## 4. Hareket ve çarpışma

`tile_is_free(x, y)` = `arena[x,y] == 0` **VE** o karede bomba yok **VE** aktif ajan yok.

- Bombalar ve ajanlar hareketi **bloklar**
- Coin bloklamaz (altındaki arena zaten 0) — üstüne gidince toplanır
- Bomba bıraktıktan sonra kendi bombanızın **üstünde durursunuz**; inebilirsiniz ama geri çıkamazsınız
- Geçersiz hareket → `INVALID_ACTION` (adım boşa gider, ölümcül olabilir)
- Aksiyonlar **rastgele permütasyonla** uygulanır → aynı kareye giden iki ajandan
  permütasyonda önce gelen kazanır, diğeri `INVALID_ACTION` alır

---

## 5. Tahta yapısı

- 17 × 17 (`COLS = ROWS = 17`)
- Duvarlar: kenarlar **+** `x` ve `y`'nin **ikisi de çift** olan her kare
  (`(x+1)*(y+1) % 2 == 1` ⟺ x çift ve y çift)
- Başlangıç köşeleri: `(1,1), (1,15), (15,1), (15,15)` — her birinin 4-komşuluğu sandıktan temizlenir
- Köşe ataması **rastgele permütasyon**

### Coin yerleşimi
```python
coin_positions = concat(permute(crate_positions), permute(free_positions))[:COIN_COUNT]
```
→ Coin'ler **öncelikle sandık altına** konur. `classic`'te (yoğunluk 0.75, 9 coin)
pratikte **tüm coin'ler başta gizlidir.** Sandık yıkılınca `collectable = True` olur.

| Senaryo | CRATE_DENSITY | COIN_COUNT |
|---|---|---|
| `empty` | 0 | 0 |
| `coin-heaven` | 0 | 50 |
| `loot-crate` | 0.75 | 50 |
| **`classic`** ← turnuva | **0.75** | **9** |

---

## 6. Skor ve olaylar

- Coin: **+1** · Rakip öldürme: **+5** · İntihar: **0** (ceza yok, ama tur biter)
- Aynı blast'ta hem kendini hem rakibi öldürürsen: `KILLED_SELF` **ve** `KILLED_OPPONENT` (+5)

### ⚠️ 2 NUMARALI TUZAK — çifte olay

`evaluate_explosions` içinde:
```python
if a is explosion.owner: a.add_event(KILLED_SELF)
else:                    owner.add_event(KILLED_OPPONENT)
...
for a in agents_hit: a.add_event(GOT_KILLED)     # <-- HERKES için, intiharda DA
```

> **İntihar `KILLED_SELF` VE `GOT_KILLED` olaylarının ikisini birden üretir.**

Ödül tablonuzda ikisine de büyük negatif verirseniz intihar cezasını farkında olmadan
**iki katına** çıkarırsınız. Bilinçli yapın veya birine 0 verin. `EXPERIMENTS.md`'de not düşün.

### ⚠️ 3 NUMARALI TUZAK — ölüm transition'ı `end_of_round`'da gelir

`send_game_events` yalnızca `if a.train and not a.dead` için `game_events_occurred` çağırır.
Öldüğünüz adımın transition'ı **`game_events_occurred`'a HİÇ gelmez**; `end_of_round`'a
`(last_game_state, last_action, events)` olarak gelir.

> Yani **ölüm cezasını taşıyan tek transition** `end_of_round`'dadır.
> Orada buffer'a yazmazsanız ajanınız ölmemeyi asla öğrenemez.

`--train N ≥ 1` iken eğitilen ajan ölünce tur **hemen biter** (`continue_without_training`
verilmediyse), yani `end_of_round` o adımda tetiklenir.

---

## 7. Tur bitiş koşulları (`time_to_stop`)

1. Hiç aktif ajan kalmadı
2. Tek ajan kaldı **ve** sandık yok **ve** toplanabilir coin yok **ve** bomba/patlama yok
3. Eğitim modunda (`continue_without_training` yoksa) hiç eğitilen ajan hayatta değil
4. `step >= MAX_STEPS (400)`

`SURVIVED_ROUND` yalnızca tur bitiminde **hayatta olan** ajanlara verilir.

---

## 8. Zaman limiti

- `TIMEOUT = 0.5s` (turnuva) · `TRAIN_TIMEOUT = inf` (eğitim)
- Aşarsan: aksiyon **`WAIT`**'e zorlanır **ve** `available_think_time` bir sonraki adım için
  aşım kadar azalır → **kümülatif çığ riski**
- `available_think_time <= 0` ise ajan o adım hiç çağrılmaz, `WAIT` sayılır
- `game_events_occurred` / `end_of_round` **zaman limitsiz**

---

## 9. `game_state` sözlüğü — kesin şema

| Anahtar | Tip | Not |
|---|---|---|
| `round` | `int` | 1'den başlar |
| `step` | `int` | 1'den başlar |
| `field` | `np.ndarray (17,17) int` | `-1` duvar, `0` boş, `1` sandık. **(x, y) image koordinatı** |
| `bombs` | `[((x,y), t)]` | `t == 0` → **bu adım sonunda patlar** |
| `explosion_map` | `np.ndarray (17,17) float` | `{0, 1}`. `1` → bu adım öldürücü |
| `coins` | `[(x,y)]` | yalnızca `collectable` olanlar |
| `self` | `(name, score, bombs_left, (x,y))` | `bombs_left: bool` |
| `others` | `[(name, score, bombs_left, (x,y))]` | yalnızca hayatta olanlar |
| `user_input` | `str \| None` | GUI |

`field[x, y]` — `print` edince GUI'ye göre **transpoze** görünür. Feature'da x/y karıştırmak
en sık yapılan sessiz hatadır; `state_to_features` içinde bir kez `assert` koyun.

---

## 10. CLI hatırlatmaları (`main.py`)

```bash
python main.py play --agents A B C D --train 1 --scenario classic --no-gui --n-rounds 1000 --seed 42 --save-stats results/run.json
```

- `--train N` → `--agents`'ın **ilk N**'i eğitilir
- `--train 0` → `continue_without_training` otomatik `True` olur
- `--n-rounds` varsayılan **10**
- `--seed` yalnızca dünyayı sabitler; **ajan rastgeleliği ayrı**
- `--save-stats <path>` → JSON (`by_agent`: score/coins/kills/suicides/…, `by_round`)
  ← **eval harness'ın veri kaynağı bu**
