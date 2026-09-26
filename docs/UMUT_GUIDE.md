# Umut için rehber — araçlar, ölçüm ve bulgular

> Irmak'ın kurduğu ölçüm altyapısını sen de kullanabilirsin. Bu dosya
> **ne yaptığımızı**, **nasıl koşacağını** ve **neden böyle ölçtüğümüzü** anlatır.
>
> Detaylı deney kayıtları: [EXPERIMENTS.md](EXPERIMENTS.md) · Durum: [STATUS.md](STATUS.md)
> Oyunun kesin kuralları: [GAME_MECHANICS.md](GAME_MECHANICS.md)

---

## 0. Kurulum (bir kez)

```
git pull
```

Bizim ajan **PyTorch** kullanıyor (seninki sadece numpy). Yoksa:

```
pip install torch numpy pyyaml matplotlib
```

Kontrol:

```
python -c "import torch, numpy, yaml, matplotlib; print('hepsi tamam')"
```

Bütün komutlar repo kökünden (`bomberman_rl/`) çalıştırılır.

---

## 1. En önemli araç: `bbrl.eval` — ortak cetvel

**Neden var:** tek bir oyunun sonucu gürültüdür. Bu araç ajanı **birden fazla
seed'de, yüzlerce turda** koşar ve %95 güven aralığı verir.

```
python -m bbrl.eval --agent Qlearned_agent --card EK-2 --seeds 10
```

### Değerlendirme kartları

| kart | senaryo | rakipler | hangi Task |
|---|---|---|---|
| **EK-2** | classic | 3× `rule_based` | **Task 4 — TURNUVA VEKİLİ, birincil metrik** |
| EK-1 | classic | 1× `rule_based` | Task 4 teşhis |
| EK-3 | coin-heaven | yok | Task 1 (navigasyon) |
| EK-4 | classic | `peaceful` + `coin_collector` | Task 3 (avlanma) |
| EK-5 | classic | yok | Task 2 (bomba + kaçış) |

### Referans değerler (EXP-000'de ölçüldü, donduruldu)

| kart | `rule_based_agent` |
|---|---|
| **EK-2** | **3.317 ± 0.247** ← geçilmesi gereken sayı |
| EK-1 | 4.67 ± 0.21 |
| EK-3 | 50.00 coin / 125.19 adım |
| EK-4 | 8.60 ± 0.34 |
| EK-5 | 8.53 ± 0.06 |

### İki sonucu karşılaştırmak

```
python -m bbrl.eval --compare runs/eval/A/result.json runs/eval/B/result.json
```

Welch testi yapar. **Güven aralığı sıfırı içermiyorsa fark gerçek**, içeriyorsa
"belki şans".

---

## 2. `bbrl.duel` — iki ajanı kapıştır

```
python -m bbrl.duel --a Qlearned_agent --b irmak_umut --seeds 10
```

Üç ölçüm birden:
1. **1v1** — aynı tahtada ikisi
2. **2v2** — ikişer kopya
3. **Ortak cetvel** — ikisi de ayrı ayrı `rule_based`'e karşı

> **Üçüncüsü kritik.** Düelloda kazanmak turnuvada kazanmak demek değil —
> turnuvada karşımızda **başka takımların** ajanları olacak, birbirimiz değil.
> Ortak referans olmadan sıralama yanıltır.

### 4 ajanı aynı masaya koymak

```
python -c "import sys; sys.path.insert(0,'.'); from bbrl.duel import mixed_match; mixed_match(['Qlearned_agent','irmak_umut','rule_based_agent','rule_based_agent'], rounds=100, seeds=range(10))"
```

Tek koşudan **dört ajanın da** istatistiğini okur — hepsi aynı tahtalarda,
aynı turlarda. Ayrı ayrı koşmaktan hem ucuz hem daha dürüst.

---

## 3. `bbrl.trace` — ajan NEDEN kötü, adım adım

Skor "ajan kötü" der, **nedenini söylemez**. Bu araç her adımı kaydeder ve
9 tipik hatayı otomatik işaretler:

```
python -m bbrl.trace --agent Qlearned_agent --scenario classic --opponents rule_based_agent --rounds 10 --csv runs/trace.csv
```

| işaret | ne demek |
|---|---|
| `FREEZE` | tek karede çakılı kalma |
| `OSCILLATION` | A-B-A-B gidip gelme ← **senin "looping" dediğin** |
| `AVOIDABLE_TRAP` | kurtuluş varken ölüme giden hamle |
| `TOWARD_BOMB` | kendi patlama hattında bombaya **doğru** gitmek |
| `SELF_TRAP` | bomba bırakınca kaçış yollarının kapanması |
| `SUICIDE_BOMB` | kaçışı olmayan bomba |
| `USELESS_BOMB` | 0 sandık 0 rakip vuran bomba ← **"irrelevant places"** |
| `STAY_IN_DANGER` | tehlikedeyken kaçış varken beklemek |
| `DEATH` | ölüm + **öncesindeki 8 adımın dökümü** |

CSV'yi Excel'de açabilirsin — her adım bir satır.

---

## 4. `bbrl.plots` — rapor §6 grafikleri

```
python -m bbrl.plots --training e19
```

```
python -m bbrl.plots --convergence e19
```

**`--convergence` her sonuçtan önce çalıştırılmalı.** Son %20 ile önceki %20'yi
karşılaştırır. Eğri hâlâ yükseliyorsa ölçtüğün şey bir **sıralama değil**,
"kim daha hızlı öğreniyor".

---

## 5. Canlı izleme

```
.\watch.ps1 -Agents Qlearned_agent -TurnBased
```

`-TurnBased` = her tuşa basışta bir adım. Hata avlamak için en iyisi.
(PowerShell'de `&&` ve `VAR=x komut` yok; script bu yüzden var.)

---

## 6. Üç metodoloji kuralı — bunlar bize pahalıya mal oldu

### 6.1 Eğitim metriği YALAN SÖYLER

Bu bize **dört kez** oldu:

| deney | eğitimde | eval'de (eps=0) |
|---|---|---|
| EXP-002 | 48.67 coin | **26.12** |
| EXP-009 | intihar %92 | **%9** |
| EXP-011 (lineer) | 7.1 sandık | **0.00** |
| EXP-012 | — | CNN 3× düştü |

Sebep: eğitimde `eps=0.05` rastgelelik var ve **politikayı o taşıyor.**
`eps=0` (gerçek davranış) bambaşka olabiliyor.

> **Kural: performansı DAİMA `bbrl.eval` ile ölç.** Eğitim logundaki skora bakma.

`experiments.txt`'inde *"suicides are again prevented"* yazıyor ama EK-2
ölçümünde ajanın tur başına **0.399 intihar** ediyor (`rule_based` 0.528,
bizimki 0.274). Yani düşmüş ama bitmemiş — muhtemelen aynı tuzak: eğitim
logunda gördüğün oran `eps=0.05` altındaki davranış.

### 6.2 Tek koşu yeterli değil — dar bir aralık bile

Aynı config'i iki kez koştuk:

| koşu | 4 seed sonucu |
|---|---|
| A | 4.157 ± **0.053** |
| B | 3.912 ± **0.472** |

8 seed'i havuzlayınca gerçek değer **4.035 ± 0.197**. Yani **dar bir güven
aralığı da artefakt olabilir.** Küçük etkileri ayırt etmek için **5–8 eğitim
seed'i** gerekiyor.

### 6.3 Tek kartta iyi olmak yetmiyor

Ajanımız `rule_based`'e karşı harika ama **rakipsiz kalınca çöküyor**
(EK-5'te referansın %37'si). Ve `rule_based` turların **%53'ünde intihar
ettiği için EK-2 turlarının yarısında yalnız kalıyoruz.**

---

## 7. Şu ana kadarki ana bulgular

| # | bulgu | kanıt |
|---|---|---|
| 1 | **Model darboğazdı** — DQN, tablosal modeli **13.7×** geçti | EXP-011 |
| 2 | **Ödülün YÖNÜ büyüklüğünden önemli** — sandık ödülünü 10× artırmak hiçbir şey yapmadı; 10× AZALTIP coini yükseltmek skoru **1.9 katına** çıkardı | EXP-010, EXP-017 |
| 3 | Ajan shaped ödülün **%96'sını sandıktan** alıyordu, sandığın gerçek skora katkısı **sıfır** — sandık çiftliği botu | EXP-016 |
| 4 | **Müfredat işe yaramıyor** (iki farklı model sınıfında, −1.913) | EXP-009, EXP-015, EXP-018R |
| 5 | **Rakip çeşitliliği kazandırıyor** (+0.254, anlamlı) | EXP-019 |
| 6 | **Simetri augmentation** CNN'i 3× yaptı (8× veri, bedava) | EXP-012 |
| 7 | `explosion_map` tek başına öldürücü karelerin **%60'ını** kaçırıyor | `test_gamelogic` |
| 8 | Potansiyel shaping **yanlış tasarlanırsa ters teper** — "en yakın sandığa mesafe" potansiyeli, amaç sandığı YOK ETMEK olduğu için sandık yıkmayı cezalandırıyordu | EXP-010 |
| 9 | **Çok-aşamalı eğitim bozuktu** — aşamalar arası epsilon 1.0'a sıfırlanıyordu, üç deneyi geçersiz kıldı | EXP-018R |

### Senin bulduklarınla örtüşenler

`experiments.txt`'inde yazdığın üç sorunu biz de **bağımsız olarak** bulduk:
**intihar**, **döngüye girme** (bizde `OSCILLATION`), ve **coin odaklanmaması**
(bizde ödül hizalaması, EXP-017'de +1.865 kazandırdı).

**İki bağımsız implementasyon, aynı üç duvar** — rapor §6 için çok değerli.

---

## 8. Mevcut durum

EK-2, 10 değerlendirme seed'i, 100'er tur:

| ajan | skor | intihar | kill | coin | referansa göre |
|---|---|---|---|---|---|
| **`irmak_umut`** (DQN) | **4.210 ± 0.186** | **0.284** | **0.352** | **2.450** | **%127** |
| *`rule_based_agent`* | *3.317 ± 0.247* | *0.528* | *0.214* | *2.247* | *referans* |
| `Qlearned_agent` (senin) | 2.953 ± 0.207 | 0.399 | 0.170 | 2.103 | %89 |

> Bizim satır **kasten kötümser**: gönderdiğimiz model seçim seed'lerinde
> (0–9) 4.416 alıyor, ama o seed'lere **bakarak** seçildiği için sayı şişkin.
> Tabloda hiç kullanılmamış seed'lerdeki (10–19) skoru var. Seçim primi 0.206.
> Sen de kendi en iyi seed'ini seçtiğinde aynı düzeltmeyi yapmalısın.

Kafa kafaya (4 ölçüm, hepsi anlamlı): 1v1 **+0.612** · 2v2 **+0.371** ·
EK-2 **+1.214** · 4'lü karma masa **+0.848**.

---

## 9. Senin ajanını nasıl güçlendirebilirsin

Bizim ölçümlerimizden çıkan, **lineer modele de uygulanabilir** olanlar:

1. **Ödülü gerçek skora hizala** — bizde en büyük tek kazanç (+1.865).
   Sandık ödülü coinden büyükse ajan sandık çiftçisi olur. Gerçek skor
   `coin×1 + kill×5`; sandığın katkısı sıfır.
2. **Rakip çeşitliliğiyle eğit** — `rule_based` + `coin_collector` + `irmak_umut`.
   Hepsi **aynı anda** masada olsun; aşamalar arasında değiştirmek zarar veriyor
   (felaket unutma, −0.490).
3. **`bbrl.trace` ile döngüleri bul** — `OSCILLATION` işaretleri tam olarak
   senin "looping" dediğin şeyi yakalıyor, hangi karede olduğunu da veriyor.
4. **Ölüm transition'ını yakaladığından emin ol.** Framework, ajan öldüğü adımda
   `game_events_occurred`'ı **çağırmıyor** — o deneyim yalnızca `end_of_round`'a
   geliyor ve **ölüm cezasını taşıyan tek transition** o.
   (`GAME_MECHANICS.md` §6, tuzak 3)
5. **Tehlikeyi `explosion_map`'ten hesaplama** — öldürücü karelerin %60'ını
   kaçırıyor. Patlamak üzere olan bombayı hiç göstermiyor.

---

## 10. Hazır kullanabileceğin kod

`bbrl/gamelogic.py` — hepsi gerçek environment'a karşı doğrulanmış (758 adımda
birebir eşleşme):

| fonksiyon | ne yapar |
|---|---|
| `danger_map(gs)` | her kare kaç adım sonra öldürücü (`SAFE` = güvenli) |
| `escape_directions(gs, extra_bomb_at=(x,y))` | hangi yöne kaçarsan kurtulursun / buraya bomba atarsam kurtulur muyum |
| `bfs_distances(passable, starts)` | labirenti dolaşan gerçek mesafe |
| `direction_to_nearest(gs, targets)` | en yakın hedefe one-hot yön |
| `bomb_impact(gs, x, y)` | buraya bomba atarsam kaç sandık / rakip |
| `blast_coords(field, x, y)` | bu bomba nereleri vurur (**sandıklar durdurmaz**) |

```python
import sys
sys.path.insert(0, ".")
from bbrl.gamelogic import danger_map, escape_directions, bomb_impact
```

---

## 11. Testler

```
python tests/test_gamelogic.py
```

```
python tests/test_features.py
```

```
python tests/test_dqn.py
```

```
python tests/test_multistage.py
```

Kod değiştirirsen bunları koş. `test_features.py`'deki simetri testi gerçek bir
bug yakalamıştı (dönüşüm ters yazılmıştı, augment edilen veriyi sessizce
zehirliyordu).

---

## Sorular

`EXPERIMENTS.md` her deneyin **hipotezini, sonucunu ve kararını** taşıyor —
negatif sonuçlar ve çürüyen hipotezler dahil. Rapor §6'nın iskeleti o.
