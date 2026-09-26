# Deney Günlüğü

> Bu dosya raporun **§6 Experiments and Results** bölümünün iskeletidir ve
> "sistematik / bilimsel yaklaşım" kriterinin **kanıtıdır**. Notun en ağır kalemi burası.
>
> **Kural: sonradan doldurulmaz.** Her koşu, koşulduğu gün yazılır.
> Sorumlu: haftanın rotasyon kişisi (Sprint 1: **Umut**, Sprint 2: **Irmak**).

---

## Nasıl kullanılır

1. Deneyden **önce** hipotezi yaz. Sonucu görmeden. (Aksi halde post-hoc rasyonalizasyon olur.)
2. Koş. `runs/<run_id>/` altına otomatik kaydedilenler: config, git SHA, seed'ler, metrikler.
3. Sonucu ve **kararı** yaz. Negatif sonuçlar da yazılır — rapor için en değerlileri onlar.
4. Her Pazar Deney Konseyi'nde gözden geçir.

**Değerlendirme daima `bbrl/eval.py` ile, orijinal env'de, `self.train=False`, çok-seed,
gerçek oyun skoruyla.** Shaped ödülle asla performans raporlanmaz.

---

## ⚠️ METODOLOJİK UYARI — EXP-007..010 sonuçları YAKINSAMAMIŞ ajanlara ait

**Tarih:** 2026-08-17 · Eğitim eğrileri çizilince (`runs/figs/e10_training.png`) fark edildi.

6000 turluk EXP-010 koşularında dört varyantın da eğrisi **hâlâ yükseliyordu**:

| config | sandık (önceki %20) | sandık (son %20) | büyüme | yakınsadı? |
|---|---|---|---|---|
| `e10a_ratio` | 12.33 | 14.56 | **+18%** | HAYIR |
| `e10b_potential` | 10.44 | 10.25 | −2% | evet |
| `e10c_goodbomb` | 11.34 | 12.63 | **+11%** | HAYIR |
| `e10d_bombidle` | 11.89 | 13.36 | **+12%** | HAYIR |

Ayrıca eğrilerde **~2700. tura kadar neredeyse hiçbir şey olmuyor** — tam olarak
`eps_decay_rounds = 3000`'in bittiği yer. Yani 6000 turluk bütçenin **yarısı**
büyük ölçüde rastgele keşifle geçiyor.

**Bunun anlamı:** EXP-009 ve EXP-010'daki karşılaştırmalar bir **sıralama değil**,
"6000 turda kim daha hızlı öğreniyor" ölçümü. Özellikle EXP-010'daki
*"ödül büyüklüğü darboğaz değil"* çıkarımı **fazla erken** olabilir — ödül farkı
kendini gösterecek kadar eğitim yapılmamış olabilir.

**Alınan önlemler:**
1. `python -m bbrl.plots --convergence <önek>` eklendi — son %20 ile önceki %20'yi
   karşılaştırıp plato kontrolü yapıyor.
2. **Kural:** bir sonuç rapora girmeden önce bu kontrolden geçecek.
3. `eps_decay_rounds` toplam turun **%25–30'u** olmalı (%50 değil).
4. EXP-011 sonrası **uzun koşu** (≥25000 tur) yapılıp gerçek platonun nerede
   olduğu ölçülecek → EXP-013.

Bu, raporun §6'sında açıkça anlatılacak: erken karşılaştırmalarımızın hangi
kısmının geçerli, hangi kısmının bütçeye bağlı olduğu.

---

## Standart değerlendirme kartı (EK)

Her model bu 4 koşudan geçer, hep aynı seed'lerle:

| Kod | Senaryo | Rakipler | Tur | Seed |
|---|---|---|---|---|
| **EK-1** | classic | 1× rule_based | 200 | 0–4 |
| **EK-2** | classic | 3× rule_based | 200 | 0–4 | ← turnuva vekili |
| **EK-3** | coin-heaven | yok | 100 | 0–4 | ← Task 1 regresyon |
| **EK-4** | classic | peaceful + coin_collector | 200 | 0–4 |

Referans tabanlar: **EXP-000'de ölçüldü ve DONDURULDU** (aşağıya bak).

---

## Şablon (kopyala-yapıştır)

```markdown
### EXP-NNN — <kısa başlık>
**Tarih:** YYYY-MM-DD · **Kim:** Irmak / Umut · **Git:** `<sha>` · **Run:** `runs/<run_id>/`

**Hipotez:** <ne bekliyoruz ve NEDEN — mekanizmayı yaz>

**Değişken:** <sadece bu değişti> · **Sabit:** <geri kalan her şey>

**Kurulum:** config `configs/X.yaml` · seed'ler `[0,1,2,3,4]` · eval kartı `EK-?`

**Sonuç:**
| Metrik | Taban | Yeni | Δ (±%95 GA) |
|---|---|---|---|
| score | | | |
| suicide_rate | | | |
| win_rate | | | |
| p99_act_time_ms | | | |

**Yorum:** <hipotez doğrulandı mı? beklenmedik ne oldu?>

**Karar:** ✅ tut / ❌ at / 🔁 varyantını dene → **sonraki deney:** EXP-NNN+1
```

---

## Deneyler

### EXP-000 — Referans tabanların ölçümü ✅
**Tarih:** 2026-08-16 · **Kim:** Irmak · **Git:** `74fa822` · **Run:** `runs/eval/20260816-15*`

**Hipotez:** Ölçüm değil, kalibrasyon. Sağlanan ajanların EK-1..EK-4 üzerindeki performansını
sabitliyoruz ki sonraki her karşılaştırmanın bir sıfır noktası olsun.

**Kurulum:** 100 tur × 10 seed (EK-3: 50 tur × 10 seed), `--train 0`, orijinal env.
Değerler `ortalama ± %95 GA` (seed-düzeyi ortalamalar üzerinden, t-tabanlı).

#### EK-1 — classic, vs 1× rule_based
| ajan | score | win% | coin | kill | suicide | steps | act_ms |
|---|---|---|---|---|---|---|---|
| `rule_based_agent` | **4.67 ± 0.21** | 0.60 ± 0.37 | 4.47 | 0.04 | 0.38 | 279.0 | 1.03 |
| `coin_collector_agent` | 4.01 ± 0.18 | 0.00 | 3.87 | 0.03 | 0.61 | 221.7 | 1.21 |
| `random_agent` | 0.00 ± 0.00 | 0.00 | 0.00 | 0.00 | **1.00** | 18.4 | 0.10 |

#### EK-2 — classic, vs 3× rule_based ⭐ TURNUVA VEKİLİ
| ajan | score | win% | coin | kill | suicide | steps | act_ms |
|---|---|---|---|---|---|---|---|
| `rule_based_agent` | **3.32 ± 0.25** | 0.30 ± 0.35 | 2.25 | 0.21 | **0.53** | 228.4 | 1.27 |
| `coin_collector_agent` | 2.95 ± 0.14 | 0.00 | 2.57 | 0.08 | 0.49 | 204.7 | 1.06 |
| `random_agent` | 0.00 ± 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 18.0 | 0.11 |

#### EK-3 — coin-heaven, tek başına (Task 1)
| ajan | score | coin | suicide | **steps** | act_ms |
|---|---|---|---|---|---|
| `rule_based_agent` | 50.00 ± 0.00 | 50/50 | 0.00 | **125.19 ± 0.98** | 0.69 |
| `coin_collector_agent` | 50.00 ± 0.00 | 50/50 | 0.00 | 125.60 ± 1.01 | 0.66 |
| `random_agent` | 1.65 ± 0.21 | 1.65 | 1.00 | 21.4 | 0.05 |

#### EK-4 — classic, vs peaceful + coin_collector (Task 3)
| ajan | score | win% | coin | kill | suicide | steps | act_ms |
|---|---|---|---|---|---|---|---|
| `rule_based_agent` | **8.60 ± 0.34** | **1.00** | 4.96 | 0.73 | 0.20 | 300.1 | 1.37 |
| `coin_collector_agent` | 6.67 ± 0.29 | 0.50 | 4.48 | 0.44 | 0.24 | 322.3 | 1.17 |
| `random_agent` | 0.00 ± 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 18.1 | 0.11 |

**Tüm kartlarda `timeout_violations = 0`.**

**Yorum — dört sonuç doğrudan tasarımı etkiliyor:**

1. **Yenmemiz gereken sayı: EK-2'de 3.32 ± 0.25.** Turnuva vekili bu. Ve `rule_based_agent`
   EK-1'de kendine karşı `win% = 0.60 ± 0.37` alıyor — simetri gereği 0.50 olmalı, gözlem
   onunla tutarlı. Harness'ın kendi doğrulaması.

2. **⚠️ EK-3'te skor METRİK OLARAK ÖLÜ.** Hem `rule_based` hem `coin_collector` 50/50 coin
   topluyor → skor tavana yapışıyor. Task 1'de ayırt eden tek metrik **`steps`**:
   **125.19 ± 0.98**. Çok dar bir aralık, yani gürültü değil gerçek bir hedef.
   Task 1 hedefimiz: *50 coin'i ≤ 125 adımda topla.*

3. **`rule_based_agent` EK-2'de turların %53'ünde kendini öldürüyor** (EK-1'de %38).
   Güçlü referans ajanın en büyük zaafı bu. Sadece **intihar etmeyen** bir ajan bile
   ciddi avantaj kazanır — Task 2'ye (kaçış) ağırlık vermenin somut gerekçesi.

4. **Zaman bütçesi rahat:** en yavaş referans 1.37 ms/adım, limit 500 ms. **~365× marj**
   var. CNN/DQN inference'ı CPU'da rahatça sığar; kısıt yakınsama süresi, model boyutu değil.

**Karar:** Bu sayılar **donduruldu**. Raporun her grafiğinde referans çizgi olarak görünecek.
`random_agent` (score 0.00, suicide 1.00, 18 adım) mutlak taban; ilk modelimizin ilk hedefi
bunu net geçmek.

---

### EXP-001 — fastenv parity
**Tarih:** 2026-08-__ · **Kim:** Irmak

**Hipotez:** Vektörize env, orijinal `BombeRLeWorld` ile 10.000 episode boyunca adım adım
bit-bit aynı sonucu üretir.

**Kabul kriteri:** tek fark = FAIL. Ayrıca throughput ölçümü (adım/sn, orijinale karşı hızlanma).

**Sonuç:** _(doldur)_

**Karar:** Yeşil değilse **hiçbir eğitim koşusu başlamaz.**

---

### EXP-002 — Task 1 baseline: tabular Q, coin-heaven ✅ (bulgu: politika donuyor)
**Tarih:** 2026-08-16 · **Kim:** Irmak · **Git:** `7ad75a0` · **Run:** `runs/eval/20260816-205939_irmak_umut_EK-3`

**Hipotez:** `v0_minimal` feature setiyle tablosal Q, `coin-heaven`'da 3000 turda
`random_agent`'ı belirgin şekilde geçer ve boru hattının uçtan uca çalıştığını kanıtlar.

**Kurulum:** `m01_tabular_q`, 3000 tur, eps 1.0→0.05 (1500 turda), lr 0.15, γ 0.95,
ödül `r01_minimal`. Değerlendirme EK-3, 50 tur × 10 seed, **eps=0**.

**Sonuç:**
| metrik | eğitim (eps=0.05) | **eval (eps=0)** | `rule_based` hedefi |
|---|---|---|---|
| coin/tur | 48.67 | **26.12 ± 1.14** | 50.00 |
| adım/tur | 169 | **352.6 ± 12.6** | **125.19 ± 0.98** |
| suicide | — | 0.000 | 0.000 |
| invalid | — | 0.000 | 0.000 |
| act süresi | — | 0.494 ms | 0.69 ms |

**🔴 Ana bulgu — greedy politika donuyor.** Eğitim metriği (48.67) ile eval (26.12)
arasındaki uçurum gerçek. 3 turluk adım-adım izleme:

```
tur 1: 400 adım, 3 benzersiz kare — TEK karede 398 adım
tur 2: 400 adım, 3 benzersiz kare — TEK karede 398 adım
tur 3: 82 benzersiz kare, ama tek karede 312 adım
```

Ajan bir kareye gelip `WAIT`'e kilitleniyor (bomba 0, geçersiz 0 → eleme yoluyla WAIT).

**Mekanizma:** `coin-heaven`'da `WAIT` seçilince dünyada hiçbir şey değişmez — bomba yok,
rakip yok, coin'ler sabit. Aynı state → aynı argmax → yine `WAIT`. **Kendini emen döngü.**
Q(WAIT) = −c + γ·Q(WAIT) sabit noktasına oturuyor ve hareket alternatiflerinin altında kalmıyor.

Eğitimde görünmemesinin sebebi: %5 keşif gürültüsü ajanı döngüden çıkarıyordu.
**Yani politika iyi değildi, keşif onu taşıyordu.**

**Yorum:** Boru hattı çalışıyor (feature → model → ödül → buffer → update → kayıt → eval)
ve `random_agent`'ı (1.65 coin) 16× geçtik. Ama bu sonuç aynı zamanda eval harness'ının
neden şart olduğunun kanıtı: **eğitim metriğine baksaydık "48.67, harika" deyip yanlış
modelle devam edecektik.**

**Karar → EXP-007:** Üç düzeltme adayı, sırayla ve tek tek ölçülecek:
1. **Adım başına yaşam cezası** (`r02_living_cost`, −0.02/adım) — beklemeyi ve dolanmayı
   pahalı yapar; ayrıca Task 1'in gerçek metriği olan *adım sayısını* doğrudan optimize eder
2. **İyimser başlatma** (`optimistic_init > 0`) — denenmemiş aksiyonları cazip kılar
3. **Daha zengin feature** (`v1_handcrafted`, mesafe bilgisi dahil) — 703 durum çok fakir

Hangisinin ne kadar katkı verdiği `python -m bbrl.eval --compare A.json B.json` ile
Welch farkıyla raporlanacak.

---

### EXP-007 — Donma tedavisi: yaşam cezası × iyimser başlatma (2×2) ✅
**Tarih:** 2026-08-16 · **Kim:** Irmak · **Run:** `runs/eval/20260816-2310*`
**Configler:** `configs/e07{a,b,c,d}.yaml`

**Hipotez:** EXP-002'deki donma, `WAIT`'in state'i değiştirmemesinden kaynaklanıyor
(Q(WAIT) = −c/(1−γ), c=0 iken 0). İki bağımsız tedavi adayı:
(1) **yaşam cezası** — hareket −0.02, bekleme −0.15 → sabit nokta −3.0'a iner;
(2) **iyimser başlatma** — denenmemiş aksiyonlar cazip görünür.
Beklenti: ikisi de yardımcı olur, birlikte en iyisi.

**Değişken:** ödül şeması (r01/r02) × `optimistic_init` (0.0/1.0).
**Sabit:** model, feature, lr, γ, eps programı, tur sayısı, buffer — hepsi birebir aynı.

**Sonuç** (EK-3, 50 tur × 10 seed, eps=0):

| config | coin/tur | adım/tur | tabana fark (coin) | tabana fark (adım) |
|---|---|---|---|---|
| `e07a_base` (r01, opt=0) | 41.67 ± 1.26 | 200.55 ± 11.96 | — | — |
| **`e07b_living` (r02, opt=0)** | **50.00 ± 0.00** | **125.83 ± 0.65** | **+8.33** ✱ | **−74.73** ✱ |
| `e07c_optimistic` (r01, opt=1) | 18.82 ± 1.84 | 380.71 ± 8.93 | −22.85 ✱ | +180.16 ✱ |
| `e07d_both` (r02, opt=1) | 35.05 ± 1.46 | 307.56 ± 9.67 | −6.62 ✱ | +107.01 ✱ |

✱ = %95 GA sıfırı içermiyor (Welch)

**`rule_based_agent`'a karşı (EK-3):**
| metrik | e07b | rule_based | fark [%95 GA] |
|---|---|---|---|
| coin | 50.00 | 50.00 | +0.00 → eşit |
| adım | 125.83 | 125.19 | **+0.64 [−0.46, +1.75] → istatistiksel olarak EŞİT** |

**Davranış izi (`bbrl/trace.py`, e07b, 3 tur):** `Patoloji bulunamadi.`
WAIT %0 (öncesi %92), dört yön de ~%25, FREEZE 0 (öncesi 1108 adım kaybı).

**Yorum — hipotez YARIM doğrulandı, bu asıl bulgu:**

1. **Yaşam cezası tek başına Task 1'i çözdü.** 50/50 coin ve `rule_based` ile
   istatistiksel olarak eşit adım sayısı. Donma tamamen ortadan kalktı.
2. **İyimser başlatma tek başına ZARAR VERDİ** (−22.85 coin, +180 adım) ve
   yaşam cezasıyla birleştiğinde onun faydasını da yedi (50.00 → 35.05).
   İkisi toplanmıyor, çelişiyor. Muhtemel mekanizma: lr=0.15 ile iyimser değerler
   yavaş sönüyor, ajan sürekli "denenmemişi dene" moduna sıkışıp coin peşinden
   gitmiyor. **Bu, "iki iyi fikir birleşince daha iyi olur" sezgisinin yanlış
   çıktığı somut bir örnek** — tek tek ölçmeseydik göremezdik.

**⚠️ Metodolojik uyarı — rapora girmeli:** `e07a_base` (41.67 coin) ile EXP-002'deki
`m01` (26.12 coin) **aynı config**, farklı eğitim koşusu. Aradaki 15 coin'lik fark,
bazı ablation etkilerinden büyük. Yani **config başına tek eğitim koşusu yeterli değil**;
eval çok-seed ama *eğitim* tek-seed. e07b'nin etkisi (±0.00 varyansla tavan) bu
gürültünün çok ötesinde olduğu için bu sonuç güvenli, ama küçük etkiler için
**config başına 3–5 eğitim seed'i** şart. → **C14 olarak task listesine eklendi.**

**Karar:** ✅ `r02_living_cost` varsayılan oldu · ❌ `optimistic_init` terk edildi
(negatif sonuç, rapor §4'te "denenip bırakılan yaklaşımlar" bölümüne gidecek)
→ **sonraki:** Task 2 (sandıklı `classic`, bomba + kaçış)

---

### EXP-008 — Task 1 → Task 2 transferi: **NEGATİF** ✅ (beklenmedik)
**Tarih:** 2026-08-16 · **Kim:** Irmak · **Kart:** EK-5 (`classic`, rakipsiz), 100 tur × 10 seed

**Hipotez:** Task 1'i çözen `e07b` (50/50 coin, `rule_based` ile eşit), Task 2'de kötü
ama en azından `random_agent`'tan iyi olur — navigasyonu öğrendi, bombayı bilmiyor.

**Sonuç — hipotez ÇÜRÜDÜ, ajanımız rastgeleden DAHA KÖTÜ:**

| ajan | coin/tur | sandık | **intihar** | **adım** |
|---|---|---|---|---|
| `rule_based_agent` | **8.53 ± 0.06** / 9 | 116.9 | **0.000** | 399.1 |
| `coin_collector_agent` | 8.51 ± 0.06 | 116.9 | 0.000 | 399.1 |
| `random_agent` | 0.006 | 2.95 | 1.000 | **19.3 ± 0.6** |
| **`irmak_umut` (e07b)** | 0.003 | 2.96 | **1.000** | **9.5 ± 0.1** ← rastgelenin YARISI |

**Mekanizma (davranış izinden, 8 tur / 73 adım):**
`SUICIDE_BOMB` 11 kez · `AVOIDABLE_TRAP` 10 kez · `SELF_TRAP` 10 kez · ölüm 8/8

1. `coin-heaven`'da eğitim boyunca ajan **hiç bomba görmedi**. Tehlike/kaçış
   feature'ları hep "güvenli" değerindeydi.
2. Tablosal Q **sıfır genelleme yapar.** Task 2'deki her durum tabloda YOK →
   tüm Q değerleri 0 → 6 aksiyon arasında **eşit kura**.
3. Eşit kura BOMB'u da içeriyor: ölçülen bomba oranı **%18**.
   `random_agent`'ın kodu ise `p=[.23,.23,.23,.23,.08]` — yalnızca **%8** bomba,
   hiç WAIT yok.
4. Başlangıç köşesinde (1,1) bomba bırakmak = kaçış yok = kesin ölüm.

→ **İki kattan fazla bomba atmak, iki kat hızlı ölmek demek.** Rastgeleden kötü
olmamızın tamamı bu.

**Yorum — üç sonuç doğrudan tasarımı değiştiriyor:**

1. **Tablosal modelin sıfır genellemesi artık ÖLÇÜLDÜ, iddia değil.** Bu, hem
   Umut'un `v1_handcrafted`'ı (C3) hem de fonksiyon yaklaşımı (linear/DQN) için
   en güçlü gerekçe. Lineer model ve DQN görülmemiş durumda en azından
   *interpolasyon* yapar; tablo hiçbir şey yapmaz.
2. **Müfredat `classic` içermek ZORUNDA.** Sadece `coin-heaven`'da eğitip
   "sonra öğrenir" varsaymak yanlış — transfer negatif.
3. **`SUICIDAL_BOMB` custom event'i zaten hazır** (`bbrl/rewards.py`) ama
   coin-heaven'da hiç tetiklenmiyor. Bombalı ortamda eğitim başlar başlamaz
   devreye girecek; EXP-007'nin yaşam cezası gibi doğrudan hedefli bir sinyal.

**Bir başka gözlem:** `rule_based_agent` **rakipsizken hiç intihar etmiyor** (0.000),
ama 4 ajanlı EK-2'de %53. Yani o intiharlar rakip baskısı altında ortaya çıkıyor.
Task 2'de temiz bomba kullanımı, Task 4'teki asıl kazancın ön koşulu.

**Karar:** → **C15** (Task 2 müfredatı: `classic`'te eğitim). Hedef sırası:
(a) intihar 1.000 → < 0.2, (b) sandık 3 → > 50, (c) coin → `rule_based`'in 8.53'üne yaklaş.

---

### EXP-009 — Task 2: müfredat etkisi **TAM SIFIR** + ajan pasifist oldu ✅
**Tarih:** 2026-08-17 · **Kim:** Irmak · **Kart:** EK-5 · **3 eğitim seed'i × 10 eval seed'i**
**Configler:** `configs/m02{a,b,c}.yaml` · **Log:** `runs/exp009.log`

**Hipotez:** (a) `coin-heaven → classic` müfredatı, doğrudan `classic`'ten iyidir;
(b) `v0b_task2` + `r03_task2` ile intihar 1.000 → <0.2 iner.

**Sonuç:**

| | `m02a` doğrudan | `m02b` müfredat | `rule_based` |
|---|---|---|---|
| coin/tur | 0.409 ± 0.520 | **0.409 ± 0.520** | 8.53 |
| sandık/tur | 15.74 ± 10.93 | **15.74 ± 10.93** | 116.9 |
| intihar | 0.091 ± 0.159 | **0.091 ± 0.159** | 0.000 |
| adım | 369.6 ± 49.8 | **369.6 ± 49.8** | 399.1 |

#### Bulgu 1 — müfredat etkisi tam sıfır, ve SEBEBİ ÖLÇÜLDÜ

İki config'in sonuçları **virgülden sonra her hanede aynı**. Bug sanıp Q tablolarını
karşılaştırdım:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `m02a` durum sayısı | 5094 | 5195 | 5108 |
| `m02b` durum sayısı | 5629 | 5723 | 5636 |
| ortak durum | 5094 | 5195 | 5108 |
| **ortakların değerleri aynı** | **5094** | **5195** | **5108** |

`m02b` ⊃ `m02a`, ve ortak her durumun Q değeri **birebir aynı**. Müfredattan gelen
fazladan 535 durum `classic`'te hiç ziyaret edilmiyor.

Sebep, feature yapısında: müfredattan gelen durumların **%100'ünde "sandık yönü = YOK"**
(coin-heaven'da sandık yok), `classic` durumlarının ise **%0'ında**. Tek bir bit iki
senaryonun state uzayını **tamamen ayırıyor**.

> **Tablosal modelde müfredat, state uzayları ayrık olduğu sürece SIFIR bilgi aktarır.**
> Bu, EXP-008'in "tablo sıfır genelleme yapar" bulgusunun devamı ve
> fonksiyon yaklaşımı (linear/DQN) için ikinci bağımsız gerekçe.

#### Bulgu 2 — "intihar" rakamı eğitimde yanıltıcıydı

Eğitim logu son 200 turda `intihar ≈ 0.92` gösteriyordu; eval'de **0.091**.
Fark `eps=0.05` keşif gürültüsü: rastgele `BOMB` aksiyonları ajanı öldürüyor,
greedy politika ise neredeyse hiç bomba atmıyor.
**Eğitim metriği yine yanılttı** (EXP-002'deki 48.67 vs 26.12 ile aynı ders).

#### Bulgu 3 — ajan pasifist oldu, salınıma kaçtı

Davranış izi (12 tur, 4800 adım, `runs/trace_m02.csv`):

```
ölüm 0 | BOMB %2 | WAIT %1
OSCILLATION 13 kez — 4074 adım kaybı = tümünün %85'i
  tur 2: (1,15) <-> (1,14) arasında 392 adım (400'ün 392'si)
```

`r03_task2`'de ölümü çok pahalı yaptım (`KILLED_SELF −15`, `SUICIDAL_BOMB −3`).
Ajan rasyonel olanı yaptı: **hiç bomba atma.** Sandık 15.7/116.9 (%13),
coin 0.409/8.53 (%5).

Ayrıca EXP-007'nin yaşam cezasında bir açık varmış: **`WAIT` cezalandırıldı (−0.15)
ama ileri-geri gitmek cezalandırılmadı** (−0.02, üretken hareketle aynı). Ajan
donmayı bıraktı, yerine **salınmayı** buldu. Ödülün etrafından dolandı.

#### Bulgu 4 — C14 kendini haklı çıkardı

`coin/tur = 0.409 ± 0.520` — güven aralığı ortalamadan büyük. Tek eğitim seed'iyle
bakan biri 0.25 veya 0.65 görüp kesin bir hüküm verirdi. Üç seed olmasa bu
belirsizliği göremezdik.

**Karar → EXP-010:**
1. **Potansiyel-tabanlı shaping**, Φ = −(en yakın sandığa BFS mesafesi).
   İleri-geri gidişte teleskopik olarak sıfırlanır → salınım hiçbir şey kazandırmaz.
   Optimal politikayı değiştirmediği kanıtlı biçim (Ng et al. 1999; PDF de bunu veriyor).
2. `SUICIDAL_BOMB` cezasını düşür, `CRATE_DESTROYED` ödülünü artır →
   bombanın beklenen değerini pozitife çevir.
3. `linear_q` koşuları numpy/BLAS çökmesi yüzünden düştü (ortam düzeltildi) — tekrarla.
   Ayrık state uzayı sorununa karşı asıl testi bu verecek.

---

### EXP-010 — Ödül ablation: **dört tahminin dördü de yanlış** ✅
**Tarih:** 2026-08-17 · **Kim:** Irmak · **Kart:** EK-5 · **3 eğitim × 10 eval seed'i**
**Configler:** `configs/e10{a,b,c,d}.yaml` · **Log:** `runs/exp010.log`

**Peşinen yazılan tahminler** (EXP-010 başlamadan önce kaydedildi):

| varyant | tahmin | **gerçek** | |
|---|---|---|---|
| `e10a` oran düzeltmesi | coin 0.4 → **3–5** | **0.294 ± 0.287** | ❌ |
| `e10b` + potansiyel shaping | +0.5–1.5 ⬆️ | **0.077 ± 0.040** | ❌ ters yön |
| `e10c` + GOOD_BOMB | +0–1 ⬆️ | 0.078 ± 0.095 | ❌ |
| `e10d` + BOMB_IDLE | ≈ veya ⬇️ | 0.135 ± 0.246 | ✅ kısmen |

Tam sonuç:

| config | coin | sandık | intihar | adım |
|---|---|---|---|---|
| **`e10a_ratio`** | **0.294 ± 0.287** | **14.68 ± 7.74** | 0.233 | 323.7 |
| `e10b_potential` | 0.077 ± 0.040 | 3.82 ± 3.54 ✱ | 0.019 ✱ | 393.1 ✱ |
| `e10c_goodbomb` | 0.078 ± 0.095 | 6.09 ± 2.17 ✱ | 0.039 ✱ | 386.3 ✱ |
| `e10d_bombidle` | 0.135 ± 0.246 | 8.26 ± 7.47 | 0.079 ✱ | 373.6 ✱ |

✱ = tabana göre fark anlamlı (Welch %95)

#### 🔴 Bulgu 1 — potansiyel shaping KENDİ KENDİNİ BALTALIYOR

`e10b` sandığı 14.68 → **3.82**'ye düşürdü (anlamlı). Mekanizmayı yapay bir state
kurup ölçtüm:

```
yakın sandık varken       Φ = −1.0   (en yakın sandık 1 kare)
sandık YIKILDIKTAN sonra  Φ = −5.0   (en yakın sandık 5 kare)
→ shaping ödülü = −0.600     (CRATE_DESTROYED +1.5'in %40'ı)
```

Φ = −(en yakın sandığa mesafe) seçmiştim. Ama **hedefimiz sandıkları yok etmek** —
yani amaca ulaşmak potansiyeli DÜŞÜRÜYOR. Φ'yi maksimize eden politika
*"sandığın yanında dur, yıkma"*. Davranış izi birebir bunu gösterdi:
355 adım tek noktada salınım, 799 adım donma, 189 boş bomba, 6/6 turda 0 ölüm.

> **Genel ders: "en yakın X'e mesafe" potansiyeli, amaç X'i ORTADAN KALDIRMAK
> olduğunda ters çalışır.** Potansiyel ilerlemeyi de içermeli.

**Düzeltme** (`p04_crate_progress` = mesafe 0.15 + kalan sandık sayısı 1.0):
sandık yıkmak artık **+0.400** getiriyor (`p05_progress_only`: +1.000).
`EXP-011`'de `e11d` olarak test ediliyor.

#### 🔴 Bulgu 2 — ödül BÜYÜKLÜĞÜ darboğaz değil

| şema | `CRATE_DESTROYED` | sonuç sandık |
|---|---|---|
| `r03_task2` (EXP-009) | 0.15 | 15.74 |
| `r04_ratio` (EXP-010) | **1.5** (10×) | **14.68** |

**Sandık ödülünü 10 kat artırmak hiçbir şey değiştirmedi.** Literatüre göre
düzelttiğim oran (ölüm:sandık 100:1 → 10:1) da ölçülebilir bir kazanç vermedi.

İki bağımsız ödül müdahalesi (oran + shaping) sandık sayısını `rule_based`'in
116.9'una yaklaştıramadı. **Darboğaz ödül değil.** Kalan iki aday: model ve feature.
→ EXP-011 bunu test ediyor.

#### Bulgu 3 — `BOMB_IDLE` tahminim kısmen tuttu

Zarar vermesini bekliyordum; `e10a`'ya göre gerçekten daha kötü (0.135 vs 0.294),
ama `e10b`/`e10c`'den **daha iyi** — çünkü onların shaping hatasını taşımıyor
denecek kadar basit değil, o da `p01_crate` kullanıyor. Yani asıl zarar veren
shaping'di, `BOMB_IDLE` onu bir miktar dengeledi (bomba atmaya zorlayarak).
Temiz bir yargı için `p04` üzerinde tekrar ölçülmeli.

**Karar:**
- ✅ `r04_ratio` tut (en iyi varyant `e10a`, shaping'siz)
- ❌ `p01_crate` **terk edildi** — rapor §4 "denenip bırakılan yaklaşımlar"
- 🔧 `p04_crate_progress` yazıldı, EXP-011'de test ediliyor
- ➡️ **EXP-011: model karşılaştırması** — darboğazın ödül olmadığı ölçüldü

---

### EXP-011 — 🏆 **MODEL DARBOĞAZMIŞ: DQN 13.7× öne geçti**
**Tarih:** 2026-08-17 · **Kim:** Irmak · **Kart:** EK-5 · **3 eğitim × 10 eval seed'i**
**Log:** `runs/exp011.log` · **Grafik:** `runs/figs/e11_ablation.png`

**Hipotez:** EXP-008 (tablo sıfır genelleme yapar) ve EXP-009 (müfredat sıfır transfer,
ayrık state uzayları) iki bağımsız kanıtla fonksiyon yaklaşımını işaret etti.
Feature ve ödül SABİT tutulup yalnızca model değiştirilirse fark ölçülebilir olmalı.

**Değişken:** yalnızca model. **Sabit:** `v0b_task2` + `r04_ratio` + shaping yok + 6000 tur.

| config | coin/tur | sandık/tur | intihar | adım | yakınsadı? |
|---|---|---|---|---|---|
| `e11a_tabular` | 0.294 ± 0.287 | 14.68 ± 7.74 | 0.233 | 323.7 | ❌ (+%18) |
| `e11b_linear` | **0.000** | **0.000** | 0.000 | 400.0 | ✅ |
| **`e11c_dqn_mlp`** | **4.028 ± 1.610** ✱ | **76.24 ± 21.38** ✱ | **0.000** ✱ | 400.0 | ✅ (−%2) |
| `e11d_tab_p04` | 0.000 ✱ | 2.97 ✱ | **1.000** ✱ | 5.0 ✱ | ✅ |
| *`rule_based` (EXP-000)* | *8.53* | *116.9* | *0.000* | *399.1* | — |

✱ = tabana (`e11a`) göre fark anlamlı (Welch %95)

#### 🏆 Ana sonuç

**DQN, tablosal modeli 13.7 kata katladı** (0.294 → 4.028 coin, fark **+3.734** anlamlı).
Sandıkta 14.7 → 76.2 (`rule_based`'in **%65'i**), **intihar sıfır**, 400 adımın tamamını
hayatta bitiriyor. Coin'de `rule_based`'in **%47'sine** ulaştık.

EXP-008 ve EXP-009'un işaret ettiği yer doğruymuş: **darboğaz ödül değil, modeldi.**
EXP-010'daki *"ödül büyüklüğü darboğaz değil"* çıkarımı da böylece desteklendi —
ama sebebi ödülün önemsizliği değil, **modelin ödülü kullanamamasıydı.**

#### `e11b_linear` eval'de TAMAMEN çöktü — 0.000 / 0.000 / 0.000

Eğitimde (eps=0.05) 7.1 sandık kırıyordu; eval'de (eps=0) **hiç bomba atmıyor**,
0 sandık, 0 coin, 400 adım hayatta. Yine aynı ders: **keşif gürültüsü politikayı
taşıyordu** (EXP-002 ve EXP-009'daki gibi).

⚠️ Bu, "lineer Q çalışmaz" demek DEĞİL — *bizim* lineer implementasyonumuzun bu
feature seti ve hiperparametrelerle çöktüğü anlamına gelir. Umut'un lineer Q ile
`rule_based`'i yendiği iddiası bu yüzden ayrıca ölçülmeli; farkı yaratan büyük
ihtimalle **feature seti**.

#### `p04` shaping düzeltmesi de çöktü — bu sefer ters uçta

EXP-010'da `p01_crate` sandık yıkmayı **cezalandırıyordu** (aşırı pasif ajan).
Düzeltmede kalan-sandık potansiyeline ağırlık **1.0** verdim; şimdi tam tersi oldu:
intihar **1.000**, ajan **5 adımda** ölüyor. Sandık yıkmak o kadar cazip ki
pervasızca bomba atıyor.

> İki shaping denemesi de aşırı uca savurdu. Ağırlık 1.0 değil ~0.2–0.3 olmalı.
> Rapor §4 için: potansiyel-tabanlı shaping "güvenli" diye biliniyor ama
> **potansiyelin TASARIMI kritik** — yanlış tasarlanmışsa optimal politikayı
> değiştirmese bile öğrenmeyi mahvediyor.

**Karar → EXP-012 (koşuyor):** DL rotası.
- `e12a`: DQN + **`v2_planes` CNN** (GPU) — e11c ile aynı bütçe, doğrudan karşılaştırma
- `e12b`: + D4 simetri augmentation (8× veri)
- `e12c`: e11c'nin 2× uzun koşusu (12000 tur) — bütçe gerçekten katkı veriyor mu

---

### EXP-012 — DL rotası: CNN elle tasarlanmışı geçemedi, simetri 3× kazandırdı
**Tarih:** 2026-08-17 · **Kim:** Irmak · **Kart:** EK-5 · **3 eğitim × 10 eval seed'i**
**Log:** `runs/exp012.log` · **Grafik:** `runs/figs/e12_ablation.png`

**Hipotez:** CNN ham 17×17 tahtayı okuyup kendi feature'ını öğrenerek bizim elle
tasarladığımız `v0b_task2`'yi geçer.

| config | coin/tur | sandık | intihar | adım | yakınsadı? |
|---|---|---|---|---|---|
| `e12a_cnn` (6000) | 1.589 ± 0.180 | 45.20 | 0.166 | 352.5 | ❌ **+%43** |
| `e12b_cnn_sym` (6000) | 4.671 ± 1.915 ✱ | **92.81** ✱ | 0.252 | 325.8 | ❌ +%14 |
| **`e12c_mlp_long`** (12000) | **5.205 ± 0.462** ✱ | 86.61 ✱ | **0.000** ✱ | 400.0 | ✅ +%1 |
| *`e11c_mlp`* (6000) | *4.028 ± 1.610* | *76.24* | *0.000* | *400.0* | ✅ |
| *`rule_based`* | *8.53* | *116.9* | *0.000* | *399.1* | — |

✱ = `e12a` tabanına göre anlamlı (Welch %95)

#### Bulgu 1 — CNN, aynı bütçede elle tasarlanmış feature'ları GEÇEMEDİ

`e12a_cnn` 1.589; aynı bütçedeki `e11c_mlp` 4.028. Yani ham tahtadan feature
öğrenmek, bizim `gamelogic`'ten türettiğimiz (BFS mesafeleri, kaçış yönleri,
bomba kurtulabilirliği) bilgiyi 6000 turda yakalayamadı.

⚠️ **AMA bu bir sıralama DEĞİL:** CNN yakınsamamış, son beşte birde **hâlâ %43
büyüyor**. `e12b` de %14. Sadece MLP platoda (%1). Yani ölçtüğümüz şey
*"6000 turda kim daha hızlı öğreniyor"* — ve elle tasarlanmış feature'lar
**öğrenmeyi hızlandırıyor**, bu zaten beklenen. Uzun bütçede CNN geçebilir;
ayrı bir uzun koşu gerekiyor.

#### Bulgu 2 — 🎯 Simetri augmentation CNN'i **3 katına** çıkardı

`e12a` 1.589 → `e12b` 4.671 (**+3.082**, anlamlı). Sandık 45.2 → 92.8.
Aynı deneyimden D4 grubuyla 8 kopya üretmek, örnek verimliliğini bu kadar
artırıyor. Bu, projedeki **en ucuz kazanç** — ek hesap maliyeti neredeyse yok.

#### Bulgu 3 — Bütçeyi ikiye katlamak MLP'ye az şey kattı

`e11c` (6000) 4.028 ± 1.610 → `e12c` (12000) 5.205 ± 0.462. Nokta tahmini
yükseldi ama **CI'lar örtüşüyor** — anlamlı fark değil. Asıl kazanç
**varyansın daralması** (±1.610 → ±0.462): daha uzun eğitim daha *tutarlı*
politika üretiyor.

#### 🔧 Bulgudan doğan iş: simetri MLP'de çalışmıyordu

Simetri kodum yalnızca `(C,H,W)` düzlem feature'larını destekliyordu; düz
vektörde dönüşüm tanımsızdı. Yani **en iyi modelimiz (MLP) o 3× kazançtan
faydalanamıyordu.**

`FeatureSet.dir_slots` eklendi: hangi indeks aralıklarının yön one-hot'u olduğunu
tanımlıyor (`v0b_task2` için 4 dilim). `transform_flat` o dilimleri permute ediyor,
yön dışı alanlar (tehlikede miyim, bomba atabilir miyim…) değişmiyor.

Tutarlılık testi eklendi (`test_flat_symmetry_matches_action`): feature "coin
YUKARIDA" diyorsa, dönüşümden sonra `transform_action(UP)` hangi yönü veriyorsa
orayı işaretlemeli. Düzlem versiyonunda bu hatayı bir kez yapmıştım (UP→LEFT);
aynı tuzağa düşmemek için önce testi yazdım.

**Karar → EXP-013 (koşuyor):** `e13a` MLP + simetri (12000), `e13b` MLP + simetri
(20000), `e13c` kontrol (simetri yok, aynı kod/koşu).
**Ayrıca gerekli:** CNN'in uzun koşusu — %43 büyümeyle kesilmiş bir eğriye bakıp
"CNN kaybetti" demek erken olur.

---

### EXP-013 — En iyi ajanımız: **6.43 coin** (rule_based'in %75'i) ⚠️ ama fark anlamlı değil
**Tarih:** 2026-08-17 · **Kim:** Irmak · **Kart:** EK-5 · **3 eğitim × 10 eval seed'i**
**Log:** `runs/exp013.log`

**Hipotez:** Simetri, CNN'i 3× yaptı (EXP-012); artık düz vektörde de çalıştığına
göre MLP'yi de benzer oranda iyileştirir.

| config | coin/tur | sandık | intihar | yakınsadı? |
|---|---|---|---|---|
| `e13a_mlp_sym` (12000) | 4.999 ± 2.695 | 78.95 | 0.000 | ✅ +%4 |
| **`e13b_mlp_sym_long`** (20000) | **6.433 ± 1.302** | **98.68** | **0.000** | ✅ +%1 |
| `e13c_mlp_base` (12000, simetri YOK) | 3.383 ± 2.093 | 66.74 | 0.000 | ✅ −%0 |
| *`rule_based`* | *8.53* | *116.9* | *0.000* | — |

**Hepsi yakınsadı** (%0–4) — daha uzun bütçe ve düzeltilmiş eps decay işe yaradı.

#### 🏆 En iyi ajan: `e13b`

**coin 6.433 = `rule_based`'in %75'i · sandık 98.68 = %84 · intihar 0.000 ·
400 adımın tamamı hayatta.**

#### ⚠️ Ama simetrinin katkısı İSTATİSTİKSEL OLARAK GÖSTERİLEMEDİ

`e13a` (simetri) 4.999 vs `e13c` (kontrol) 3.383 → **+1.616**, yani nokta
tahmininde **%48 daha iyi** — ama **yıldız yok**, güven aralıkları örtüşüyor
(±2.695 ve ±2.093).

Bu, EXP-012'deki CNN sonucuyla (simetri +3.082, **anlamlı**) çelişiyor gibi
görünüyor. Dürüst okuma: **simetri CNN'de kanıtlandı, MLP'de kanıtlanamadı.**
Yön işaret ediyor ama 3 seed bu varyansta yetmiyor.

#### 🔴 Asıl sorun: eğitim seed'leri arası varyans baskın

| config | coin |
|---|---|
| `e12c_mlp_long` (12000, simetri yok) | **5.205 ± 0.462** |
| `e13c_mlp_base` (12000, simetri yok) | **3.383 ± 2.093** |

**Neredeyse aynı config, iki farklı koşu → 5.205 vs 3.383.** Aradaki fark,
ölçmeye çalıştığımız etkilerden büyük.

C14'te 3 eğitim seed'ine geçmiştik; **bu ölçekte 3 de yetmiyor.** Küçük etkileri
(simetri, bütçe) ayırt etmek için **5–8 eğitim seed'i** gerekiyor. Rapor §6'da
bu açıkça yazılacak: hangi sonuçlarımız istatistiksel olarak sağlam, hangileri
yalnızca yön gösteriyor.

**Karar:**
- ✅ `e13b` (MLP + simetri, 20000 tur) şu anki **en iyi ajanımız**
- ⏭️ **EXP-015 (koşuyor): Task 4'e geçiş.** Ajanımız hiç rakip görmedi ve EXP-008
  transferin negatif olabileceğini gösterdi — turnuva vekili EK-2'de ölçüyoruz
- 📌 Borç: `e13b` üzerinde 5–8 seed ile tekrar (simetri sorusunu kapatmak için)
  ve CNN'in uzun koşusu

---

### EXP-015 — 🎯 Task 4: turnuva vekilinde **rule_based'in %75'i**
**Tarih:** 2026-08-17 · **Kim:** Irmak · **Kart:** EK-2 (turnuva vekili) · **3 eğitim × 10 eval seed'i**
**Log:** `runs/exp015.log`

**Hipotez:** Ajanımız hiç rakip görmedi. EXP-008 transferin negatif olabileceğini
gösterdi → rakiple eğitim gerekli. Ayrıca müfredat (önce tek başına, sonra rakiple)
doğrudan eğitimden iyi olmalı.

| ajan | skor | coin | kill | intihar | adım |
|---|---|---|---|---|---|
| **`e15b_direct_opp`** (3 rakiple 12000) | **2.487 ± 0.676** | 1.76 | 0.14 | 0.591 | 210 |
| `e15a_curric_opp` (6000 solo → 6000 1 rakip) | 1.785 ± 0.419 | 1.21 | 0.12 | 0.661 | 200 |
| `e15c_solo_control` (rakip görmeden 12000) | 1.695 ± 0.459 | 1.06 | 0.13 | 0.773 | 206 |
| *`rule_based_agent`* | *3.317 ± 0.247* | *2.25* | *0.21* | *0.528* | *228* |

Hepsi yakınsadı (−%4…+%4).

**İki anlamlı bulgu:**

| karşılaştırma | fark | sonuç |
|---|---|---|
| rakip maruziyeti (`e15b` − `e15c`) | **+0.792 [+0.265, +1.319]** | ✅ ANLAMLI |
| doğrudan vs müfredat (`e15b` − `e15a`) | **+0.702 [+0.114, +1.289]** | ✅ ANLAMLI |

#### 🏆 Turnuva vekilinde **%75**: 2.487 / 3.317

#### Bulgu 1 — rakiple eğitim şart, hipotez doğrulandı
Rakip görmeden eğitilen ajan EK-2'de 1.695; 3 rakiple eğitilen 2.487. Transfer
beklemek yanlıştı (EXP-008'in dersi tekrar).

#### Bulgu 2 — müfredat İKİNCİ KEZ başarısız, hipotez çürüdü
"Önce kolay, sonra zor" doğrudan eğitimden **anlamlı ölçüde kötü** (−0.702).
EXP-009'da tablosal modelde de sıfır transfer vermişti; orada sebep ayrık state
uzaylarıydı. Şimdi genelleme yapan bir ağda da işe yaramadı.
**İki farklı model sınıfı, aynı sonuç** — rapor §4 için sağlam bir negatif bulgu.

#### 🔴 Bulgu 3 — açığımız İNTİHARDA, ve `rule_based`'den KÖTÜYÜZ

| | Task 2 (rakipsiz) | EK-2 (3 rakip) |
|---|---|---|
| bizim intihar | **0.000** | **0.591** |
| `rule_based` | 0.000 | 0.528 |

Rakipsizken hiç intihar etmeyen ajan, rakip gelince `rule_based`'i bile geçiyor.

**Teşhis:** `escape_directions` rakipleri **şu anki yerlerinde sabit** varsayıyor.
Ama rakipler hareket eder — bomba bırakırken "kaçışım var" diyoruz, rakip yolu
kesiyor, ölüyoruz.

**Karar → EXP-016 (koşuyor):** `v0c_task4` feature seti yazıldı —
kaçış yolu **sayısı** (bool değil), rakipler komşuluklarıyla bloklanırsa
**temkinli kaçış** kalıyor mu, rakip yönü ve yakınlığı. Tek değişken feature;
kontrol grubu `v0b_task2` ile aynı koşuda, **4 eğitim seed'i** ile
(3 seed'in yetmediğini EXP-013'te görmüştük).

---

### EXP-016 — Teşhis doğruydu, tedavi işe yaradı, **skor değişmedi**
**Tarih:** 2026-08-17 · **Kim:** Irmak · **Kart:** EK-2 · **4 eğitim × 10 eval seed'i**
**Log:** `runs/exp016.log`

**Hipotez:** EXP-015'te intihar 0.591 (rule_based 0.528) çıkmıştı çünkü
`escape_directions` rakipleri sabit varsayıyor. Rakip farkındalıklı feature'lar
(`v0c_task4`) intiharı düşürüp skoru yükseltir.

| config | skor | coin | sandık | **intihar** | adım |
|---|---|---|---|---|---|
| `e16a_v0b_base` (kontrol) | 2.280 ± 0.401 | 1.724 | 33.52 | 0.481 | 229.7 |
| `e16b_v0c_opp` (rakip farkındalıklı) | 2.170 ± 0.484 | 1.407 ✱ | 36.39 ✱ | **0.298** ✱ | 247.6 |

✱ = anlamlı (Welch %95). İkisi de yakınsadı.

**Tedavi tam hedefini vurdu:** intihar **0.481 → 0.298** (−%38, anlamlı),
`rule_based`'in 0.528'inin **altına** indik. Sandık +2.87 (anlamlı), adım +17.9.

**Ama skor değişmedi** (−0.110, anlamsız) — çünkü coin **düştü** (−0.317, anlamlı).
Ajan güvenli oynamayı öğrendi, karşılığında saldırganlığından verdi.

> **Ders:** Teşhis edilen bir zaafı gidermek, hedefi otomatik iyileştirmez.
> Ajan onu başka bir boyutla takas edebilir.

#### 🔴 Asıl sorun ölçüldü: yanlış şeyi optimize ediyoruz

Bir turdaki shaped ödülün nereden geldiğini hesapladım:

| | sandıktan | coinden | kill'den | **sandık payı** |
|---|---|---|---|---|
| `e16a` | 50.3 | 1.72 | 0.56 | **%96** |
| `e16b` | 54.6 | 1.41 | 0.76 | **%96** |

Gerçek oyun skoru ise `coin×1 + kill×5`; **sandığın skora katkısı SIFIR.**

`r04_ratio`'da `CRATE_DESTROYED = 1.5` vardı — Task 2 için doğruydu (sandık
kırmak orada tek işti) ama Task 4'e taşıyınca **bir sandık çiftliği botu**
ürettik. Ajan turnuvada hiçbir puan getirmeyen şeyi maksimize ediyor.

**Karar → EXP-017 (koşuyor):** ödülü gerçek hedefe hizala.
- `r07_objective`: sandık 1.5 → **0.15** (araç), `COIN_FOUND` 0.5 → **2.0**
  (coini AÇAN sandık değerli), coin 1.0 → **2.0**. Yeni dağılım: sandık %40.
- `r08_sparse`: neredeyse saf oyun ödülü — DQN'in gücü (EXP-011) yoğun
  shaping'i gereksiz kılıyor olabilir, ve shaping yoksa ona overfit de yok.
- `e17a`: kontrol (`r04_ratio`).

---

### EXP-017 — 🏆 **RULE_BASED GEÇİLDİ**: ödülü hedefe hizalamak +1.865 getirdi
**Tarih:** 2026-08-18 · **Kim:** Irmak · **Kart:** EK-2 (turnuva vekili) · **4 eğitim × 10 eval seed'i**
**Log:** `runs/exp017.log`

**Hipotez:** EXP-016 ölçtü ki shaped ödülün %96'sı sandıktan geliyor, oysa
sandığın gerçek skora katkısı sıfır. Ödülü gerçek hedefe (`coin×1 + kill×5`)
hizalarsak skor yükselir.

| config | skor | coin | sandık | intihar | adım |
|---|---|---|---|---|---|
| `e17a_r04_control` | 2.113 ± 0.189 | 1.494 | 36.26 | 0.330 | 257.9 |
| **`e17b_objective`** | **3.979 ± 0.377** ✱ | **2.424** ✱ | 34.64 | **0.199** ✱ | **338.9** ✱ |
| `e17c_sparse` | 1.923 ± 0.985 | 1.699 | 11.71 ✱ | 0.181 ✱ | 254.3 |
| *`rule_based_agent`* | *3.317 ± 0.247* | *2.247* | — | *0.528* | *228.4* |

✱ = kontrole göre anlamlı. Üçü de yakınsadı.

#### 🏆 `rule_based_agent`'i HER metrikte geçtik

| metrik | `e17b` | `rule_based` | |
|---|---|---|---|
| **skor** | **3.979 ± 0.377** | 3.317 ± 0.247 | **GA'lar örtüşmüyor → anlamlı** |
| coin | 2.424 ± 0.194 | 2.247 ± 0.085 | örtüşüyor (berabere) |
| kill | **0.311** | 0.214 | önde |
| intihar | **0.199** | 0.528 | çok önde |
| adım | **338.9** | 228.4 | çok önde |

**Oran: %120.** PDF'in kendi ölçütü: *"you must be able to beat the
rule_based_agent in order to have any chance of winning the tournament."*

#### Ödül hizalamasının katkısı: **+1.865 [+1.497, +2.234]**

Projenin **en büyük tek kazancı**. Tek değişen şey ödül tablosundaki üç sayı:
`CRATE_DESTROYED` 1.5→0.15, `COIN_FOUND` 0.5→2.0, `COIN_COLLECTED` 1.0→2.0.
Model, feature, mimari, bütçe — hepsi aynı.

> **Rapor için ana ders:** ödülün BÜYÜKLÜĞÜ değil, YÖNÜ belirleyici.
> EXP-010'da sandık ödülünü 10× artırmak hiçbir şey değiştirmemişti; burada
> onu 10× AZALTMAK (ve coini yükseltmek) skoru 1.9 kat yaptı. Optimize edilen
> şeyin gerçek amaçla hizalı olması, sinyalin gücünden önemli.

#### `r08_sparse` başarısız — yoğun shaping GEREKLİ

Neredeyse saf oyun ödülüyle sandık 36 → **11.7** (anlamlı düşüş), skor 1.923
ve güven aralığı çok geniş (±0.985). Ajan kazmayı öğrenemiyor; `classic`'te
9 coin sandık altında olduğu için kazmadan skor da yok.

Yani shaping gerekli — ama **doğru yöne** shaping.

**Karar:**
- ✅ `e17b_objective` seed 2 (EK-2'de **4.203**) **TURNUVA ADAYI** oldu
- ✅ Ajanın varsayılan config'i güncellendi, `vendor.py` ile paketlendi (287 KB),
  bağımsızlık doğrulaması geçti
- ❌ `r08_sparse` terk edildi (rapor §4: denenip bırakılanlar)
- ⏭️ Sıradaki: Task 3 (EK-4), self-play, ve diğer takımların ajanlarına karşı merdiven

**⚠️ Not:** Turnuva adayımız `classic`+rakip için özelleşti; `coin-heaven`'da
yalnızca 2.4 coin/tur alıyor (Task 1 ajanımız 50/50 alıyordu). PDF görevlerin
alt küme olduğunu söylüyor ama biz uzmanlaşmış bir ajan ürettik. Turnuva
`classic` olduğu için sorun değil; rapora bu tercih olarak yazılacak.

---

### EXP-018R — Hipotezim çürüdü, ama confounded bir sonucu KURTARDI
**Tarih:** 2026-08-24 · **Kim:** Irmak · **Kart:** EK-2 · **4 eğitim × 10 eval seed'i**
**Log:** `runs/exp018R.log` · Çok-aşamalı bug DÜZELTİLDİKTEN sonra koşuldu.

**Neden yeniden koşuldu:** `callbacks.py` epsilon'u `game_state["round"]`'dan
hesaplıyordu; bu sayaç her `main.py` sürecinde 1'den başlıyor. Müfredat aşamaları
ayrı süreçler olduğu için **her aşamada keşif 1.0'a sıfırlanıyordu.** İlk EXP-018'de
`e18b` (18 aşama) epsilon'u hiç %68'in altına indiremedi — 24000 turun tamamı
rastgeleye yakın geçti. Aynı bug `e15a_curric_opp` ve `m02b_tab_curric`'i de vurmuştu.

| config | skor | coin | sandık | intihar | adım |
|---|---|---|---|---|---|
| **`e18a_fixed3_control`** (hep 3 rakip) | **4.157 ± 0.053** | 2.592 | 36.68 | 0.178 | 335.5 |
| `e18b_mixed_opp` (karma rakip sayısı) | 3.668 ± 0.435 | 2.494 | 38.19 | 0.218 | 321.2 |
| `e15a_curric_opp` (müfredat) | 2.244 ± 0.356 | 1.478 ✱ | 33.36 ✱ | 0.637 ✱ | 196.9 ✱ |

| karşılaştırma | fark | |
|---|---|---|
| karma rakip sayısı − kontrol | **−0.490** | ✱ ANLAMLI, **ZARAR** |
| müfredat − kontrol | **−1.913** | ✱ ANLAMLI, **ZARAR** |

#### ❌ Hipotez 1 çürüdü — karma rakip sayısı ZARAR verdi

Beklentim: `rule_based` turların %53'ünde intihar ettiği için EK-2 turlarının
yarısında yalnız kalıyoruz ve orada zayıfız (EK-5'te %37); karma rakip sayısıyla
eğitmek bunu düzeltmeli.

**Tam tersi oldu** (−0.490, anlamlı). Ve `e18b` yakınsamadı — son beşte birde
**küçülüyor** (−%7), yani sadece "yetersiz eğitim" değil, aktif bozulma var.
Muhtemel sebep: rakipsiz bloklar ajanı rakipsiz politikaya çekiyor ve 3-rakipli
bloklarda öğrendiğini bozuyor — **felaket unutma (catastrophic forgetting)**.

#### ✅ Ama müfredat sonucumuz KURTULDU

EXP-015'te "müfredat doğrudan eğitimden kötü" demiştik ve bug bulununca bu
**confounded** olmuştu. Düzeltilmiş kodla tekrar koşunca fark **daha da büyüdü**
(−0.702 → **−1.913**). Yani orijinal sonuç doğruymuş, sadece güvenli
gösterilmemişti. Artık gösterildi.

> Bu, bug'ı bulmanın iki kere kazandırdığı bir durum: hem geçersiz bir deneyi
> attık, hem asılı kalan bir sonucu sağlamlaştırdık.

#### 📌 Bütçe: skor değil, TUTARLILIK alıyor

| | tur | skor |
|---|---|---|
| `e17b_objective` | 20000 | 4.167 ± **0.256** |
| `e18a_fixed3_control` | 24000 | 4.157 ± **0.053** |

Ortalama aynı, **güven aralığı 5 kat daraldı** (seed'ler: 4.108–4.181). EXP-012'de
de aynı örüntüyü görmüştük (±1.610 → ±0.462). Daha uzun eğitim daha *yüksek* değil,
daha *öngörülebilir* ajan üretiyor — turnuvada tek maç oynanacağı için bu değerli.

**Karar:**
- ✅ `e18a_fixed3_control` yeni referans taban (en dar GA)
- ❌ Karma rakip **sayısı** terk edildi
- ❌ Müfredat kesin olarak terk edildi (artık sağlam kanıtla)
- ⏭️ **EXP-019: rakip ÇEŞİTLİLİĞİ** — bu farklı bir eksen. `e18b` "kaç rakip"
  sorusunu test etti ve kaybetti; EXP-019 "hangi rakipler" sorusunu test edecek
  (`coin_collector`, `Qlearned_agent`, kendi checkpoint'lerimiz). Turnuvada
  karşımızda `rule_based` değil, başka takımların ajanları olacak.

---

### EXP-019 — 🎯 Rakip TİPİ çeşitliliği kanıtlandı (+0.254, anlamlı)
**Tarih:** 2026-08-24 · **Kim:** Irmak · **Kart:** EK-2 · **4 eğitim × 10 eval seed'i**
**Log:** `runs/exp019.log`

**Hipotez:** Şimdiye kadar SADECE `rule_based`'e karşı eğitildik ve o ajan turların
%53'ünde intihar ediyor. Turnuvada karşımızda başka takımların ajanları olacak.
Rakip **tipini** çeşitlendirmek genelleme kazandırır.

**EXP-018R'den kritik tasarım farkı:** orada rakip **sayısı** aşamalar arasında
değişiyordu ve felaket unutma oldu (−0.490). Burada tüm rakipler **aynı anda, tek
aşamada** masada — politika bir yöne çekilip geri bozulmuyor.

| config | rakipler | skor | coin | intihar |
|---|---|---|---|---|
| `e19a_rb_only` | 3× rule_based | 3.912 ± 0.472 | 2.425 | 0.157 |
| **`e19b_types`** | rule_based + coin_collector + **Qlearned_agent** | **4.288 ± 0.155** | 2.590 ✱ | 0.177 |
| `e19c_league` 🔴 | ~~pool_v1~~ **RASTGELE AĞ** (bkz. aşağıdaki karar notu) | 4.376 ± 0.677 | 2.642 ✱ | 0.234 |

Üçü de yakınsadı.

#### ⚠️ ÖNEMLİ DÜZELTME — "bütçe tutarlılık alıyor" yorumum yanlıştı

`e18a_fixed3_control` ve `e19a_rb_only` **birebir aynı config.** İki farklı koşu:

| koşu | seed'ler | ortalama ± GA |
|---|---|---|
| `e18a` | 4.164, 4.181, 4.176, 4.108 | 4.157 ± **0.053** |
| `e19a` | 4.231, 3.993, 3.517, 3.908 | 3.912 ± **0.472** |

EXP-018R'de `e18a`'nın ±0.053'üne bakıp *"daha uzun eğitim daha öngörülebilir ajan
üretiyor"* demiştim. **Yanlıştı.** O dar aralık 4 seed'in şansla kümelenmesiydi.

**8 seed'i havuzlayınca gerçek değer: 4.035 ± 0.197** (aralık 3.517–4.231).

> **Ders:** 4 seed'den çıkan DAR bir güven aralığı da bir artefakt olabilir.
> Aynı config'i iki kez koşmadan "bu config tutarlı" denemez.

#### Sonuç — 8 seed'lik havuzlanmış kontrole karşı

| config | ortalama | **min seed** | fark | |
|---|---|---|---|---|
| **`e19b_types`** | 4.288 | **4.183** | **+0.254 [+0.04, +0.47]** | ✅ **ANLAMLI** |
| `e19c_league` 🔴 | 4.376 | 3.986 | +0.341 [−0.29, +0.98] | anlamsız **VE geçersiz** |

**Rakip tipi çeşitliliği kanıtlandı.** Self-play ligi daha yüksek ortalama veriyor ama
varyansı (±0.677) etkiyi yutuyor — 4 seed yetmiyor.

**Turnuva açısından `e19b` daha da iyi:** tek maç oynanacağı için **taban** önemli.
`e19b`'nin en kötü seed'i (4.183) kontrolün ORTALAMASINI (4.035) geçiyor.

#### ✅ Taze eval seed'lerinde doğrulandı (seçim yanlılığı kontrolü)

`e19b_types_s2` seçim seed'lerinde (0–9) 4.416 aldı. **Hiç görülmemiş** seed'lerde (10–19):

| metrik | değer |
|---|---|
| **skor** | **4.210 ± 0.186** |
| coin | 2.450 ± 0.086 |
| intihar | 0.284 ± 0.040 |
| timeout ihlali | **0** |

Seçim yanlılığı ≈ 0.2 puan — beklenen büyüklükte, ve 4.210 hâlâ `rule_based`'in
3.317'sinin **%127'si**. Aday sağlam.

**Karar:**
- ✅ `e19b_types` (seed 2) **yeni turnuva adayı** — 31 Ağustos'ta
  `agent_code/irmak_umut/config.yaml` bu config'e çevrildi ve
  `model_e19b_types_s2.bin` repoya zorla eklendi (`git add -f`).
  `e17b`'nin modeli fallback olarak duruyor.
- ⏭️ EXP-020: PER × n-step tam faktöriyel, `e19b` tabanı üzerinde
- 🔴 **`e19c_league` GEÇERSİZ — 8 Eylül'de bulundu.** Bu kol "kendi geçmişimize karşı
  self-play" olacaktı; olmadı. `agent_code/pool_v1/logs/pool_v1.log`:

  > `model_e19c_league_s1.bin bulunamadi ve egitim modunda degiliz: egitilmemis model ile oynanacak!`

  ~~Havuz ajanının `config.json`'u `e19c_league_s1` diyordu ama o isimli model dosyası
  kopyalanmamıştı.~~ **Bu teşhis YANLIŞTI — 17 Eylül'de düzeltildi, aşağıya bakın.**
  `callbacks.setup()` dosyayı bulamayınca **rastgele ağırlıklarla**
  oynadı ve yalnızca bir WARNING satırı bıraktı. Kimse okumadı.

  **Yani tablodaki en yüksek ortalama (4.376) self-play değil, bir rastgele ağın
  masaya eklenmesiydi.** Kol raporda self-play kanıtı olarak KULLANILAMAZ.
  *(Bu sonuç doğru kaldı; yanlış olan sebepti.)*

  #### 17 Eylül düzeltmesi — gerçek sebep: `BBRL_CONFIG` rakiplere sızıyordu

  `e19c_league_s1` havuz ajanının adı **değil**, **eğitim koşusunun** adı. Havuz
  ajanının `config.json`'u başka bir şey diyordu (`e17b_objective_s2`). O isim
  havuz ajanına şuradan geldi: `BBRL_CONFIG` bir **ortam değişkeni**, yani
  `main.py` sürecindeki **her** ajana miras kalıyor — sadece eğitilene değil.
  Havuz ajanı eğitim koşusunun config'ini okudu, o isimde bir modeli kendi
  klasöründe aradı, bulamadı, rastgele oynadı.

  **Aynı hata EXP-019D'yi de vurdu.** Dalga 1 sırasında `pool_e19b` şunu yazdı:

  > `model_e19d_selfplay_s3.bin bulunamadi ve egitim modunda degiliz: egitilmemis model ile oynanacak!`

  Yani `e19d_selfplay` **da self-play değildi.** Bunu önlemek için kurduğum üç
  kapı (`selfplay.verify()`) hatayı yakalayamadı, çünkü havuz ajanını **tek
  başına** test ediyordu — orada `BBRL_CONFIG` hiç set edilmiyor. Sızıntı yalnızca
  bir eğitim koşusunun içinde oluyor. Yanlış teşhis, yanlış yere kapı koydurdu.

  **Düzeltme (commit `8497b0b`):** `BBRL_CONFIG_AGENT` config'in hangi ajan klasörü
  için olduğunu söylüyor; `_config_path()` başkasını hedefleyen `BBRL_CONFIG`'i yok
  sayıyor. `train_loop` ve `resume` eğitim süreçlerinde set ediyor, `eval.run_one`
  varsayılan olarak değerlendirilen ajana (`agent_dirs[0]`) set ediyor.

  `tests/test_config_scope.py` **iki yönlü**: hedef yokken uyarıyı birebir yeniden
  üretiyor (`model_umut_dueling.bin bulunamadi`), hedefle rakip kendi modelini
  yüklüyor. Tek yönlü bir test — "düzeltmeyle uyarı yok" — hatayı hiç göremese de
  geçerdi; `verify()`'ın başına gelen tam olarak buydu.

  > **Ders:** bir kazanın sebebini "makul görünen" ilk açıklamayla kapatma. Buradaki
  > açıklama yanlıştı ama semptomla tutarlıydı, ve üzerine inşa edilen önlem
  > gerçek sebebi hiç test etmediği için bir sonraki deneyi de korumadı.

  **Self-play ekseni bu projede hiç doğru test edilmedi.** Havuz ajanlarının
  (`pool_v1`, `pool_e19b`) vendor'lanmış `callbacks.py`'leri eski kodu taşıyor;
  yeniden kullanılacaksa `selfplay.snapshot()` ile yeniden üretilmeleri gerekir.

---

### EXP-020 — PER × n-step tam faktöriyel *(koşuyor)*
**Tarih:** 2026-08-31 · **Kim:** Irmak

**Hipotez:** `bbrl/buffer.py`'daki prioritized replay **yazıldı, test edildi ama hiçbir
config'de kullanılmadı** — bütün configler `uniform`. n-step return de yeni eklendi.
İkisi de seyrek ödülde standart kazanç sağlar; bizde `KILLED_OPPONENT` (5.0 puan) çok
seyrek. 2×2 faktöriyel, çünkü ikisinin **etkileşimi** olabilir: PER zaten yüksek-TD
transition'ları öne çıkarıyor, n-step de ödülü zamanda geriye taşıyor — üst üste
binebilirler.

| config | buffer | n |
|---|---|---|
| `e20a_base` | uniform | 1 |
| `e20b_per` | prioritized | 1 |
| `e20c_nstep` | uniform | 3 |
| `e20d_both` | prioritized | 3 |

Taban `e19b_types`, 4 eğitim seed'i, eşit bütçe, EK-2 ile ölçülüyor.

**Not — ilk deneme yanlış teşhis edildi:** ilk başlatmada 16 koşunun 7'si boş
`stderr` ile düştü ve bunu "16 worker fazla, süreç sert çöküyor" diye yorumladım.
**Gerçek sebep: koşuyu ben elle durdurmuştum.** Yani worker sayısı hakkında
çıkardığım sonuç dayanaksız.

Buradan çıkan gerçek ders koda girdi (`bbrl/train_loop.py`): süreç python'a söz
vermeden ölürse `stderr` boş kalır ve hata **sebepsiz** görünür — bu da insanı
uydurma bir teşhise iter. Artık en azından çıkış kodu basılıyor.

**Sonuç:** _(koşuyor)_

---

---

### EXP-020..024 — 16 saatlik kampanya: ONCEDEN KAYITLI TAHMINLER
**Tarih:** 2026-09-08 · **Kim:** Irmak · **Durum:** Dalga 1 kosuyor

> Bu bolum **kosudan ONCE** yazildi. Sonuclar geldikce doldurulacak ama
> tahminler ve MDE degistirilmeyecek.

#### Kampanyanin durustce ilan edilmis sinirlari

sigma (egitim seed'i, EK-2) = **0.235** (8 seed havuzlanmis kontrol).
4 seed x 4 seed Welch, aile duzeltmeli **MDE ~ 0.66**.

**Bu projede olculmus en buyuk odul-disi etki: +0.254.**

Yani optimizer ekseninde tespit tabanimiz, tarihsel gercek etkinin **2.5 kati**.
En olasi sonuc **null**. Bu bir tasarim hatasi degil, butcenin siniri -- ama
onceden yazilmazsa null "bir sey bulamadik" diye okunur. Null cikan her satirin
yanina "bu tasarim X'ten kucuk etkiyi goremezdi" yazilacak.

Onlem: ikili kontrast yerine **doz-yanit trendi** (3 doz, 12 kosu) MDE'yi
0.79 -> 0.62'ye indiriyor ve bedava. Dort aile, aile ICINDE Holm.

#### Dalga 1 -- 4 config x 4 egitim seed'i = 16 kosu

| config | TEK degisken | tahmin (EK-2, kontrole gore) | mekanizma |
|---|---|---|---|
| `e20a_base` | KONTROL, 14000 tur | e19b'ye (24000) gore **etkisiz**, -0.10..+0.10; ama **sd artar** (e19b sd 0.098 -> beklenen 0.15-0.30) | EXP-012: butce x2 ortalamayi degil GA'yi oynatiyordu (+-1.610 -> +-0.462) |
| `e19d_selfplay` 🔴 GECERSIZ | ucuncu rakip: `Qlearned_agent` -> `pool_e19b` **(pool_e19b sizan config yuzunden RASTGELE oynadi, bkz. e19c notu)** | **yukseltir, 0.00..+0.40** | Kaldirac tablosunda skoru oynatan her sey "neye karsi egitildigi" oldu. Havuzun en zayif halkasi Qlearned (2.953); kendi 4.210'umuzla degistirmek kanitlanmis eksende dogrudan doz artisi |
| `e21b_eps01` | eps_end 0.05 -> 0.01 | **yukseltir, -0.10..+0.25** | Projenin en pahali tekrarlayan hatasi: eps=0.05 politikayi tasiyor (EXP-002, 009, 011, 016 -- dort ayri vaka). Turda ~8.6 rastgele aksiyon; egitim dagilimi, eps=0 politikasinin hic uretmedigi olumleri iceriyor |
| `e23b_ratio2x` | train_every 4 -> 2 (51 -> 94 update/tur) | **isaret BELIRSIZ, -0.20..+0.30** | Yukari: gradyan duvar suresinin sadece %6'si (olculdu: update 2.167 ms, tur 1.83 s), yani 1.84x ogrenme +%7.6 maliyete geliyor. Asagi: `target_sync=1000` sabit kaldigi icin hedef ag 2x sik senkronlaniyor -> primacy bias |

#### ⛔ Kampanyanin akibeti (17 Eylul)

**Dalga 1 hic degerlendirilmedi.** 16 kosunun 12'si 14000 tura, `e21b_eps01`'in 4'u
11600 tura ulasti; 8 Eylul aksami durduruldu, `bbrl.resume` ile surduruldu ama o koşu
basinda kesildi. EK-2 ozeti yok. Teslim tarihi yuzunden kampanya burada birakildi.

Degerlendirilse bile **`e19d_selfplay` kolu gecersiz olurdu** -- rakip `pool_e19b`
sizan `BBRL_CONFIG` yuzunden rastgele agirliklarla oynadi (bkz. e19c notu, 17 Eylul
duzeltmesi). Kalan uc kol (`e20a_base`, `e21b_eps01`, `e23b_ratio2x`) bundan
ETKILENMEDI: rakipleri `rule_based`, `coin_collector` ve `Qlearned_agent` -- hicbiri
`BBRL_CONFIG` okumuyor. Modeller diskte duruyor; zaman olursa degerlendirilebilirler.

#### Gemi kurali (onceden sabit -- kazananin laneti icin)

- Secim: EK-2, seed **0-9**
- Karar: EK-2, seed **20-29** (0-9 ve 10-19 `e19b` uzerinde yakildi)
- Degisim ancak **ikisi birden** saglanirsa: (a) holdout > 4.210, **ve**
  (b) adayin config'inin **en kotu** egitim seed'i 4.288'i geciyor
- Saglanmazsa `e19b_types_s2` korunur -- bu bir SONUCTUR, basarisizlik degil

Emsal: `e19b_s2` secim seed'lerinde 4.416, taze seed'lerde 4.210 -> **0.206 buharlasti**.

#### Kosudan once kapatilan uc olcum borcu

| # | ne | neden onemliydi |
|---|---|---|
| **P0** | `train_loop.train()` artik `model_<name>.bin`'i de siliyor | Silinmiyordu; `setup()` kosulsuz yukluyor. Yeni kosu eski agirliklardan devam ediyordu -- epsilon 1.0'a, buffer sifira donmus halde. **Yanlilik YONLU**: mimari degistiren kollar shape hatasi alip sifirdan basliyor, mimarisi ayni olan KONTROL sicak basliyor. Olculdu: ayni config uc kez -> `updates` 2381, 3283, 4250. Regresyon testi: `tests/test_multistage.py::test_fresh_run_discards_stale_model` |
| **P1** | Diskteki 16 bayat `model_e20*` silindi | P0 yalnizca duzeltmeden SONRA baslayan kosulari korur |
| **P2** | Gonderilen ajan ilk kez EK-1/3/4/5'te olculdu | Raporda dolasan "EK-5 %37, EK-3 %10" rakamlari eski aday `e17b`'ye aitti |


---

### BULGU-01 — Ajana duvara toslamayi beklemekten UCUZA ogretmisiz
**Tarih:** 2026-09-08 · **Kim:** Irmak · **Tur:** olcum, deney degil

`r07_objective` tablosunda:

| aksiyon | odul | ortamda ne olur |
|---|---|---|
| `MOVED_*` | **-0.02** | ajan gercekten hareket eder |
| `INVALID_ACTION` | **-0.10** | adim yanar, ajan YERINDE kalir |
| `WAITED` | **-0.15** | adim yanar, ajan YERINDE kalir |

Son iki satir ortamda **birebir ayni sey** (environment.py 132-150: ikisi de
sadece bir olay ekler, baska hicbir etkisi yok). Ama biri digerinden %50 daha
ucuz. Yani ajanin "bu turu pas gec" demek icin kesin BASKIN bir yolu var.

Ogrendi mi? `trainlog_e19b_types_s2`, son 2000 tur, tur basina:

| olay | sayi | adimin %'si |
|---|---|---|
| `INVALID_ACTION` | **20.00** | %11.5 |
| `WAITED` | **2.61** | %1.5 |

**Oran 7.7x.** Ajan beklemiyor, duvara tosluyor.

#### Ama bu skoru dusuruyor mu? BILINMIYOR -- ve fazla yorumlamamak lazim

Gercek hareket (-0.02) hala en ucuzu. Yani ajan hareket YERINE toslamiyor;
**yerinde kalmak istedigi anlarda** toslamayi bekleme yerine seciyor. Tur
basina 22.5 bomba atiliyor ve her patlamanin beklenmesi gerekiyor, yani
~20 pas hamlesi kendi basina makul olabilir. Oyleyse maskeleme sadece ayni
davranisi DOGRU ETIKETLE yapar: skor degismez, `invalid_action_rate` sifira
iner.

Iki gozlem bunun tam hikaye olmadigini soyluyor:

1. **EK-3 (coin-heaven) oranı 0.297** -- orada bomba YOK, yani "patlama
   bekliyorum" aciklamasi gecerli degil. 400 adimin ~120'si tosluyor.
2. Maskeleme, Q-siralamasinda toslamadan sonra ne geliyorsa onu secer.
   Siralama [tosla, WAIT, ...] ise davranis ayni kalir; [tosla, hareket,
   WAIT] ise davranis DEGISIR.

#### Nasil ogrenilecek

| yol | maliyet | ne soyler |
|---|---|---|
| `act()` maskesi A/B (yeniden egitim YOK) | ~15 dk eval | mevcut politikada toslamanin yerini ne aliyor |
| Odul siralamasi kolu (`INVALID_ACTION` <= `WAITED`) | 4 kosu | bastan ogrenirse farkli bir politika mi cikiyor |

**Kampanya ortasinda odul tablosuna DOKUNULMAYACAK** -- 8 kolun hepsini birden
confound eder. Odul kolu Dalga 2 adayi, maske A/B ise Dalga 1 ile Dalga 2
arasindaki bosluga sigar.

> Rapor icin degeri sonuctan bagimsiz: bu, odul tasarimi derslerimizin
> ucuncusu. Birincisi "odulun YONU buyuklugunden onemli" (EXP-017),
> ikincisi "potansiyel yanlis tasarlanirsa ters teper" (EXP-010), ucuncusu
> **"ayni sonucu veren iki aksiyonu farkli fiyatlarsan ajan ucuz olani
> ogrenir -- isterse anlamsiz olsun"**.


---

### BULGU-02 — PER "en yeniyi tekrarla"ya donmemis: iddia olculdu ve dusuruldu
**Tarih:** 2026-09-08 · **Kim:** Irmak · **Tur:** olcum

Kampanya planinda dalgalar arasi bir kod isi vardi: `buffer.py:221`'deki
`max_p = max(self.max_p, td.max())` monoton, hic sonmuyor. `KILLED_SELF=-15`
yuzunden erken ~15'e cikiyor ve yeni ornekler `15**0.6 = 5.08` onceligiyle
giriyor; tipik TD ise `0.135**0.6 ~ 0.30`. Buradan "PER pratikte oncelikli
degil, EN YENIYI TEKRARLA oluyor" sonucu cikarilmisti.

**Olctum, sonuc bu degil.** Gercek egitim oranini taklit eden simulasyon
(adim basina 8 push -- simetri augmentation -- ve ~19 ornek, `train_every=4`
+ `updates_per_round=8`; 200k buffer, TD'ler N(0, 0.135) arti arada bir 15):

| olcum | plandaki iddia | olculen |
|---|---|---|
| en yeni %10'un medyan onceligi / genel medyan | ~20x | **1.4x** |
| ornekleme payi, en yeni %10 (adil pay %10) | — | **%22.7** |
| hic guncellenmemis (hala max_p'de duran) ornek | — | **%0.5** |

**Neden:** yuksek oncelikli yeni ornek hemen cekiliyor ve `update_priorities`
onun 5.08'ini gercek TD'siyle degistiriyor. Sisme GECICI. Zaten Schaul et al.
2016 yeni ornekleri kasten maksimum oncelikle koyar -- amac "en az bir kez
gorulsun"dur, kusur degil.

**Karar: dalgalar arasi PER yamasi YAPILMAYACAK.** Planin kendi kabul kapisi
T1 "en yeni / medyan oncelik orani < 3" idi; olculen deger **1.4**, yani kapi
zaten acik. `e20b_per` Dalga 2'de OLDUGU GIBI kosacak ve olctugumuz sey
gercekten PER olacak.

Kalan gercek etki 2.3x'lik bir tazelik egilimi (%22.7 / %10). Bu PER'in
dogasinda var ve is-politikasi verisini one cikardigi icin zararli olmasi da
gerekmiyor.

> **Sinir:** simulasyon TD'leri sabit bir dagilimdan cekiyor; gercekte TD
> hatalari state yeniligiyle korelasyonlu ve zamanla degisiyor. Test edilen
> mekanizma (max_p sismesi kalici mi) dogru yakalaniyor ama sayilar tam
> egitim dinamigini temsil etmiyor.
>
> **Ders:** "kodda su satir yanlis gorunuyor"dan "bu, sonucu bozuyor"a
> gecmek bir OLCUM ister. Bu iddia makuldu ve yanlisti; olcmeden yamalasaydik
> gereksiz bir degisiklik yapip Dalga 1 kontrolunu Dalga 2'ye tasiyamama
> riskini bedavaya almis olacaktik.


### EXP-003 — Danger feature: `bombs` vs `explosion_map`
**Tarih:** 2026-08-__ · **Kim:** Umut

**Hipotez:** Tehlike haritasını yalnızca `explosion_map`'ten hesaplayan varyant,
patlamak üzere olan bombayı göremediği için (bkz. `GAME_MECHANICS.md` §2)
belirgin şekilde daha yüksek `suicide_rate` gösterir.

**Neden önemli:** Bu, raporun en öğretici ablation'larından biri — mekanizması net,
sonucu ölçülebilir, ve tasarım kararımızı doğrudan gerekçelendiriyor.

**Sonuç:** _(doldur)_

---

### EXP-004 — Çifte ölüm cezası: `KILLED_SELF` + `GOT_KILLED`
**Tarih:** 2026-08-__ · **Kim:** Umut

**Hipotez:** İntihar her iki olayı da tetiklediği için (GAME_MECHANICS §6), ikisine de
tam ceza veren şema efektif olarak 2× ceza uygular; bu aşırı-kaçıngan (bomba atmayan)
bir politikaya yol açar ve `kills`/`coins` düşer.

**Varyantlar:** (a) her ikisi −5, (b) `KILLED_SELF` −5 / `GOT_KILLED` 0, (c) her ikisi −2.5

**Sonuç:** _(doldur)_

---

### EXP-005 — Ölüm transition'ı `end_of_round`'da yakalanıyor mu?
**Tarih:** 2026-08-__ · **Kim:** Irmak

**Hipotez:** `end_of_round`'daki terminal transition buffer'a yazılmazsa ajan ölüm cezasını
hiç görmez (GAME_MECHANICS §6, tuzak 3) ve `suicide_rate` düşmez.

**Kurulum:** Aynı config, tek fark: `end_of_round` transition'ı yazılıyor / yazılmıyor.

**Neden yapıyoruz:** Bu bir bug avı değil, **ablation olarak raporlanabilir bir tasarım
kararı.** Sessizce düzeltmek yerine ölçüp yazacağız.

**Sonuç:** _(doldur)_

---

### EXP-006 — Simetri augmentation (D4 grubu)
**Tarih:** 2026-08-__ · **Kim:** _(rotasyon)_

**Hipotez:** Tahta 4 dönme × 2 aynalama simetrisine sahip. Transition'ları 8 katına
çıkarmak, aynı env adımı bütçesinde örnek verimliliğini artırır.

**Dikkat:** Aksiyon etiketleri de dönüştürülmeli (`UP`→`RIGHT` vb.); `WAIT`/`BOMB` sabit.
Yanlış eşleme sessizce zehirler → önce birim testi.

**Sonuç:** _(doldur)_

---

## Negatif sonuçlar / terk edilen yaklaşımlar

> Rapor §4 bunları açıkça istiyor ("approaches you tried and abandoned later, including
> the reasons"). Buraya yazın, silmeyin.

| # | Ne denendi | Neden bırakıldı | Kanıt |
|---|---|---|---|
| | | | |

---

## Deney Konseyi tutanakları

### Konsey #1 — 2026-08-23
**Katılım:** Irmak, Umut
**Sprint 1 DoD durumu:** _/9_
**Öne çıkan bulgular:**
**Sprint 2 hipotezleri:**
**Rol takası kararı:**
