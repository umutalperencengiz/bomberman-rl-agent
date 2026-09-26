# Durum Panosu — 16 Ağustos 2026

> Canlı dosya. Her Deney Konseyi'nde güncellenir.
> Detaylar: [TEAM_PLAN.md](TEAM_PLAN.md) · [EXPERIMENTS.md](EXPERIMENTS.md) ·
> [GAME_MECHANICS.md](GAME_MECHANICS.md) · [INTERFACE.md](INTERFACE.md)

**Repo:** https://github.com/irmakerkol/mle-final-bomberman (PRIVATE)
**Kalan süre:** ajan kodu **21 Eylül** (36 gün) · rapor **28 Eylül** (43 gün)

---

## A. Ne yaptık — tamamlanan işler

| # | İş | Çıktı | Kim |
|---|---|---|---|
| A1 | Private repo + `upstream` ayrımı, `.gitignore` hijyeni | repo canlı, venv sızmıyor | Irmak |
| A2 | Proje dokümanları | 4 doküman (`docs/`) | Irmak |
| A3 | **Oyun mekaniğinin kaynak koddan çıkarılması** | `GAME_MECHANICS.md` — 3 tuzak belgelendi | Irmak |
| A4 | **Eval harness** (çok-seed, %95 GA, Welch farkı) | `bbrl/eval.py`, `bbrl/metrics.py` | Irmak |
| A5 | **EXP-000 referans tabanlar** | 4 kart × 3 ajan × 1000 oyun | Irmak |
| A6 | Oyun mantığı saf fonksiyonları | `bbrl/gamelogic.py` + 6 test | Irmak |
| A7 | Feature / ödül / replay buffer iskeleti | `bbrl/{features,rewards,buffer}.py` + 8 test | Irmak |
| A8 | Config sistemi + model zoo | `bbrl/config.py`, `bbrl/models/` (tabular, linear, DQN) | Irmak |
| A9 | **İlk gerçek ajan** + eğitim döngüsü | `agent_code/irmak_umut/` | Irmak |
| A10 | Ortam kurulumu | conda `ml_homework`: torch 2.11+cu128, GPU doğrulandı | Irmak |
| A11 | DQN testleri (CPU bütçe, GPU, kaydet/yükle) | `tests/test_dqn.py` | Irmak |
| A12 | Uyku sonrası bütünlük kontrolü | temiz, veri kaybı yok | Irmak |
| A13 | **Davranış izleyici + patoloji avcısı** | `bbrl/trace.py` (9 patoloji tipi) | Irmak |
| A14 | Eğitim/sweep koşucusu | `bbrl/train_loop.py` (paralel eğitim + Welch tablosu) | Irmak |
| A15 | **EXP-007: donma tedavisi 2×2 ablation** | **Task 1 ÇÖZÜLDÜ** | Irmak |
| A16 | Turnuva paketleyici | `bbrl/vendor.py` → `dist/` + 21 KB zip | Irmak |

**Test durumu:** `test_gamelogic.py` ✅ · `test_features.py` ✅ · `test_dqn.py` ✅

---

## B. Ne sonuç çıktı

### B1. Referans tabanlar (EXP-000) — DONDURULDU

| kart | ajan | score | suicide | steps |
|---|---|---|---|---|
| **EK-2** (4 ajan, turnuva vekili) | `rule_based_agent` | **3.32 ± 0.25** | **0.53** | 228 |
| EK-2 | `coin_collector_agent` | 2.95 ± 0.14 | 0.49 | 205 |
| EK-2 | `random_agent` | 0.00 | 1.00 | 18 |
| EK-3 (Task 1) | `rule_based` / `coin_collector` | 50/50 (tavan) | 0.00 | **125.19 ± 0.98** |
| EK-4 (Task 3) | `rule_based_agent` | 8.60 ± 0.34 | 0.20 | 300 |

**Yenmemiz gereken sayı: 3.32 ± 0.25.** Skor = oyunun kendi skoru (coin ×1 + kill ×5),
`--save-stats` JSON'undan okunur, shaped ödülle ilgisi yok.

### B2. Ölçülmüş bulgular

| # | Bulgu | Etkisi |
|---|---|---|
| B2.1 | `explosion_map` tek başına öldürücü karelerin **%60'ını** kaçırıyor | Tehlike `bombs`'tan hesaplanmalı → EXP-003'ün gerekçesi |
| B2.2 | `rule_based_agent` EK-2 turlarının **%53'ünde intihar** ediyor | Sadece intihar etmeyen ajan bile avantajlı → Task 2 önceliği |
| B2.3 | EK-3'te **skor metrik olarak ölü** (iki ajan da 50/50) | Task 1 metriği **adım sayısı**: hedef ≤125 |
| B2.4 | CNN (322k param) CPU'da p99 **0.83 ms** / limit 500 ms | **572× marj** → model boyutu kısıt değil, yakınsama süresi kısıt |
| B2.5 | Simetri aksiyon eşlemesi ters yazılmıştı | Test yakaladı; düzeltilmeseydi augment edilen 8 kopya sessizce zehirlenecekti |
| B2.6 | Son adım hem `game_events_occurred` hem `end_of_round`'a gidiyor | Çift sayım + çelişkili transition; düzeltildi |
| B2.7 | Laptop uykusu `elapsed_s`'i şişiriyor | `cpu_s` eklendi; rapordaki süre iddiaları için o kullanılacak |

### B3. İlk ajanımız (m01, tablosal Q) — EXP-002

| metrik | eğitim (eps=0.05) | **eval (eps=0)** | hedef |
|---|---|---|---|
| coin/tur | 48.67 | **26.12 ± 1.14** | 50.00 |
| adım/tur | 169 | **352.6 ± 12.6** | 125.19 |

🔴 **Greedy politika donuyor.** Ajan bir kareye gelip `WAIT`'e kilitleniyor
(400 adımın 398'i tek karede). Sebep: `coin-heaven`'da `WAIT` seçince dünya değişmez →
aynı state → aynı argmax → sonsuz döngü. Eğitimde %5 keşif gürültüsü bunu maskeliyordu.

**Yani politika iyi değildi, keşif onu taşıyordu.** Eval harness'ı olmasaydı
"48.67, harika" deyip yanlış modelle devam edecektik.

### B4. ✅ TASK 1 ÇÖZÜLDÜ — EXP-007

Yaşam cezası (`r02_living_cost`) donmayı tamamen kaldırdı:

| | önce (m01) | **sonra (e07b)** | `rule_based` |
|---|---|---|---|
| coin/tur | 26.12 ± 1.14 | **50.00 ± 0.00** | 50.00 |
| adım/tur | 352.6 ± 12.6 | **125.83 ± 0.65** | 125.19 ± 0.98 |

Fark `rule_based`'e karşı **+0.64 [−0.46, +1.75] → istatistiksel olarak EŞİT.**
Davranış izi: **sıfır patoloji**, WAIT %92 → %0.

**Sürpriz:** `optimistic_init` tek başına **zarar verdi** (−22.85 coin) ve yaşam
cezasıyla birleşince onun faydasını da yedi. "İki iyi fikir toplanır" sezgisi
yanlış çıktı — tek tek ölçmeseydik göremezdik. Rapor §4'te "denenip bırakılan
yaklaşımlar"a gidecek.

**⚠️ Metodolojik açık:** `e07a_base` (41.67 coin) ile `m01` (26.12) **aynı config**,
farklı eğitim koşusu. Eğitim run-to-run varyansı bazı ablation etkilerinden büyük.
→ **C14**

### B5. 🔴 Task 1 → Task 2 transferi NEGATİF — EXP-008

EK-5 (`classic`, rakipsiz, 100 tur × 10 seed):

| ajan | coin/tur | intihar | adım |
|---|---|---|---|
| `rule_based_agent` | **8.53 ± 0.06** / 9 | 0.000 | 399.1 |
| `coin_collector_agent` | 8.51 ± 0.06 | 0.000 | 399.1 |
| `random_agent` | 0.006 | 1.000 | **19.3** |
| **bizim `e07b`** | 0.003 | 1.000 | **9.5** ← rastgelenin yarısı |

**Sebep:** `coin-heaven`'da hiç bomba görmedik → tablosal Q'da her Task-2 durumu
YOK → Q hep 0 → 6 aksiyon eşit kura → **%18 bomba**. `random_agent` ise
`p=[.23,.23,.23,.23,.08]` ile yalnızca **%8**. İki kat bomba = iki kat hızlı ölüm.
Başlangıç köşesinde bomba = kesin intihar.

**Üç sonuç:**
1. **Tablosal modelin sıfır genellemesi ölçüldü** → `v1_handcrafted` (C3) ve
   fonksiyon yaklaşımı (linear/DQN) için en güçlü gerekçe
2. **Müfredat `classic` içermek zorunda** — transfer beklemek yanlış
3. `SUICIDAL_BOMB` event'i hazır, bombalı ortamda devreye girecek

**Not:** `rule_based` rakipsizken **hiç** intihar etmiyor (0.000) ama 4 ajanlı
EK-2'de %53. O intiharlar rakip baskısıyla ortaya çıkıyor.

### B6. 🌙 Gece çalışması (17 Ağustos, 01:50–)

**EXP-010 — ödül ablation, dört tahminin dördü de yanlış.**

| varyant | tahmin | gerçek |
|---|---|---|
| `e10a` oran düzeltmesi | coin 3–5 | **0.294 ± 0.287** |
| `e10b` + potansiyel shaping | +0.5–1.5 | **0.077** (ters yön) |
| `e10c` + GOOD_BOMB | +0–1 | 0.078 |
| `e10d` + BOMB_IDLE | ≈ / ⬇️ | 0.135 (tek doğru tahmin) |

🔴 **Potansiyel shaping kendi kendini baltalıyordu.** Φ = −(en yakın sandığa mesafe)
seçmiştim; ama amaç sandıkları **yok etmek**, yani amaca ulaşmak Φ'yi düşürüyor.
Ölçüldü: sandık yıkmak shaping'den **−0.600** getiriyor (ödülün +1.5'inin %40'ı).
Φ'yi maksimize eden politika *"sandığın yanında dur, yıkma"*. İz doğruladı:
355 adım salınım, 799 adım donma, 189 boş bomba, 0 ölüm.
**Ders:** "en yakın X'e mesafe" potansiyeli, amaç X'i yok etmekse ters çalışır.
Düzeltme (`p04_crate_progress`) yazıldı, +0.400 veriyor.

🔴 **Ödül büyüklüğü darboğaz değil:** sandık ödülünü 10× artırmak (0.15→1.5)
sandık sayısını değiştirmedi (15.74 → 14.68).

### B7. ⚠️ Yakınsama uyarısı — önceki sonuçları yeniden yorumlamak gerek

Eğitim eğrilerini çizince görüldü: **6000. turda hiçbir varyant yakınsamamış**,
üçü son beşte birde hâlâ %11–18 büyüyor. Ve ~2700. tura kadar neredeyse hiçbir şey
olmuyor — `eps_decay_rounds=3000` tam orada bitiyor, yani bütçenin yarısı
rastgele keşifle geçiyor.

**Yani EXP-009/010 bir sıralama değil, "6000 turda kim daha hızlı öğreniyor" ölçümü.**
Özellikle "ödül darboğaz değil" çıkarımı erken olabilir.

Önlem: `python -m bbrl.plots --convergence <önek>` eklendi; artık hiçbir sonuç
plato kontrolünden geçmeden rapora girmeyecek. EXP-013 (15000 tur, eps decay %27)
gerçek platoyu ölçecek.

### B8. Rapor grafikleri hazır

`bbrl/plots.py` — eğitim eğrileri (çok seed'li band) + ablation barları (%95 GA,
EXP-000 referans çizgileriyle) + yakınsama raporu. `runs/figs/` altına PNG.

### B9. 🏆 18 Ağustos — `rule_based_agent` GEÇİLDİ (EXP-017)

**Turnuva vekilinde (EK-2) referansın %120'si.** Güven aralıkları örtüşmüyor.

| metrik | bizim `e17b` | `rule_based` |
|---|---|---|
| **skor** | **3.979 ± 0.377** | 3.317 ± 0.247 |
| coin | 2.424 | 2.247 |
| kill | **0.311** | 0.214 |
| intihar | **0.199** | 0.528 |
| adım | **338.9** | 228.4 |

**Bunu getiren tek şey: ödül tablosunda üç sayı.** `CRATE_DESTROYED` 1.5→0.15,
`COIN_FOUND` 0.5→2.0, `COIN_COLLECTED` 1.0→2.0 → **+1.865 [+1.497, +2.234]**,
projenin en büyük tek kazancı. Model/feature/mimari/bütçe aynı kaldı.

> **Ders:** ödülün BÜYÜKLÜĞÜ değil YÖNÜ belirleyici. EXP-010'da sandık ödülünü
> 10× artırmak hiçbir şey değiştirmemişti; 10× azaltmak skoru 1.9 katına çıkardı.

Sebep EXP-016'da ölçülmüştü: shaped ödülün **%96'sı sandıktan** geliyordu ve
sandığın gerçek skora katkısı **sıfır** — bir sandık çiftliği botu eğitmişiz.

**Turnuva adayı hazır:** `e17b_objective` seed 2 (EK-2'de 4.203), `vendor.py`
ile 287 KB paket, bağımsızlık doğrulaması geçti, model repoda.

### B10. Tam kart profili — ve turnuvada asıl önemli olan

Gönderilen aday `e19b_types_s2`, 10 eval seed'i, eps=0:

| kart | Task | bizim | `rule_based` | oran |
|---|---|---|---|---|
| **EK-2** (3 rakip) | 4 | **4.210 ± 0.186** | 3.317 | **%127** ✅ |
| EK-1 (1 rakip) | 4 | **4.974 ± 0.135** | 4.67 | %106 ✅ |
| EK-4 (av) | 3 | **7.518 ± 0.348** | 8.60 | %87 |
| **EK-5** (rakipsiz) | 2 | **5.227 ± 0.277** | 8.53 | **%61** |
| **EK-3** (coin-heaven) | 1 | **24.078 ± 1.395** | 50.00 | **%48** |

> **8 Eylül'de düzeltildi.** Bu satırlar daha önce EK-5 için %37, EK-3 için %10
> diyordu. O rakamlar **eski aday `e17b`'ye** aitti; gönderilen `e19b_types_s2`
> bu kartlarda hiç ölçülmemişti. Gerçek profil yukarıda.
>
> Asıl bulgu skorlarda değil: **EK-5'te ajan hiç ölmüyor** (intihar 0.000,
> hayatta kalma 400.0/400) ve 74.3 sandık kırıyor. Yani "yalnız kalınca çöküyor"
> teşhisi yanlıştı — açık **hayatta kalmada değil, coin toplama verimliliğinde**.
>
> Ve her kartta `invalid_action_rate` yüksek: EK-1 0.121, EK-5 0.124,
> **EK-3 0.297**. Coin-heaven'da 400 adımın ~120'si duvara çarpmakla geçiyor.

Bu yalnızca "Task 1–2'de kötüyüz" değil. **`rule_based` turların %53'ünde
intihar ediyor** → EK-2 turlarının yarısında rakipler erken ölüyor ve ajanımız
**sandık dolu tahtada yalnız kalıyor** — tam olarak EK-5 durumu.

**Eski açıklama yanlıştı ve 8 Eylül'de çürütüldü.** Burada şöyle yazıyordu:
*"rakip feature'ları rakipsizken sabit kalıyor, model hiç görmediği bölgeye
düşüyor"*. Ölçüm bunu desteklemiyor — EK-5'te ajan **hiç ölmüyor** (intihar
0.000, hayatta kalma 400.0/400) ve 74.3 sandık kırıyor. Dağılım dışına düşen
bir model böyle davranmaz; gayet yetkin oynuyor.

Gerçek açık **verimlilik**: 400 adımda 74 sandık kırıp yalnızca 5.2 coin
topluyor, `rule_based` ise 8.5 topluyor. Bunun bir kısmı ölçülmüş bir israf —
`invalid_action_rate` 0.124, yani ~50 adım duvara gidiyor.

İki denenmiş ve **başarısız** yol (tekrar denenmemeli):
- **EXP-018R** — rakip SAYISINI karıştırmak (3 / 0 / 1 blokları): **−0.490**,
  felaket unutma. Bu paragrafın eskiden işaret ettiği çözümdü; zarar verdi.
- Sandık ödülünü artırmak: EXP-010/016/017, sandık çiftçisi ajan üretiyor.

Açık duruyor. Bu kampanyada **bilerek** bütçe ayrılmadı: EK-5 turnuva vekili
değil ve iki bariz çözümü de ölçülmüş biçimde geri tepiyor.

---

### B11. 📦 31 Ağustos — gönderilen config `e19b_types_s2` oldu

`agent_code/irmak_umut/config.yaml` artık **`e17b_objective_s2` değil**.

| | eğitim rakipleri | EK-2 (seed 0–9) | EK-2 (seed 10–19, taze) |
|---|---|---|---|
| `e17b_objective_s2` | 3× `rule_based` | 4.203 | — |
| **`e19b_types_s2`** | `rule_based` + `coin_collector` + `Qlearned_agent` | **4.416** | **4.210** |

**Neden değiştirdik:** `e17b` tek bir rakip tarzına karşı eğitildi. Turnuvada
karşımızda `rule_based` değil **başka takımların ajanları** olacak ve bu risk
somut olarak ölçüldü — Umut'un `Qlearned_agent`'ına karşı kafa kafaya
karşılaşmada bizden **daha çok** öldürüyordu (1v1'de 0.12 vs 0.07), oysa
`rule_based`'e karşı biz iki katı öldürüyoruz. Genel bir avcılık üstünlüğü
değil, **o ajana özel bir açığımız** vardı. EXP-019 üç tipe aynı anda maruz
kalmanın **+0.254** getirdiğini ölçtü.

**Seçim yanlılığı düzeltildi:** seed 2, EK-2 skoruna *bakılarak* seçildi;
bu yüzden hiç kullanılmamış eval seed'lerinde (10–19) yeniden ölçüldü →
**4.210 ± 0.186**. Seçim primi 0.206, referansın (3.317 ± 0.247) üstünde
kalmaya devam ediyor.

| metrik | `e19b_types_s2` (taze seed) | `rule_based` | `Qlearned_agent` |
|---|---|---|---|
| **skor** | **4.210 ± 0.186** | 3.317 ± 0.247 | 2.953 ± 0.207 |
| coin | **2.450** | 2.247 | 2.103 |
| kill | **0.352** | 0.214 | 0.170 |
| intihar | **0.284** | 0.528 | 0.399 |

`model_e19b_types_s2.bin` repoya **zorla eklendi** (`.gitignore` model
dosyalarını atıyor). `e17b`'nin modeli de fallback olarak duruyor —
11 Eylül karar noktasında geri dönmek gerekirse diye.

> **Not:** `e19c_league` (self-play havuzu dahil) ortalamada daha yüksek
> (4.376) ama güven aralığı **±0.677** — dört seed'i 3.986 ile 4.881 arasında
> saçılıyor. `e19b` hem yüksek hem kararlı (±0.155), o yüzden o gönderiliyor.
> `e19c`'yi daha çok seed'le tekrar etmek bekleyen işler arasında.


## C. Şimdi ne yapacağız — öncelikli task listesi

### 🔥 Bu hafta (Sprint 1 kalanı)

| # | Task | Kim | Bitti sayılır: | Bağımlılık |
|---|---|---|---|---|
| ~~C1~~ | ~~`r02_living_cost` — donma düzeltmesi~~ | Irmak | ✅ **BİTTİ** — EK-3'te 50 coin / 125.8 adım | — |
| ~~C2~~ | ~~A/B ablation, Welch farkı~~ | Irmak | ✅ **BİTTİ** — EXP-007 | — |
| ~~C14~~ | ~~Config başına çoklu eğitim seed'i~~ | Irmak | ✅ **BİTTİ** — hiyerarşik CI, ilk koşuda kendini kanıtladı | — |
| **C15** | **Task 2: `classic` müfredatı** — sandık + bomba kaçışı | Irmak | EK-5'te intihar < 0.2, sandık > 50, coin > 4 | C14 |
| **C3** | **`v1_handcrafted` feature seti** (INTERFACE §1'deki 10 madde) | **Umut** | `test_features.py` yeşil + EK-3'te v0'ı geçiyor | — |
| **C4** | `curriculum.py` — Task 1→4 aşama tanımları | **Umut** | `classic` senaryosunda eğitim koşuyor | — |
| ~~C5~~ | ~~`plots.py` — §6 grafikleri~~ | Irmak | ✅ **BİTTİ** — eğriler + ablation + yakınsama kontrolü | — |
| **C6** | DQN config (`m05`, `v2_planes`) + ilk GPU eğitim koşusu | Irmak | `coin-heaven`'da öğreniyor | C1 |
| **C7** | `fastenv.py` + `test_fastenv_parity.py` | Irmak | 10k episode bit-bit eşleşiyor | — |
| ~~C8~~ | ~~`vendor.py` — turnuva klasörü bağımsız çalışsın~~ | Irmak | ✅ **BİTTİ** — `dist/` + 21 KB zip, coin-heaven'da 50 coin/tur doğrulandı | — |

### 📅 Sprint 2 (24–30 Ağustos)

| # | Task | Kim |
|---|---|---|
| C9 | **Rakip merdiveni** — Discord'dan diğer takımların ajanlarını indir, gece round-robin | Irmak |
| C10 | Task 2 (kaçış) odaklı eğitim — B2.2 gereği en yüksek getirili iş | Umut |
| C11 | EXP-003 (danger feature ablation) + EXP-004 (çift ölüm cezası) + EXP-005 | Umut |
| C12 | Simetri augmentation'ı DQN'de aç, örnek verimliliği ölç (EXP-006) | Irmak |
| C13 | **Rol takası** (TEAM_PLAN §1) | ikisi |

### ⏰ Sabit tarihler

| Tarih | Ne |
|---|---|
| **11 Eylül** | **Karar noktası:** DQN `rule_based`'i yenmiyorsa m01/m02 ile devam |
| **17 Eylül 21:00** | MaMPF submission test (Docker ön koşu) |
| **21 Eylül 21:00** | Ajan kodu teslimi — `git add -f agent_code/irmak_umut/model_*.bin` unutma |
| **28 Eylül 21:00** | Rapor + repo **PUBLIC** yapılacak |

### Umut'un hemen başlayabileceği (bağımlılığı yok)

`C3` → `C5` → `C4` sırası. Üçü de hazır primitiflerin üstüne oturuyor:
`bbrl/gamelogic.py` içindeki `danger_map`, `bfs_distances`, `escape_directions`,
`bomb_impact` test edilmiş ve kullanıma hazır. `INTERFACE.md` §1'de hangi feature'ın
hangi primitifle yazılacağı madde madde yazılı.
