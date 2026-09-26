# Takım Planı — Irmak & Umut

**Hedef:** Turnuvada 1. olmak **ve** A almak.
**Takım:** 2 kişi (Irmak, Umut) — "4 kişilik" tempoda çalışacağız.

| Kilometre taşı | Tarih |
|---|---|
| Sprint 1 (bu hafta) | 17–23 Ağustos |
| Submission test (MaMPF) | **17 Eylül, 21:00** |
| **Ajan kodu deadline** | **21 Eylül, 21:00** |
| **Rapor deadline** | **28 Eylül, 21:00** |

---

## 0. Neden "sen DL, ben Q-learning" yapmıyoruz

Proje tanımı, sayfa 1:

> *"do not split labor such that every team member works on their separate model —
> we attach great importance to real teamwork!"*

Bu, isim vererek yasaklanan tek iş bölümü şekli. Repo public olacağı için commit
geçmişinden bir bakışta görülür.

Pratik sebep daha da önemli: 1. haftada modele göre ayrılırsak hafta sonunda **iki farklı
feature imzası, iki farklı ödül formatı, iki farklı metrik tanımı** olur — ve raporun en ağır
bölümü olan §6 (Experiments) **yazılamaz hale gelir**, çünkü iki model hiç aynı cetvelden
geçmemiştir.

**Bunun yerine katmana göre bölüyoruz.** Her iki model de aynı feature arayüzünü, aynı
buffer'ı, aynı eval harness'ını kullanır. İkimiz de her iki modele dokunuruz.

**Ayırt edici test:** *Hafta sonunda ikimiz de her iki modeli savunabiliyor muyuz?*
Evet → gerçek takım çalışması. Hayır → yasak bölünme.

---

## 1. Rol dağılımı

| | **IRMAK** | **UMUT** |
|---|---|---|
| **Sprint 1–2** | **R1 Infra** + **R3 Learning**<br>`gamelogic`, `eval`+`metrics`, `buffer`, `fastenv`, eğitim döngüsü, model zoo, config, DQN | **R2 Representation** + **R4 Reward**<br>`v1_handcrafted` feature seti, ödül şemaları r02+, custom event'ler, `curriculum.py`, `plots.py` (§6 grafikleri), ablation deneyleri |
| **Sprint 3–4 (TAKAS)** | R2 + R4 | R1 + R3 |
| Donanım | RTX 5070 laptop + RunPod | CPU sweep'leri (RunPod CPU pod) |

**Takas 3. sprintte zorunlu.** Hem "her iki modeli de savunabilme" testini geçirir,
hem raporda farklı bölümlere farklı isim yazabilmemizi sağlar.

**Yatay iş — haftalık rotasyon:** "Bu haftanın deney sorumlusu" sırayla. O kişi hafta sonu
`EXPERIMENTS.md`'yi yazar ve konseyi yönetir. Sprint 1: **Umut**. Sprint 2: **Irmak**.

---

## 2. Sprint 1 — 17–23 Ağustos

> **Sprint hedefi:** Model yapmak DEĞİL. **Uçtan uca çalışan bir araştırma tezgahı** kurmak,
> üstüne iki öğrenen baseline oturtmak. Bu hafta doğru kurulursa kalan 4 hafta 3 kat hızlı geçer.

### Definition of Done (Pazar akşamı hepsi ✅ olmalı)

- [ ] `INTERFACE.md` donmuş, ikimiz de onaylamış
- [ ] `test_fastenv_parity.py` **yeşil** — 10k episode, orijinalle bit-bit aynı
- [ ] `bbrl/eval.py` tek komutla çalışıyor, çok-seed + %95 güven aralığı üretiyor
- [ ] Config-driven model zoo: `python -m bbrl.train configs/X.yaml` çalışıyor
- [ ] **Model 1** (tabular/linear Q) `coin-heaven`'da öğreniyor — Task 1 çözülmüş
- [ ] **Model 2** (DQN) pipeline'ı `coin-heaven`'da öğreniyor (yakınsama şart değil; skor artıyor)
- [ ] RunPod sweep launcher'ı çalıştı, en az 1 gecelik sweep koşuldu
- [ ] `EXPERIMENTS.md`'de ≥ 6 kayıtlı deney
- [ ] `p99_act_time_ms` ölçüldü ve < 50 ms (0.5 s bütçesinin %10'u)

### Günlük plan

#### 🟦 Pazartesi 17 — BİRLİKTE (3–4 saat, ayrılmadan önce)
Bu oturum atlanamaz. Ayrılmadan önce:
1. `docs/GAME_MECHANICS.md`'yi **beraber okuyun.** Özellikle §2 (bomba zamanlaması) ve
   §6 (çifte olay + ölüm transition'ı). Buradaki 3 tuzak, bu projede en çok zaman yakan şeyler.
2. `INTERFACE.md`'yi dondurun. İmzalar üzerinde anlaşın, dosyanın durumunu 🟢 yapın, commit'leyin.
3. Repo hijyeni: `ml_homework/` ve `runs/` → `.gitignore`. Public repo'yu açın (fork).
4. `agent_code/irmak_umut/` klasörünü `tpl_agent`'tan kopyalayın.
5. Karar: ortak `bbrl/` kodunun turnuva klasörüne nasıl **vendor**'lanacağı (INTERFACE §0 uyarısı).

Sonra ayrılın.

| | Irmak | Umut |
|---|---|---|
| **Pzt (kalan)** | `fastenv.py` iskeleti — batch'li arena, bomba/patlama tensörleri | `features.py` — BFS altyapısı + danger map (`bombs`'tan!) |
| **Salı 18** | `fastenv` tamam + `test_fastenv_parity.py` v1 | `v1_handcrafted` feature seti (INTERFACE §1'deki 10 madde) |
| **Çarş 19** | `buffer.py` + `train_loop.py` + `models/REGISTRY` + `config.py` | `rewards.py` şemaları r01–r04 + custom event'ler + `metrics.py` |
| **🟨 Perş 20** | **ENTEGRASYON GÜNÜ — yarım gün pair.** Tabular Q'yu beraber yazın, `coin-heaven`'da öğrendiğini görün. Bu haftanın en önemli commit'i (`Co-authored-by:`). Öğleden sonra: Irmak → RunPod sweep launcher; Umut → `eval.py` + değerlendirme kartı |
| **Cuma 21** | `models/dqn.py` (double + dueling) + **CPU inference bütçe testi** (`torch.set_num_threads(1)`) | Task 2 müfredatı (`classic`, sandıklı) + kaçış-odaklı ödül şeması + feature ablation scaffold |
| **Cmt 22** | Gece sweep'ini başlat (≥ 20 config × 3 seed) | İlk gerçek ablation koşusu: feature setleri × ödül şemaları |
| **🟩 Paz 23** | **DENEY KONSEYİ #1** (birlikte, 1 saat): sonuçları oku → `EXPERIMENTS.md`'ye yaz → Sprint 2 hipotezlerini belirle → rol takası kararı |

---

## 3. "4'ten fazla model" stratejisi

Proje **en az 2** model istiyor. Daha fazlası serbest — ama dikkat:

> **Not, model sayısıyla değil, karşılaştırmanın titizliğiyle geliyor.**
> Özensiz karşılaştırılan 6 model, titiz ablation'lı 2 modelden **daha kötü** not alır.

Çözüm: **model zoo'yu config-driven yapın.** O zaman "model" = YAML dosyası, ve ablation
matrisi kendiliğinden oluşur:

| ID | Model | Features | Rewards | Not |
|---|---|---|---|---|
| `m01` | tabular Q | v1 (küçültülmüş) | r01 baseline | dersten teknik ✅ |
| `m02` | linear Q (SARSA/Q) | v1 | r02 escape-heavy | |
| `m03` | gradient-boosted Q (sklearn) | v1 | r02 | fitted Q-iteration |
| `m04` | DQN (MLP) | v1 | r02 | |
| `m05` | DQN (CNN, dueling+double) | v2 planes | r03 | 🎯 turnuva adayı |
| `m06` | m05 + prioritized replay + n-step | v2 | r03 | |
| `m07` | m06 + self-play ligi | v2 | r04 | 🎯 turnuva adayı |

`m01` **daima çalışan güvenli tabanınız.** Laptop'ta eğitilir, CPU'da uçar, "en az bir model
dersten teknik" şartını karşılar. DQN yakınsamazsa turnuvaya bu gider.

**Karar noktası: 11 Eylül.** O tarihte DQN `rule_based_agent`'ı yenmiyorsa `m01/m02`'yi
cilalayıp gönderiyoruz. Bu tarihi takvime yazın.

---

## 4. Hesaplama stratejisi — **karar: şimdilik kiralama YOK**

**Darboğaz GPU değil, environment throughput'u.** Orijinal env saf Python, tek thread —
saniyede birkaç yüz adım. DQN 10⁶–10⁸ adım ister.

Irmak'ın laptopu (ölçüldü, 2026-08-16):

| | |
|---|---|
| CPU | **Intel i7-14650HX — 16 çekirdek / 24 thread** |
| RAM | **31.6 GB** |
| GPU | RTX 5070 (Blackwell, sm_120) |

24 thread, kiralık bir CPU pod'un bu projede yapacağı işin çoğunu yapar. Vektörize `fastenv`
gelince paralel rollout + sweep tek makinede koşar; 31 GB RAM büyük bir replay buffer'a bol.
GPU zaten darboğaz değil — buradaki ağlar küçük.

### Kiralama tetikleyicisi (tahmin değil, ölçüm)

`EXP-001`'de fastenv throughput'u ölçülecek. **RunPod'a ancak şu ikisinden biri olursa çıkılır:**
1. Tam bir ablation sweep'i laptopta **> 10 saat** sürüyor, veya
2. Self-play ligi **> 24 eşzamanlı** maç istiyor

O güne kadar para harcanmaz. Karar `EXPERIMENTS.md`'ye yazılır.

**Notlar:**
- Eğitimde multiprocessing **serbest** (PDF s.2) — sweep'leri 24 thread'e yayın.
  ⚠️ **Final ajanda multiprocessing YASAK** — sadece eğitimde.
- ⚠️ **RTX 5070 = Blackwell (sm_120).** PyTorch'un **cu128** wheel'i gerekiyor, eski CUDA
  build'leri çalışmaz. İlk iş bunu doğrulayın:
  `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
- ⚠️ Turnuva **CPU'da, tek thread, 0.5 s/adım.** `callbacks.py`'de `torch.set_num_threads(1)`
  koyun ve `setup()`'ta bir **warm-up inference** yapın (ilk torch çağrısı yavaştır,
  ilk adımda timeout yersiniz ve ceza kümülatiftir).
- Checkpoint atın — spot instance kesilir.

---

## 5. Risk kaydı

| Risk | Olasılık | Etki | Önlem |
|---|---|---|---|
| fastenv orijinalden sapıyor | orta | **ölümcül** | Parity testi yeşil olmadan eğitim yok |
| DQN deadline'a yakınsamıyor | **yüksek** (PDF uyarıyor) | yüksek | `m01/m02` güvenli taban; 11 Eylül karar noktası |
| Yardımcı ödüllere overfit | yüksek | yüksek | Değerlendirme **daima** gerçek skorla, shaped ödülle değil |
| 0.5 s timeout aşımı | düşük | **ölümcül** | `p99_act_time_ms` her eval'de ölçülür; `set_num_threads(1)` |
| Mutlak yol hatası (Docker) | orta | **ölümcül** | Sadece göreli yol; 17 Eylül submission test'i |
| İki haftalık iş 1 haftaya sığmıyor | orta | orta | Perşembe entegrasyon günü kontrol noktası; sığmıyorsa `m05+`'i Sprint 2'ye at, tezgahı asla kısma |
| §6 için yeterli veri yok | orta | **not için ölümcül** | Deney logger'ı 1. günden itibaren otomatik |

---

## 6. Rapor sorumluluk haritası

Rapor bölüm bölüm **bireysel** notlandırılıyor ("mark each chapter with its main author").
Ama kimse "benim modelim" başlıklı bölüm yazmıyor — **katmana ve deneye** göre bölüyoruz.

| Bölüm | Ana yazar |
|---|---|
| §1 Introduction | Umut |
| §2 Background (RL yaklaşımları) | Irmak |
| §3 Project planning (organizasyon + donanım/RunPod) | Irmak |
| §4.1 Problem formülasyonu (MDP, state/action/reward) | Irmak |
| §4.2 Feature design | **Umut** |
| §4.3 Learning algorithms | **Irmak** |
| §4.4 Evaluation methodology + metrikler | **Umut** |
| §5.1 Training infrastructure (fastenv, sweep) | Irmak |
| §5.2 Reward shaping & curriculum | **Umut** |
| §6.1 Ablation: feature setleri | Umut |
| §6.2 Ablation: ödül şemaları | Umut |
| §6.3 Ablation: algoritma varyantları | Irmak |
| §6.4 Head-to-head + turnuva merdiveni | rotasyon sorumlusu |
| §7 Conclusion & outlook | ortak |

**Hacim:** ~4000 kelime/kişi → toplam ~8000 kelime.
**Kurallar:** üniversite logosu **yok** · rapor kod reposuna **yüklenmez** ·
her başlığın altına yazar adı · public repo URL'i raporda geçmeli.

---

## 7. Haftalık ritim (Sprint 1'den sonra da devam)

- **Pazartesi 30 dk:** sprint planlama, hipotezleri yaz
- **Her gün ~10 dk:** async standup (Discord/WhatsApp) — dün / bugün / engel
- **Perşembe:** entegrasyon günü, pair
- **Pazar 1 saat:** **Deney Konseyi** — sonuçlar → `EXPERIMENTS.md` → sonraki hipotezler

`EXPERIMENTS.md` raporun §6'sının iskeletidir ve "sistematik/bilimsel yaklaşım" kanıtımızdır.
Sonradan uydurulamaz. **Her koşu, ilk günden itibaren kaydedilir.**

---

## 8. Hemen yapılacaklar (Pazartesi oturumundan önce)

**Irmak:**
- [x] Repo açıldı: **https://github.com/irmakerkol/mle-final-bomberman** (PRIVATE)
- [ ] Umut'u collaborator olarak davet et
- [ ] PyTorch cu128 kurulumunu doğrula (Blackwell / sm_120)
- [ ] Takımı `tinyurl.com/fml-final-project-teams`'e kaydet (**havalı takım adı!**)
- [ ] MaMPF: Anzeigename + Name in Uebungsgruppen = gerçek ad (muesli ile aynı)

**⚠️ 28 Eylül'den önce repo PUBLIC yapılacak** — rapor şartı. Takvime yaz.

### Git kurulumu

`origin` = bizim private repo · `upstream` = ukoethe/bomberman_rl (framework güncellemeleri)

```bash
git clone https://github.com/irmakerkol/mle-final-bomberman.git
cd mle-final-bomberman
git remote add upstream https://github.com/ukoethe/bomberman_rl
```

Framework güncellemesi geldiğinde (PDF Dockerfile için `git pull` diyor):
```bash
git fetch upstream && git merge upstream/master
```

Branch akışı: `main`/`master` korumalı, herkes `irmak/...` veya `umut/...` branch'inde çalışır,
her PR karşı taraf tarafından review'lanır.

**Umut:**
- [ ] `docs/GAME_MECHANICS.md`'yi oku (özellikle §2, §6)
- [ ] `agent_code/rule_based_agent/callbacks.py`'yi satır satır oku — en iyi referans
- [ ] `python main.py play --agents user_agent rule_based_agent --turn-based` ile 2–3 tur oyna
- [ ] MaMPF: ad ayarları + takım davet koduyla ortak teslime katıl
