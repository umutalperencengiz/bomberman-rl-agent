# INTERFACE — Donmuş Sözleşme

**Durum:** 🔴 TASLAK — Pazartesi ortak oturumda dondurulacak
**Donduktan sonra kural:** Bu dosyadaki imzalar **ikimizin onayı olmadan değişmez.**
Değişiklik gerekiyorsa: PR aç, karşı taraf onaylasın, `EXPERIMENTS.md`'ye "interface change" notu düş.

**Neden:** Bu sözleşme donarsa Irmak ve Umut haftanın geri kalanında birbirini **hiç beklemez**.
Asıl hız buradan gelir — modeli bölmekten değil.

---

## 0. Dizin yapısı

```
bomberman_rl/
├─ agent_code/
│  ├─ irmak_umut/            # TURNUVAYA GİDECEK TEK KLASÖR
│  │  ├─ callbacks.py         # setup(self), act(self, game_state)
│  │  ├─ train.py             # setup_training, game_events_occurred, end_of_round
│  │  ├─ model.pt             # eğitilmiş parametreler (GÖRELİ YOL ile yüklenir!)
│  │  ├─ avatar.png           # 30x30, ödül töreni için
│  │  └─ bomb.png             # 30x30
│  └─ ...
├─ bbrl/                      # ORTAK KÜTÜPHANE — asıl işin yapıldığı yer
│  ├─ gamelogic.py         ✅ # oyun kurallarının saf-fonksiyon hâli        [IRMAK]
│  ├─ features.py          ◐  # registry + v0_minimal + v2_planes           [IRMAK]
│  │                          #   └─ v1_handcrafted (10 feature)            [UMUT]
│  ├─ rewards.py           ◐  # registry + custom event motoru + r00/r01    [IRMAK]
│  │                          #   └─ şemalar r02+ ve ablation'ları          [UMUT]
│  ├─ metrics.py           ✅ # metrikler + güven aralıkları                [IRMAK]
│  ├─ eval.py              ✅ # evaluation harness                          [IRMAK]
│  ├─ buffer.py            ✅ # replay buffer (uniform + prioritized)        [IRMAK]
│  ├─ fastenv.py              # vektörize environment                       [IRMAK]
│  ├─ models/                 # model zoo (registry)                        [IRMAK]
│  ├─ train_loop.py           # ortak eğitim döngüsü                        [IRMAK]
│  ├─ config.py               # Config dataclass + YAML yükleme             [IRMAK]
│  ├─ curriculum.py           # Task 1→4 müfredat tanımları                 [UMUT]
│  └─ plots.py                # §6 için grafik üretimi                      [UMUT]
├─ configs/                   # her model = bir YAML dosyası
├─ runs/                      # deney çıktıları (gitignore)
├─ tests/
│  ├─ test_gamelogic.py    ✅ # danger_map / blast / free_mask parity       [IRMAK]
│  ├─ test_features.py     ✅ # sözleşme + simetri eşlemesi + buffer        [IRMAK]
│  └─ test_fastenv_parity.py  # diferansiyel test                           [IRMAK]
└─ docs/
```

✅ = yazıldı ve testleri geçiyor · ◐ = iskelet hazır, içerik TODO

**Sahiplik notu:** Irmak `gamelogic`/`eval`/`metrics`/`buffer`'ı yazdığı için bunlar
Irmak'a geçti. Denge için Umut'a `curriculum.py` ve `plots.py` eklendi — `plots.py`
raporun §6'sının tüm grafiklerini üretir, yani not ağırlığı yüksek bir parça.

> ⚠️ `agent_code/irmak_umut/` içindeki kod `bbrl`'i **import edemez** — turnuvada sadece o klasör
> kopyalanır. Çözüm: `callbacks.py` gerekli feature/model kodunu **kendi içinde** taşır
> (build script'i `bbrl`'den kopyalar) veya `bbrl`'in ilgili dosyaları o klasöre vendor'lanır.
> **Bunu Perşembe entegrasyon gününde çöz, sona bırakma.**

---

## 1. Feature arayüzü  `bbrl/features.py`  — sahibi: UMUT

```python
import numpy as np

FEATURE_SETS: dict[str, "FeatureSet"] = {}   # registry: isim -> FeatureSet

class FeatureSet:
    name: str
    shape: tuple[int, ...]      # örn. (24,) veya (7, 17, 17)
    dtype: np.dtype             # np.float32

    def __call__(self, game_state: dict | None) -> np.ndarray | None:
        """game_state None ise None döner (tur başı/sonu)."""
```

**Kurallar:**
- Çıktı **her zaman** `shape` ve `dtype`'a uyar. `assert` ile kontrol edilir.
- `game_state is None` → `None` döner.
- **Yan etkisiz** (state'i mutate etmez, global tutmaz) → paralel rollout'ta güvenli.
- ⚠️ **Yasak:** deterministik olarak "en iyi aksiyonu" döndüren feature (proje kuralı, PDF s.9).
  Yön/mesafe/tehlike bilgisi ver, kararı verme.

**v1'de olması gerekenler** (`FEATURE_SETS["v1_handcrafted"]`):
| # | Feature | Not |
|---|---|---|
| 1 | 4 yön geçilebilir mi | `tile_is_free` mantığı |
| 2 | En yakın coin yönü (BFS, one-hot 4+1) | duvarları dolaşan gerçek mesafe |
| 3 | En yakın sandık yönü (BFS) | |
| 4 | En yakın rakip yönü + mesafe | |
| 5 | **Danger map** — her komşu kare kaç adım sonra ölümcül | `bombs`'tan hesapla, `explosion_map`'ten DEĞİL (bkz. GAME_MECHANICS §2) |
| 6 | Şu an tehlikede miyim + kaç adım kaldı | |
| 7 | Güvenli kaçış yönü var mı (BFS, tehlike zamanına karşı) | hayat kurtarıcı #1 |
| 8 | Bomba atabilir miyim (`bombs_left`) | |
| 9 | Buraya bomba atarsam kaçabilir miyim | intiharı doğrudan engeller |
| 10 | Bomba atarsam kaç sandık / rakip vurur | |

**v2 (CNN için)** `FEATURE_SETS["v2_planes"]`: `(C, 17, 17)` kanallar —
duvar / sandık / coin / kendi / rakipler / bomba-timer / danger-time.

---

## 2. Ödül arayüzü  `bbrl/rewards.py`  — sahibi: UMUT

```python
REWARD_SCHEMES: dict[str, dict[str, float]] = {}   # isim -> {event: reward}

def custom_events(old_state: dict, action: str, new_state: dict,
                  events: list[str]) -> list[str]:
    """Yalnızca ek event ADLARI döndürür. `events`'i MUTATE ETMEZ."""

def reward_from_events(events: list[str], scheme: str) -> float:
    """Toplam ödül."""
```

**Kurallar:**
- Ödül şeması **veri**, kod değil → YAML'den seçilebilir → ablation ücretsiz.
- ⚠️ `KILLED_SELF` **ve** `GOT_KILLED` intiharda birlikte gelir (GAME_MECHANICS §6).
  Şemada bunu bilerek ele al.
- Potential-based tercih: yardımcı ödül mümkün olduğunca **state**'e bağlı olsun, aksiyona değil.
- ⚠️ Yardımcı ödüller turnuvada **yok** → onlara bağımlı politika turnuvada çöker.
  Değerlendirme **her zaman** gerçek skorla yapılır, shaped ödülle değil.

---

## 3. Transition ve buffer  `bbrl/buffer.py`  — sahibi: IRMAK

```python
Transition = namedtuple("Transition", ["state", "action", "next_state", "reward", "done"])
# state, next_state : np.ndarray | None   (None = terminal)
# action            : int  (ACTIONS index)
# reward            : float
# done              : bool

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]   # sıra DEĞİŞMEZ
```

`ACTIONS` sırası dondurulmuştur — kaydedilmiş modeller bu indekslemeye bağlı.

---

## 4. Model arayüzü  `bbrl/models/`  — sahibi: IRMAK

```python
REGISTRY: dict[str, type["Model"]] = {}    # "dqn" -> DQN sınıfı

class Model(Protocol):
    def __init__(self, feature_shape, n_actions, cfg): ...

    def act(self, features, eps: float) -> int:
        """Aksiyon indeksi. eps=0 -> greedy (değerlendirme)."""

    def update(self, batch: list[Transition]) -> dict[str, float]:
        """Bir gradyan/güncelleme adımı. Loglanacak skaler metrikleri döner."""

    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

`update` **her zaman** bir `dict` döner (boş olabilir) → logger tek kod yolunda çalışır.

---

## 5. Config  `bbrl/config.py`  — sahibi: IRMAK

Bir model = bir YAML. **Model sayısı kod değil, config sayısıdır.**

```yaml
# configs/m03_dqn_dueling.yaml
name: m03_dqn_dueling
model: dqn                    # REGISTRY anahtarı
features: v2_planes           # FEATURE_SETS anahtarı
rewards: r02_escape_heavy     # REWARD_SCHEMES anahtarı
curriculum: [coin-heaven, classic]
hparams:
  lr: 3.0e-4
  gamma: 0.95
  batch_size: 256
  buffer_size: 200000
  target_sync: 1000
  dueling: true
  double: true
  eps_start: 1.0
  eps_end: 0.05
  eps_decay_steps: 300000
seeds: [0, 1, 2, 3, 4]
n_rounds: 20000
```

---

## 6. Evaluation harness  `bbrl/eval.py`  — sahibi: UMUT

```python
def evaluate(agent_dir: str, opponents: list[str], scenario: str,
             n_rounds: int, seeds: list[int]) -> EvalResult:
    """self.train=False ile ORİJİNAL env'de koşar. Tek doğru cetvel budur."""
```

**Zorunlu metrikler** (hepsi ortalama ± %95 GA, çoklu seed):

| Metrik | Neden |
|---|---|
| `score` | Turnuvanın tanımı |
| `win_rate` | Rakibe karşı üstünlük |
| `coins` | Task 1–2 ilerlemesi |
| `kills` | Task 3–4 ilerlemesi |
| `suicide_rate` | 🔴 en teşhis edici metrik |
| `survival_steps` | Kaçış yeteneği |
| `invalid_action_rate` | Politika sağlığı |
| `mean_act_time_ms` / `p99_act_time_ms` | 🔴 0.5 s bütçesi |

**Standart değerlendirme kartı** (her model için, her zaman aynı):
1. `classic`, 1× rule_based_agent, 200 tur × 5 seed
2. `classic`, 3× rule_based_agent, 200 tur × 5 seed  ← **turnuva vekili**
3. `coin-heaven`, tek başına, 100 tur × 5 seed        ← Task 1 regresyon testi
4. `classic`, peaceful + coin_collector, 200 tur × 5 seed

> Tek seed'lik sonuç RL'de **gürültüdür.** Raporda tek seed grafiği göstermeyin.

---

## 7. Fast env  `bbrl/fastenv.py`  — sahibi: IRMAK

Orijinal env saf Python + tek thread. Batch'li numpy yeniden yazımı eğitim throughput'unu
100×+ artırır. Proje kuralı buna **açıkça izin veriyor** (PDF s.2).

**Kabul kriteri — pazarlıksız:**
```
tests/test_fastenv_parity.py
  10.000 rastgele episode, rastgele aksiyonlar, sabit seed'ler
  her adımda karşılaştır: field, bombs, explosion_map, coins, self, others, events, score
  TEK BİR fark = FAIL
```
> Bu test yeşil olmadan **hiçbir eğitim koşusu başlamaz.** Sapmış bir env, haftalarca
> yanlış oyunu öğrenmek demektir ve turnuvada fark edilir.

---

## 8. Git akışı

- `main` korumalı. Herkes feature branch'te çalışır: `irmak/fastenv`, `umut/features-v1`
- Her PR karşı taraf tarafından review'lanır (2 kişide 5 dk sürer, **kural**)
- Pair oturumlarında `Co-authored-by:` ekleyin → GitHub takım çalışmasını gösterir
- ⚠️ `.gitignore`'a ekle: `ml_homework/`, `runs/`, `*.pt` (final model hariç), `wandb/`
