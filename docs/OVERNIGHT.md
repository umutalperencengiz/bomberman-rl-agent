# Gece Programı — 17 Ağustos 2026, ~01:50

> Irmak uyurken koşacak deney zinciri. Sabah okumak için.
> Sonuçlar [EXPERIMENTS.md](EXPERIMENTS.md)'ye, özet [STATUS.md](STATUS.md)'ye işlenecek.

## Nasıl çalışıyor (dürüst açıklama)

Ben sürekli değil, **olay-tetiklemeli** çalışıyorum: arka plandaki bir iş bitince
uyanıp sonucu analiz ediyor, yazıyor ve sıradakini başlatıyorum. Yani zincir
"iş bitti → uyan → analiz et → yaz → yeni iş başlat" şeklinde ilerliyor.

**Riski azaltmak için:** sıradaki deneylerin configleri **şimdiden yazıldı**.
Bir halka koparsa sabah tek komutla kaldığı yerden devam edilebilir.

## Zincir

| # | Deney | Soru | Durum |
|---|---|---|---|
| **EXP-010** | ödül ablation (4 varyant × 3 seed) | Oran düzeltmesi + potansiyel shaping pasifizmi ve salınımı kırar mı? | 🔵 koşuyor |
| **EXP-011** | **model karşılaştırması** (tablo / lineer / DQN-MLP) | Fonksiyon yaklaşımı gerçekten fark yaratıyor mu? | ⚪ config hazır |
| **EXP-012** | hiperparametre (lr × γ, 6 config) | Kazanan modelin ayarı | ⚪ config hazır |
| **EXP-013** | feature ablation (v0 / v0b / v0c) | Feature mi model mi daha çok kazandırıyor? | ⚪ planlandı |
| **EXP-014** | DQN + `v2_planes` (GPU) | DL rotası — CNN kendi feature'ını öğrenir mi? | ⚪ planlandı |

**EXP-011 neden sırada:** EXP-008 (tablo sıfır genelleme yapar) ve EXP-009
(müfredat sıfır transfer — ayrık state uzayları) **iki bağımsız kanıtla** fonksiyon
yaklaşımını işaret etti. Ama numpy/BLAS çöktüğü için lineer modeli hiç koşamadık.
Ortam düzeldi; bu deney projenin yönünü belirleyecek.

## Sabit kurallar (gece boyunca uyulacak)

1. **Birincil metrik:** `coins_per_round`, kart EK-5, gerçek oyun skoru.
   `crates` / `suicide` yalnızca teşhis. İntihar oranını **doğrudan optimize etme** —
   pasifist ajanın intiharı zaten düşük (EXP-009).
2. **Config başına 3 eğitim seed'i.** CI eğitim seed'leri üzerinden (C14).
3. Her deneyden önce **tahmin yaz**, sonra ölç. Negatif sonuç da rapor malzemesi.
4. Her deney sonrası: `EXPERIMENTS.md` + commit + push.
5. Ödül şeması değiştiğinde **önce aritmetiği kontrol et**
   (`test_reward_ratio_sanity`) — EXP-010'da bu, 12 koşuyu bozuk ödülden kurtardı.

## Sabah kontrol listesi

```powershell
git -C "C:\Users\Irmak\Desktop\mle-final\bomberman_rl" log --oneline -15
```

```powershell
Get-Content "C:\Users\Irmak\Desktop\mle-final\bomberman_rl\docs\STATUS.md" | Select-Object -First 60
```

Zincir koptuysa kaldığı yerden devam:

```powershell
Set-Location "C:\Users\Irmak\Desktop\mle-final\bomberman_rl"; & "C:\Users\Irmak\miniconda3\envs\ml_homework\python.exe" -u -m bbrl.train_loop --sweep "configs/e11*.yaml" --card EK-5 --train-seeds 3 --seeds 10 --workers 9
```
