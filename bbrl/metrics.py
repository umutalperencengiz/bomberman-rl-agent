"""Metrikler ve guven araliklari.

Sahibi: IRMAK  (plan basta UMUT diyordu; Irmak yazdigi icin guncellendi)

Tasarim kurali: performans HER ZAMAN gercek oyun skoruyla raporlanir,
shaped odulle asla. Shaped odul turnuvada yok (bkz. docs/GAME_MECHANICS.md).

Veri kaynagi: `python main.py play ... --save-stats <path>` ciktisi.
`environment.py -> GenericWorld.end()`:

    results = {"by_agent": {name: lifetime_statistics}, "by_round": {...}}

`lifetime_statistics` anahtarlari (agents.py -> EVENT_STAT_MAP + note_stat):
    rounds, steps, score, time, moves, invalid, bombs, crates, coins,
    kills, suicides
`time` = toplam dusunme suresi (saniye, float)  -> act suresi buradan cikar,
log parse etmeye gerek yok.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# scipy varsa kullan, yoksa tabloya dus (PDF scipy'yi varsayiyor ama
# lokal venv'de olmayabilir -- harness bu yuzden scipy'siz de calisir).
try:  # pragma: no cover
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover
    _scipy_stats = None

# t dagiliminin %95 iki yonlu kritik degerleri, df = 1..30
_T95 = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
]


def t_critical(df: int, confidence: float = 0.95) -> float:
    """Iki yonlu t kritik degeri."""
    if df <= 0:
        return float("nan")
    if _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(0.5 + confidence / 2.0, df))
    if abs(confidence - 0.95) > 1e-9:
        raise ValueError("scipy yok: sadece confidence=0.95 destekleniyor")
    return _T95[df - 1] if df <= 30 else 1.96


@dataclass(frozen=True)
class MeanCI:
    """Ortalama + guven araligi. n = bagimsiz ornek (seed) sayisi."""

    mean: float
    lo: float
    hi: float
    sem: float
    n: int

    @property
    def halfwidth(self) -> float:
        return (self.hi - self.lo) / 2.0

    def __str__(self) -> str:
        if self.n < 2 or not math.isfinite(self.halfwidth):
            return f"{self.mean:.3f} (n={self.n})"
        return f"{self.mean:.3f} +/- {self.halfwidth:.3f}"

    def as_dict(self) -> dict:
        return {"mean": self.mean, "lo": self.lo, "hi": self.hi,
                "sem": self.sem, "n": self.n}


def mean_ci(values, confidence: float = 0.95) -> MeanCI:
    """Ortalama ve t-tabanli guven araligi.

    ONEMLI: `values` BAGIMSIZ orneklerdir -- bizde her eleman bir SEED'in
    (o seed'deki n_rounds turun) ortalamasidir. Tur-ici korelasyon bu
    sekilde disarida kalir, yani CI durustur.
    """
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return MeanCI(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    m = float(v.mean())
    if n == 1:
        return MeanCI(m, m, m, 0.0, 1)
    sem = float(v.std(ddof=1) / math.sqrt(n))
    h = t_critical(n - 1, confidence) * sem
    return MeanCI(m, m - h, m + h, sem, n)


def bootstrap_ci(values, confidence: float = 0.95, n_boot: int = 10000,
                 seed: int = 0) -> MeanCI:
    """Bootstrap CI -- kucuk n veya carpik dagilim icin t'ye alternatif."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return MeanCI(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    if n == 1:
        return MeanCI(float(v[0]), float(v[0]), float(v[0]), 0.0, 1)
    rng = np.random.default_rng(seed)
    boot = rng.choice(v, size=(n_boot, n), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return MeanCI(float(v.mean()), float(lo), float(hi),
                  float(boot.std(ddof=1)), n)


def diff_ci(a, b, confidence: float = 0.95) -> MeanCI:
    """Iki bagimsiz orneklem farkinin CI'si (Welch).

    `mean` = ortalama(a) - ortalama(b). CI sifiri icermiyorsa fark anlamli.
    Ablation'larda "gercekten iyilesti mi?" sorusunun cevabi budur.
    """
    x = np.asarray([v for v in a if v is not None], dtype=float)
    y = np.asarray([v for v in b if v is not None], dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        d = float(x.mean() - y.mean()) if nx and ny else float("nan")
        return MeanCI(d, float("nan"), float("nan"), float("nan"), min(nx, ny))
    vx, vy = x.var(ddof=1) / nx, y.var(ddof=1) / ny
    d = float(x.mean() - y.mean())
    sem = math.sqrt(vx + vy)
    if sem == 0.0:
        # Iki orneklem de SABIT (or. tum seed'lerde kills=0). Welch'in
        # payda'si 0/0 -> NaN. Bu durumda fark tam olarak bilinir: CI sifir
        # genislikte. `d != 0` ise fark trivially anlamli, `d == 0` ise yok.
        return MeanCI(d, d, d, 0.0, min(nx, ny))
    df = (vx + vy) ** 2 / (vx**2 / (nx - 1) + vy**2 / (ny - 1))  # Welch-Satterthwaite
    if not math.isfinite(df) or df < 1:
        df = 1.0
    h = t_critical(max(1, int(round(df))), confidence) * sem
    return MeanCI(d, d - h, d + h, sem, min(nx, ny))


# --------------------------------------------------------------------------
# Tek kosunun (bir seed) istatistikleri
# --------------------------------------------------------------------------

#: Turetilmis metrikler. Her biri: seed basina TEK bir sayi uretir.
#: Sunum adi -> (aciklama, yon)  yon=+1 buyuk iyi, -1 kucuk iyi
METRIC_INFO = {
    "score_per_round":       ("Tur basina skor (turnuvanin tanimi)", +1),
    "score_margin":          ("Skor farki: biz - en iyi rakip", +1),
    "win_rate":              ("Seed'lerin kacinda tum rakipleri gectik", +1),
    "coins_per_round":       ("Tur basina toplanan coin", +1),
    "kills_per_round":       ("Tur basina oldurulen rakip", +1),
    "crates_per_round":      ("Tur basina yikilan sandik", +1),
    "suicide_rate":          ("Turlarin kacinda kendini oldurdu", -1),
    "death_rate":            ("Turlarin kacinda oldu (intihar dahil)", -1),
    "survival_steps":        ("Tur basina hayatta kalinan adim", +1),
    "invalid_action_rate":   ("Gecersiz aksiyon orani (adim basina)", -1),
    "bombs_per_round":       ("Tur basina birakilan bomba", 0),
    "mean_act_time_ms":      ("Ortalama act() suresi (ms)", -1),
    "timeout_violations":    ("0.5 s asimi sayisi -- OLUMCUL, 0 olmali", -1),
}


@dataclass
class RunStats:
    """Tek bir `main.py play` kosusunun (bir seed) ozeti."""

    seed: int
    agent: str
    rounds: int
    raw: dict = field(repr=False)                 # by_agent[agent]
    opponents: dict = field(default_factory=dict, repr=False)  # name -> raw
    timeout_violations: int = 0

    # -- yardimcilar ------------------------------------------------------
    def _g(self, key: str, default: float = 0.0) -> float:
        return float(self.raw.get(key, default))

    @property
    def steps(self) -> float:
        return self._g("steps")

    def metric(self, name: str) -> float:
        r = max(1, self.rounds)
        st = max(1.0, self.steps)
        if name == "score_per_round":
            return self._g("score") / r
        if name == "score_margin":
            if not self.opponents:
                return float("nan")
            best = max(float(o.get("score", 0.0)) for o in self.opponents.values())
            return (self._g("score") - best) / r
        if name == "win_rate":
            if not self.opponents:
                return float("nan")
            best = max(float(o.get("score", 0.0)) for o in self.opponents.values())
            return 1.0 if self._g("score") > best else 0.0
        if name == "coins_per_round":
            return self._g("coins") / r
        if name == "kills_per_round":
            return self._g("kills") / r
        if name == "crates_per_round":
            return self._g("crates") / r
        if name == "suicide_rate":
            return self._g("suicides") / r
        if name == "death_rate":
            # SURVIVED_ROUND stat'i yok; olum = tur sayisi - hayatta bitirilen tur.
            # Yaklasik: suicides + (rakip bombasiyla olumler). Framework
            # GOT_KILLED'i stat olarak tutmuyor -> sadece intihari kesin biliyoruz.
            return float("nan")
        if name == "survival_steps":
            return self.steps / r
        if name == "invalid_action_rate":
            return self._g("invalid") / st
        if name == "bombs_per_round":
            return self._g("bombs") / r
        if name == "mean_act_time_ms":
            return 1000.0 * self._g("time") / st
        if name == "timeout_violations":
            return float(self.timeout_violations)
        raise KeyError(name)


def load_run(stats_path: str | Path, agent: str, seed: int,
             game_log: str | Path | None = None) -> RunStats:
    """Bir seed'in `--save-stats` JSON'unu (+ game.log) oku."""
    data = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    by_agent = data["by_agent"]
    if agent not in by_agent:
        raise KeyError(
            f"Ajan {agent!r} sonuclarda yok. Mevcut: {sorted(by_agent)}. "
            "Ayni ajan birden fazla kez verildiyse isimler _0,_1 diye eklenir."
        )
    raw = by_agent[agent]
    opponents = {k: v for k, v in by_agent.items() if k != agent}
    rounds = int(raw.get("rounds", len(data.get("by_round", {}))) or 1)

    violations = 0
    if game_log is not None and Path(game_log).exists():
        # environment.py: f'Agent <{name}> exceeded think time by ...'
        needle = f"Agent <{agent}> exceeded think time"
        try:
            with open(game_log, encoding="utf-8", errors="replace") as fh:
                violations = sum(1 for line in fh if needle in line)
        except OSError:
            violations = -1  # okunamadi

    return RunStats(seed=seed, agent=agent, rounds=rounds, raw=raw,
                    opponents=opponents, timeout_violations=violations)


def aggregate(runs: list[RunStats], confidence: float = 0.95) -> dict[str, MeanCI]:
    """Seed bazli ornekleri metrik bazinda ozetle."""
    out = {}
    for name in METRIC_INFO:
        vals = []
        for r in runs:
            try:
                v = r.metric(name)
            except KeyError:
                continue
            if v is not None and math.isfinite(v):
                vals.append(v)
        out[name] = mean_ci(vals, confidence) if vals else MeanCI(
            float("nan"), float("nan"), float("nan"), float("nan"), 0)
    return out


def format_table(agg: dict[str, MeanCI], title: str = "",
                 baseline: dict[str, MeanCI] | None = None) -> str:
    """Konsol tablosu. baseline verilirse delta sutunu eklenir."""
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    head = f"{'metrik':<22}{'deger (95% GA)':>26}"
    if baseline:
        head += f"{'taban':>14}{'delta':>12}"
    lines += [head, "-" * len(head)]
    for name, ci in agg.items():
        if ci.n == 0:
            continue
        row = f"{name:<22}{str(ci):>26}"
        if baseline and name in baseline and baseline[name].n:
            b = baseline[name]
            arrow = ""
            d = ci.mean - b.mean
            direction = METRIC_INFO.get(name, ("", 0))[1]
            if direction:
                arrow = " +" if d * direction > 0 else (" -" if d * direction < 0 else "")
            row += f"{b.mean:>14.3f}{d:>+11.3f}{arrow}"
        lines.append(row)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Kampanya istatistigi: coklu karsilastirma, doz-yanit, guc
# --------------------------------------------------------------------------
#
# NEDEN GEREKLI: 16 saatlik kampanyada tek bir paylasilan kontrole karsi
# 7 kontrast var. Duzeltmesiz bakarsak %95 GA'larin en az birinin sansa
# sifiri disarida birakma olasiligi ~%30. Ama Bonferroni-7 de asiri
# muhafazakar; kontrastlar dort AILEYE ayriliyor ve aile ICINDE Holm
# uygulaniyor.
#
# Ikinci mesele guc: sigma(egitim seed'i, EK-2) = 0.235 olculdu ve bu
# projede olculmus en buyuk odul-disi etki +0.254. Yani null sonuc
# BEKLENEN sonuc. `mde()` bunu her satirin yanina yazmak icin var --
# yorumlanabilir null, yorumlanamaz null'dan degerlidir.


def welch_p(a, b) -> float:
    """Iki bagimsiz orneklem icin Welch t-testi p degeri (iki yonlu)."""
    x = np.asarray([v for v in a if v is not None], dtype=float)
    y = np.asarray([v for v in b if v is not None], dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    vx, vy = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
    sem = math.sqrt(vx + vy)
    if sem == 0.0:
        # ikisi de sabit: fark ya tam olarak var ya yok
        return 0.0 if x.mean() != y.mean() else 1.0
    t = (x.mean() - y.mean()) / sem
    df = (vx + vy) ** 2 / (vx**2 / (x.size - 1) + vy**2 / (y.size - 1))
    if _scipy_stats is None:
        raise RuntimeError("welch_p scipy gerektiriyor")
    return float(2.0 * _scipy_stats.t.sf(abs(t), max(1.0, df)))


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down -- AILE ICINDE uygulanir.

    Bonferroni'den kesinlikle daha gucludur ve ayni aile-bazli hata oranini
    korur. Doner: {isim: {p, p_adj, reject, rank}}.

    NOT: aile tanimi ONCEDEN ilan edilmeli. Sonuclara bakip aile secmek
    duzeltmeyi anlamsiz kilar.
    """
    items = [(k, v) for k, v in pvals.items() if v == v]      # NaN'lari at
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj                     # step-down: monoton olmali
        out[k] = {"p": p, "p_adj": adj, "reject": adj <= alpha, "rank": i + 1}
    for k, v in pvals.items():
        if k not in out:
            out[k] = {"p": v, "p_adj": float("nan"),
                      "reject": False, "rank": -1}
    return out


def trend_test(doses, groups, confidence: float = 0.95) -> dict:
    """Doz-yanit egimi: metrigi doza karsi OLS ile regresyona sok.

    `doses`  : her kolun doz degeri, or. eps_end icin [0.01, 0.05, 0.10]
    `groups` : ayni sirada, her kolun seed basina degerleri listesi

    NEDEN IKILI KONTRASTTAN IYI: uc dozu tek teste birlestirmek MDE'yi
    ~%21 dusuruyor (0.79 -> 0.62) ve bedava. Ayrica "bir kol sansla
    yukseldi" artefaktina bagisik -- monoton bir trend sans eseri zor cikar.

    Doner: {slope, lo, hi, p, n, r2}. `slope` = doz birimi basina metrik.
    """
    xs, ys = [], []
    for d, g in zip(doses, groups):
        for v in g:
            if v is not None and np.isfinite(v):
                xs.append(float(d)); ys.append(float(v))
    x, y = np.asarray(xs), np.asarray(ys)
    n = x.size
    if n < 3 or np.allclose(x, x[0]):
        return {"slope": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p": float("nan"), "n": n, "r2": float("nan")}
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    slope = ((x - xm) * (y - ym)).sum() / sxx
    inter = ym - slope * xm
    resid = y - (inter + slope * x)
    df = n - 2
    s2 = (resid ** 2).sum() / df
    se = math.sqrt(s2 / sxx)
    h = t_critical(df, confidence) * se
    sst = ((y - ym) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / sst if sst > 0 else float("nan")
    p = (float(2.0 * _scipy_stats.t.sf(abs(slope / se), df))
         if _scipy_stats is not None and se > 0 else float("nan"))
    return {"slope": float(slope), "lo": float(slope - h),
            "hi": float(slope + h), "p": p, "n": n, "r2": float(r2)}


#: Welch kucuk-orneklem cezasi: (n, n_comparisons) -> sisirme carpani.
#:
#: NEDEN: ders kitabi MDE formulu esit-varyans df'ini (2n-2) kullanir, ama
#: biz HER YERDE Welch kosuyoruz ve Welch'in df'i kucuk orneklemde daha
#: dusuk VE degisken. Duzeltmesiz formul, n=4'te %80 dedigi yerde gercekte
#: %76 guc veriyordu -- yani MDE'yi kucuk gosterip null sonuclari
#: oldugundan guclu okutuyordu.
#:
#: Tablo simulasyonla olculdu (her hucre: 24 adimlik bisection x 8000
#: tekrar, sigma=1; oran olcekten bagimsiz cunku t-testi olcek-degismez).
#: Ureten script: scratchpad/mde_calib.py, ciktisi runs/mde_calib.log.
_WELCH_INFLATION = {
    (3, 1): 1.1559, (3, 2): 1.2447, (3, 3): 1.2493, (3, 4): 1.1652,
    (4, 1): 1.0546, (4, 2): 1.0922, (4, 3): 1.1204, (4, 4): 1.1381,
    (5, 1): 1.0309, (5, 2): 1.0506, (5, 3): 1.0638, (5, 4): 1.0716,
    (6, 1): 1.0139, (6, 2): 1.0271, (6, 3): 1.0340, (6, 4): 1.0388,
    (8, 1): 1.0062, (8, 2): 1.0146, (8, 3): 1.0193, (8, 4): 1.0235,
    (12, 1): 1.0029, (12, 2): 1.0037, (12, 3): 1.0070, (12, 4): 1.0101,
    (16, 1): 1.0042, (16, 2): 1.0049, (16, 3): 1.0060, (16, 4): 1.0071,
}
_WELCH_NS = sorted({n for n, _ in _WELCH_INFLATION})


def _welch_inflation(n: int, ncomp: int) -> float:
    """Tablodan carpan; ara n degerlerinde 1/n uzerinden interpolasyon.

    Ceza kabaca 1/n ile soner, o yuzden interpolasyon x=1/n ekseninde
    yapiliyor. n tablonun ustundeyse ceza ihmal edilebilir (1.0).
    """
    c = min(4, max(1, int(ncomp)))
    if n >= _WELCH_NS[-1]:
        return _WELCH_INFLATION[(_WELCH_NS[-1], c)]
    if n <= _WELCH_NS[0]:
        return _WELCH_INFLATION[(_WELCH_NS[0], c)]
    if (n, c) in _WELCH_INFLATION:
        return _WELCH_INFLATION[(n, c)]
    lo = max(x for x in _WELCH_NS if x < n)
    hi = min(x for x in _WELCH_NS if x > n)
    t = (1.0 / n - 1.0 / lo) / (1.0 / hi - 1.0 / lo)
    return (_WELCH_INFLATION[(lo, c)]
            + t * (_WELCH_INFLATION[(hi, c)] - _WELCH_INFLATION[(lo, c)]))


def mde(sigma: float, n_per_group: int, alpha: float = 0.05,
        power: float = 0.80, n_comparisons: int = 1,
        welch: bool = True) -> float:
    """Bu tasarimin gorebilecegi EN KUCUK etki (iki orneklem, esit n).

    MDE = (t_{alpha'/2,df} + t_{1-power,df}) * sigma * sqrt(2/n)
    `n_comparisons` > 1 ise alpha Bonferroni ile bolunur (aile ust siniri).
    `welch=True` (varsayilan) kucuk-orneklem cezasini uygular -- kampanyada
    her test Welch oldugu icin dogru olan bu.

    Null cikan her satirin yanina yazilmali: "bu tasarim X'ten kucuk etkiyi
    zaten goremezdi". Yoksa null "bir sey bulamadik" diye okunur.
    """
    if _scipy_stats is None:
        raise RuntimeError("mde scipy gerektiriyor")
    n = int(n_per_group)
    if n < 2:
        return float("nan")
    df = 2 * n - 2
    a = alpha / max(1, int(n_comparisons))
    t_a = _scipy_stats.t.ppf(1.0 - a / 2.0, df)
    t_b = _scipy_stats.t.ppf(power, df)
    base = (t_a + t_b) * sigma * math.sqrt(2.0 / n)
    if welch and abs(power - 0.80) < 1e-9:
        base *= _welch_inflation(n, n_comparisons)
    return float(base)
