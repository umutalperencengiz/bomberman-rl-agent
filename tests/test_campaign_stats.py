"""Kampanya istatistiklerinin testleri: Welch p, Holm, doz-yanit, MDE.

Sahibi: IRMAK

NEDEN ONEMLI: 16 saatlik kampanyanin BUTUN sonuclari bu dort fonksiyondan
geciyor. Biri sessizce yanlissa "anlamli" dedigimiz sey anlamli olmaz ve
rapordaki ablation tablosu curur.

Referans olarak scipy kullaniliyor -- kendi formulumuzu kendi formulumuzle
dogrulamak bir sey kanitlamaz. MDE icin referans yok, o yuzden MONTE CARLO
ile dogruluyoruz: hesaplanan MDE buyuklugundeki gercek bir etki, iddia
edilen guc kadar (%80) yakalanmali.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bbrl.metrics import holm, mde, trend_test, welch_p   # noqa: E402


def test_welch_matches_scipy(trials=200, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        na, nb = int(rng.integers(3, 12)), int(rng.integers(3, 12))
        a = rng.normal(rng.uniform(-2, 2), rng.uniform(0.2, 2.0), na)
        b = rng.normal(rng.uniform(-2, 2), rng.uniform(0.2, 2.0), nb)
        want = float(st.ttest_ind(a, b, equal_var=False).pvalue)
        got = welch_p(a, b)
        worst = max(worst, abs(got - want))
    assert worst < 1e-10, f"scipy'den sapma {worst:.2e}"
    print(f"  ok  welch_p            {trials} kosum, max sapma {worst:.1e}")


def test_holm_properties(trials=300, seed=1):
    """Holm: monoton, tek testte degismez, Bonferroni'den kucuk-esit."""
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        m = int(rng.integers(1, 7))
        pv = {f"k{i}": float(rng.uniform(0, 1)) for i in range(m)}
        out = holm(pv)

        # 1) tek test -> duzeltme yok
        if m == 1:
            k = next(iter(pv))
            assert abs(out[k]["p_adj"] - pv[k]) < 1e-12, "tek testte duzeltildi"

        # 2) siralamayi koru: p kucukse p_adj de kucuk-esit
        ordered = sorted(pv.items(), key=lambda kv: kv[1])
        prev = -1.0
        for k, _ in ordered:
            assert out[k]["p_adj"] >= prev - 1e-12, "p_adj monoton degil"
            prev = out[k]["p_adj"]

        # 3) Bonferroni'den daha muhafazakar OLMAMALI
        for k, p in pv.items():
            assert out[k]["p_adj"] <= min(1.0, m * p) + 1e-12, \
                "Holm Bonferroni'den daha muhafazakar cikti"

    # 4) bilinen el hesabi
    out = holm({"a": 0.01, "b": 0.04, "c": 0.30})
    assert abs(out["a"]["p_adj"] - 0.03) < 1e-12    # 3*0.01
    assert abs(out["b"]["p_adj"] - 0.08) < 1e-12    # 2*0.04
    assert abs(out["c"]["p_adj"] - 0.30) < 1e-12    # 1*0.30
    print(f"  ok  holm               {trials} kosum + el hesabi")


def test_trend_matches_scipy(trials=100, seed=2):
    rng = np.random.default_rng(seed)
    worst_s = worst_p = 0.0
    for _ in range(trials):
        doses = sorted(rng.uniform(0, 1, 3))
        groups = [list(rng.normal(2 + 3 * d, 0.4, 4)) for d in doses]
        got = trend_test(doses, groups)
        x = np.repeat(doses, 4)
        y = np.concatenate([np.asarray(g) for g in groups])
        ref = st.linregress(x, y)
        worst_s = max(worst_s, abs(got["slope"] - ref.slope))
        worst_p = max(worst_p, abs(got["p"] - ref.pvalue))
    assert worst_s < 1e-9, f"egim sapmasi {worst_s:.2e}"
    assert worst_p < 1e-9, f"p sapmasi {worst_p:.2e}"
    print(f"  ok  trend_test         {trials} kosum, scipy.linregress ile ayni")


def test_mde_is_honest(sigma=0.235, n=4, reps=4000, seed=3):
    """MONTE CARLO: MDE buyuklugundeki gercek etki ~%80 yakalanmali.

    Bu testin amaci sayiyi degil IDDIAYI dogrulamak. Rapor "bu tasarim
    X'ten kucugunu goremezdi" diyecek; X gercekten %80 gucun esigi degilse
    o cumle yanlis olur.
    """
    for ncomp in (1, 2, 3):
        eff = mde(sigma, n, n_comparisons=ncomp)
        alpha = 0.05 / ncomp
        rng = np.random.default_rng(seed)
        hits = 0
        for _ in range(reps):
            a = rng.normal(eff, sigma, n)
            b = rng.normal(0.0, sigma, n)
            if welch_p(a, b) <= alpha:
                hits += 1
        power = hits / reps
        # Kalibrasyondan SONRA iddia edilen %80'e yakin cikmali. Bant DAR
        # tutuluyor: gevsek bir bant duzeltmenin ise yarayip yaramadigini
        # gizlerdi -- duzeltmesiz hali 0.761 / 0.728 / 0.706 veriyordu ve
        # o hali "bu tasarim X'ten kucugunu goremezdi" cumlesini yanlis
        # yapardi (X oldugundan kucuk gosteriliyordu).
        assert 0.77 <= power <= 0.84, \
            f"ncomp={ncomp}: MDE={eff:.3f} icin olculen guc {power:.3f}"
        print(f"  ok  mde ncomp={ncomp}        MDE={eff:.3f} -> "
              f"olculen guc {power:.3f}")


def test_mde_shrinks_with_n():
    """Daha cok seed -> daha kucuk MDE (yon kontrolu)."""
    vals = [mde(0.235, n) for n in (4, 6, 8, 16)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)), vals
    print(f"  ok  mde ~ n            n=4:{vals[0]:.3f} n=8:{vals[2]:.3f} "
          f"n=16:{vals[3]:.3f}")


def main() -> int:
    print("kampanya istatistigi testleri")
    print("-" * 60)
    try:
        test_welch_matches_scipy()
        test_holm_properties()
        test_trend_matches_scipy()
        test_mde_is_honest()
        test_mde_shrinks_with_n()
    except AssertionError as ex:
        print(f"  FAIL {ex}")
        print("-" * 60); print("BASARISIZ"); return 1
    except Exception as ex:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print(f"  ERR  {type(ex).__name__}: {ex}")
        print("-" * 60); print("BASARISIZ"); return 1
    print("-" * 60); print("TUMU GECTI"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
