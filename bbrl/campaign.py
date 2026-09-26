"""Kampanya analizi -- ONCEDEN ILAN EDILMIS aileler, Holm, doz-yanit, MDE.

Sahibi: IRMAK

NEDEN AYRI BIR MODUL: `train_loop.multi_table` her kolu tabana karsi
duzeltmesiz Welch ile kiyasliyor. Tek bir deneyde dogru, ama 16 saatlik
kampanyada tek paylasilan kontrole karsi 7 kontrast var -- duzeltmesiz
bakarsak birinin sansa sifiri disarida birakma olasiligi ~%30.

Bu modul uc sey ekliyor:

  1. AILE ICINDE Holm. Aileler kodda, sonuclara BAKMADAN ONCE tanimli
     (asagidaki FAMILIES). Sonucu gorup aile secmek duzeltmeyi anlamsiz
     kilar; o yuzden burada duruyorlar, komut satirinda degil.

  2. DOZ-YANIT trendi. Uc dozu tek OLS testine birlestirmek MDE'yi ~%21
     dusuruyor ve "bir kol sansla yukseldi" artefaktina bagisik.

  3. Her satirin yaninda MDE. sigma(egitim seed'i) = 0.235 olculdu ve bu
     projede olculmus en buyuk odul-disi etki +0.254 -- yani NULL BEKLENEN
     SONUC. MDE yazilmazsa null "bir sey bulamadik" diye okunur; yazilirsa
     "bu tasarim bundan kucugunu zaten goremezdi" diye okunur.

Kullanim:
    python -m bbrl.campaign --summary runs/summaries/e20.json
    python -m bbrl.campaign --summary runs/summaries/e20.json runs/summaries/e21.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bbrl import metrics as M  # noqa: E402

CONTROL = "e20a_base"

#: Tarihsel sigma: 8 seed havuzlanmis kontrol (EXP-013/019, 24000 tur).
#: 4 seed'den kestirilen sd bunun cok altina inebiliyor ve o zaman MDE
#: yaniltici derecede kucuk gorunuyor -- bkz. analyse() icindeki not.
SIGMA_HIST = 0.235

#: ONCEDEN ILAN EDILMIS aileler. Sonuclara bakilmadan yazildi (2026-09-08).
#: Holm her ailenin ICINDE uygulanir; aileler arasi duzeltme YOK.
FAMILIES: dict[str, dict] = {
    "F1 kesif (eps_end)": {
        "contrasts": ["e21b_eps01", "e21c_eps10"],
        "dose": {"e21b_eps01": 0.01, CONTROL: 0.05, "e21c_eps10": 0.10},
        "dose_unit": "eps_end",
    },
    "F2 replay orani": {
        "contrasts": ["e23c_upr16", "e23b_ratio2x"],
        # tur basina guncelleme: 172.5/train_every + updates_per_round
        "dose": {CONTROL: 51.1, "e23c_upr16": 59.1, "e23b_ratio2x": 94.3},
        "dose_unit": "update/tur",
    },
    "F3 kredi atamasi": {
        "contrasts": ["e20c_nstep", "e20b_per", "e20d_both"],
        "dose": None,
    },
    "F4 rakip dagilimi": {
        "contrasts": ["e19d_selfplay"],
        "dose": None,
    },
    "F5 ufuk (gamma)": {
        "contrasts": ["e22b_gamma98", "e22c_gamma99"],
        "dose": {CONTROL: 0.95, "e22b_gamma98": 0.98, "e22c_gamma99": 0.99},
        "dose_unit": "gamma",
    },
    "F6 kapasite": {
        "contrasts": ["e24b_wide", "e24c_deep"],
        "dose": None,
    },
}


def load(paths, metric="score_per_round") -> dict[str, list[float]]:
    """Ozet JSON'larindan config -> seed basina metrik degerleri."""
    out: dict[str, list[float]] = {}
    for p in paths:
        blob = json.loads(Path(p).read_text(encoding="utf-8"))
        for c in blob.get("configs", []):
            vals = (c.get("per_run") or {}).get(metric) or []
            vals = [v for v in vals if v is not None]
            if vals:
                out.setdefault(c["config"], []).extend(vals)
    return out


def analyse(data: dict[str, list[float]], metric="score_per_round",
            alpha=0.05) -> str:
    if CONTROL not in data:
        return (f"KONTROL '{CONTROL}' ozette yok -- kampanya analizi "
                f"paylasilan kontrole dayaniyor. Bulunanlar: "
                f"{sorted(data)}")
    base = data[CONTROL]
    sigma = float(M.np.std(base, ddof=1)) if len(base) > 1 else float("nan")

    L = []
    L.append(f"KAMPANYA ANALIZI  --  metrik: {metric}")
    L.append("=" * 78)
    L.append(f"kontrol {CONTROL}: {M.mean_ci(base)}  (n={len(base)} egitim seed'i)")
    L.append(f"kontrolun seed'ler arasi sd = {sigma:.3f}")
    L.append("")

    # -- her kolun ham durumu -------------------------------------------
    L.append(f"{'config':<20}{'n':>3}{'ortalama':>22}{'kontrole gore':>24}")
    L.append("-" * 78)
    L.append(f"{CONTROL:<20}{len(base):>3}{str(M.mean_ci(base)):>22}"
             f"{'(taban)':>24}")
    for name in sorted(k for k in data if k != CONTROL):
        v = data[name]
        d = M.diff_ci(v, base)
        L.append(f"{name:<20}{len(v):>3}{str(M.mean_ci(v)):>22}{str(d):>24}")
    L.append("")

    # -- aile ici Holm ---------------------------------------------------
    for fam, spec in FAMILIES.items():
        arms = [a for a in spec["contrasts"] if a in data]
        if not arms:
            continue
        pv = {a: M.welch_p(data[a], base) for a in arms}
        hol = M.holm(pv, alpha=alpha)
        n_min = min([len(base)] + [len(data[a]) for a in arms])
        m_obs = (M.mde(sigma, n_min, n_comparisons=len(arms))
                 if sigma == sigma else float("nan"))
        m_hist = M.mde(SIGMA_HIST, n_min, n_comparisons=len(arms))
        # !!! EXP-019 DERSI !!!
        # 4 seed'den kestirilen sd'nin KENDISI gurultulu. Ayni config iki kez
        # kosuldu: bir kez +-0.053, bir kez +-0.472. 8 seed havuzlaninca
        # gercek deger +-0.197 cikti. Yani dar bir sd, kolun tutarli oldugunu
        # DEGIL, sansli dort cekilis oldugunu gosterebilir. Bu yuzden karar
        # esigi olarak IKISININ BUYUGUNU aliyoruz.
        m = max(m_obs, m_hist) if m_obs == m_obs else m_hist

        L.append(f"{fam}   (aile ici Holm, {len(arms)} kontrast)")
        L.append(f"  MDE: bu kosunun sd'siyle {m_obs:.3f}, "
                 f"tarihsel sigma={SIGMA_HIST} ile {m_hist:.3f} "
                 f"-> KARAR ESIGI {m:.3f}")
        L.append(f"      [n={n_min}, %80 guc, alpha={alpha}/{len(arms)}; "
                 f"buyugu aliniyor cunku 4 seed'lik sd tahmini kendisi "
                 f"gurultulu -- EXP-019]")
        L.append(f"  {'kol':<18}{'etki':>10}{'%95 GA':>22}{'p':>9}{'p_holm':>9}  karar")
        for a in arms:
            d = M.diff_ci(data[a], base)
            h = hol[a]
            if h["reject"]:
                verdict = "ANLAMLI"
                if abs(d.mean) < m_hist:
                    # Holm reddetti ama etki tarihsel MDE'nin ALTINDA. Bu
                    # ancak bu kosunun sd'si tarihsel sigma'dan cok kucukse
                    # olur -- yani tam olarak EXP-019'da yanildigimiz durum
                    # (ayni config +-0.053 ve +-0.472 verdi). Kolu gemiye
                    # almadan once EK SEED ile tekrarla.
                    verdict += "  <-- DIKKAT: etki tarihsel MDE'nin altinda, "\
                               "dar sd sansli cekilis olabilir (EXP-019); "\
                               "ek seed ile tekrarla"
            else:
                verdict = "null"
                if abs(d.mean) < m:
                    verdict += " (|etki| karar esiginin altinda)"
            L.append(f"  {a:<18}{d.mean:>+10.3f}"
                     f"{f'[{d.lo:+.3f}, {d.hi:+.3f}]':>22}"
                     f"{h['p']:>9.3f}{h['p_adj']:>9.3f}  {verdict}")

        # -- doz-yanit ---------------------------------------------------
        dose = spec.get("dose")
        if dose:
            have = [(v, data[k]) for k, v in dose.items() if k in data]
            if len(have) >= 3:
                have.sort(key=lambda t: t[0])
                tr = M.trend_test([h[0] for h in have], [h[1] for h in have])
                sig = "ANLAMLI" if tr["p"] == tr["p"] and tr["p"] < alpha else "null"
                L.append(f"  doz-yanit ({spec['dose_unit']}, {len(have)} doz): "
                         f"egim {tr['slope']:+.4f} "
                         f"[{tr['lo']:+.4f}, {tr['hi']:+.4f}] "
                         f"p={tr['p']:.3f} r2={tr['r2']:.2f}  {sig}")
            else:
                L.append(f"  doz-yanit: {len(have)}/3 doz mevcut -- test yok")
        L.append("")

    L.append("NOT: aileler ve tahminler kosudan ONCE ilan edildi "
             "(docs/EXPERIMENTS.md). Sonuca bakip aile secmek duzeltmeyi "
             "anlamsiz kilar.")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summary", nargs="+", required=True,
                    help="runs/summaries/*.json")
    ap.add_argument("--metric", default="score_per_round")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(argv)
    data = load(a.summary, a.metric)
    if not data:
        print("ozetlerde veri yok"); return 1
    print(analyse(data, a.metric, a.alpha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
