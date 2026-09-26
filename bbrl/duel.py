"""Iki ajani kapistirir ve durust bir karsilastirma raporu verir.

Sahibi: IRMAK

NEDEN AYRI BIR ARAC: "benim ajanim seninkini yendi" tek bir oyunda anlamsiz.
Bu arac uc olcumu birden yapar:

  1. DUELLO      A ve B ayni tahtada (1v1 ve 2v2)
  2. ORTAK CETVEL ikisi de AYRI AYRI rule_based'e karsi (EK-2)
  3. Cok seed + %95 guven araligi

(2) sart: duelloda A kazanabilir ama ikisi de rule_based'e kaybediyorsa
turnuvada ikisi de kaybeder. Ortak referans olmadan siralama yaniltir.

Kullanim:
    python -m bbrl.duel --a irmak_umut --b umut_agent
    python -m bbrl.duel --a irmak_umut --b umut_agent --seeds 20 --rounds 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bbrl import eval as E  # noqa: E402
from bbrl import metrics as M  # noqa: E402

RB = "rule_based_agent"


def _row(label, res, extra=""):
    a = res.agg
    return (f"{label:<26}"
            f"{a['score_per_round'].mean:>8.3f}±{a['score_per_round'].halfwidth:<6.3f}"
            f"{a['coins_per_round'].mean:>8.2f}"
            f"{(a['score_per_round'].mean - a['coins_per_round'].mean)/5:>8.2f}"
            f"{a['suicide_rate'].mean:>9.3f}"
            f"{a['survival_steps'].mean:>8.0f}  {extra}")


def duel(a: str, b: str, seeds=range(10), rounds: int = 100,
         python: str | None = None) -> dict:
    python = python or E.default_python()
    seeds = list(seeds)
    out = {}

    print(f"\n{'='*88}\nDUELLO: {a}  vs  {b}\n{'='*88}")
    print(f"{'kurulum':<26}{'skor (95% GA)':>15}{'coin':>8}{'kill':>8}"
          f"{'intihar':>9}{'adim':>8}")
    print("-" * 88)

    # --- 1v1 (ayni tahtada, iki ajan) -------------------------------------
    out["a_1v1"] = E.evaluate(a, [b], "classic", rounds, seeds,
                              card="duel-1v1", python=python, verbose=False)
    out["b_1v1"] = E.evaluate(b, [a], "classic", rounds, seeds,
                              card="duel-1v1", python=python, verbose=False)
    print(_row(f"{a}  (1v1)", out["a_1v1"]))
    print(_row(f"{b}  (1v1)", out["b_1v1"]))

    # --- 2v2 (ikisinden ikiser kopya -- turnuvaya daha yakin kalabalik) ---
    out["a_2v2"] = E.evaluate(a, [a, b, b], "classic", rounds, seeds,
                              card="duel-2v2", python=python, verbose=False)
    out["b_2v2"] = E.evaluate(b, [b, a, a], "classic", rounds, seeds,
                              card="duel-2v2", python=python, verbose=False)
    print(_row(f"{a}  (2v2)", out["a_2v2"]))
    print(_row(f"{b}  (2v2)", out["b_2v2"]))

    # --- ORTAK CETVEL: ikisi de 3x rule_based'e karsi ---------------------
    print("-" * 88)
    out["a_ek2"] = E.evaluate_card(a, "EK-2", seeds=seeds, python=python,
                                   verbose=False)
    out["b_ek2"] = E.evaluate_card(b, "EK-2", seeds=seeds, python=python,
                                   verbose=False)
    print(_row(f"{a}  (EK-2)", out["a_ek2"], "<- turnuva vekili"))
    print(_row(f"{b}  (EK-2)", out["b_ek2"], "<- turnuva vekili"))
    print(f"{'rule_based_agent (EK-2)':<26}{3.317:>8.3f}±{0.247:<6.3f}"
          f"{2.247:>8.2f}{0.214:>8.2f}{0.528:>9.3f}{228:>8.0f}  <- referans")

    # --- istatistik -------------------------------------------------------
    print("\n" + "=" * 88)
    print("FARKLAR (Welch %95 -- GA sifiri icermiyorsa ANLAMLI)")
    print("-" * 88)
    for key, label in (("1v1", "1v1 duelloda"), ("2v2", "2v2 duelloda"),
                       ("ek2", "rule_based'e karsi (ortak cetvel)")):
        d = M.diff_ci(out[f"a_{key}"].per_seed["score_per_round"],
                      out[f"b_{key}"].per_seed["score_per_round"])
        verdict = ("ANLAMLI" if (d.lo > 0 or d.hi < 0) else "anlamli DEGIL")
        who = a if d.mean > 0 else b
        print(f"  {label:<38}{d.mean:+8.3f} [{d.lo:+.3f},{d.hi:+.3f}]  "
              f"{verdict:<14}{('-> ' + who) if verdict == 'ANLAMLI' else ''}")

    print("\nNOT: duelloda kazanmak turnuvada kazanmak demek degil. Turnuvada"
          "\nkarsimizda BASKA takimlarin ajanlari olacak; ortak cetvel (EK-2)"
          "\nikisinin de o ortama ne kadar hazir oldugunu gosterir.")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m bbrl.duel")
    p.add_argument("--a", required=True, help="agent_code/ altindaki klasor")
    p.add_argument("--b", required=True)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--python", default=None)
    args = p.parse_args(argv)

    for name in (args.a, args.b):
        if not (REPO / "agent_code" / name / "callbacks.py").is_file():
            print(f"HATA: agent_code/{name}/callbacks.py yok")
            return 1
    duel(args.a, args.b, range(args.seeds), args.rounds, args.python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------
# Karma masa: birden fazla ajan AYNI oyunlarda
# --------------------------------------------------------------------------
#
# `--save-stats` JSON'u masadaki TUM ajanlarin istatistigini tasiyor. Her
# ajani ayri kosuda olcmek yerine tek kosudan hepsini okumak hem 4x ucuz
# hem de daha durust: ayni tahtalar, ayni tur, ayni rakipler.

def mixed_match(agents, scenario: str = "classic", rounds: int = 100,
                seeds=range(10), python: str | None = None,
                out_name: str = "mixed") -> dict:
    """Verilen ajanlari ayni masada oynatir, HEPSININ metrigini dondurur."""
    import json
    from datetime import datetime
    python = python or E.default_python()
    agents = list(agents)
    seeds = list(seeds)
    names = E.resolve_agent_names(agents)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = E.RUNS / f"{stamp}_{out_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*92}\nKARMA MASA: {' | '.join(names)}")
    print(f"{scenario}, {rounds} tur x {len(seeds)} seed -- hepsi AYNI oyunlarda")
    print("=" * 92)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os as _os
    workers = max(1, min(len(seeds), (_os.cpu_count() or 4) - 2))
    per_agent = {n: [] for n in names}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(E.run_one, agents[0], agents[1:], scenario,
                            rounds, s, out_dir, python): s for s in seeds}
        done = 0
        for fut in as_completed(futs):
            oc = fut.result(); done += 1
            if not oc.ok:
                print(f"  seed {oc.seed}: FAIL {oc.error.splitlines()[0]}")
                continue
            # ayni JSON'dan TUM ajanlari oku
            sp = out_dir / f"seed_{oc.seed:03d}" / "stats.json"
            data = json.loads(sp.read_text(encoding="utf-8"))
            for n in names:
                raw = data["by_agent"].get(n)
                if raw:
                    per_agent[n].append(
                        M.RunStats(seed=oc.seed, agent=n,
                                   rounds=int(raw.get("rounds", rounds)),
                                   raw=raw,
                                   opponents={k: v for k, v in
                                              data["by_agent"].items() if k != n}))
            print(f"  [{done}/{len(seeds)}] seed {oc.seed} ok ({oc.duration_s:.0f}s)")

    print(f"\n{'ajan':<26}{'skor (95% GA)':>17}{'coin':>8}{'kill':>8}"
          f"{'intihar':>9}{'adim':>8}{'kazanma':>9}")
    print("-" * 92)
    agg = {}
    for n in names:
        runs = per_agent[n]
        if not runs:
            continue
        a = M.aggregate(runs)
        agg[n] = {"agg": a, "per_seed": {m: [r.metric(m) for r in runs]
                                         for m in ("score_per_round", "win_rate")}}
        print(f"{n:<26}{a['score_per_round'].mean:>10.3f}±"
              f"{a['score_per_round'].halfwidth:<6.3f}"
              f"{a['coins_per_round'].mean:>8.2f}"
              f"{a['kills_per_round'].mean:>8.2f}"
              f"{a['suicide_rate'].mean:>9.3f}"
              f"{a['survival_steps'].mean:>8.0f}"
              f"{a['win_rate'].mean:>9.2f}")
    return agg
