"""Rapor §6 grafikleri.

Sahibi: IRMAK (Umut'tan devralindi)

Iki grafik ailesi:
  1. EGITIM EGRISI   -- `trainlog_*.json`'dan, cok seed'li band ile
  2. ABLATION        -- `runs/summaries/*.json`'dan, %95 GA hata cubuklu

KURAL: grafikte gosterilen performans DAIMA gercek oyun skoru (eval kartlari).
Shaped odul egrisi ayri gosterilir ve "ogrenme sinyali" diye etiketlenir --
ikisini ayni eksende karistirmak raporu yaniltir.

Kullanim:
    python -m bbrl.plots --summary runs/summaries/e10.json
    python -m bbrl.plots --training e10 --out runs/figs
    python -m bbrl.plots --all
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # basliksiz ortam; GUI acmaz
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FIGS = REPO / "runs" / "figs"
AGENT_DIR = REPO / "agent_code" / "irmak_umut"
SUMMARIES = REPO / "runs" / "summaries"

#: EXP-000'de olculup DONDURULAN referans degerler (kart -> metrik -> deger).
#: Her grafikte referans cizgi olarak gorunur.
BASELINES = {
    "EK-5": {"coins_per_round": 8.53, "crates_per_round": 116.9,
             "suicide_rate": 0.000, "survival_steps": 399.1},
    "EK-3": {"coins_per_round": 50.0, "survival_steps": 125.19},
    "EK-2": {"score_per_round": 3.32, "suicide_rate": 0.53},
    "EK-1": {"score_per_round": 4.67, "suicide_rate": 0.38},
}

NICE = {
    "coins_per_round": "coin / tur",
    "score_per_round": "skor / tur",
    "crates_per_round": "sandık / tur",
    "suicide_rate": "intihar oranı",
    "survival_steps": "hayatta kalınan adım",
    "invalid_action_rate": "geçersiz aksiyon oranı",
    "mean_act_time_ms": "act süresi (ms)",
}


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


# --------------------------------------------------------------------------
# 1) Ablation grafigi
# --------------------------------------------------------------------------

def plot_ablation(summary_path, metrics=None, out_dir=None) -> list[Path]:
    """Sweep ozetinden metrik basina bar + %95 GA hata cubugu."""
    blob = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    card = blob.get("card", "")
    cfgs = blob["configs"]
    if not cfgs:
        return []
    metrics = metrics or ["coins_per_round", "crates_per_round",
                          "suicide_rate", "survival_steps"]
    out_dir = Path(out_dir or FIGS)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp = blob.get("experiment", Path(summary_path).stem)

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6))
    axes = np.atleast_1d(axes)
    labels = [c["config"] for c in cfgs]
    short = [l.split("_", 1)[-1] if "_" in l else l for l in labels]

    for ax, m in zip(axes, metrics):
        means, errs = [], []
        for c in cfgs:
            g = c["aggregate"].get(m)
            if not g or g.get("n", 0) == 0 or not np.isfinite(g["mean"]):
                means.append(np.nan); errs.append(0.0); continue
            means.append(g["mean"])
            hw = (g["hi"] - g["lo"]) / 2.0
            errs.append(hw if np.isfinite(hw) else 0.0)
        x = np.arange(len(cfgs))
        ax.bar(x, means, yerr=errs, capsize=4, color="#4C78A8",
               edgecolor="#2A4A6F", linewidth=0.8, alpha=0.9)
        base = BASELINES.get(card, {}).get(m)
        if base is not None:
            ax.axhline(base, color="#E45756", linestyle="--", linewidth=1.2,
                       label=f"rule_based = {base:g}")
            ax.legend(fontsize=7, frameon=False)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=30, ha="right", fontsize=7)
        _style(ax, NICE.get(m, m), "", NICE.get(m, m))

    n_seeds = len(cfgs[0].get("train_seeds", []))
    fig.suptitle(f"{exp} — {card} ({n_seeds} eğitim seed'i, hata çubuğu %95 GA)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out_dir / f"{exp}_ablation.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return [p]


# --------------------------------------------------------------------------
# 2) Egitim egrisi
# --------------------------------------------------------------------------

def _load_trainlogs(prefix: str) -> dict[str, list[dict]]:
    """`trainlog_<prefix>*.json` dosyalarini config adina gore gruplar."""
    out: dict[str, list[dict]] = {}
    for p in sorted(AGENT_DIR.glob(f"trainlog_{prefix}*.json")):
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = blob.get("config", p.stem[9:])
        base = name.rsplit("_s", 1)[0]       # seed son ekini at
        out.setdefault(base, []).append(blob.get("rounds", []))
    return out


def _smooth(y, k=51):
    y = np.asarray(y, dtype=float)
    if len(y) < k or k < 3:
        return y
    kern = np.ones(k) / k
    return np.convolve(y, kern, mode="valid")


def plot_training(prefix: str, metrics=None, out_dir=None) -> list[Path]:
    """Tur bazli egitim egrileri; her config icin seed'ler arasi band."""
    groups = _load_trainlogs(prefix)
    if not groups:
        return []
    metrics = metrics or ["game_score", "steps", "e_KILLED_SELF", "e_CRATE_DESTROYED"]
    nice = {"game_score": "gerçek oyun skoru", "steps": "tur uzunluğu (adım)",
            "e_KILLED_SELF": "intihar (tur başına)",
            "e_CRATE_DESTROYED": "kırılan sandık", "reward": "shaped ödül (öğrenme sinyali)"}
    out_dir = Path(out_dir or FIGS)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6))
    axes = np.atleast_1d(axes)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 3)))

    for ax, m in zip(axes, metrics):
        for (name, runs), col in zip(sorted(groups.items()), colors):
            curves = []
            for rounds in runs:
                if not rounds:
                    continue
                curves.append(_smooth([r.get(m, 0.0) for r in rounds]))
            if not curves:
                continue
            L = min(len(c) for c in curves)
            arr = np.stack([c[:L] for c in curves])
            x = np.arange(L)
            mu = arr.mean(axis=0)
            ax.plot(x, mu, color=col, linewidth=1.4,
                    label=f"{name} (n={len(curves)})")
            if arr.shape[0] > 1:
                lo, hi = arr.min(axis=0), arr.max(axis=0)
                ax.fill_between(x, lo, hi, color=col, alpha=0.15, linewidth=0)
        _style(ax, nice.get(m, m), "tur", nice.get(m, m))
        ax.legend(fontsize=7, frameon=False)

    fig.suptitle(f"{prefix} — eğitim ilerlemesi (51 tur kayan ortalama, "
                 f"band = seed'ler arası min–maks)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = out_dir / f"{prefix}_training.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return [p]


# --------------------------------------------------------------------------

def convergence_report(prefix: str, metric: str = "e_CRATE_DESTROYED",
                       tol: float = 0.05) -> list[dict]:
    """Egriler platoya ulasti mi? Son %20 ile onceki %20'yi karsilastirir.

    NEDEN VAR: EXP-010'un egitim egrilerine bakinca 6000. turda dort varyantin
    da HALA yukseldigi goruldu (sandik %11-18 buyumeye devam ediyordu). Yani
    o ana kadarki tum karsilastirmalar EGITIMI BITMEMIS ajanlar arasindaydi --
    olculen farklar gercek bir siralama degil, "hangisi 6000 turda daha hizli
    ogreniyor" olabilir.

    Bir sonuc rapora girmeden ONCE bu kontrolden gecmeli.
    """
    groups = _load_trainlogs(prefix)
    out = []
    for name, runs in sorted(groups.items()):
        firsts, lasts = [], []
        for rs in runs:
            if len(rs) < 10:
                continue
            q = len(rs) // 5
            firsts.append(np.mean([r.get(metric, 0.0) for r in rs[-2 * q:-q]]))
            lasts.append(np.mean([r.get(metric, 0.0) for r in rs[-q:]]))
        if not firsts:
            continue
        a, b = float(np.mean(firsts)), float(np.mean(lasts))
        growth = (b - a) / max(abs(a), 1e-9)
        out.append({"config": name, "prev": a, "last": b,
                    "growth": growth, "converged": abs(growth) <= tol})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m bbrl.plots")
    p.add_argument("--summary", help="runs/summaries/<exp>.json")
    p.add_argument("--training", help="config on eki, or. e10 / m02")
    p.add_argument("--metrics", nargs="*", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--all", action="store_true",
                   help="tum ozetler + tum egitim gunlukleri")
    p.add_argument("--convergence", help="config on eki -- plato kontrolu")
    a = p.parse_args(argv)

    if a.convergence:
        rows = convergence_report(a.convergence)
        if not rows:
            print("egitim gunlugu bulunamadi"); return 1
        print(f"{'config':<26}{'onceki %20':>12}{'son %20':>12}"
              f"{'buyume':>10}{'yakinsadi?':>12}")
        print("-" * 72)
        for r in rows:
            print(f"{r['config']:<26}{r['prev']:>12.2f}{r['last']:>12.2f}"
                  f"{r['growth']*100:>9.0f}%"
                  f"{('EVET' if r['converged'] else 'HAYIR'):>12}")
        if any(not r["converged"] for r in rows):
            print()
            print("UYARI: yakinsamamis kosular var -- karsilastirma bir "
                  "SIRALAMA degil, 'kim daha hizli ogreniyor' olur.")
        return 0

    made = []
    if a.all:
        for s in sorted(SUMMARIES.glob("*.json")):
            made += plot_ablation(s, a.metrics, a.out)
        prefixes = sorted({Path(f).stem[9:].rsplit("_s", 1)[0][:3]
                           for f in glob.glob(str(AGENT_DIR / "trainlog_*.json"))})
        for pre in prefixes:
            made += plot_training(pre, None, a.out)
    else:
        if a.summary:
            made += plot_ablation(a.summary, a.metrics, a.out)
        if a.training:
            made += plot_training(a.training, a.metrics, a.out)
    if not made:
        print("grafik uretilmedi (--summary / --training / --all ver)")
        return 1
    for m in made:
        print(f"  {m.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
