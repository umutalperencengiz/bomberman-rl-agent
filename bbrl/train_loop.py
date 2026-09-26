"""Egitim ve sweep koscusu.

Sahibi: IRMAK (infra)

Egitim dongusunu BIZ kurmuyoruz -- framework kuruyor. Bu modul yalnizca
`main.py play --train 1` cagrilarini mufredat asamalarina gore siraya dizer,
sonra ayni degerlendirme kartindan gecirir.

    python -m bbrl.train_loop --config configs/e07b.yaml
    python -m bbrl.train_loop --sweep configs/e07*.yaml --card EK-3

Paralellik: her config AYRI bir surectir; model dosyalari `model_<name>.bin`
diye adlandirildigi icin ayni ajan klasorunde cakismazlar.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bbrl import eval as E  # noqa: E402
from bbrl import metrics as M  # noqa: E402
from bbrl.config import Config  # noqa: E402

#AGENT_DIR = "irmak_umut"
AGENT_DIR = "umut_dueling_agent"


def _stage_cmd(python: str, cfg: Config, stage: dict, seed: int | None):
    agents = [AGENT_DIR] + list(stage.get("opponents", []))
    cmd = [python, "main.py", "play",
           "--agents", *agents,
           "--train", "1",
           "--no-gui",
           "--scenario", stage.get("scenario", "classic"),
           "--n-rounds", str(int(stage.get("n_rounds", 1000)))]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd


def train(cfg_path: str, python: str | None = None, seed: int | None = None,
          verbose: bool = True) -> dict:
    """Bir config'i mufredat asamalari boyunca egitir."""
    python = python or sys.executable
    cfg = Config.load(cfg_path)
    cfg.validate()

    env = dict(os.environ)
    # Ajan callback'leri bu env degiskeninden config'i bulur.
    env["BBRL_CONFIG"] = str(Path(cfg_path).as_posix())
    # Config yalnizca EGITILEN ajan icin -- rakip olarak masada oturan
    # irmak_umut / pool_* onu okursa rastgele agirliklarla oynar.
    if not LEAK_CONFIG_TO_OPPONENTS:
        env["BBRL_CONFIG_AGENT"] = AGENT_DIR
    env.setdefault("BBRL_LOG_GAME", "WARNING")
    env.setdefault("BBRL_LOG_AGENT_WRAPPER", "CRITICAL")
    env.setdefault("BBRL_LOG_AGENT_CODE", "WARNING")

    # YENI kosu -> onceki kosudan kalan kumulatif durumu temizle.
    # (Aksi halde epsilon/tur sayaci eski kosudan devam eder ve
    #  "24000 tur egittim" derken aslinda 48000. asamadan baslamis olur.)
    #
    # !!! MODEL DOSYASI DA SILINMELI !!!
    # callbacks.setup() `model_<name>.bin` varsa KOSULSUZ yukler. Model
    # burada silinmezse yeni kosu eski agirliklardan devam eder ama epsilon
    # 1.0'a, buffer sifira doner -- ne temiz baslangic ne duzgun devam.
    # Daha kotusu YONLU: mimari degistiren kollar (hidden boyutu farkli)
    # load'da shape hatasi alip sessizce sifirdan baslar, mimarisi ayni olan
    # KONTROL ise sicak baslar. Yani yanlilik kontrol lehine, rastgele degil.
    # Olculdu: ayni config'i uc kez kosunca `updates` 2381 -> 3283 -> 4250.
    agent_path = REPO / "agent_code" / AGENT_DIR
    for f in (agent_path / f"model_{cfg.name}.bin",
              agent_path / f"progress_{cfg.name}.json",
              agent_path / f"buffer_{cfg.name}.npz",
              agent_path / f"trainlog_{cfg.name}.json"):
        if f.exists():
            f.unlink()

    t0 = time.time()
    stages = []
    for i, stage in enumerate(cfg.curriculum, 1):
        cmd = _stage_cmd(python, cfg, stage, seed)
        if verbose:
            print(f"  [{cfg.name}] asama {i}/{len(cfg.curriculum)}: "
                  f"{stage.get('scenario')} x{stage.get('n_rounds')} "
                  f"vs {stage.get('opponents') or '-'}")
        t1 = time.time()
        p = subprocess.run(cmd, cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)
        dt = time.time() - t1
        if p.returncode != 0:
            # stderr BOS olabilir: surec python'a hic soz vermeden olurse
            # (disaridan kill/Ctrl-C, OOM, native crash) yazacak bir sey
            # bulamayiz. O zaman en azindan cikis kodunu goster -- yoksa
            # hata 'sebepsiz' gorunur ve yanlis teshise goturur.
            out = (p.stderr or "") + (p.stdout or "")
            tail = "\n".join(out.strip().splitlines()[-15:])
            if not tail:
                tail = (f"(cikti yok, cikis kodu {p.returncode} -- surec python"
                        f"'a soz vermeden oldu: kill/Ctrl-C, OOM ya da native "
                        f"crash olabilir)")
            raise RuntimeError(f"[{cfg.name}] asama {i} basarisiz:\n{tail}")
        stages.append({"stage": i, "seconds": round(dt, 1), **stage})

    return {"config": cfg.name, "path": cfg_path,
            "fingerprint": cfg.fingerprint(),
            "wall_s": round(time.time() - t0, 1), "stages": stages}


def train_and_eval(cfg_path: str, card: str = "EK-3", python: str | None = None,
                   seeds=E.DEFAULT_SEEDS, seed: int | None = None) -> dict:
    info = train(cfg_path, python=python, seed=seed, verbose=True)
    cfg = Config.load(cfg_path)
    # Degerlendirme, egitilen config ile AYNI config'i gormeli ki dogru
    # model dosyasini (model_<name>.bin) yuklesin.
    prev = os.environ.get("BBRL_CONFIG")
    os.environ["BBRL_CONFIG"] = str(Path(cfg_path).as_posix())
    try:
        res = E.evaluate_card(AGENT_DIR, card, seeds=seeds, python=python)
    finally:
        if prev is None:
            os.environ.pop("BBRL_CONFIG", None)
        else:
            os.environ["BBRL_CONFIG"] = prev
    info["eval"] = res
    return info


def sweep(cfg_paths, card: str = "EK-3", python: str | None = None,
          workers: int = 4, seeds=E.DEFAULT_SEEDS) -> list[dict]:
    """Birden fazla config'i PARALEL egitir, sonra SIRAYLA degerlendirir.

    Degerlendirme sirali cunku eval'in kendisi zaten cok surecli; ust uste
    binerse cekirdekler icin yarisirlar ve act suresi olcumu bozulur.
    """
    cfg_paths = list(cfg_paths)
    print(f"[sweep] {len(cfg_paths)} config, {workers} paralel egitim\n")

    trained, failed = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(train, p, python, None, True): p for p in cfg_paths}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                info = fut.result()
                trained.append(info)
                print(f"  egitildi: {info['config']} ({info['wall_s']:.0f}s)")
            except Exception as ex:  # noqa: BLE001
                failed.append((p, str(ex)))
                print(f"  BASARISIZ: {p}\n    {str(ex).splitlines()[0]}")

    print(f"\n[sweep] degerlendirme ({card})\n")
    results = []
    for info in sorted(trained, key=lambda d: d["config"]):
        prev = os.environ.get("BBRL_CONFIG")
        os.environ["BBRL_CONFIG"] = str(Path(info["path"]).as_posix())
        try:
            res = E.evaluate_card(AGENT_DIR, card, seeds=seeds, python=python,
                                  verbose=False)
        finally:
            if prev is None:
                os.environ.pop("BBRL_CONFIG", None)
            else:
                os.environ["BBRL_CONFIG"] = prev
        info["eval"] = res
        results.append(info)
        print(f"  {info['config']:<22} {res.out_dir}")

    if failed:
        print(f"\n{len(failed)} config basarisiz.")
    return results


# --------------------------------------------------------------------------
# Cok eğitim seed'i  (C14)
# --------------------------------------------------------------------------
#
# NEDEN: EXP-007'de `e07a_base` 41.67 coin, `m01` ise AYNI config ile farkli
# bir egitim kosusunda 26.12 coin aldi. Egitim run-to-run varyansi bazi
# ablation etkilerinden buyuk. Eval cok-seed'di ama EGITIM tek-seed'di.
#
# DOGRU ISTATISTIK (hiyerarsik):
#   her egitim kosusu -> eval seed'leri uzerinden ortalama -> TEK bir sayi
#   CI bu sayilar (egitim kosulari) uzerinden hesaplanir
# Tum (egitim, eval) ciftlerini havuzlamak YANLIS olur -- ayni egitim
# kosusundan gelen eval'ler bagimsiz degil.

SEED_CFG_DIR = REPO / "configs" / "_seeds"


def make_seed_variant(cfg_path: str, seed: int) -> Path:
    """Config'in bu egitim seed'ine ozel kopyasi.

    Adi degistirmek SART: model dosyasi `model_<name>.bin` oldugu icin ayni
    ad kullanilirsa kosular birbirinin modelini ezer.
    """
    cfg = Config.load(cfg_path)
    base = cfg.name
    cfg.name = f"{base}_s{seed}"
    cfg.seed = seed
    SEED_CFG_DIR.mkdir(parents=True, exist_ok=True)
    return cfg.save(SEED_CFG_DIR / f"{cfg.name}.yaml")


def train_eval_multi(cfg_path: str, train_seeds=(0, 1, 2), card: str = "EK-5",
                     eval_seeds=E.DEFAULT_SEEDS, python: str | None = None,
                     workers: int = 3) -> dict:
    """Bir config'i N egitim seed'i ile egitir, her birini degerlendirir."""
    python = python or E.default_python()
    base_name = Config.load(cfg_path).name
    variants = [(s, make_seed_variant(cfg_path, s)) for s in train_seeds]

    print(f"[multi] {base_name}: {len(variants)} egitim seed'i "
          f"({workers} paralel)")
    ok = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(train, str(p), python, s, False): (s, p)
                for s, p in variants}
        for fut in as_completed(futs):
            s, p = futs[fut]
            try:
                fut.result()
                ok.append((s, p))
                print(f"  seed {s}: egitildi")
            except Exception as ex:  # noqa: BLE001
                print(f"  seed {s}: BASARISIZ -- {str(ex).splitlines()[0]}")

    runs = []
    for s, p in sorted(ok):
        prev = os.environ.get("BBRL_CONFIG")
        os.environ["BBRL_CONFIG"] = str(Path(p).relative_to(REPO).as_posix())
        try:
            res = E.evaluate_card(AGENT_DIR, card, seeds=eval_seeds,
                                  python=python, verbose=False)
        finally:
            if prev is None:
                os.environ.pop("BBRL_CONFIG", None)
            else:
                os.environ["BBRL_CONFIG"] = prev
        runs.append({"train_seed": s, "eval": res})
        print(f"  seed {s}: degerlendirildi")

    # Hiyerarsik toplama: kosu-ici ortalama -> kosular arasi CI
    agg, per_run = {}, {}
    for m in M.METRIC_INFO:
        vals = [r["eval"].agg[m].mean for r in runs
                if r["eval"].agg.get(m) and r["eval"].agg[m].n]
        per_run[m] = vals
        if vals:
            agg[m] = M.mean_ci(vals)
    return {"config": base_name, "path": cfg_path, "card": card,
            "train_seeds": [r["train_seed"] for r in runs],
            "agg": agg, "per_run": per_run, "runs": runs}


def train_eval_multi_many(cfg_paths, train_seeds=(0, 1, 2), card: str = "EK-5",
                          eval_seeds=E.DEFAULT_SEEDS, python: str | None = None,
                          workers: int | None = None) -> list[dict]:
    """Birden fazla config x birden fazla egitim seed'i -- HEPSI tek havuzda.

    Config'leri sirayla kosmak yerine tum (config, seed) ciftlerini ayni
    havuza atar. Her egitim tek thread oldugu icin 9 kosu 24 cekirdekte
    rahat siğar ve duvar suresi ~3x kisalir.

    Degerlendirme yine SIRALI: eval'in kendisi cok surecli, ust uste binerse
    cekirdek icin yarisirlar ve act suresi olcumu bozulur.
    """
    python = python or E.default_python()
    cfg_paths = list(cfg_paths)
    workers = workers or max(1, min(len(cfg_paths) * len(train_seeds),
                                    (os.cpu_count() or 4) - 2))

    jobs = []          # (base_name, cfg_path, seed, variant_path)
    for cp in cfg_paths:
        base = Config.load(cp).name
        for s in train_seeds:
            jobs.append((base, cp, s, make_seed_variant(cp, s)))

    print(f"[multi] {len(cfg_paths)} config x {len(train_seeds)} egitim seed'i "
          f"= {len(jobs)} kosu, {workers} paralel\n")

    t0 = time.time()
    ok = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(train, str(v), python, s, False): (b, cp, s, v)
                for b, cp, s, v in jobs}
        done = 0
        for fut in as_completed(futs):
            b, cp, s, v = futs[fut]
            done += 1
            try:
                fut.result()
                ok.append((b, cp, s, v))
                print(f"  [{done}/{len(jobs)}] {b} seed {s}: egitildi "
                      f"({time.time()-t0:.0f}s)")
            except Exception as ex:  # noqa: BLE001
                print(f"  [{done}/{len(jobs)}] {b} seed {s}: BASARISIZ -- "
                      f"{str(ex).splitlines()[0]}")

    print(f"\n[multi] egitim bitti ({time.time()-t0:.0f}s), degerlendirme...\n")

    by_cfg: dict[str, dict] = {}
    for b, cp, s, v in sorted(ok, key=lambda j: (j[0], j[2])):
        prev = os.environ.get("BBRL_CONFIG")
        os.environ["BBRL_CONFIG"] = str(Path(v).relative_to(REPO).as_posix())
        try:
            res = E.evaluate_card(AGENT_DIR, card, seeds=eval_seeds,
                                  python=python, verbose=False)
        finally:
            if prev is None:
                os.environ.pop("BBRL_CONFIG", None)
            else:
                os.environ["BBRL_CONFIG"] = prev
        d = by_cfg.setdefault(b, {"config": b, "path": cp, "card": card,
                                  "runs": []})
        d["runs"].append({"train_seed": s, "eval": res})
        print(f"  {b} seed {s}: degerlendirildi")

    results = []
    for cp in cfg_paths:                      # girdi sirasini koru
        b = Config.load(cp).name
        d = by_cfg.get(b)
        if not d:
            continue
        agg, per_run = {}, {}
        for m in M.METRIC_INFO:
            vals = [r["eval"].agg[m].mean for r in d["runs"]
                    if r["eval"].agg.get(m) and r["eval"].agg[m].n]
            per_run[m] = vals
            if vals:
                agg[m] = M.mean_ci(vals)
        d["agg"], d["per_run"] = agg, per_run
        d["train_seeds"] = [r["train_seed"] for r in d["runs"]]
        results.append(d)

    # Ozeti diske yaz -- grafikler ve sonradan karsilastirma icin.
    # (Eval dizinleri yalnizca zaman damgasi tasidigi icin config eslesmesi
    # kayboluyordu.)
    save_summary(results)
    return results


SUMMARY_DIR = REPO / "runs" / "summaries"


def save_summary(results, name: str | None = None) -> Path | None:
    """Sweep sonucunu tek bir JSON'a yaz: config -> per_run metrikleri."""
    if not results:
        return None
    import json as _json
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    # deney adi: config adlarinin ortak oneki (e10a_ratio, e10b_... -> e10)
    names = [r["config"] for r in results]
    pref = os.path.commonprefix(names).rstrip("_abcdefgh") or "sweep"
    out = SUMMARY_DIR / f"{name or pref}.json"
    blob = {
        "experiment": name or pref,
        "card": results[0].get("card", ""),
        "configs": [{
            "config": r["config"],
            "path": r.get("path", ""),
            "train_seeds": r.get("train_seeds", []),
            "aggregate": {k: v.as_dict() for k, v in r["agg"].items()},
            "per_run": r["per_run"],
            "eval_dirs": [x["eval"].out_dir for x in r["runs"]],
        } for r in results],
    }
    out.write_text(_json.dumps(blob, indent=2), encoding="utf-8")
    print(f"  ozet -> {out.relative_to(REPO)}")
    return out


def multi_table(results, metrics) -> str:
    """Cok-seed sonuclarini yan yana; ilk config taban."""
    if not results:
        return "(sonuc yok)"
    base = results[0]
    hdr = f"{'config':<24}{'n':>3}" + "".join(f"{m[:17]:>20}" for m in metrics)
    lines = [hdr, "-" * len(hdr)]
    for r in results:
        row = f"{r['config']:<24}{len(r['train_seeds']):>3}"
        for m in metrics:
            ci = r["agg"].get(m)
            row += f"{str(ci):>20}" if ci else f"{'-':>20}"
        lines.append(row)
    lines += ["", f"TABANA GORE FARK (Welch, taban = {base['config']})",
              "-" * len(hdr)]
    for r in results[1:]:
        row = f"{r['config']:<24}{'':>3}"
        for m in metrics:
            a, b = r["per_run"].get(m), base["per_run"].get(m)
            if not a or not b or min(len(a), len(b)) < 2:
                row += f"{'-':>20}"
                continue
            d = M.diff_ci(a, b)
            sig = "*" if (d.lo > 0 or d.hi < 0) else " "
            row += f"{d.mean:>+15.3f}{sig:<5}"
        lines.append(row)
    lines += ["", "* = %95 GA sifiri icermiyor -> ANLAMLI",
              "n = EGITIM seed sayisi (CI bunlar uzerinden; eval seed'leri "
              "kosu icinde ortalandi)"]
    return "\n".join(lines)


def compare_table(results, metrics=("score_per_round", "coins_per_round",
                                    "survival_steps", "suicide_rate",
                                    "invalid_action_rate")) -> str:
    """Sweep sonuclarini yan yana koy; ilk config taban kabul edilir."""
    if not results:
        return "(sonuc yok)"
    base = results[0]
    lines = []
    hdr = f"{'config':<22}" + "".join(f"{m[:16]:>20}" for m in metrics)
    lines += [hdr, "-" * len(hdr)]
    for info in results:
        row = f"{info['config']:<22}"
        for m in metrics:
            ci = info["eval"].agg.get(m)
            row += f"{str(ci):>20}" if ci and ci.n else f"{'-':>20}"
        lines.append(row)
    lines.append("")
    lines.append(f"TABANA GORE FARK (Welch, taban = {base['config']})")
    lines.append("-" * len(hdr))
    for info in results[1:]:
        row = f"{info['config']:<22}"
        for m in metrics:
            a = info["eval"].per_seed.get(m)
            b = base["eval"].per_seed.get(m)
            if not a or not b:
                row += f"{'-':>20}"
                continue
            d = M.diff_ci(a, b)
            sig = "*" if (d.lo > 0 or d.hi < 0) else " "
            row += f"{d.mean:>+14.2f}{sig:<1}{'':>5}" if d.n >= 2 else f"{'-':>20}"
        lines.append(row)
    lines.append("")
    lines.append("* = %95 guven araligi sifiri icermiyor -> fark ANLAMLI")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m bbrl.train_loop")
    p.add_argument("--config", help="tek config egit + degerlendir")
    p.add_argument("--sweep", nargs="*", help="config glob'lari")
    p.add_argument("--card", default="EK-2", choices=sorted(E.EVAL_CARDS))
    p.add_argument("--python", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, default=len(E.DEFAULT_SEEDS))
    p.add_argument("--train-seed", type=int, default=None)
    p.add_argument("--train-seeds", type=int, default=None,
                   help="C14: config basina kac EGITIM seed'i (CI bunlar "
                        "uzerinden hesaplanir)")
    p.add_argument("--metrics", nargs="*", default=None)
    p.add_argument("--agent", default=None,
                   help="egitilecek ajan klasoru (varsayilan: irmak_umut)")
    p.add_argument("--leak-config-to-opponents", action="store_true",
                   help="17 Eylul oncesi HATALI davranisi yeniden uret: config "
                        "rakiplere sizar, bizim-ajan rakipler rastgele oynar. "
                        "Yalnizca eski bir sonucu tekrarlamak icin.")
    a = p.parse_args(argv)
    global AGENT_DIR, LEAK_CONFIG_TO_OPPONENTS
    if a.agent:
        AGENT_DIR = a.agent
        os.environ["BBRL_AGENT_DIR"] = a.agent
    if a.leak_config_to_opponents:
        LEAK_CONFIG_TO_OPPONENTS = True
        print("[train_loop] !! UYARI: config rakiplere SIZDIRILIYOR "
              "(eski hata, bilerek yeniden uretiliyor)")
    print(f"[train_loop] ajan: agent_code/{AGENT_DIR}")

    python = a.python or E.default_python()
    seeds = list(range(a.seeds))

    # --- C14: cok egitim seed'li mod ------------------------------------
    if a.train_seeds:
        paths = []
        for g in (a.sweep or ([a.config] if a.config else ["configs/*.yaml"])):
            paths.extend(sorted(glob.glob(g)))
        if not paths:
            print("config bulunamadi")
            return 1
        metrics = a.metrics or ["score_per_round", "coins_per_round",
                                "crates_per_round", "suicide_rate",
                                "survival_steps"]
        results = train_eval_multi_many(
            paths, tuple(range(a.train_seeds)), a.card, seeds, python,
            workers=a.workers)
        print()
        print(multi_table(results, metrics))
        return 0

    if a.sweep is not None:
        paths = []
        for g in (a.sweep or ["configs/*.yaml"]):
            paths.extend(sorted(glob.glob(g)))
        if not paths:
            print("config bulunamadi")
            return 1
        res = sweep(paths, a.card, python, a.workers, seeds)
        print()
        print(compare_table(res))
        return 0

    if not a.config:
        p.error("--config veya --sweep gerekli")
    info = train_and_eval(a.config, a.card, python, seeds, a.train_seed)
    print()
    print(info["eval"].table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
