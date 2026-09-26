"""Yarim kalmis bir sweep'i KALDIGI YERDEN surdur.

Sahibi: IRMAK

NEDEN GEREKLI: `train_loop.train()` yeni bir kosuya baslarken
`model_/progress_/buffer_/trainlog_` dosyalarinin HEPSINI siler -- ki bu
dogru (bkz. test_multistage.test_fresh_run_discards_stale_model: silinmezse
yeni kosu eski agirliklardan devam eder ve yanlilik YONLU olur, kontrolu
kayirir). Ama bunun bedeli su: 8 saatlik bir sweep durdurulursa, ayni
komutu tekrar calistirmak ilerlemeyi SILER.

Bu modul o bosluğu kapatiyor. Cok-asamali egitim duzeltmesi sayesinde
kumulatif durum zaten diskte:

  progress_<ad>.json  -> kac tur egitildi (epsilon bunu okuyor)
  buffer_<ad>.npz     -> replay buffer
  model_<ad>.bin      -> ag + hedef ag + Adam momentleri + updates sayaci

Yani `main.py play --train 1 --n-rounds <kalan>` dogrudan cagrilirsa kosu
tam kaldigi yerden devam eder. Burada yapilan tek sey: kalani hesaplamak,
temizligi ATLAMAK, sonra normal degerlendirme + ozet yolunu kosturmak.

Kullanim:
    python -m bbrl.resume --checkpoint runs/wave1_checkpoint.json --workers 16
    python -m bbrl.resume --sweep "configs/e2*.yaml" --train-seeds 4 --workers 16
    python -m bbrl.resume --checkpoint runs/wave1_checkpoint.json --eval-only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bbrl import eval as E                      # noqa: E402
from bbrl import metrics as M                   # noqa: E402
from bbrl import train_loop as TL               # noqa: E402
from bbrl.config import Config                  # noqa: E402

def _agent_path():
    return REPO / "agent_code" / TL.AGENT_DIR


def rounds_done(name: str) -> int:
    p = _agent_path() / f"progress_{name}.json"
    try:
        return int(json.loads(p.read_text(encoding="utf-8"))["rounds_done"])
    except (OSError, ValueError, KeyError):
        return 0


def scan(cfg_paths, train_seeds) -> list[dict]:
    """Her (config, seed) icin diskteki duruma bak."""
    out = []
    for cp in cfg_paths:
        cfg = Config.load(cp)
        total = cfg.total_rounds
        for s in train_seeds:
            v = TL.make_seed_variant(cp, s)
            name = Config.load(v).name
            done = rounds_done(name)
            out.append({
                "name": name, "base": cfg.name, "seed": s,
                "variant": str(Path(v).relative_to(REPO).as_posix()),
                "path": str(cp), "done": done, "total": total,
                "remaining": max(0, total - done),
                "model": (_agent_path() / f"model_{name}.bin").is_file(),
                "buffer": (_agent_path() / f"buffer_{name}.npz").is_file(),
            })
    return out


def _continue_one(job: dict, python: str, verbose: bool = True) -> dict:
    """Bir kosuyu kalan tur kadar surdur. TEMIZLIK YAPMAZ."""
    cfg = Config.load(REPO / job["variant"])
    stage = dict(cfg.curriculum[-1])          # son asamanin rakip/senaryosu
    stage["n_rounds"] = job["remaining"]

    env = dict(os.environ)
    env["BBRL_CONFIG"] = job["variant"]
    env["BBRL_CONFIG_AGENT"] = TL.AGENT_DIR       # rakiplere sizmasin
    env.setdefault("BBRL_LOG_GAME", "WARNING")
    env.setdefault("BBRL_LOG_AGENT_WRAPPER", "CRITICAL")
    env.setdefault("BBRL_LOG_AGENT_CODE", "WARNING")

    cmd = TL._stage_cmd(python, cfg, stage, job["seed"])
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                       env=env)
    if p.returncode != 0:
        out = (p.stderr or "") + (p.stdout or "")
        tail = "\n".join(out.strip().splitlines()[-15:])
        if not tail:
            tail = (f"(cikti yok, cikis kodu {p.returncode} -- surec python'a "
                    f"soz vermeden oldu: kill/Ctrl-C, OOM ya da native crash)")
        raise RuntimeError(f"[{job['name']}] surdurme basarisiz:\n{tail}")

    after = rounds_done(job["name"])
    return {**job, "seconds": round(time.time() - t0, 1), "done_after": after}


def resume(cfg_paths, train_seeds, card="EK-2", eval_seeds=E.DEFAULT_SEEDS,
           python=None, workers=None, eval_only=False,
           summary_name=None) -> list[dict]:
    python = python or E.default_python()
    jobs = scan(cfg_paths, train_seeds)
    todo = [j for j in jobs if j["remaining"] > 0]
    workers = workers or max(1, min(len(todo) or 1, (os.cpu_count() or 4) - 2))

    print(f"[resume] {len(jobs)} kosu; {len(jobs)-len(todo)} tanesi zaten bitmis, "
          f"{len(todo)} tanesi surdurulecek")
    for j in todo:
        print(f"    {j['name']:24s} {j['done']:6d}/{j['total']} "
              f"-> {j['remaining']} tur daha")
    kayip = [j["name"] for j in jobs if j["done"] and not (j["model"] and j["buffer"])]
    if kayip:
        print(f"  !! model/buffer EKSIK, bunlar sifirdan baslar: {kayip}")

    if todo and not eval_only:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_continue_one, j, python): j for j in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                j = futs[fut]
                try:
                    r = fut.result()
                    print(f"  [{i}/{len(todo)}] {j['name']}: "
                          f"{r['done_after']}/{j['total']} tur "
                          f"({time.time()-t0:.0f}s)")
                except Exception as ex:  # noqa: BLE001
                    print(f"  [{i}/{len(todo)}] {j['name']}: BASARISIZ -- "
                          f"{str(ex).splitlines()[0]}")
        print(f"\n[resume] egitim bitti ({time.time()-t0:.0f}s)\n")

    # -- degerlendirme: train_loop ile AYNI yol -------------------------
    print("[resume] degerlendirme...\n")
    by_cfg: dict[str, dict] = {}
    for j in sorted(jobs, key=lambda x: (x["base"], x["seed"])):
        if rounds_done(j["name"]) <= 0:
            print(f"  {j['name']}: egitilmemis, atlaniyor")
            continue
        prev = os.environ.get("BBRL_CONFIG")
        os.environ["BBRL_CONFIG"] = j["variant"]
        try:
            res = E.evaluate_card(TL.AGENT_DIR, card, seeds=eval_seeds,
                                  python=python, verbose=False)
        finally:
            if prev is None:
                os.environ.pop("BBRL_CONFIG", None)
            else:
                os.environ["BBRL_CONFIG"] = prev
        d = by_cfg.setdefault(j["base"], {"config": j["base"], "path": j["path"],
                                          "card": card, "runs": []})
        d["runs"].append({"train_seed": j["seed"], "eval": res})
        print(f"  {j['name']}: degerlendirildi "
              f"({res.agg['score_per_round'].mean:.3f})")

    results = []
    for cp in cfg_paths:
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

    TL.save_summary(results, name=summary_name)
    print()
    print(TL.multi_table(results, ["score_per_round", "coins_per_round",
                                   "suicide_rate", "invalid_action_rate"]))
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", help="runs/*_checkpoint.json (config listesi buradan)")
    ap.add_argument("--sweep", nargs="*", help="config glob'lari")
    ap.add_argument("--train-seeds", type=int, default=4)
    ap.add_argument("--card", default="EK-2", choices=sorted(E.EVAL_CARDS))
    ap.add_argument("--seeds", type=int, default=len(E.DEFAULT_SEEDS))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--python", default=None)
    ap.add_argument("--name", default=None, help="ozet dosya adi")
    ap.add_argument("--eval-only", action="store_true",
                    help="egitimi surdurme, sadece degerlendir")
    ap.add_argument("--agent", default=None,
                    help="ajan klasoru (varsayilan: irmak_umut)")
    a = ap.parse_args(argv)
    if a.agent:
        TL.AGENT_DIR = a.agent
        os.environ["BBRL_AGENT_DIR"] = a.agent

    paths: list[str] = []
    if a.sweep:
        for g in a.sweep:
            paths.extend(sorted(glob.glob(g)))
    if a.checkpoint:
        blob = json.loads(Path(a.checkpoint).read_text(encoding="utf-8"))
        bases = []
        for r in blob.get("runs", []):
            b = r["name"].rsplit("_s", 1)[0]
            if b not in bases:
                bases.append(b)
        paths.extend(f"configs/{b}.yaml" for b in bases)
    paths = [p for p in dict.fromkeys(paths) if Path(p).is_file()]
    if not paths:
        print("config bulunamadi (--sweep ya da --checkpoint ver)"); return 1

    resume(paths, list(range(a.train_seeds)), card=a.card,
           eval_seeds=tuple(range(a.seeds)), python=a.python,
           workers=a.workers, eval_only=a.eval_only, summary_name=a.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
