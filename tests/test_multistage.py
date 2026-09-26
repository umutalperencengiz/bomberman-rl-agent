"""Cok-asamali egitim sureklilik testi.

Sahibi: IRMAK

NEDEN VAR:
Mufredat asamalari AYRI `main.py` surecleridir. Asamalar arasi tasinmasi
gereken bes sey vardi ve HICBIRI tasinmiyordu:

  1. epsilon programi  -> her asamada kesif 1.0'a sifirlaniyordu  <- en agiri
  2. replay buffer     -> her asamada bosaliyordu
  3. Adam momentleri   -> optimizer sifirdan basliyordu
  4. `updates` sayaci  -> target network senkronu basa donuyordu
  5. trainlog          -> uzerine yaziliyor, yalnizca son asama kaliyordu

`e18b_mixed_opp` (18 asama, 800-2400 turluk bloklar, eps_decay=7200) bu
yuzden 24000 turun tamamini %68'in altina hic inmeyen kesifle gecirdi.
EXP-009 ve EXP-015'in mufredat kollari da ayni sebeple confounded.

Bu test iki asamali bir kosunun, tek asamali esdegeriyle AYNI kumulatif
duruma ulastigini dogrular.

Kosum:
    <env>/python tests/test_multistage.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

AGENT = REPO / "agent_code" / "irmak_umut"
CFG_DIR = REPO / "configs" / "_test"
NAME = "_multistage_test"


def _write_cfg(stages):
    from bbrl.config import Config
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    c = Config(name=NAME, model="dqn", features="v0b_task2",
               rewards="r04_ratio", shaping="p00_none", buffer="uniform",
               buffer_size=20000, curriculum=stages,
               eps_start=1.0, eps_end=0.05, eps_decay_rounds=400,
               gamma=0.95, batch_size=32, train_every=8, symmetry=False,
               hparams={"lr": 3e-4, "device": "cpu", "hidden": [64, 64],
                        "updates_per_round": 2, "save_every_rounds": 50},
               seed=0, notes="cok-asamali sureklilik testi")
    return c.save(CFG_DIR / f"{NAME}.yaml")


def _clean():
    for pat in (f"progress_{NAME}.json", f"buffer_{NAME}.npz",
                f"trainlog_{NAME}.json", f"model_{NAME}.bin"):
        f = AGENT / pat
        if f.exists():
            f.unlink()


def _run(cfg_path, stage):
    env = dict(os.environ)
    env["BBRL_CONFIG"] = str(Path(cfg_path).relative_to(REPO).as_posix())
    env["BBRL_CONFIG_AGENT"] = "irmak_umut"
    env["BBRL_LOG_GAME"] = "CRITICAL"
    env["BBRL_LOG_AGENT_WRAPPER"] = "CRITICAL"
    env["BBRL_LOG_AGENT_CODE"] = "CRITICAL"
    cmd = [sys.executable, "main.py", "play", "--agents", "irmak_umut",
           "--train", "1", "--no-gui", "--scenario", stage["scenario"],
           "--n-rounds", str(stage["n_rounds"]), "--seed", "1"]
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                       env=env, timeout=1800)
    if p.returncode != 0:
        tail = "\n".join((p.stderr or p.stdout or "").splitlines()[-12:])
        raise RuntimeError(f"asama basarisiz:\n{tail}")


def test_two_stages_continue():
    """Iki 150-turluk asama = tek 300-turluk kosu ile ayni kumulatif durum."""
    from bbrl.config import Config
    _clean()
    stages = [{"scenario": "classic", "opponents": [], "n_rounds": 150},
              {"scenario": "classic", "opponents": [], "n_rounds": 150}]
    cfg_path = _write_cfg(stages)
    cfg = Config.load(cfg_path)

    for st in stages:
        _run(cfg_path, st)

    # 1) kumulatif tur sayaci
    prog = json.loads((AGENT / f"progress_{NAME}.json").read_text(encoding="utf-8"))
    got = prog["rounds_done"]
    assert got == 300, f"kumulatif tur {got}, 300 bekleniyordu"

    # 2) epsilon asama sinirinda SICRAMAMALI
    log = json.loads((AGENT / f"trainlog_{NAME}.json").read_text(encoding="utf-8"))
    rounds = log["rounds"]
    assert len(rounds) == 300, f"trainlog {len(rounds)} kayit, 300 bekleniyordu"
    eps = [r["eps"] for r in rounds]
    jumps = [(i, eps[i - 1], eps[i]) for i in range(1, len(eps))
             if eps[i] > eps[i - 1] + 1e-9]
    assert not jumps, (
        f"epsilon {len(jumps)} kez YUKARI sicradi (asama sinirinda sifirlanma): "
        f"{jumps[:3]}")
    assert eps[-1] < eps[0], f"epsilon dusmedi: {eps[0]} -> {eps[-1]}"

    # 3) buffer asama sinirinda BOSALMAMALI
    bufs = [r["buffer"] for r in rounds]
    drops = [(i, bufs[i - 1], bufs[i]) for i in range(1, len(bufs))
             if bufs[i] < bufs[i - 1] * 0.5]
    assert not drops, f"buffer {len(drops)} kez yariya dustu: {drops[:3]}"

    # 4) checkpoint optimizer durumunu tasiyor mu
    import torch
    blob = torch.load(AGENT / f"model_{NAME}.bin", map_location="cpu",
                      weights_only=False)
    for key in ("net", "target", "opt", "updates"):
        assert key in blob, f"checkpoint'te '{key}' yok"
    assert blob["updates"] > 0, "updates sayaci sifir kalmis"

    print(f"  ok  kumulatif tur      300")
    print(f"  ok  epsilon surekli    {eps[0]:.3f} -> {eps[-1]:.3f}, sicrama yok")
    print(f"  ok  buffer surekli     {bufs[0]} -> {bufs[-1]}, bosalma yok")
    print(f"  ok  checkpoint tam     opt + target + updates={blob['updates']}")
    _clean()




# ---------------------------------------------------------------------------
# Regresyon: YENI kosu eski model dosyasini DEVRALMAMALI
# ---------------------------------------------------------------------------

SENTINEL = 999_999


def test_fresh_run_discards_stale_model():
    """`train_loop.train()` yeni kosuda `model_<name>.bin`'i de silmeli.

    NEDEN: callbacks.setup() model dosyasi varsa KOSULSUZ yukler. Model
    silinmezse "yeni" kosu eski agirliklardan devam eder -- ama epsilon 1.0'a,
    buffer sifira donmus olarak. Ne temiz baslangic, ne duzgun devam.

    Daha kotusu YONLU bir yanlilik: mimarisi degisen kollar (farkli `hidden`)
    load'da shape hatasi alip sessizce sifirdan baslar, mimarisi ayni olan
    KONTROL ise sicak baslar. Yani hata tedavi kollarini degil kontrolu
    kayirir -- ablation'i tersine cevirebilir.

    Olculdu (duzeltmeden once): ayni config uc kez kosuldu, `updates`
    sayaci 2381 -> 3283 -> 4250, hic sifirlanmadi.

    Test: gecerli bir checkpoint'e SENTINEL bir `updates` degeri yazip
    train()'i cagiriyoruz. Model devralinsaydi sayac SENTINEL'den BUYUK
    kalirdi; temiz baslangicta cok kucuk olmali.
    """
    import torch
    from bbrl import train_loop as TL

    _clean()
    cfg_path = _write_cfg([{"scenario": "coin-heaven", "opponents": [], "n_rounds": 30}])

    # 1) gercek bir checkpoint uret
    TL.train(cfg_path, python=sys.executable, seed=1, verbose=False)
    mp = AGENT / f"model_{NAME}.bin"
    assert mp.is_file(), "ilk kosu model uretmedi"

    # 2) sayaci isaretle
    blob = torch.load(mp, map_location="cpu", weights_only=False)
    first = int(blob["updates"])
    blob["updates"] = SENTINEL
    torch.save(blob, mp)

    # 3) ayni config'i YENIDEN kos -- temiz baslamali
    TL.train(cfg_path, python=sys.executable, seed=1, verbose=False)
    after = int(torch.load(mp, map_location="cpu",
                           weights_only=False)["updates"])

    assert after < SENTINEL, (
        f"eski model devralindi: updates={after} >= sentinel={SENTINEL}. "
        f"train_loop.train() model_<name>.bin'i silmiyor.")
    assert after < 3 * max(first, 1), (
        f"updates={after}, ilk kosudakinin ({first}) 3 katindan buyuk -- "
        f"birikme var gibi gorunuyor")

    print(f"  ok  bayat model atildi  sentinel={SENTINEL} -> updates={after} "
          f"(ilk kosu {first})")
    _clean()


def main() -> int:
    print("cok-asamali egitim sureklilik testi")
    print("-" * 60)
    try:
        test_two_stages_continue()
        test_fresh_run_discards_stale_model()
    except AssertionError as ex:
        print(f"  FAIL {ex}")
        print("-" * 60); print("BASARISIZ"); return 1
    except Exception as ex:  # noqa: BLE001
        print(f"  ERR  {type(ex).__name__}: {ex}")
        print("-" * 60); print("BASARISIZ"); return 1
    print("-" * 60); print("TUMU GECTI"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
