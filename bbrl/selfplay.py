"""Self-play havuzu -- ajanin anlik kopyasini BAGIMSIZ bir rakip yapar.

Sahibi: IRMAK

`agent_code/pool_<tag>/` altina, `vendor.py`'nin urettigi gibi tam bagimsiz
bir kopya cikarir (kendi config.json + model dosyasi + vendor'lanmis `_bbrl`).
Boylece framework onu sirandan bir rakip gibi yukler:

    python main.py play --agents irmak_umut pool_v1 rule_based_agent ...

!!! EXP-018R DERSI !!!
Rakipleri ASAMALAR ARASINDA degistirmek zarar veriyor (karma rakip sayisi
-0.490, ve egri kuculuyordu -- felaket unutma). Havuzu boyle KULLANMA.
Dogru kullanim: TUM rakipler AYNI ANDA masada, TEK asamada. Ajan her turda
cesitli rakip gorur ama politika bir yone cekilip geri bozulmaz.

Kullanim:
    python -m bbrl.selfplay --snapshot v1
    python -m bbrl.selfplay --list
    python -m bbrl.selfplay --remove v1
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bbrl import vendor as V  # noqa: E402

AGENT_CODE = REPO / "agent_code"
PREFIX = "pool_"


def _assert_model_matches_config(dst: Path) -> str:
    """Havuz klasorundeki config'in ISTEDIGI model dosyasi gercekten var mi.

    callbacks.py model yolunu `model_{cfg.name}.bin` diye kurar. Dosya yoksa
    ajan patlamaz -- sessizce egitilmemis agirliklarla oynar. Bu fonksiyon
    o sessizligi bir istisnaya cevirir.
    """
    import json
    cfgp = dst / "config.json"
    if not cfgp.is_file():
        raise FileNotFoundError(f"{dst.name}: config.json yok")
    name = json.loads(cfgp.read_text(encoding="utf-8")).get("name")
    if not name:
        raise ValueError(f"{dst.name}: config.json icinde 'name' yok")
    want = dst / f"model_{name}.bin"
    if not want.is_file():
        have = sorted(p.name for p in dst.glob("model_*.bin"))
        raise FileNotFoundError(
            f"{dst.name}: config 'name={name}' diyor ama {want.name} YOK. "
            f"Klasorde bulunanlar: {have or '(hicbiri)'}. "
            f"Bu ajan egitilmemis agirliklarla oynardi -- bkz. EXP-019 kazasi.")
    return name


def snapshot(tag: str, agent: str = "irmak_umut", verbose: bool = True) -> Path:
    """Mevcut ajani `agent_code/pool_<tag>/` altina bagimsiz kopyala.

    `vendor.vendor()` zaten importlari `._bbrl`'e cevirip config'i JSON'a
    doner; kopyayi oradan alip agent_code'a tasiyoruz. Havuz ajani egitim
    dosyasi tasimaz (train.py silinir) -- yalnizca oynar.
    """
    dist = V.vendor(agent, verbose=False)
    dst = AGENT_CODE / f"{PREFIX}{tag}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(dist, dst)

    # Havuz ajani ASLA egitilmez -> train.py gereksiz, kafa karistirici.
    (dst / "train.py").unlink(missing_ok=True)

    models = list(dst.glob("model_*.bin"))
    if not models:
        shutil.rmtree(dst)
        raise FileNotFoundError(
            f"{agent} icin egitilmis model yok -- snapshot anlamsiz")

    # !!! EXP-019 KAZASI -- BURADA YAKALANMALIYDI !!!
    # `e19c_league` kolu bir havuz ajanina karsi egitildi ama havuzun
    # config.json'u `e19c_league_s1` diyordu ve o isimli model dosyasi
    # kopyalanmamisti. callbacks.setup() dosyayi bulamayinca RASTGELE
    # agirliklarla oynadi; yalnizca bir WARNING satiri birakti:
    #   "model_e19c_league_s1.bin bulunamadi ... egitilmemis model ile oynanacak"
    # Kimse okumadi. Sonuc: en yuksek ortalamayi veren kol (4.376) aslinda
    # SELF-PLAY DEGILDI. Deney sessizce gecersiz oldu.
    #
    # Bir dosya adi karsilastirmasi bunu engellerdi. Artik engelliyor.
    _assert_model_matches_config(dst)

    if verbose:
        size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
        print(f"  {dst.relative_to(REPO)}: {models[0].name}, {size/1024:.0f} KB")
    return dst


def pool_list() -> list[str]:
    return sorted(d.name for d in AGENT_CODE.glob(f"{PREFIX}*")
                  if (d / "callbacks.py").is_file())


def remove(tag: str) -> bool:
    d = AGENT_CODE / f"{PREFIX}{tag}"
    if not d.is_dir():
        return False
    shutil.rmtree(d)
    return True


def verify(tag: str, python: str | None = None) -> bool:
    """Havuz ajani gercekten EGITILMIS haliyle oynuyor mu.

    Uc kapi (ucu de gecmeden False doner):
      1. config.json'un istedigi model dosyasi var mi (statik)
      2. izole kopyada 2 tur cokmeden kosuyor mu (bbrl gorunmez halde)
      3. kosu sirasinda "egitilmemis model ile oynanacak" UYARISI cikti mi

    (3) EXP-019 kazasinin ta kendisi: ajan kosuyordu, sadece rastgele
    agirliklarla. Sadece (2)'ye bakan bir verify bunu GECIRIR.
    """
    import subprocess
    import tempfile
    python = python or sys.executable
    name = f"{PREFIX}{tag}"

    try:
        cfg_name = _assert_model_matches_config(AGENT_CODE / name)
        print(f"    kapi 1 OK: model_{cfg_name}.bin yerinde")
    except (FileNotFoundError, ValueError) as ex:
        print(f"    kapi 1 BASARISIZ: {ex}")
        return False

    tmp = Path(tempfile.mkdtemp(prefix="bbrl_pool_"))
    try:
        work = tmp / "bomberman_rl"
        work.mkdir()
        for f in ("main.py", "environment.py", "agents.py", "items.py",
                  "events.py", "settings.py", "fallbacks.py", "replay.py"):
            shutil.copy2(REPO / f, work / f)
        shutil.copytree(REPO / "assets", work / "assets")
        (work / "logs").mkdir()
        ac = work / "agent_code"
        ac.mkdir()
        shutil.copytree(AGENT_CODE / name, ac / name)
        shutil.copytree(AGENT_CODE / "random_agent", ac / "random_agent")
        p = subprocess.run(
            [python, "main.py", "play", "--no-gui", "--n-rounds", "2",
             "--agents", name, "random_agent", "--train", "0"],
            cwd=str(work), capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            for line in (p.stderr or p.stdout or "").splitlines()[-10:]:
                print("   ", line)
            return False
        print("    kapi 2 OK: izole kopyada 2 tur cokmeden kostu")

        # Kapi 3: ajan kendi log'una "egitilmemis" diye yazdi mi?
        blob = (p.stderr or "") + (p.stdout or "")
        for lg in (work / "logs").glob("*.log"):
            blob += lg.read_text(encoding="utf-8", errors="replace")
        for lg in (ac / name / "logs").glob("*.log"):
            blob += lg.read_text(encoding="utf-8", errors="replace")
        if "egitilmemis model" in blob or "bulunamadi ve egitim modunda" in blob:
            print("    kapi 3 BASARISIZ: ajan EGITILMEMIS agirliklarla oynadi "
                  "(EXP-019 kazasinin aynisi)")
            return False
        print("    kapi 3 OK: 'egitilmemis model' uyarisi yok")
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m bbrl.selfplay")
    p.add_argument("--snapshot", metavar="TAG", help="mevcut ajani havuza ekle")
    p.add_argument("--agent", default="irmak_umut")
    p.add_argument("--list", action="store_true")
    p.add_argument("--remove", metavar="TAG")
    p.add_argument("--python", default=None)
    a = p.parse_args(argv)

    if a.snapshot:
        print(f"[selfplay] snapshot: {a.agent} -> {PREFIX}{a.snapshot}")
        snapshot(a.snapshot, a.agent)
        print("[selfplay] dogrulama (bbrl gorunmez)...")
        ok = verify(a.snapshot, a.python)
        print(f"[selfplay] {'CALISIYOR' if ok else 'BASARISIZ'}")
        return 0 if ok else 1
    if a.remove:
        print("silindi" if remove(a.remove) else "bulunamadi")
        return 0
    names = pool_list()
    print(f"havuz ({len(names)}): {', '.join(names) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
