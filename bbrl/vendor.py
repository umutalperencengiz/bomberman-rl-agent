"""Turnuva paketleyicisi -- bagimsiz bir ajan klasoru uretir.

Sahibi: IRMAK (infra)

SORUN: teslimde YALNIZCA ajan klasoru kopyalaniyor (PDF s.11: "Search for
the first directory that contains a callbacks.py. Copy this directory to our
agent_code."). Bizim `callbacks.py` ise `from bbrl import ...` diyor ->
turnuvada ImportError.

COZUM: `dist/<ajan>/` altina bagimsiz bir kopya uret -- ajan dosyalari +
gerekli bbrl modulleri `_bbrl/` alt paketi olarak, importlar yeniden yazilmis.

!!! KAYNAK DOSYALAR DEGISTIRILMEZ. Ilk surum `agent_code/` icindeki
callbacks.py/train.py'yi yerinde yeniden yaziyordu; bu, gelistirme sirasinda
`bbrl`'e yapilan her degisikligi bayatlatiyordu. Artik kaynak `bbrl`'i
import etmeye devam eder, paketleme yalnizca `dist/` uretir.

Kullanim:
    python -m bbrl.vendor --agent irmak_umut
    python -m bbrl.vendor --agent irmak_umut --zip
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIST = REPO / "dist"

#: Ajanin ihtiyac duydugu bbrl modulleri.
#:   callbacks -> features, models, config, gamelogic
#:   config.validate() -> features, rewards, models
#:   train (yalnizca egitimde) -> features, rewards, buffer, gamelogic
#: Egitim-only araclar (eval, metrics, trace, train_loop, vendor) BILEREK
#: disarida -- turnuva paketi kucuk ve bagimsiz kalsin.
RUNTIME_MODULES = [
    "__init__.py",
    "gamelogic.py",
    "features.py",
    "rewards.py",
    "buffer.py",
    "config.py",
    "models/__init__.py",
    "models/tabular_q.py",
    "models/linear_q.py",
    "models/dqn.py",
    # models/__init__.py bu ikisini DOGRUDAN import ediyor (umut_main
    # birlestirmesinden beri). Listede olmasalar paket ACILISTA ImportError
    # ile coker -- sadece Umut'un ajani degil, irmak_umut da.
    "models/umut_linear_q.py",
    "models/umut_dueling_q.py",
]

VENDOR_PKG = "_bbrl"

#: KAYNAKTAN kopyalarken atlanacaklar. `_bbrl` burada cunku kaynak klasorde
#: eski bir vendor kalintisi olabilir -- onu tasimayiz, yenisini uretiriz.
SKIP_FROM_SOURCE = {"__pycache__", "logs", VENDOR_PKG}

#: ZIP'lerken atlanacaklar. `_bbrl` BURADA OLMAMALI -- o, paketin calismasi
#: icin gereken kodun ta kendisi.
#: (Ilk surumde ayni kume ikisi icin de kullaniliyordu ve zip `_bbrl`'siz
#: cikiyordu: 4 dosya, turnuvada garantili ImportError. `check()` de dist/
#: klasorunu test ettigi icin bunu hic gormuyordu -- yani TESLIM EDILECEK
#: SEY hic dogrulanmiyordu. Artik check() dogrudan ZIP'i acip kosuyor.)
SKIP_FROM_ZIP = {"__pycache__", "logs"}


def _rewrite(text: str, prefix: str) -> str:
    text = re.sub(r"\bfrom bbrl\.(\w+) import", rf"from {prefix}.\1 import", text)
    text = re.sub(r"\bfrom bbrl import\b", f"from {prefix} import", text)
    return text


def vendor(agent: str, verbose: bool = True) -> Path:
    """dist/<agent>/ altina bagimsiz kopya uret. Kaynagi DEGISTIRMEZ."""
    agent_dir = REPO / "agent_code" / agent
    if not (agent_dir / "callbacks.py").is_file():
        raise FileNotFoundError(f"{agent_dir}/callbacks.py yok")

    out = DIST / agent
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Aktif config hangi modeli kullaniyor? Yalnizca ONU paketle -- ablation
    # sirasinda biriken diger model_*.bin'ler teslime girmemeli.
    active_model = None
    cfg_file = agent_dir / "config.yaml"
    if cfg_file.is_file():
        try:
            import yaml
            active_model = yaml.safe_load(
                cfg_file.read_text(encoding="utf-8")).get("name")
        except Exception:
            pass

    # 1) ajan dosyalari (importlari ._bbrl'e cevrilerek)
    n_files = 0
    for f in sorted(agent_dir.rglob("*")):
        if not f.is_file() or any(p in SKIP_FROM_SOURCE for p in f.parts):
            continue
        if f.suffix == ".pyc":
            continue
        # EGITIM ARTIFAKTLARI teslime girmez. buffer_*.npz ozellikle onemli:
        # replay buffer'i 500k transition'da ~60 MB ve turnuvada TAMAMEN
        # gereksiz -- ajan yalnizca oynar, ogrenmez.
        if f.name.startswith(("trainlog_", "buffer_", "progress_")):
            continue
        if (active_model and f.name.startswith("model_")
                and f.name != f"model_{active_model}.bin"):
            continue                      # yalnizca aktif model
        rel = f.relative_to(agent_dir)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix == ".py":
            text = _rewrite(f.read_text(encoding="utf-8"), f".{VENDOR_PKG}")
            # Paket YAML'e bagimli OLMASIN: config JSON olarak gidiyor.
            # (Dockerfile `pyaml` kuruyor, `PyYAML` degil -- dolayli geliyor
            #  ama garanti degil. JSON stdlib, risk sifir.)
            text = text.replace('DEFAULT_CONFIG = "config.yaml"',
                                'DEFAULT_CONFIG = "config.json"')
            dst.write_text(text, encoding="utf-8")
        elif f.name == "config.yaml":
            import json as _json
            import yaml as _yaml
            data = _yaml.safe_load(f.read_text(encoding="utf-8"))
            (dst.parent / "config.json").write_text(
                _json.dumps(data, indent=2), encoding="utf-8")
        else:
            shutil.copy2(f, dst)
        n_files += 1

    # 2) bbrl modulleri (paket ici mutlak import -> goreli)
    pkg = out / VENDOR_PKG
    (pkg / "models").mkdir(parents=True)
    for rel in RUNTIME_MODULES:
        src = REPO / "bbrl" / rel
        if not src.is_file():
            raise FileNotFoundError(f"vendor edilecek modul yok: {src}")
        (pkg / rel).write_text(_rewrite(src.read_text(encoding="utf-8"), "."),
                               encoding="utf-8")

    if verbose:
        models = list(out.glob("model_*.bin"))
        print(f"  {out.relative_to(REPO)}: {n_files} ajan dosyasi + "
              f"{len(RUNTIME_MODULES)} bbrl modulu")
        print(f"  model dosyalari: {[m.name for m in models] or 'YOK (!)'}")
    return out


def check(agent: str, python: str | None = None, verbose: bool = True,
          zip_path: str | Path | None = None) -> bool:
    """Paketi `bbrl` GORUNMEZ halde calistirarak dogrula.

    `zip_path` verilirse ZIP ACILIR ve o test edilir -- yani TESLIM EDILECEK
    ARTIFACT'IN KENDISI. (Ilk surum dist/ klasorunu test ediyordu; zip
    `_bbrl` olmadan uretilirken bu fark edilmedi.)

    Temiz bir framework kopyasi kurulur, `bbrl/` oraya KOPYALANMAZ --
    turnuva prosedurunun taklidi (PDF s.11).
    """
    python = python or sys.executable
    src = DIST / agent
    if not src.is_dir():
        print(f"  {src} yok -- once vendor et")
        return False

    tmp = Path(tempfile.mkdtemp(prefix="bbrl_vendor_"))
    try:
        work = tmp / "bomberman_rl"
        work.mkdir()
        for name in ("main.py", "environment.py", "agents.py", "items.py",
                     "events.py", "settings.py", "fallbacks.py", "replay.py"):
            shutil.copy2(REPO / name, work / name)
        shutil.copytree(REPO / "assets", work / "assets")
        (work / "logs").mkdir()
        ac = work / "agent_code"
        ac.mkdir()
        shutil.copytree(REPO / "agent_code" / "random_agent", ac / "random_agent")
        if zip_path is not None:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(ac)
            if not (ac / agent / "callbacks.py").is_file():
                print(f"  ZIP icinde {agent}/callbacks.py yok")
                return False
            n = sum(1 for _ in (ac / agent).rglob("*.py"))
            if verbose:
                print(f"  ZIP acildi: {n} .py dosyasi")
        else:
            shutil.copytree(src, ac / agent)

        # (a) TURNUVA TAKLIDI: PDF s.11'deki prosedurun aynisi --
        #     tek oyun, self.train=False, 3x random_agent. Cokmemeli.
        cmd = [python, "main.py", "play", "--no-gui", "--n-rounds", "3",
               "--agents", agent, "random_agent", "random_agent",
               "random_agent", "--train", "0", "--save-stats", "stats.json"]
        p = subprocess.run(cmd, cwd=str(work), capture_output=True, text=True,
                           timeout=900)
        if p.returncode != 0:
            print("  DOGRULAMA BASARISIZ (turnuva taklidi, cikis kodu):")
            for line in (p.stderr or p.stdout or "").strip().splitlines()[-15:]:
                print("   ", line)
            return False

        # (b) MODEL GERCEKTEN YUKLENDI MI: coin-heaven'da egitilmis politika
        #     coin toplamali. Cokmeme testi bunu gostermez -- egitilmemis bir
        #     model de cokmeden oynar. Bu adim onu ayirir.
        p2 = subprocess.run(
            [python, "main.py", "play", "--no-gui", "--n-rounds", "5",
             "--agents", agent, "--train", "0", "--scenario", "coin-heaven",
             "--save-stats", "coins.json"],
            cwd=str(work), capture_output=True, text=True, timeout=900)
        if p2.returncode != 0:
            print("  DOGRULAMA BASARISIZ (coin-heaven):")
            for line in (p2.stderr or p2.stdout or "").strip().splitlines()[-12:]:
                print("   ", line)
            return False

        errs = []
        logs = ac / agent / "logs"
        if logs.is_dir():
            for lf in logs.glob("*.log"):
                for line in lf.read_text(encoding="utf-8",
                                         errors="replace").splitlines():
                    if "ERROR" in line or "Traceback" in line:
                        errs.append(line)
                    if "bulunamadi ve egitim modunda degiliz" in line:
                        errs.append("MODEL DOSYASI YUKLENMEDI: " + line)
        if errs:
            print("  ajan logunda sorun:")
            for line in errs[:5]:
                print("   ", line)
            return False

        import json
        st = json.loads((work / "stats.json").read_text(encoding="utf-8"))
        me = st["by_agent"].get(agent, {})
        ch = json.loads((work / "coins.json").read_text(encoding="utf-8"))
        mych = ch["by_agent"].get(agent, {})
        coins = float(mych.get("coins", 0)) / max(1, int(mych.get("rounds", 1)))

        if verbose:
            print(f"  turnuva taklidi (3 tur, 3x random): skor {me.get('score', 0)}, "
                  f"adim {me.get('steps', 0)}")
            print(f"  coin-heaven (5 tur): {coins:.1f} coin/tur")

        if coins < 1.0:
            print(f"  MODEL YUKLENMEMIS OLABILIR: coin-heaven'da yalnizca "
                  f"{coins:.1f} coin/tur (egitilmemis ajan seviyesi). "
                  f"model_*.bin paketlendi mi, config adiyla esliyor mu?")
            return False
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_zip(agent: str, out: str | None = None) -> Path:
    """Teslim zip'i -- dist kopyasindan, tek klasor icerecek sekilde."""
    src = DIST / agent
    out_path = Path(out or (REPO / "final-project-agent-code.zip"))
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.rglob("*")):
            if not f.is_file() or any(p in SKIP_FROM_ZIP for p in f.parts):
                continue
            if f.suffix == ".pyc":
                continue
            z.write(f, str(Path(agent) / f.relative_to(src)))
    print(f"  {out_path.name} ({out_path.stat().st_size/1024:.0f} KB)")
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m bbrl.vendor")
    p.add_argument("--agent", default="irmak_umut")
    p.add_argument("--zip", action="store_true", help="teslim zip'i uret")
    p.add_argument("--python", default=None)
    a = p.parse_args(argv)

    print(f"[vendor] {a.agent} -> dist/")
    vendor(a.agent)
    if not list((DIST / a.agent).glob("model_*.bin")):
        print("  UYARI: model_*.bin YOK -- egitilmemis ajan paketleniyor!")

    # ZIP'i HER ZAMAN uret ve ONU dogrula. Teslim edilecek artifact ile
    # test edilen artifact ayni olmali.
    zp = make_zip(a.agent)
    print("[vendor] ZIP dogrulamasi (bbrl gorunmez, turnuva taklidi)...")
    ok = check(a.agent, a.python, zip_path=zp)
    print(f"[vendor] {'ZIP BAGIMSIZ CALISIYOR' if ok else 'BASARISIZ'}")
    if not ok:
        return 1
    if not a.zip:
        zp.unlink(missing_ok=True)
        print("  (--zip verilmedi, gecici zip silindi)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
