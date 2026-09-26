"""BBRL_CONFIG'in rakip ajanlara sizmadigini dogrular.

Sahibi: IRMAK

NEDEN VAR: BBRL_CONFIG bir ortam degiskeni, yani `main.py` surecindeki HER
ajana miras kalir. Egitilen ajan icin set edilen config'i masadaki bir
bizim-ajan rakip (irmak_umut, pool_*) de okursa, o isimde bir modeli KENDI
klasorunde arar, bulamaz ve RASTGELE AGIRLIKLARLA oynar. Tek iz bir WARNING:

    model_<egitim-kosusunun-adi>.bin bulunamadi ... egitilmemis model ile oynanacak!

Bu iki deneyi sessizce gecersiz kildi (EXP-019 e19c_league, Dalga 1
e19d_selfplay) ve Umut'un umut_dueling mufredatinin 2. ve 3. asamasini da
vuruyordu: o asamalarda rakip olan irmak_umut rastgele oynuyordu.

TEST IKI YONLU: yalnizca "duzeltmeyle uyari yok" diye bakmak yetmez -- test
hatayi zaten goremiyorsa da gecer. O yuzden once hatayi YENIDEN URETIYORUZ
(hedef belirtilmeden uyari CIKMALI), sonra duzeltmeyi dogruluyoruz.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "agent_code" / "irmak_umut" / "logs" / "irmak_umut.log"
WARN = "egitilmemis model ile oynanacak"


def _play(scope: str | None) -> str:
    """umut_dueling_agent egitilirken irmak_umut rakip; irmak_umut log'unu dondur."""
    env = dict(os.environ)
    env["BBRL_CONFIG"] = "configs/umut_dueling.yaml"
    env.pop("BBRL_CONFIG_AGENT", None)
    if scope is not None:
        env["BBRL_CONFIG_AGENT"] = scope
    env["BBRL_LOG_AGENT_CODE"] = "WARNING"
    env["BBRL_LOG_GAME"] = "CRITICAL"
    env["BBRL_LOG_AGENT_WRAPPER"] = "CRITICAL"
    if LOG.exists():
        LOG.unlink()
    cmd = [sys.executable, "main.py", "play", "--no-gui", "--n-rounds", "1",
           "--agents", "umut_dueling_agent", "irmak_umut",
           "--train", "1", "--scenario", "classic", "--seed", "3"]
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                       env=env, timeout=600)
    assert p.returncode == 0, (p.stderr or p.stdout)[-1500:]
    return LOG.read_text(encoding="utf-8", errors="replace") if LOG.exists() else ""


def _cleanup():
    d = REPO / "agent_code" / "umut_dueling_agent"
    for pat in ("model_umut_dueling.bin", "progress_umut_dueling.json"):
        f = d / pat
        if f.exists():
            f.unlink()


def test_bug_reproduces_without_scope():
    """Hedef YOKKEN rakip sizan config'i okur ve rastgele oynar (eski hata)."""
    log = _play(scope=None)
    assert WARN in log, (
        "hedef belirtilmeden bile uyari cikmadi -- ya hata baska yerde "
        "duzeltilmis ya da bu test hatayi GOREMIYOR (o zaman asagidaki "
        "'duzeltme calisiyor' sonucu da anlamsiz olur)")
    line = next(l for l in log.splitlines() if WARN in l)
    assert "model_umut_dueling.bin" in line, line
    print(f"  ok  hata yeniden uretildi  irmak_umut: "
          f"'{line.split('WARNING:')[-1].strip()[:60]}...'")


def test_scope_keeps_opponent_on_its_own_model():
    """Hedef egitilen ajansa, rakip KENDI config'ini ve modelini kullanir."""
    log = _play(scope="umut_dueling_agent")
    assert WARN not in log, (
        f"rakip hala egitilmemis oynuyor:\n"
        f"{[l for l in log.splitlines() if WARN in l]}")
    print("  ok  kapsam calisiyor     irmak_umut kendi modelini yukledi, uyari yok")


def main() -> int:
    print("BBRL_CONFIG kapsam testi")
    print("-" * 60)
    try:
        test_bug_reproduces_without_scope()
        test_scope_keeps_opponent_on_its_own_model()
    except AssertionError as ex:
        print(f"  FAIL {ex}")
        print("-" * 60); print("BASARISIZ"); return 1
    except Exception as ex:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print("-" * 60); print("BASARISIZ"); return 1
    finally:
        _cleanup()
    print("-" * 60); print("TUMU GECTI"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
