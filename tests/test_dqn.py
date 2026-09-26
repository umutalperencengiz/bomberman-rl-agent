"""DQN testleri -- ozellikle TURNUVA ZAMAN BUTCESI.

Sahibi: IRMAK

Kosum:
    <env>/python tests/test_dqn.py

Turnuva kosullari (PDF s.3 + settings.py):
    tek CPU thread'i, adim basina 0.5 s, asim KUMULATIF ceza getiriyor.
EXP-000'de en yavas referans ajan 1.37 ms/adim. Bizim CNN'in de ayni
mertebede kalmasi gerekiyor -- burada olcuyoruz.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import torch  # noqa: E402

from bbrl import features as F  # noqa: E402
from bbrl.buffer import make_buffer  # noqa: E402
from bbrl.config import Config  # noqa: E402
from bbrl.gamelogic import ACTIONS  # noqa: E402
from bbrl.models import build  # noqa: E402

TIMEOUT_S = 0.5          # settings.TIMEOUT
BUDGET_MS = 50.0         # kendi kendimize koydugumuz tavan: butcenin %10'u


def _cfg(**kw):
    # hparams'i ONCE birlestir: `dict(device="cpu", **kw["hparams"])` bicimi
    # cagirici device verdiginde "multiple values" hatasi veriyor.
    hp = {"lr": 3e-4, "device": "cpu"}
    hp.update(kw.pop("hparams", {}))
    base = dict(name="test", model="dqn", features="v2_planes",
                gamma=0.95, batch_size=32, seed=0, hparams=hp)
    base.update(kw)
    return Config(**base)


def test_shapes_and_learning():
    """MLP ve CNN govdeleri: forward + geriye yayilim calisiyor, loss dusuyor."""
    for fname in ("v0_minimal", "v2_planes"):
        fs = F.get(fname)
        cfg = _cfg(features=fname)
        m = build("dqn", fs.shape, len(ACTIONS), cfg)

        buf = make_buffer("uniform", 2000, fs.shape, seed=0)
        rng = np.random.default_rng(0)
        # Ogrenilebilir oyuncak gorev: odul YALNIZCA aksiyona bagli.
        for _ in range(600):
            s = rng.normal(size=fs.shape).astype(np.float32)
            a = int(rng.integers(len(ACTIONS)))
            buf.push(s, a, rng.normal(size=fs.shape).astype(np.float32),
                     1.0 if a == 0 else -1.0, True)

        first = np.mean([m.update(buf.sample(32))["loss"] for _ in range(5)])
        for _ in range(200):
            m.update(buf.sample(32))
        last = np.mean([m.update(buf.sample(32))["loss"] for _ in range(5)])

        assert np.isfinite(first) and np.isfinite(last), f"{fname}: NaN loss"
        assert last < first, (
            f"{fname}: loss dusmedi ({first:.4f} -> {last:.4f}) -- "
            "ogrenme dongusu bozuk olabilir")

        # ogrendi mi: en iyi aksiyon 0 olmali
        qs = np.mean([m.act(rng.normal(size=fs.shape).astype(np.float32), 0.0)
                      for _ in range(50)])
        print(f"  ok  {fname:<12} loss {first:.4f} -> {last:.4f} "
              f"(ort. secilen aksiyon {qs:.2f}, 0 bekleniyor)")


def test_cpu_inference_budget(n=500):
    """EN KRITIK: tek thread CPU'da act() suresi butceye siğiyor mu."""
    torch.set_num_threads(1)
    fs = F.get("v2_planes")
    cfg = _cfg(features="v2_planes")
    m = build("dqn", fs.shape, len(ACTIONS), cfg)

    rng = np.random.default_rng(0)
    xs = [rng.normal(size=fs.shape).astype(np.float32) for _ in range(n)]

    for x in xs[:20]:            # isinma (ilk cagrilar yavas)
        m.act(x, 0.0)

    times = []
    for x in xs:
        t0 = time.perf_counter()
        m.act(x, 0.0)
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)

    mean, p99, mx = times.mean(), np.percentile(times, 99), times.max()
    params = sum(p.numel() for p in m.net.parameters())
    print(f"  ok  CPU inference (1 thread)  ort {mean:.2f} ms | "
          f"p99 {p99:.2f} ms | maks {mx:.2f} ms | {params/1000:.0f}k parametre")
    print(f"      turnuva limiti {TIMEOUT_S*1000:.0f} ms -> "
          f"marj {TIMEOUT_S*1000/max(mx, 1e-6):.0f}x")

    assert p99 < BUDGET_MS, (
        f"p99 {p99:.1f} ms > kendi tavanimiz {BUDGET_MS} ms. "
        "Ag kucultulmeli veya feature sadelestirilmeli.")


def test_save_load_roundtrip(tmp="runs/_dqn_test.pt"):
    """Kaydet/yukle: ayni girdi ayni aksiyonu vermeli (CPU'ya map dahil)."""
    Path("runs").mkdir(exist_ok=True)
    fs = F.get("v2_planes")
    cfg = _cfg(features="v2_planes")
    m1 = build("dqn", fs.shape, len(ACTIONS), cfg)
    rng = np.random.default_rng(1)
    xs = [rng.normal(size=fs.shape).astype(np.float32) for _ in range(30)]
    before = [m1.act(x, 0.0) for x in xs]

    m1.save(tmp)
    m2 = build("dqn", fs.shape, len(ACTIONS), _cfg(features="v2_planes", seed=99))
    m2.load(tmp)
    after = [m2.act(x, 0.0) for x in xs]

    assert before == after, "kaydet/yukle sonrasi aksiyonlar degisti"
    os.remove(tmp)
    print("  ok  kaydet/yukle roundtrip")


def test_gpu_available_and_trains():
    """GPU varsa: egitim GPU'da calismali (kurulum dogrulamasi)."""
    if not torch.cuda.is_available():
        print("  --  GPU yok, atlandi")
        return
    fs = F.get("v2_planes")
    cfg = _cfg(features="v2_planes", hparams=dict(device="cuda"))
    m = build("dqn", fs.shape, len(ACTIONS), cfg)
    buf = make_buffer("uniform", 500, fs.shape, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        buf.push(rng.normal(size=fs.shape).astype(np.float32),
                 int(rng.integers(len(ACTIONS))),
                 rng.normal(size=fs.shape).astype(np.float32), 1.0, False)
    out = m.update(buf.sample(64))
    assert np.isfinite(out["loss"])
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"  ok  GPU egitimi               {name} sm_{cap[0]}{cap[1]} | "
          f"torch {torch.__version__}")


def main() -> int:
    print("DQN testleri")
    print("-" * 66)
    failed = 0
    for t in (test_shapes_and_learning, test_cpu_inference_budget,
              test_save_load_roundtrip, test_gpu_available_and_trains):
        try:
            t()
        except AssertionError as ex:
            failed += 1
            print(f"  FAIL {t.__name__}\n       {ex}")
        except Exception as ex:  # noqa: BLE001
            failed += 1
            print(f"  ERR  {t.__name__}: {type(ex).__name__}: {ex}")
    print("-" * 66)
    print("TUMU GECTI" if not failed else f"{failed} TEST BASARISIZ")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
