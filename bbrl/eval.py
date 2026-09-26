"""Evaluation harness -- tek dogru cetvel.

Sahibi: IRMAK  (plan basta UMUT diyordu; Irmak yazdigi icin guncellendi)

Kurallar (docs/INTERFACE.md 6):
  * Degerlendirme HER ZAMAN orijinal env'de, `--train 0` (self.train=False) ile.
  * Performans gercek oyun skoruyla raporlanir, shaped odulle asla.
  * Tek seed'lik sonuc gurultudur -> her kart cok-seed kosar, CI uretir.

Istatistiksel tasarim:
  Her seed BAGIMSIZ bir orneklem. Bir seed = `n_rounds` turluk bir kosu ve
  o kosunun ortalamasi tek bir sayi verir. CI bu seed-ortalamalari uzerinden
  hesaplanir -> tur-ici korelasyon CI'yi sisirmiyor.

Kullanim:
    python -m bbrl.eval --baselines
    python -m bbrl.eval --agent irmak_umut --card EK-2
    python -m bbrl.eval --agent irmak_umut --opponents rule_based_agent \
        --scenario classic --n-rounds 100 --seeds 10
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from . import metrics as M

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs" / "eval"


# --------------------------------------------------------------------------
# Kart tanimlari -- her model AYNI kartlardan gecer, hep ayni seed'lerle
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Card:
    name: str
    scenario: str
    opponents: tuple[str, ...]
    n_rounds: int
    note: str = ""


#: Kartlarin PDF bolum 4'teki gorevlere esleniyor:
#:   Task 1 (navigasyon, sandiksiz/rakipsiz)      -> EK-3
#:   Task 2 (sandikli, rakipsiz, bomba + kacis)   -> EK-5
#:   Task 3 (peaceful + coin_collector avi)       -> EK-4
#:   Task 4 (rule_based'e karsi, turnuva)         -> EK-2 (birincil), EK-1 (teshis)
EVAL_CARDS: dict[str, Card] = {
    "EK-1": Card("EK-1", "classic", ("rule_based_agent",), 100,
                 "Task 4 teshis -- 1v1 rule_based"),
    "EK-2": Card("EK-2", "classic", ("rule_based_agent",) * 3, 100,
                 "Task 4 -- 3x rule_based, TURNUVA VEKILI, birincil metrik"),
    "EK-3": Card("EK-3", "coin-heaven", (), 50,
                 "Task 1 -- tek basina coin-heaven, navigasyon"),
    "EK-4": Card("EK-4", "classic", ("peaceful_agent", "coin_collector_agent"), 100,
                 "Task 3 -- avlanma"),
    "EK-5": Card("EK-5", "classic", (), 100,
                 "Task 2 -- sandikli, rakipsiz: bomba kullan, kendini oldurme"),
}

DEFAULT_SEEDS = tuple(range(10))


# --------------------------------------------------------------------------
# Yardimcilar
# --------------------------------------------------------------------------

def default_python() -> str:
    """Oyunu kosacak yorumlayici.

    Varsayilan: harness'i calistiran python (`sys.executable`). Boylece
    `<env>/python -m bbrl.eval` dedigin ortam alt sureclere de gecer --
    makineye ozel yol gomulmez, Umut'ta da ayni calisir.

    BBRL_PYTHON ile ezilebilir.

    ORTAM SARTI: kosacak python'da numpy + (torch, eger ajan torch
    kullaniyorsa) bulunmali. `check_python()` bunu dogrular.
    """
    return os.environ.get("BBRL_PYTHON") or sys.executable


def check_python(python: str, need_torch: bool = False) -> list[str]:
    """Alt surec yorumlayicisinda eksik paketleri dondurur (bos = sorun yok)."""
    probe = ("import importlib.util as u;"
             "mods=['numpy']" + ("+['torch']" if need_torch else "") + ";"
             "print(','.join(m for m in mods if u.find_spec(m) is None))")
    try:
        out = subprocess.run([python, "-c", probe], capture_output=True,
                             text=True, timeout=120)
        missing = out.stdout.strip()
        return [m for m in missing.split(",") if m]
    except Exception:  # noqa: BLE001
        return ["<yorumlayici calistirilamadi>"]


def resolve_agent_names(agent_dirs: list[str]) -> list[str]:
    """environment.BombeRLeWorld.setup_agents isimlendirmesini birebir taklit et.

    Ayni klasor birden fazla kez verilirse isimler `<dir>_0`, `<dir>_1` olur.
    Sonuc JSON'unu dogru anahtarla okumak icin gerekli.
    """
    names, seen = [], {}
    total = {d: agent_dirs.count(d) for d in set(agent_dirs)}
    for d in agent_dirs:
        if total[d] > 1:
            names.append(f"{d}_{seen.get(d, 0)}")
        else:
            names.append(d)
        seen[d] = seen.get(d, 0) + 1
    return names


@dataclass
class SeedOutcome:
    seed: int
    ok: bool
    duration_s: float
    error: str = ""
    stats: M.RunStats | None = field(default=None, repr=False)


# --------------------------------------------------------------------------
# Tek kosu
# --------------------------------------------------------------------------

def run_one(agent: str, opponents: list[str], scenario: str, n_rounds: int,
            seed: int, out_dir: Path, python: str,
            timeout_s: float = 3600.0, quiet_logs: bool = True) -> SeedOutcome:
    """Tek seed icin `main.py play` kosar ve istatistigi okur.

    quiet_logs=True ise BBRL_LOG_* ile framework loglari susturulur (settings.py
    bu env degiskenlerini okuyor). Kosu basina ~19 MB game.log + ajan loglari
    yazilmasini engeller. `timeout_violations` metriginin game.log'a ihtiyaci
    oldugu icin LOG_GAME WARNING'de birakilir -- asim uyarisi WARNING seviyede.
    """
    agent_dirs = [agent] + list(opponents)
    names = resolve_agent_names(agent_dirs)
    our_name = names[0]

    seed_dir = out_dir / f"seed_{seed:03d}"
    log_dir = seed_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)      # FileHandler dizini yaratmaz
    stats_path = seed_dir / "stats.json"

    cmd = [
        python, "main.py", "play",
        "--agents", *agent_dirs,
        "--train", "0",
        "--no-gui",
        "--scenario", scenario,
        "--n-rounds", str(n_rounds),
        "--seed", str(seed),
        "--save-stats", str(stats_path),
        "--log-dir", str(log_dir),
        "--match-name", f"{our_name}-s{seed}",
    ]

    env = dict(os.environ)
    # BBRL_CONFIG yalnizca DEGERLENDIRILEN ajan icindir (agent_dirs[0]).
    # Hedef belirtilmezse masadaki bizim-ajan rakipler (irmak_umut, pool_*)
    # onu okuyup kendi klasorlerinde olmayan bir modeli arar ve RASTGELE
    # oynar. Bkz. irmak_umut/callbacks.py::_config_path.
    if env.get("BBRL_CONFIG") and not env.get("BBRL_CONFIG_AGENT"):
        env["BBRL_CONFIG_AGENT"] = agent_dirs[0]
    if quiet_logs:
        # WARNING: timeout asim uyarisi bu seviyede -> metrik korunur.
        env["BBRL_LOG_GAME"] = "WARNING"
        env["BBRL_LOG_AGENT_WRAPPER"] = "CRITICAL"
        env["BBRL_LOG_AGENT_CODE"] = "CRITICAL"

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True,
                              text=True, timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired:
        return SeedOutcome(seed, False, time.time() - t0,
                           f"timeout ({timeout_s}s)")
    dt = time.time() - t0

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        return SeedOutcome(seed, False, dt,
                           f"exit {proc.returncode}\n" + "\n".join(tail))
    if not stats_path.exists():
        return SeedOutcome(seed, False, dt, "stats.json uretilmedi")

    try:
        rs = M.load_run(stats_path, our_name, seed, game_log=log_dir / "game.log")
    except Exception as ex:  # noqa: BLE001
        return SeedOutcome(seed, False, dt, f"{type(ex).__name__}: {ex}")
    return SeedOutcome(seed, True, dt, stats=rs)


# --------------------------------------------------------------------------
# Kart kosumu
# --------------------------------------------------------------------------

@dataclass
class EvalResult:
    agent: str
    card: str
    scenario: str
    opponents: list[str]
    n_rounds: int
    seeds: list[int]
    agg: dict[str, M.MeanCI]
    per_seed: dict[str, list[float]]
    failures: list[tuple[int, str]]
    wall_s: float
    out_dir: str
    # Hangi config ile olculdugu -- BBRL_CONFIG'ten. Dizin adi yalnizca
    # zaman damgasi tasiyor, sonuclari sonradan eslestirmek imkansizdi.
    config: str = ""

    def table(self, baseline: "EvalResult | None" = None) -> str:
        title = (f"{self.agent}  |  {self.card}  |  {self.scenario}  |  "
                 f"vs {', '.join(self.opponents) or '-'}  |  "
                 f"{self.n_rounds} tur x {len(self.seeds)} seed")
        return M.format_table(self.agg, title,
                              baseline.agg if baseline else None)

    def to_json(self) -> dict:
        return {
            "agent": self.agent, "card": self.card, "scenario": self.scenario,
            "opponents": self.opponents, "n_rounds": self.n_rounds,
            "seeds": self.seeds, "wall_s": round(self.wall_s, 1),
            "config": self.config,
            "failures": [{"seed": s, "error": e} for s, e in self.failures],
            "aggregate": {k: v.as_dict() for k, v in self.agg.items()},
            "per_seed": self.per_seed,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return p


def evaluate(agent: str, opponents=None, scenario: str = "classic",
             n_rounds: int = 100, seeds=DEFAULT_SEEDS, card: str = "custom",
             workers: int | None = None, python: str | None = None,
             out_dir: Path | None = None, keep_raw: bool = False,
             quiet_logs: bool = True, verbose: bool = True) -> EvalResult:
    """Bir ajani cok-seed degerlendirir.

    workers: paralel subprocess sayisi. UYARI -- agents.py agent loglarini
    `agent_code/<dir>/logs/` altina SABIT yolla yazar (`--log-dir`'e uymaz),
    yani paralel kosularda o dosya cakisir. Zararsizdir (biz okumuyoruz) ama
    Windows'ta nadiren PermissionError'a yol acabilir; o seed FAIL olur ve
    digerleri devam eder. Sorun cikarsa --workers 1.
    """
    opponents = list(opponents or [])
    seeds = list(seeds)
    python = python or default_python()
    workers = workers or max(1, min(len(seeds), (os.cpu_count() or 4) - 2))

    # !!! KLASOR ADI TEK BASINA BENZERSIZ OLMALI !!!
    # Eskiden `<saniye>_<ajan>_<kart>` idi. Ayni ajanin 8 farkli modeli ayni
    # saniyede degerlendirilince (17 Eylul, Umut'un modelleri, seed 20-29)
    # hepsi AYNI klasore yazdi: result.json en son bitenin oldu ve her seed'in
    # `seed_NNN/stats.json`'u 8 surec tarafindan saniyeler arayla yazilip
    # okundu -- yani loglarin bastigi skorlar da capraz kirlendi. Sonuc: 8
    # farkli model birebir ayni sayiyi verdi ve olcum tamamen cope gitti.
    #
    # Config adi (BBRL_CONFIG'ten) okunabilirlik icin, mikrosaniye + pid
    # benzersizlik icin. exist_ok=False: yine de cakisirsa SESSIZCE
    # paylasmak yerine patlasin.
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        cfg = os.environ.get("BBRL_CONFIG")
        tag = f"_{Path(cfg).stem}" if cfg else ""
        out_dir = RUNS / f"{stamp}_{agent}{tag}_{card}_p{os.getpid()}"
        Path(out_dir).mkdir(parents=True, exist_ok=False)
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[eval] {agent} | {card} | {scenario} | vs "
              f"{opponents or '-'} | {n_rounds} tur x {len(seeds)} seed | "
              f"{workers} paralel")

    t0 = time.time()
    outcomes: list[SeedOutcome] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, agent, opponents, scenario, n_rounds,
                            s, out_dir, python,
                            quiet_logs=quiet_logs): s for s in seeds}
        for i, fut in enumerate(as_completed(futs), 1):
            oc = fut.result()
            outcomes.append(oc)
            if verbose:
                mark = "ok " if oc.ok else "FAIL"
                print(f"  [{i:>2}/{len(seeds)}] seed {oc.seed:<3} {mark} "
                      f"{oc.duration_s:6.1f}s"
                      + ("" if oc.ok else f"  <- {oc.error.splitlines()[0]}"))
    wall = time.time() - t0

    outcomes.sort(key=lambda o: o.seed)
    good = [o.stats for o in outcomes if o.ok and o.stats is not None]
    failures = [(o.seed, o.error) for o in outcomes if not o.ok]

    if not good:
        raise RuntimeError(
            "Hicbir seed basarili olmadi. Ilk hata:\n"
            + (failures[0][1] if failures else "?"))

    agg = M.aggregate(good)
    per_seed = {}
    for name in M.METRIC_INFO:
        vals = []
        for rs in good:
            try:
                vals.append(rs.metric(name))
            except KeyError:
                vals.append(float("nan"))
        per_seed[name] = vals
    per_seed["_seeds"] = [rs.seed for rs in good]

    res = EvalResult(agent, card, scenario, opponents, n_rounds,
                     [rs.seed for rs in good], agg, per_seed, failures,
                     wall, str(out_dir),
                     config=os.environ.get("BBRL_CONFIG", ""))
    res.save(out_dir / "result.json")

    if not keep_raw:
        for o in outcomes:  # ham loglar buyuk -- ozet kaydedildi, gerisi silinir
            shutil.rmtree(out_dir / f"seed_{o.seed:03d}", ignore_errors=True)

    return res


def evaluate_card(agent: str, card: str, seeds=DEFAULT_SEEDS, **kw) -> EvalResult:
    c = EVAL_CARDS[card]
    return evaluate(agent, list(c.opponents), c.scenario, c.n_rounds,
                    seeds, card=c.name, **kw)


# --------------------------------------------------------------------------
# EXP-000: referans tabanlar
# --------------------------------------------------------------------------

BASELINE_AGENTS = ["rule_based_agent", "coin_collector_agent", "random_agent"]


def measure_baselines(cards=("EK-2",), agents=None, seeds=DEFAULT_SEEDS,
                      **kw) -> dict[str, dict[str, EvalResult]]:
    """EXP-000 -- sagladiklari ajanlarin referans sayilari.

    Bu sayilar bir kez olculur, dondurulur ve raporun her grafiginde
    referans cizgi olarak gorunur.
    """
    agents = list(agents or BASELINE_AGENTS)
    out: dict[str, dict[str, EvalResult]] = {}
    for card in cards:
        out[card] = {}
        c = EVAL_CARDS[card]
        for a in agents:
            # Kart rakipleri her ajan icin AYNI kalir -- karsilastirilabilirlik
            # sarti bu. Olculen ajan rakiplerle ayni klasorse (rule_based vs
            # 3x rule_based) framework isimleri _0.._3 diye ayirir ve biz _0'i
            # olcariz; simetri geregi bu "ortalama rule_based" demektir.
            res = evaluate(a, list(c.opponents), c.scenario, c.n_rounds, seeds,
                           card=card, **kw)
            out[card][a] = res
            print()
            print(res.table())
            print()
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bbrl.eval",
        description="Bomberman ajan degerlendirme harness'i (cok-seed + %95 GA)")
    p.add_argument("--agent", help="agent_code/ altindaki klasor adi")
    p.add_argument("--card", choices=sorted(EVAL_CARDS), help="hazir kart")
    p.add_argument("--opponents", nargs="*", default=None)
    p.add_argument("--scenario", default="classic")
    p.add_argument("--n-rounds", type=int, default=100)
    p.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS),
                   help="0..N-1 seed'leri kullan")
    p.add_argument("--seed-list", type=int, nargs="*", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--python", default=None)
    p.add_argument("--keep-raw", action="store_true")
    p.add_argument("--baselines", action="store_true",
                   help="EXP-000: rule_based / coin_collector / random olc")
    p.add_argument("--baseline-cards", nargs="*", default=["EK-2"])
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                   help="iki result.json'u karsilastir (Welch CI)")
    a = p.parse_args(argv)

    if a.compare:
        ja = json.loads(Path(a.compare[0]).read_text(encoding="utf-8"))
        jb = json.loads(Path(a.compare[1]).read_text(encoding="utf-8"))
        print(f"A = {ja['agent']} ({ja['card']})   B = {jb['agent']} ({jb['card']})")
        print(f"{'metrik':<22}{'A-B (95% GA)':>28}{'anlamli?':>12}")
        print("-" * 62)
        for name in M.METRIC_INFO:
            va, vb = ja["per_seed"].get(name), jb["per_seed"].get(name)
            if not va or not vb:
                continue
            d = M.diff_ci(va, vb)
            if d.n < 2:
                continue
            sig = "EVET" if (d.lo > 0 or d.hi < 0) else "hayir"
            print(f"{name:<22}{str(d):>28}{sig:>12}")
        return 0

    seeds = a.seed_list if a.seed_list is not None else list(range(a.seeds))

    if a.baselines:
        measure_baselines(cards=tuple(a.baseline_cards), seeds=seeds,
                          workers=a.workers, python=a.python,
                          keep_raw=a.keep_raw)
        return 0

    if not a.agent:
        p.error("--agent veya --baselines gerekli")

    if a.card:
        res = evaluate_card(a.agent, a.card, seeds, workers=a.workers,
                            python=a.python, keep_raw=a.keep_raw)
    else:
        res = evaluate(a.agent, a.opponents, a.scenario, a.n_rounds, seeds,
                       workers=a.workers, python=a.python,
                       keep_raw=a.keep_raw)
    print()
    print(res.table())
    print()
    if res.failures:
        print(f"UYARI: {len(res.failures)} seed basarisiz:")
        for s, e in res.failures[:3]:
            print(f"  seed {s}: {e.splitlines()[0]}")
    print(f"-> {res.out_dir}\\result.json   ({res.wall_s:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
