"""Adim adim hareket kaydi + patoloji avcisi.

Sahibi: IRMAK (infra)

NEDEN: skor "ajan kotu" der ama NEDEN kotu oldugunu soylemez. Bu arac her
adimi kaydeder ve tipik olumcul davranislari OTOMATIK isaretler:

    FREEZE            tek karede cakili kalma        (bizim m01'in hastaligi)
    OSCILLATION       A-B-A-B gidip gelme
    STEP_INTO_DANGER  guvenli komsu VARKEN tehlikeli kareye adim
    TOWARD_BOMB       kendi patlama hattinda bombaya DOGRU gitme
    STAY_IN_DANGER    tehlikedeyken kacis varken beklemek
    SUICIDE_BOMB      kacisi olmayan bomba birakma
    USELESS_BOMB      hicbir sandiga/rakibe degmeyen bomba
    SELF_TRAP         bomba biraktiktan sonra kacis yollarinin kapanmasi
    DEATH             olum + oncesindeki 8 adimin dokumu

Kullanim:
    python -m bbrl.trace --agent irmak_umut --scenario coin-heaven --rounds 3
    python -m bbrl.trace --agent irmak_umut --scenario classic \
        --opponents rule_based_agent --rounds 5 --csv runs/trace.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import settings as s  # noqa: E402
from environment import BombeRLeWorld, WorldArgs  # noqa: E402

from .gamelogic import (  # noqa: E402
    SAFE, DELTAS, MOVES, blast_coords, bomb_impact, danger_map,
    escape_directions, free_mask,
)

FREEZE_MIN = 8          # bu kadar ardisik ayni kare -> FREEZE
OSC_MIN = 6             # bu kadar adimlik A-B-A-B -> OSCILLATION


# --------------------------------------------------------------------------
# Kayit
# --------------------------------------------------------------------------

def trace_game(agent: str, opponents=None, scenario: str = "classic",
               seed: int = 0, n_rounds: int = 3, verbose: bool = True):
    """Oyunu kosar ve her adim icin bir kayit dondurur."""
    opponents = list(opponents or [])
    tmp = tempfile.mkdtemp(prefix="bbrl_trace_")
    args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                     save_replay=False, replay=None, make_video=False,
                     continue_without_training=True, log_dir=tmp,
                     save_stats=False, match_name=None, seed=seed,
                     silence_errors=False, scenario=scenario)
    world = BombeRLeWorld(args, [(a, False) for a in [agent] + opponents])

    # Ayni klasor birden fazla verilmisse framework isimleri _0.._N yapar
    me = world.agents[0]
    records = []
    try:
        for rnd in range(n_rounds):
            world.new_round()
            n_actions_before = 0
            while world.running:
                if not hasattr(world, "user_input"):
                    world.user_input = None
                if me.dead or me not in world.active_agents:
                    break
                st = world.get_state_for_agent(me)
                if st is None:
                    break

                pos = (int(me.x), int(me.y))
                d = danger_map(st)
                passable = free_mask(st)
                esc = escape_directions(st)
                bomb_here = escape_directions(st, extra_bomb_at=pos)
                crates_hit, enemies_hit = bomb_impact(st, *pos)

                # bu karede tehlikedeyiz -> kac adim sonra oldurur
                danger_here = int(d[pos])
                # guvenli komsu var mi
                safe_moves = []
                for mv in MOVES:
                    dx, dy = DELTAS[mv]
                    nx, ny = pos[0] + dx, pos[1] + dy
                    if passable[nx, ny] and int(d[nx, ny]) >= SAFE:
                        safe_moves.append(mv)

                # Secilen aksiyon hayatta kalinabilir mi? escape_directions
                # zaman-genisletilmis BFS ile "bu ilk hamleden sonra tum
                # tehlikeler gecene kadar hayatta kalinabiliyor mu" der.
                esc5 = {"UP": esc[0], "RIGHT": esc[1], "DOWN": esc[2],
                        "LEFT": esc[3], "WAIT": esc[4]}
                any_ok = bool(esc[:5].any()) or bool(bomb_here.any())

                world.do_step()

                acts = world.replay["actions"][me.name]
                action = acts[-1] if len(acts) > n_actions_before else "?"
                # act() None dondurebiliyor (framework bunu gecersiz sayar);
                # kayitta string olmasi garanti edilir.
                action = "NONE" if action is None else str(action)
                n_actions_before = len(acts)
                events = list(me.events)

                if action == "BOMB":
                    act_ok = bool(bomb_here.any())
                elif action in esc5:
                    act_ok = bool(esc5[action])
                else:
                    act_ok = True          # NONE/? -> degerlendirme disi

                records.append({
                    "round": rnd + 1,
                    "step": world.step,
                    "x": pos[0], "y": pos[1],
                    "action": action,
                    "new_x": int(me.x), "new_y": int(me.y),
                    "moved": int((me.x, me.y) != pos),
                    "danger_here": danger_here if danger_here < SAFE else -1,
                    "n_safe_moves": len(safe_moves),
                    "safe_moves": "|".join(safe_moves),
                    "escape_dirs": int(esc[:4].sum()),
                    "bomb_here_survivable": int(bool(bomb_here.any())),
                    "bomb_here_crates": crates_hit,
                    "bomb_here_enemies": enemies_hit,
                    "can_bomb": int(bool(st["self"][2])),
                    "n_bombs": len(st["bombs"]),
                    "coins_visible": len(st["coins"]),
                    "dest_danger": int(d[me.x, me.y]) if int(d[me.x, me.y]) < SAFE else -1,
                    "action_escape_ok": int(act_ok),
                    "any_escape_ok": int(any_ok),
                    "dist_to_bomb": _dist_to_nearest_bomb(st, pos),
                    "dead": int(me.dead),
                    "events": "|".join(events),
                })
            if verbose:
                n = sum(1 for r in records if r["round"] == rnd + 1)
                last = records[-1] if records else {}
                print(f"  tur {rnd+1}: {n} adim, "
                      f"{'OLDU' if last.get('dead') else 'hayatta'}")
    finally:
        try:
            world.end()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)
    return records


def _dist_to_nearest_bomb(st, pos):
    if not st["bombs"]:
        return -1
    return min(abs(bx - pos[0]) + abs(by - pos[1])
               for (bx, by), _t in st["bombs"])


# --------------------------------------------------------------------------
# Patoloji avi
# --------------------------------------------------------------------------

def find_pathologies(records):
    """Kayitlari tarayip suphe listesi cikarir."""
    issues = []
    by_round = defaultdict(list)
    for r in records:
        by_round[r["round"]].append(r)

    for rnd, rs in sorted(by_round.items()):
        # --- FREEZE: ardisik ayni kare -------------------------------------
        run_start, run_len = 0, 1
        for i in range(1, len(rs) + 1):
            same = (i < len(rs)
                    and (rs[i]["x"], rs[i]["y"]) == (rs[i - 1]["x"], rs[i - 1]["y"]))
            if same:
                run_len += 1
                continue
            if run_len >= FREEZE_MIN:
                acts = Counter(r["action"] for r in rs[run_start:run_start + run_len])
                issues.append({
                    "type": "FREEZE", "round": rnd,
                    "step": rs[run_start]["step"], "length": run_len,
                    "detail": f"({rs[run_start]['x']},{rs[run_start]['y']}) "
                              f"karesinde {run_len} adim; aksiyonlar {dict(acts)}",
                })
            run_start, run_len = i, 1

        # --- OSCILLATION: A-B-A-B ------------------------------------------
        i = 0
        while i < len(rs) - 3:
            p = [(r["x"], r["y"]) for r in rs[i:i + 4]]
            if p[0] == p[2] and p[1] == p[3] and p[0] != p[1]:
                j = i
                while (j + 2 < len(rs)
                       and (rs[j]["x"], rs[j]["y"]) == (rs[j + 2]["x"], rs[j + 2]["y"])):
                    j += 1
                if j - i + 2 >= OSC_MIN:
                    issues.append({
                        "type": "OSCILLATION", "round": rnd,
                        "step": rs[i]["step"], "length": j - i + 2,
                        "detail": f"{p[0]} <-> {p[1]} arasinda {j-i+2} adim",
                    })
                i = j + 1
            else:
                i += 1

        # --- adim bazli kontroller -----------------------------------------
        for i, r in enumerate(rs):
            ev = r["events"]

            if "SUICIDE" not in ev and r["action"] == "BOMB":
                if not r["bomb_here_survivable"]:
                    issues.append({
                        "type": "SUICIDE_BOMB", "round": rnd, "step": r["step"],
                        "length": 1,
                        "detail": f"({r['x']},{r['y']})'de bomba birakti, "
                                  f"GUVENLI KACIS YOK",
                    })
                elif r["bomb_here_crates"] == 0 and r["bomb_here_enemies"] == 0:
                    issues.append({
                        "type": "USELESS_BOMB", "round": rnd, "step": r["step"],
                        "length": 1,
                        "detail": f"({r['x']},{r['y']})'de bomba: 0 sandik, 0 rakip",
                    })

            # tehlikedeyken kacis varken beklemek
            if (r["danger_here"] >= 0 and r["action"] == "WAIT"
                    and r["n_safe_moves"] > 0):
                issues.append({
                    "type": "STAY_IN_DANGER", "round": rnd, "step": r["step"],
                    "length": 1,
                    "detail": f"({r['x']},{r['y']}) {r['danger_here']} adim sonra "
                              f"olumcul; guvenli hamle vardi: {r['safe_moves']}",
                })

            # ONLENEBILIR TUZAK: kurtulus VARDI ama kurtulmayan aksiyon secildi.
            # (Eski "tehlikeli kareye adim" dedektoru gurultuluydu -- iyi ajan
            # patlayacak kareden bilerek GECER; onemli olan gecisin sonunda
            # hayatta kalinip kalinmadigi, tek bir karenin tehlikeli olmasi degil.
            # rule_based_agent eski dedektorde 34 kez isaretleniyordu ama
            # yalnizca 3 kez oldu.)
            if not r["action_escape_ok"] and r["any_escape_ok"]:
                issues.append({
                    "type": "AVOIDABLE_TRAP", "round": rnd,
                    "step": r["step"], "length": 1,
                    "detail": f"({r['x']},{r['y']})'de {r['action']} secti -- "
                              f"bu hamleden kurtulus YOK, ama kurtaran hamle vardi",
                })

            # KENDI patlama hattindayken bombaya DOGRU gitmek
            if (i + 1 < len(rs) and r["moved"] and r["danger_here"] >= 0
                    and r["dist_to_bomb"] > 0
                    and 0 <= rs[i + 1]["dist_to_bomb"] < r["dist_to_bomb"]):
                issues.append({
                    "type": "TOWARD_BOMB", "round": rnd, "step": r["step"],
                    "length": 1,
                    "detail": f"tehlikedeyken bombaya yaklasti "
                              f"({r['dist_to_bomb']} -> {rs[i+1]['dist_to_bomb']} kare)",
                })

            # bomba birakildiktan sonra kacis yollarinin kapanmasi
            if (i > 0 and rs[i - 1]["action"] == "BOMB"
                    and r["escape_dirs"] == 0 and r["danger_here"] >= 0):
                issues.append({
                    "type": "SELF_TRAP", "round": rnd, "step": r["step"],
                    "length": 1,
                    "detail": "bomba birakildiktan sonra guvenli kacis yonu kalmadi",
                })

            if r["dead"]:
                ctx = rs[max(0, i - 7):i + 1]
                path = " -> ".join(f"({c['x']},{c['y']}){str(c['action'])[:1]}"
                                   for c in ctx)
                issues.append({
                    "type": "DEATH", "round": rnd, "step": r["step"], "length": 1,
                    "detail": f"olaylar: {ev} | son 8 adim: {path}",
                })
    return issues


def summarize(records, issues) -> str:
    """Konsol raporu."""
    lines = []
    n_rounds = len({r["round"] for r in records})
    steps = len(records)
    deaths = sum(1 for i in issues if i["type"] == "DEATH")
    acts = Counter(r["action"] for r in records)

    lines.append(f"{steps} adim / {n_rounds} tur | olum {deaths}")
    lines.append("aksiyon dagilimi: " + ", ".join(
        f"{k} %{100*v/max(steps,1):.0f}" for k, v in acts.most_common()))
    lines.append("")

    counts = Counter(i["type"] for i in issues)
    if not counts:
        lines.append("Patoloji bulunamadi.")
        return "\n".join(lines)

    lines.append("PATOLOJILER")
    lines.append("-" * 62)
    # FREEZE/OSCILLATION icin toplam kayip adim da onemli
    for t, n in counts.most_common():
        lost = sum(i["length"] for i in issues if i["type"] == t and i["length"] > 1)
        extra = f"  ({lost} adim kaybi = tumunun %{100*lost/max(steps,1):.0f}'i)" if lost else ""
        lines.append(f"  {t:<18} {n:>4} kez{extra}")
    lines.append("")
    lines.append("ORNEKLER (her turden ilk 3)")
    lines.append("-" * 62)
    for t in counts:
        for i in [x for x in issues if x["type"] == t][:3]:
            lines.append(f"  [{t}] tur {i['round']} adim {i['step']}: {i['detail']}")
    return "\n".join(lines)


def write_csv(records, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    return p


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bbrl.trace",
        description="Ajanin her adimini kaydeder ve patolojileri isaretler")
    p.add_argument("--agent", required=True)
    p.add_argument("--opponents", nargs="*", default=None)
    p.add_argument("--scenario", default="classic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--csv", default=None, help="adim adim CSV cikti yolu")
    a = p.parse_args(argv)

    print(f"[trace] {a.agent} | {a.scenario} | vs {a.opponents or '-'} | "
          f"{a.rounds} tur | seed {a.seed}")
    recs = trace_game(a.agent, a.opponents, a.scenario, a.seed, a.rounds)
    if not recs:
        print("Kayit uretilemedi.")
        return 1
    issues = find_pathologies(recs)
    print()
    print(summarize(recs, issues))
    if a.csv:
        print(f"\n-> {write_csv(recs, a.csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
