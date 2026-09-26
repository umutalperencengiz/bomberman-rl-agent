"""Config sistemi -- bir model = bir YAML dosyasi.

Sahibi: IRMAK (infra)

NEDEN: proje "en az 2 model" istiyor, biz daha fazlasini denemek istiyoruz.
Modeli KOD yerine KONFIG yaparsak, "7 model denedik" demek 7 dosya yazmak
olur ve ablation matrisi kendiliginden dogar. Ayrica her kosunun tam olarak
neyle yapildigi tek dosyada durur -> rapor icin tekrarlanabilirlik.

Kullanim:
    cfg = Config.load("configs/m01_tabular_q.yaml")
    cfg.hp("lr", 1e-3)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """Bir egitim kosusunun TAM tanimi."""

    name: str
    model: str = "tabular_q"          # bbrl.models.REGISTRY anahtari
    features: str = "v0_minimal"      # bbrl.features.FEATURE_SETS anahtari
    rewards: str = "r01_minimal"      # bbrl.rewards.REWARD_SCHEMES anahtari
    shaping: str = "p00_none"         # bbrl.rewards.POTENTIAL_SCHEMES anahtari
    shaping_gamma: float = 1.0        # bkz. rewards.py -- bilincli olarak 1.0
    buffer: str = "uniform"           # uniform | prioritized
    buffer_size: int = 100_000

    # Mufredat: sirayla kosulacak asamalar.
    #   {scenario, opponents, n_rounds}
    curriculum: list[dict] = field(default_factory=lambda: [
        {"scenario": "coin-heaven", "opponents": [], "n_rounds": 2000},
    ])

    # Kesif
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_rounds: int = 1000

    gamma: float = 0.95
    batch_size: int = 64
    train_every: int = 4              # kac adimda bir update
    n_step: int = 1                   # n-step return (1 = klasik)
    symmetry: bool = False            # D4 augmentation

    # act() sirasinda GECERSIZ aksiyonlari maskele (duvara giris, bombasiz BOMB).
    # Olculdu: gonderilen ajanin gecersiz aksiyon orani EK-1'de 0.121,
    # EK-5'te 0.124, EK-3'te 0.297 -- coin-heaven'da 400 adimin ~120'si
    # duvara carpmakla geciyor. Varsayilan KAPALI: acmak politikayi
    # degistirir, o yuzden A/B ile olculmeden gonderilmez.
    action_mask: bool = False

    hparams: dict = field(default_factory=dict)
    seed: int = 0
    notes: str = ""

    # -- yardimcilar -------------------------------------------------------
    def hp(self, key: str, default=None):
        return self.hparams.get(key, default)

    def epsilon(self, round_idx: int) -> float:
        """Lineer eps azaltma. round_idx 0'dan baslar."""
        if self.eps_decay_rounds <= 0:
            return self.eps_end
        frac = min(1.0, round_idx / float(self.eps_decay_rounds))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    @property
    def total_rounds(self) -> int:
        return sum(int(st.get("n_rounds", 0)) for st in self.curriculum)

    def fingerprint(self) -> str:
        """Config'in icerik hash'i -- deney kaydinda kosuyu tanimlar."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]

    # -- I/O ---------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "Config":
        p = Path(path)
        if not p.is_absolute():
            # Ajan callback'leri calisirken cwd AJAN KLASORUDUR (agents.py
            # os.chdir yapar) -> once cwd'ye gore bak, sonra repo koku.
            for cand in (Path.cwd() / p, REPO / p):
                if cand.is_file():
                    p = cand
                    break
            else:
                raise FileNotFoundError(
                    f"config bulunamadi: {path!r}. Bakilan yerler: "
                    f"{Path.cwd() / Path(path)} ve {REPO / Path(path)}")
        text = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            if yaml is None:
                raise RuntimeError("pyyaml kurulu degil: pip install pyyaml")
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"{p.name}: bilinmeyen alan(lar) {sorted(unknown)}. "
                f"Gecerli alanlar: {sorted(known)}")
        return cls(**data)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        if p.suffix in (".yaml", ".yml") and yaml is not None:
            p.write_text(yaml.safe_dump(data, sort_keys=False,
                                        allow_unicode=True), encoding="utf-8")
        else:
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return p

    def validate(self) -> None:
        """Kayitli isimleri dogrula -- egitim baslamadan hata versin."""
        from . import features as F
        from . import rewards as R
        from . import models as Mo

        F.get(self.features)
        if self.rewards not in R.REWARD_SCHEMES:
            raise KeyError(f"bilinmeyen odul semasi {self.rewards!r}. "
                           f"Mevcut: {sorted(R.REWARD_SCHEMES)}")
        if self.shaping not in R.POTENTIAL_SCHEMES:
            raise KeyError(f"bilinmeyen potansiyel semasi {self.shaping!r}. "
                           f"Mevcut: {sorted(R.POTENTIAL_SCHEMES)}")
        Mo.get(self.model)
        if self.buffer not in ("uniform", "prioritized", "per"):
            raise KeyError(f"bilinmeyen buffer {self.buffer!r}")
        for i, st in enumerate(self.curriculum):
            if "scenario" not in st:
                raise KeyError(f"curriculum[{i}]: 'scenario' eksik")
