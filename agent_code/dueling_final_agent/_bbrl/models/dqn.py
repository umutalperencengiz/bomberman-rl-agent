"""Deep Q-Network -- sinir agli beyin (Double + Dueling).

Sahibi: IRMAK

Iki govde:
  * duz feature vektoru (F,)      -> MLP
  * duzlem yigini (C, 17, 17)     -> CNN   <- asil hedef, v2_planes ile

Neden Double + Dueling:
  * Double DQN: hedef hesaplarken aksiyonu ONLINE ag secer, degerini TARGET
    ag verir. Duz DQN'in Q degerlerini sistematik olarak SISIRMESI (max
    operatorunun iyimserlik yanliligi) boylece kirilir. Bomberman'de bu
    onemli, cunku sisirilmis Q "bomba at" gibi olumcul aksiyonlari cazip
    gosterebiliyor.
  * Dueling: Q = V(s) + (A(s,a) - mean_a A(s,a)). Durum degeri ile aksiyon
    avantajini ayirir. Bomberman'de cogu karede hangi hamleyi yaptigin pek
    fark etmez ama DURUM cok fark eder (tehlikede misin, degil misin) --
    tam olarak dueling'in kazandigi yapi.

TURNUVA NOTU: inference tek CPU thread'inde, adim basina 0.5 s butcesiyle.
EXP-000'de en yavas referans 1.37 ms -> ~365x marj var, bu boyutta bir CNN
rahat sigar. Kisit hiz degil, deadline'a kadar YAKINSAMA.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import register, _pick


def _pick_device(requested: str | None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _MLP(nn.Module):
    def __init__(self, n_in: int, n_actions: int, hidden=(256, 256),
                 dueling: bool = True):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.dueling = dueling
        if dueling:
            self.adv = nn.Linear(prev, n_actions)
            self.val = nn.Linear(prev, 1)
        else:
            self.head = nn.Linear(prev, n_actions)

    def forward(self, x):
        z = self.body(x)
        if not self.dueling:
            return self.head(z)
        a = self.adv(z)
        return self.val(z) + a - a.mean(dim=1, keepdim=True)


class _CNN(nn.Module):
    """17x17 icin kucuk, padding'li conv govdesi -- boyut korunur."""

    def __init__(self, c_in: int, n_actions: int, channels=(32, 64, 64),
                 dueling: bool = True):
        super().__init__()
        convs, prev = [], c_in
        for c in channels:
            convs += [nn.Conv2d(prev, c, 3, padding=1), nn.ReLU(inplace=True)]
            prev = c
        self.body = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool2d(4)          # (C,17,17) -> (C,4,4)
        n_flat = prev * 16
        self.fc = nn.Sequential(nn.Linear(n_flat, 256), nn.ReLU(inplace=True))
        self.dueling = dueling
        if dueling:
            self.adv = nn.Linear(256, n_actions)
            self.val = nn.Linear(256, 1)
        else:
            self.head = nn.Linear(256, n_actions)

    def forward(self, x):
        z = self.fc(torch.flatten(self.pool(self.body(x)), 1))
        if not self.dueling:
            return self.head(z)
        a = self.adv(z)
        return self.val(z) + a - a.mean(dim=1, keepdim=True)


@register("dqn")
class DQN:
    def __init__(self, feature_shape, n_actions: int, cfg):
        self.shape = tuple(feature_shape)
        self.n_actions = int(n_actions)
        self.gamma = float(cfg.gamma)
        self.double = bool(cfg.hp("double", True))
        self.dueling = bool(cfg.hp("dueling", True))
        self.target_sync = int(cfg.hp("target_sync", 1000))
        self.grad_clip = float(cfg.hp("grad_clip", 10.0))

        # Egitimde GPU, inference/turnuvada CPU. `act` icin ayri cihaz
        # tutmuyoruz -- turnuvada zaten cuda yok, otomatik CPU olur.
        self.device = _pick_device(cfg.hp("device", "auto"))
        torch.manual_seed(cfg.seed)

        def make():
            if len(self.shape) == 1:
                return _MLP(self.shape[0], n_actions,
                            tuple(cfg.hp("hidden", (256, 256))), self.dueling)
            if len(self.shape) == 3:
                return _CNN(self.shape[0], n_actions,
                            tuple(cfg.hp("channels", (32, 64, 64))), self.dueling)
            raise ValueError(f"dqn (F,) veya (C,H,W) bekler, {self.shape} aldi")

        self.net = make().to(self.device)
        self.target = make().to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()

        self.opt = torch.optim.Adam(self.net.parameters(),
                                    lr=float(cfg.hp("lr", 3e-4)))
        self.updates = 0
        self._rng = np.random.default_rng(cfg.seed)

        # Turnuvada tek thread -- multiprocessing yasak, thread thrash'i de onle
        torch.set_num_threads(1)

    # -- sozlesme ----------------------------------------------------------
    @torch.no_grad()
    def act(self, features, eps: float = 0.0, mask=None) -> int:
        x = torch.as_tensor(np.asarray(features, dtype=np.float32),
                            device=self.device).unsqueeze(0)
        q = self.net(x)[0].cpu().numpy()
        return _pick(self._rng, q, mask, self.n_actions, eps)

    def update(self, batch) -> dict:
        # n-step: son alan her transition'in kac adimlik oldugu. Hedefte
        # gamma yerine gamma**n kullanilir. Seyrek odulde (rakip oldurme
        # 5 puan) krediyi n adim geriye tasir.
        states, actions, rewards, next_states, dones, _idx, weights = batch[:7]
        nsteps = batch[7] if len(batch) > 7 else None
        dev = self.device
        s = torch.as_tensor(np.asarray(states, np.float32), device=dev)
        sn = torch.as_tensor(np.asarray(next_states, np.float32), device=dev)
        a = torch.as_tensor(np.asarray(actions, np.int64), device=dev)
        r = torch.as_tensor(np.asarray(rewards, np.float32), device=dev)
        d = torch.as_tensor(np.asarray(dones, np.float32), device=dev)
        w = torch.as_tensor(np.asarray(weights, np.float32), device=dev)

        q = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double:
                # aksiyonu ONLINE ag secer, degerini TARGET ag verir
                a_star = self.net(sn).argmax(dim=1, keepdim=True)
                q_next = self.target(sn).gather(1, a_star).squeeze(1)
            else:
                q_next = self.target(sn).max(dim=1).values
            if nsteps is None:
                disc = self.gamma
            else:
                n = torch.as_tensor(np.asarray(nsteps, np.float32), device=dev)
                disc = self.gamma ** n
            target = r + disc * (1.0 - d) * q_next

        td = target - q
        # Huber: aykiri TD'lerde gradyani sinirlar -> seyrek buyuk odullerde
        # (rakip oldurme +5) egitimi stabil tutar
        loss = (w * Fn.smooth_l1_loss(q, target, reduction="none")).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.opt.step()

        self.updates += 1
        if self.updates % max(1, self.target_sync) == 0:
            self.target.load_state_dict(self.net.state_dict())

        return {"loss": float(loss.item()),
                "q_mean": float(q.mean().item()),
                "td_abs_mean": float(td.abs().mean().item()),
                "td_errors": td.detach().abs().cpu().numpy()}

    # -- kalicilik ---------------------------------------------------------
    def save(self, path) -> None:
        # Optimizer momentleri ve `updates` sayaci da yazilir: mufredat
        # asamalari AYRI SURECLER, bunlar tasinmazsa her asamada Adam
        # sifirdan baslar ve target network senkronu basa doner.
        torch.save({"net": self.net.state_dict(),
                    "target": self.target.state_dict(),
                    "opt": self.opt.state_dict(),
                    "updates": self.updates,
                    "shape": self.shape,
                    "n_actions": self.n_actions,
                    "dueling": self.dueling}, path)

    def load(self, path) -> None:
        # Turnuvada GPU yok -> her zaman CPU'ya map'le, sonra cihaza tasi.
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(blob["net"])
        self.net.to(self.device)
        # Eski checkpoint'lerde bu alanlar yok -> geriye uyumlu kal.
        if "target" in blob:
            self.target.load_state_dict(blob["target"])
        else:
            self.target.load_state_dict(self.net.state_dict())
        self.target.to(self.device)
        if "opt" in blob:
            try:
                self.opt.load_state_dict(blob["opt"])
            except (ValueError, KeyError):
                pass                      # mimari degismis -> temiz optimizer
        self.updates = int(blob.get("updates", 0))
