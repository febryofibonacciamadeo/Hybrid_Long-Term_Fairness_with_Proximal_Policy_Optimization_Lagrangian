"""
network.py
==========
Arsitektur jaringan saraf tiruan untuk PPO-Lagrangian.

Terdiri dari tiga jaringan terpisah:
  - Actor      : π_θ(a|s)  — kebijakan stokastik kontinu
  - CriticR    : V^R(s)    — value function untuk reward
  - CriticC    : V^C(s)    — value function untuk fairness cost

Pemisahan dua kritik diperlukan karena PPO-Lagrangian
membutuhkan estimasi GAE terpisah untuk A_t^R dan A_t^C
(Persamaan 3.43–3.44 BAB III).

Arsitektur menggunakan shared backbone opsional untuk
efisiensi komputasi, dengan head terpisah untuk setiap output.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from typing import Tuple, Optional

# ─────────────────────────────────────────────────────────────
# Inisialisasi bobot — orthogonal init (standar PPO)
# ─────────────────────────────────────────────────────────────


def layer_init(
    layer: nn.Module,
    std: float = np.sqrt(2),
    bias: float = 0.0,
) -> nn.Module:
    """
    Orthogonal initialization untuk lapisan Linear.
    Digunakan secara konsisten di seluruh arsitektur
    agar pelatihan lebih stabil dari awal.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


# ─────────────────────────────────────────────────────────────
# MLP Backbone — shared feature extractor
# ─────────────────────────────────────────────────────────────


class MLPBackbone(nn.Module):
    """
    Shared feature extractor yang digunakan bersama oleh
    aktor dan kritik. Memungkinkan representasi state yang
    lebih kaya dengan biaya komputasi yang lebih rendah.

    Arsitektur: Linear → Tanh → Linear → Tanh
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for i in range(n_layers):
            out_dim = hidden_dim
            layers.append(layer_init(nn.Linear(in_dim, out_dim)))
            layers.append(nn.Tanh())
            in_dim = out_dim

        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# Actor — kebijakan stokastik kontinu π_θ(a|s)
# ─────────────────────────────────────────────────────────────


class Actor(nn.Module):
    """
    Aktor mengimplementasikan kebijakan Gaussian:
      π_θ(a|s) = N(μ_θ(s), σ_θ(s))

    Output μ dan log_σ digunakan untuk:
    - Sampling aksi (eksplorasi selama training)
    - Menghitung log π_θ(a|s) untuk rasio r_t(θ) — Pers. 2.12
    - Menghitung entropy untuk bonus eksplorasi — Pers. 2.16

    Aksi mentah dari aktor akan diproyeksikan ke F_t
    melalui action projection sebelum dieksekusi ke lingkungan.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Backbone bersama
        self.backbone = MLPBackbone(obs_dim, hidden_dim, n_layers)

        # Head μ — std kecil untuk output yang terpusat di awal
        self.mean_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)

        # Log standar deviasi — parameter yang dapat dipelajari
        # Diinisialisasi ke 0 → σ = exp(0) = 1
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs : tensor observasi (batch, obs_dim)
        Returns:
            mean    : mean kebijakan μ_θ(s)
            log_std : log standar deviasi log σ_θ
        """
        features = self.backbone(obs)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        log_std = log_std.expand_as(mean)
        return mean, log_std

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """Kembalikan distribusi Normal dari observasi."""
        mean, log_std = self.forward(obs)
        return Normal(mean, log_std.exp())

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample aksi dan hitung log-probability.

        Returns:
            action   : aksi yang di-sample, shape (batch, act_dim)
            log_prob : log π_θ(a|s), shape (batch,)
            entropy  : entropy distribusi H[π_θ(·|s)], shape (batch,)
        """
        dist = self.get_distribution(obs)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()  # reparameterization trick

        log_prob = dist.log_prob(action).sum(dim=-1)  # Σ log π per dimensi
        entropy = dist.entropy().sum(dim=-1)  # Σ H per dimensi

        return action, log_prob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluasi log-prob dan entropy untuk aksi yang sudah ada.
        Digunakan saat menghitung rasio r_t(θ) pada update PPO.

        Returns:
            log_prob : log π_θ(a|s)
            entropy  : H[π_θ(·|s)]
        """
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ─────────────────────────────────────────────────────────────
# Critic — value function tunggal
# ─────────────────────────────────────────────────────────────


class Critic(nn.Module):
    """
    Value function V(s) untuk satu sinyal (reward atau cost).

    PPO-Lagrangian membutuhkan DUA kritik terpisah:
    - CriticR: V^R(s) untuk sinyal reward r_t
    - CriticC: V^C(s) untuk sinyal fairness cost c_t^fair

    Keduanya menggunakan arsitektur identik tapi dilatih
    dengan target yang berbeda (Pers. 3.43–3.44).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        self.backbone = MLPBackbone(obs_dim, hidden_dim, n_layers)
        self.value_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            value : estimasi nilai V(s), shape (batch, 1)
        """
        features = self.backbone(obs)
        return self.value_head(features)


# ─────────────────────────────────────────────────────────────
# ActorCritic — wrapper seluruh jaringan
# ─────────────────────────────────────────────────────────────


class ActorCritic(nn.Module):
    """
    Wrapper yang menggabungkan Actor, CriticR, dan CriticC
    dalam satu modul untuk kemudahan pengelolaan parameter
    dan pemindahan ke device (CPU/GPU).

    Struktur:
        Actor    → π_θ(a|s)
        CriticR  → V^R(s)   — untuk GAE reward (Pers. 3.43)
        CriticC  → V^C(s)   — untuk GAE cost   (Pers. 3.44)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        device: str = "cpu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)

        self.actor = Actor(obs_dim, act_dim, hidden_dim, n_layers)
        self.criticR = Critic(obs_dim, hidden_dim, n_layers)  # reward value
        self.criticC = Critic(obs_dim, hidden_dim, n_layers)  # cost value

        self.to(self.device)

    # ── Aksi ─────────────────────────────────────────────────

    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Inference: dari observasi numpy ke aksi numpy.
        Digunakan di dalam loop env.step().

        Returns:
            action   : aksi numpy (act_dim,)
            log_prob : skalar float
            entropy  : skalar float
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, entropy = self.actor.get_action(obs_t, deterministic)
        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(entropy.item()),
        )

    # ── Nilai ─────────────────────────────────────────────────

    def get_values(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hitung V^R(s) dan V^C(s) secara bersamaan.

        Returns:
            v_r : reward value, shape (batch, 1)
            v_c : cost value,   shape (batch, 1)
        """
        v_r = self.criticR(obs)
        v_c = self.criticC(obs)
        return v_r, v_c

    # ── Evaluasi untuk update ─────────────────────────────────

    def evaluate(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluasi lengkap untuk satu batch selama PPO update.

        Returns:
            log_prob : log π_θ(a|s), shape (batch,)
            entropy  : H[π], shape (batch,)
            v_r      : V^R(s), shape (batch, 1)
            v_c      : V^C(s), shape (batch, 1)
        """
        log_prob, entropy = self.actor.evaluate_actions(obs, action)
        v_r = self.criticR(obs)
        v_c = self.criticC(obs)
        return log_prob, entropy, v_r, v_c

    # ── Utilitas ──────────────────────────────────────────────

    def count_parameters(self) -> dict:
        """Hitung jumlah parameter per komponen."""

        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "actor": count(self.actor),
            "criticR": count(self.criticR),
            "criticC": count(self.criticC),
            "total": count(self),
        }

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"  Model disimpan → {path}")

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
        print(f"  Model dimuat ← {path}")
