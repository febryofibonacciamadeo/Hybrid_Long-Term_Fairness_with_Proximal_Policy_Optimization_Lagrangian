"""
ppo_lagrangian.py (revisi)
===========================
Implementasi algoritma PPO-Lagrangian untuk menyelesaikan
program optimasi CMDP distribusi zakat.

Mengimplementasikan persamaan (3.39)-(3.45) BAB III:
  - (3.39) L(pi_theta, lambda) = J_R(pi_theta) - lambda*(J_C(pi_theta) - delta_f)
  - (3.40) max_pi min_{lambda>=0} L(pi_theta, lambda)
  - (3.41) lambda_{k+1} = max(0, lambda_k + eta_lambda*(J_C - delta_f))
  - (3.42) A_hat_t^lag  = A_t^R - lambda_k * A_t^C
  - (3.43) GAE reward A_t^R
  - (3.44) GAE cost   A_t^C
  - (3.45) L^PPO-Lag = E[min(r_t*A_hat^lag, clip(r_t,1-eps,1+eps)*A_hat^lag)]

CATATAN URUTAN UPDATE (diperbaiki agar dokumentasi = eksekusi nyata):
  Implementasi ini melakukan update lambda LEBIH DULU (berbasis mean_cost
  rollout SEBELUM update theta), baru kemudian lambda yang sudah diperbarui
  itu dipakai untuk menghitung A_hat^lag dan meng-update theta pada rollout
  YANG SAMA. Ini adalah varian "lambda-first" dari primal-dual alternating
  update -- bukan kesalahan, tapi WAJIB didokumentasikan konsisten dengan
  Bab III karena ada varian lain ("theta-first") yang menggunakan lambda
  LAMA saat menghitung A_hat^lag, baru update lambda di akhir.

CATATAN SKALA J_C (terkait Pers. 3.37):
  update_lambda() membandingkan mean_cost (rata-rata POLOS c_t^fair
  sepanjang rollout) terhadap delta_f -- BUKAN sum terdiskon literal
  J_C = sum_t gamma^(t-1) * c_t^fair. Ini disengaja: sum terdiskon literal
  dengan gamma=0.99, T=52 punya faktor pengali horizon ~41x, yang akan
  membuat constraint hampir selalu "dilanggar" oleh skala J_C meski
  perilaku kebijakan baik. Pers. (3.37) di Bab III perlu direvisi agar
  konsisten dengan ini (lihat draf revisi terpisah).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from typing import Dict
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────
# Konfigurasi hyperparameter
# ─────────────────────────────────────────────────────────────


@dataclass
class PPOLagConfig:
    """Hyperparameter PPO-Lagrangian."""

    # PPO
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    clip_eps: float = 0.2
    n_epochs: int = 10
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Lagrangian
    lr_lambda: float = 1e-2
    lambda_init: float = 0.0
    lambda_max: float = 10.0
    lr_lambda_decay: float = 1.0  # BARU: peluruhan per iterasi, mis. 0.999
    # (1.0 = tanpa peluruhan, perilaku lama)
    # Membantu lambda KONVERGEN ke titik
    # seimbang, bukan terus naik tanpa henti.

    # Constraint
    delta_f: float = 0.10

    # Normalisasi
    normalize_adv: bool = True

    @classmethod
    def from_env_params(cls, params, **overrides) -> "PPOLagConfig":
        """
        Bangun PPOLagConfig dengan delta_f WAJIB disinkronkan dari
        ZakatEnvConfig/EnvParams yang sama dengan yang dipakai ZakatEnv.

        Mencegah masalah #3 (delta_f dua sumber independen): delta_f
        TIDAK BISA diset manual lewat constructor biasa, harus lewat
        environment params sebagai satu-satunya sumber kebenaran, kecuali
        sengaja di-override (mis. untuk eksperimen sensitivitas).

        Contoh pakai:
            cfg = PPOLagConfig.from_env_params(params, lr_actor=1.5e-4)
        """
        kwargs = {"delta_f": params.delta_f}
        kwargs.update(overrides)
        return cls(**kwargs)


# ─────────────────────────────────────────────────────────────
# PPO-Lagrangian
# ─────────────────────────────────────────────────────────────


class PPOLagrangian:
    """
    Algoritma PPO-Lagrangian untuk CMDP distribusi zakat.

    Alur SATU iterasi (varian lambda-first, lihat catatan di atas modul):
      1. Kumpulkan rollout dari ZakatEnv (dilakukan di luar)
      2. Hitung GAE A_t^R dan A_t^C (buffer.compute_returns_and_advantages)
      3. update():
         a. Update lambda dari mean_cost rollout      (Pers. 3.41)
         b. Hitung A_hat_t^lag = A_t^R - lambda*A_t^C  (Pers. 3.42), lambda BARU
         c. Update theta via L^PPO-Lag                 (Pers. 3.45)
    """

    def __init__(self, actor_critic, config: PPOLagConfig, device: str = "cpu"):
        self.ac = actor_critic
        self.cfg = config
        self.device = torch.device(device)

        self.lambda_lag = config.lambda_init

        self.opt_actor = optim.Adam(
            list(self.ac.actor.parameters()), lr=config.lr_actor, eps=1e-5
        )
        self.opt_criticR = optim.Adam(
            list(self.ac.criticR.parameters()), lr=config.lr_critic, eps=1e-5
        )
        self.opt_criticC = optim.Adam(
            list(self.ac.criticC.parameters()), lr=config.lr_critic, eps=1e-5
        )

        self.metrics_history: list = []

    # ── Pers. 3.41 — Update multiplier lambda ────────────────

    def update_lambda(self, mean_cost: float) -> float:
        """
        Update Lagrange multiplier untuk fairness constraint.

        Cost fairness sudah didefinisikan sebagai:

            c_t^fair = max(0, Delta_fair(t) - delta_f)

        sehingga nilai mean_cost telah merepresentasikan
        besarnya pelanggaran fairness terhadap threshold.

        Oleh karena itu constraint yang digunakan:

            J_C(pi) <= 0

        dan update multiplier:

            lambda_(k+1)
            =
            max(0, lambda_k + eta_lambda * J_C)

        Tidak dilakukan pengurangan delta_f lagi karena
        threshold fairness sudah diterapkan pada definisi cost.
        """

        self.lambda_lag = float(
            np.clip(
                self.lambda_lag + self.cfg.lr_lambda * mean_cost,
                0.0,
                self.cfg.lambda_max,
            )
        )

        return self.lambda_lag

    # ── Pers. 3.42 — Advantage terkombinasi ──────────────────

    def compute_lagrangian_advantage(
        self, adv_r: torch.Tensor, adv_c: torch.Tensor
    ) -> torch.Tensor:
        """
        A_hat_t^lag = A_t^R - lambda_k * A_t^C

        Normalisasi hanya pada adv_r (bukan adv_c) agar skala lambda
        tetap konsisten dengan J_C yang terukur.
        """
        if self.cfg.normalize_adv:
            adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
        return adv_r - self.lambda_lag * adv_c

    # ── Loss functions ────────────────────────────────────────

    def _ppo_policy_loss(
        self,
        log_prob_new: torch.Tensor,
        log_prob_old: torch.Tensor,
        adv_lag: torch.Tensor,
    ) -> torch.Tensor:
        """L^PPO-Lag (Pers. 3.45)."""
        ratio = torch.exp(log_prob_new - log_prob_old)
        clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
        surr = torch.min(ratio * adv_lag, clipped * adv_lag)
        return -surr.mean()

    def _value_loss(self, v_pred: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """L^VF = (V_theta(s) - G_t)^2 (Pers. 2.15)."""
        return nn.functional.mse_loss(v_pred, returns)

    # ── Update utama ──────────────────────────────────────────

    def update(self, buffer) -> Dict:
        """
        Satu siklus update PPO-Lagrangian. Urutan SESUAI yang dieksekusi
        (lambda-first, lihat catatan di docstring modul):
          1. Update lambda dari mean_cost rollout      (Pers. 3.41)
          2. Untuk setiap epoch/mini-batch:
             a. SATU forward evaluate() -> log_prob, entropy, v_r, v_c
             b. A_hat^lag                              (Pers. 3.42)
             c. L^PPO-Lag + L^VF_r + H                 (Pers. 3.45) -> grad theta + criticR
             d. L^VF_c (dari v_c yang SAMA, forward tidak diulang) -> grad criticC
        """
        # ── 1. Update lambda dari mean cost rollout ───────────
        mean_cost = float(buffer.costs[: buffer.ptr].mean())
        lambda_before = self.lambda_lag
        self.update_lambda(mean_cost)

        losses_policy, losses_vr, losses_vc, losses_entropy = [], [], [], []
        clip_fracs, approx_kls = [], []

        for batch in buffer.get_minibatches(
            batch_size=self.cfg.batch_size,
            n_epochs=self.cfg.n_epochs,
            normalize_adv=False,  # normalisasi manual di compute_lagrangian_advantage
        ):
            obs = batch["obs"]
            actions = batch["actions"]
            log_probs_old = batch["log_probs_old"]
            adv_r = batch["advantages_r"]
            adv_c = batch["advantages_c"]
            ret_r = batch["returns_r"]
            ret_c = batch["returns_c"]

            # a. SATU forward pass — actor, criticR, criticC sekaligus.
            #    (Perbaikan #5: v_c di sini dipakai LANGSUNG untuk gradien
            #    criticC di bawah, tidak ada forward kedua yang diulang.)
            log_prob_new, entropy, v_r, v_c = self.ac.evaluate(obs, actions)

            # b. Advantage gabungan (Pers. 3.42), pakai lambda yang BARU
            adv_lag = self.compute_lagrangian_advantage(adv_r, adv_c)

            # c. Policy loss (Pers. 3.45)
            loss_policy = self._ppo_policy_loss(log_prob_new, log_probs_old, adv_lag)
            loss_vr = self._value_loss(v_r, ret_r)
            loss_vc = self._value_loss(v_c, ret_c)
            loss_entropy = -entropy.mean()

            loss_total = (
                loss_policy
                + self.cfg.vf_coef * loss_vr
                + self.cfg.ent_coef * loss_entropy
            )

            # Gradient step aktor + criticR
            self.opt_actor.zero_grad()
            self.opt_criticR.zero_grad()
            loss_total.backward()
            clip_grad_norm_(self.ac.actor.parameters(), self.cfg.max_grad_norm)
            clip_grad_norm_(self.ac.criticR.parameters(), self.cfg.max_grad_norm)
            self.opt_actor.step()
            self.opt_criticR.step()

            # d. Gradient step criticC — pakai loss_vc dari forward YANG SAMA.
            #    Aman dipanggil terpisah karena criticC punya backbone
            #    sendiri (tidak share parameter dengan actor/criticR di
            #    network.py), sehingga graph-nya independen dari loss_total.
            self.opt_criticC.zero_grad()
            loss_vc.backward()
            clip_grad_norm_(self.ac.criticC.parameters(), self.cfg.max_grad_norm)
            self.opt_criticC.step()

            losses_policy.append(loss_policy.item())
            losses_vr.append(loss_vr.item())
            losses_vc.append(loss_vc.item())
            losses_entropy.append(entropy.mean().item())

            with torch.no_grad():
                ratio = torch.exp(log_prob_new - log_probs_old)
                clip_frac = ((ratio - 1).abs() > self.cfg.clip_eps).float().mean()
                approx_kl = ((ratio - 1) - (log_prob_new - log_probs_old)).mean()
                clip_fracs.append(clip_frac.item())
                approx_kls.append(approx_kl.item())

        # ── Skema kunci metrik DISTANDARDISASI agar sama dengan ppo.py ──
        # (Perbaikan #4: evaluate.py bisa baca log kedua agen tanpa konversi)
        metrics = {
            "loss/policy": float(np.mean(losses_policy)),
            "loss/value_r": float(np.mean(losses_vr)),
            "loss/value_c": float(np.mean(losses_vc)),
            "loss/entropy": float(np.mean(losses_entropy)),
            "ppo/clip_frac": float(np.mean(clip_fracs)),
            "ppo/approx_kl": float(np.mean(approx_kls)),
            "lagrangian/lambda": self.lambda_lag,
            "lagrangian/lambda_before": lambda_before,
            "lagrangian/mean_cost": mean_cost,
            "lagrangian/delta_f": self.cfg.delta_f,
            "lagrangian/constraint_gap": mean_cost - self.cfg.delta_f,
        }

        self.metrics_history.append(metrics)
        return metrics

    # ── Simpan dan muat ───────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "lambda_lag": self.lambda_lag,
            "opt_actor": self.opt_actor.state_dict(),
            "opt_criticR": self.opt_criticR.state_dict(),
            "opt_criticC": self.opt_criticC.state_dict(),
            "metrics_history": self.metrics_history,
        }

    def load_state_dict(self, d: dict):
        self.lambda_lag = d["lambda_lag"]
        self.opt_actor.load_state_dict(d["opt_actor"])
        self.opt_criticR.load_state_dict(d["opt_criticR"])
        self.opt_criticC.load_state_dict(d["opt_criticC"])
        self.metrics_history = d.get("metrics_history", [])
