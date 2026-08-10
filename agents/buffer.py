"""
buffer.py
=========
Rollout buffer on-policy untuk PPO-Lagrangian.

Menyimpan pengalaman (s, a, r, c, done, log_prob, v_r, v_c)
dari satu rollout penuh, lalu menghitung GAE secara paralel
untuk sinyal reward (A_t^R) dan fairness cost (A_t^C).

Mengimplementasikan:
  - GAE reward  A_t^R (Pers. 3.43)
  - GAE cost    A_t^C (Pers. 3.44)
  - Return terdiskonto G_t^R dan G_t^C
"""

import numpy as np
import torch
from typing import Generator, Tuple, Optional


class RolloutBuffer:
    """
    Buffer pengalaman on-policy untuk satu rollout.

    Setiap entry menyimpan:
        obs      : s_t
        actions  : a_t  (aksi mentah sebelum proyeksi)
        rewards  : r_t  (reward utama — utilitas mustahik)
        costs    : c_t^fair  (fairness cost)
        dones    : flag terminal episode
        log_probs: log π_{θ_old}(a_t|s_t)
        values_r : V^R(s_t)
        values_c : V^C(s_t)

    Setelah rollout selesai, compute_returns_and_advantages()
    mengisi:
        advantages_r : A_t^R  via GAE (Pers. 3.43)
        advantages_c : A_t^C  via GAE (Pers. 3.44)
        returns_r    : G_t^R  target untuk V^R
        returns_c    : G_t^C  target untuk V^C
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        act_dim: int,
        device: str = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.reset()

    # ─────────────────────────────────────────────────────────
    # Reset dan alokasi
    # ─────────────────────────────────────────────────────────

    def reset(self):
        """Alokasi ulang semua array ke nol."""
        n, o, a = self.buffer_size, self.obs_dim, self.act_dim

        self.obs = np.zeros((n, o), dtype=np.float32)
        self.actions = np.zeros((n, a), dtype=np.float32)
        self.rewards = np.zeros(n, dtype=np.float32)
        self.costs = np.zeros(n, dtype=np.float32)
        self.dones = np.zeros(n, dtype=np.float32)
        self.log_probs = np.zeros(n, dtype=np.float32)
        self.values_r = np.zeros(n, dtype=np.float32)
        self.values_c = np.zeros(n, dtype=np.float32)

        # Diisi setelah rollout selesai
        self.advantages_r = np.zeros(n, dtype=np.float32)
        self.advantages_c = np.zeros(n, dtype=np.float32)
        self.returns_r = np.zeros(n, dtype=np.float32)
        self.returns_c = np.zeros(n, dtype=np.float32)

        self.ptr = 0  # pointer posisi saat ini
        self.full = False

    # ─────────────────────────────────────────────────────────
    # Tambah satu langkah pengalaman
    # ─────────────────────────────────────────────────────────

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        cost: float,
        done: bool,
        log_prob: float,
        value_r: float,
        value_c: float,
    ):
        """
        Tambah satu transisi (s_t, a_t, r_t, c_t, done_t, ...) ke buffer.
        """
        if self.ptr >= self.buffer_size:
            raise RuntimeError(
                f"Buffer penuh ({self.buffer_size} steps). "
                f"Panggil compute_returns_and_advantages() dulu."
            )
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.costs[i] = cost
        self.dones[i] = float(done)
        self.log_probs[i] = log_prob
        self.values_r[i] = value_r
        self.values_c[i] = value_c

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    # ─────────────────────────────────────────────────────────
    # GAE — Generalized Advantage Estimation
    # Pers. 3.43–3.44
    # ─────────────────────────────────────────────────────────

    def compute_returns_and_advantages(
        self,
        last_value_r: float = 0.0,
        last_value_c: float = 0.0,
    ):
        """
        Hitung GAE advantage dan return terdiskonto untuk
        reward dan fairness cost secara paralel.

        Persamaan GAE (Pers. 3.43 untuk reward):
          δ_t^R   = r_t + γ·V^R(s_{t+1})·(1-done) - V^R(s_t)
          A_t^R   = Σ_{l≥0} (γλ)^l δ_{t+l}^R

        Persamaan GAE (Pers. 3.44 untuk cost):
          δ_t^C   = c_t + γ·V^C(s_{t+1})·(1-done) - V^C(s_t)
          A_t^C   = Σ_{l≥0} (γλ)^l δ_{t+l}^C

        Args:
            last_value_r : V^R(s_{T+1}) — bootstrap value reward
            last_value_c : V^C(s_{T+1}) — bootstrap value cost
        """
        T = self.ptr
        gamma = self.gamma
        lam = self.gae_lambda

        gae_r, gae_c = 0.0, 0.0

        for t in reversed(range(T)):
            mask = 1.0 - self.dones[t]

            # Nilai state berikutnya
            next_vr = last_value_r if t == T - 1 else self.values_r[t + 1]
            next_vc = last_value_c if t == T - 1 else self.values_c[t + 1]
            next_vr *= mask
            next_vc *= mask

            # TD error — Pers. 3.43
            delta_r = self.rewards[t] + gamma * next_vr - self.values_r[t]
            gae_r = delta_r + gamma * lam * mask * gae_r
            self.advantages_r[t] = gae_r

            # TD error — Pers. 3.44
            delta_c = self.costs[t] + gamma * next_vc - self.values_c[t]
            gae_c = delta_c + gamma * lam * mask * gae_c
            self.advantages_c[t] = gae_c

        # Return = advantage + value (untuk target kritik)
        self.returns_r = self.advantages_r[:T] + self.values_r[:T]
        self.returns_c = self.advantages_c[:T] + self.values_c[:T]

    # ─────────────────────────────────────────────────────────
    # Normalisasi advantage
    # ─────────────────────────────────────────────────────────

    def normalize_advantages(self, eps: float = 1e-8):
        """
        Normalisasi advantage reward ke mean=0, std=1.
        Advantage cost TIDAK dinormalisasi agar skala λ
        tetap konsisten dengan J_C yang terukur.
        """
        T = self.ptr
        adv_r = self.advantages_r[:T]
        self.advantages_r[:T] = (adv_r - adv_r.mean()) / (adv_r.std() + eps)

    # ─────────────────────────────────────────────────────────
    # Generator mini-batch untuk update PPO
    # ─────────────────────────────────────────────────────────

    def get_minibatches(
        self,
        batch_size: int,
        n_epochs: int = 4,
        normalize_adv: bool = True,
    ) -> Generator:
        """
        Yield mini-batch acak dari buffer untuk beberapa epoch.

        Setiap mini-batch berisi tensor siap digunakan pada
        update PPO-Lagrangian (Pers. 3.45).

        Yields:
            dict berisi key: obs, actions, log_probs_old,
                             advantages_r, advantages_c,
                             returns_r, returns_c
        """
        T = self.ptr
        assert T > 0, "Buffer kosong."

        if normalize_adv:
            self.normalize_advantages()

        # Konversi ke tensor sekali saja
        obs_t = torch.FloatTensor(self.obs[:T]).to(self.device)
        actions_t = torch.FloatTensor(self.actions[:T]).to(self.device)
        log_probs_t = torch.FloatTensor(self.log_probs[:T]).to(self.device)
        adv_r_t = torch.FloatTensor(self.advantages_r[:T]).to(self.device)
        adv_c_t = torch.FloatTensor(self.advantages_c[:T]).to(self.device)
        ret_r_t = torch.FloatTensor(self.returns_r[:T]).to(self.device)
        ret_c_t = torch.FloatTensor(self.returns_c[:T]).to(self.device)

        for _ in range(n_epochs):
            indices = np.random.permutation(T)
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                idx = indices[start:end]
                yield {
                    "obs": obs_t[idx],
                    "actions": actions_t[idx],
                    "log_probs_old": log_probs_t[idx],
                    "advantages_r": adv_r_t[idx],
                    "advantages_c": adv_c_t[idx],
                    "returns_r": ret_r_t[idx].unsqueeze(1),
                    "returns_c": ret_c_t[idx].unsqueeze(1),
                }

    # ─────────────────────────────────────────────────────────
    # Statistik buffer
    # ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Ringkasan statistik isi buffer — berguna untuk logging."""
        T = self.ptr
        if T == 0:
            return {}
        return {
            "n_steps": T,
            "mean_reward": float(self.rewards[:T].mean()),
            "mean_cost": float(self.costs[:T].mean()),
            "mean_adv_r": float(self.advantages_r[:T].mean()),
            "mean_adv_c": float(self.advantages_c[:T].mean()),
            "std_adv_r": float(self.advantages_r[:T].std()),
            "frac_done": float(self.dones[:T].mean()),
            "mean_value_r": float(self.values_r[:T].mean()),
            "mean_value_c": float(self.values_c[:T].mean()),
        }
