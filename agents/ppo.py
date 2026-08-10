"""
ppo_baseline.py (revisi)
=========================
PPO vanilla sebagai baseline pembanding HLTF.

Identik dengan PPO-Lagrangian tapi:
  - Tidak ada multiplier lambda
  - Advantage = A_t^R saja (tanpa A_t^C)
  - Tidak ada pembaruan lambda
  - Hanya ada satu kritik (V^R), tidak ada V^C

Perbaikan #4: skema kunci metrik yang dikembalikan update() DISAMAKAN
persis dengan ppo_lagrangian.py (loss/value_r, loss/value_c, dst),
supaya evaluate.py bisa membandingkan log kedua agen tanpa konversi
manual. Field yang tidak relevan untuk baseline (loss/value_c, lambda,
constraint_gap) diisi nilai netral (0.0) bukan dihilangkan.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from typing import Dict
from dataclasses import dataclass


@dataclass
class PPOBaselineConfig:
    """Hyperparameter PPO vanilla baseline."""

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
    normalize_adv: bool = True

    # Disimpan HANYA untuk simetri logging dengan PPOLagConfig
    # (baseline tidak menggunakan ini untuk apa pun secara fungsional).
    delta_f: float = 0.10

    @classmethod
    def from_env_params(cls, params, **overrides) -> "PPOBaselineConfig":
        """Sinkronkan delta_f dengan environment, sama seperti PPOLagConfig."""
        kwargs = {"delta_f": params.delta_f}
        kwargs.update(overrides)
        return cls(**kwargs)


class PPOBaseline:
    """
    PPO vanilla tanpa fairness constraint.

    Digunakan sebagai baseline untuk evaluasi komparatif terhadap
    PPO-Lagrangian dalam empat metrik (Bab III - Evaluation):
      1. Cumulative reward
      2. Deviasi LBR antarwilayah
      3. Constraint violation rate
      4. Konvergensi kebijakan
    """

    def __init__(self, actor_critic, config: PPOBaselineConfig, device: str = "cpu"):
        self.ac = actor_critic
        self.cfg = config
        self.device = torch.device(device)

        self.opt_actor = optim.Adam(
            list(self.ac.actor.parameters()), lr=config.lr_actor, eps=1e-5
        )
        self.opt_critic = optim.Adam(
            list(self.ac.criticR.parameters()), lr=config.lr_critic, eps=1e-5
        )

        self.metrics_history: list = []

    def update(self, buffer) -> Dict:
        """
        Update PPO vanilla — hanya menggunakan sinyal reward.
        Advantage: A_hat_t = A_t^R (tanpa Lagrangian).
        """
        losses_policy, losses_value, losses_entropy = [], [], []
        clip_fracs, approx_kls = [], []

        for batch in buffer.get_minibatches(
            batch_size=self.cfg.batch_size,
            n_epochs=self.cfg.n_epochs,
            normalize_adv=self.cfg.normalize_adv,
        ):
            obs = batch["obs"]
            actions = batch["actions"]
            log_probs_old = batch["log_probs_old"]
            adv_r = batch["advantages_r"]
            ret_r = batch["returns_r"]

            log_prob_new, entropy, v_r, _ = self.ac.evaluate(obs, actions)

            ratio = torch.exp(log_prob_new - log_probs_old)
            clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
            loss_policy = -torch.min(ratio * adv_r, clipped * adv_r).mean()

            loss_value = nn.functional.mse_loss(v_r, ret_r)
            loss_entropy = -entropy.mean()

            loss_total = (
                loss_policy
                + self.cfg.vf_coef * loss_value
                + self.cfg.ent_coef * loss_entropy
            )

            self.opt_actor.zero_grad()
            self.opt_critic.zero_grad()
            loss_total.backward()
            clip_grad_norm_(self.ac.actor.parameters(), self.cfg.max_grad_norm)
            clip_grad_norm_(self.ac.criticR.parameters(), self.cfg.max_grad_norm)
            self.opt_actor.step()
            self.opt_critic.step()

            losses_policy.append(loss_policy.item())
            losses_value.append(loss_value.item())
            losses_entropy.append(entropy.mean().item())

            with torch.no_grad():
                ratio = torch.exp(log_prob_new - log_probs_old)
                clip_frac = ((ratio - 1).abs() > self.cfg.clip_eps).float().mean()
                approx_kl = ((ratio - 1) - (log_prob_new - log_probs_old)).mean()
                clip_fracs.append(clip_frac.item())
                approx_kls.append(approx_kl.item())

        mean_cost = float(buffer.costs[: buffer.ptr].mean())

        # ── Skema kunci SAMA PERSIS dengan PPOLagrangian.update() ──
        metrics = {
            "loss/policy": float(np.mean(losses_policy)),
            "loss/value_r": float(np.mean(losses_value)),
            "loss/value_c": 0.0,  # baseline tidak punya criticC
            "loss/entropy": float(np.mean(losses_entropy)),
            "ppo/clip_frac": float(np.mean(clip_fracs)),
            "ppo/approx_kl": float(np.mean(approx_kls)),
            "lagrangian/lambda": 0.0,  # baseline tidak punya lambda
            "lagrangian/lambda_before": 0.0,
            "lagrangian/mean_cost": mean_cost,
            "lagrangian/delta_f": self.cfg.delta_f,
            "lagrangian/constraint_gap": mean_cost - self.cfg.delta_f,
        }

        self.metrics_history.append(metrics)
        return metrics

    def state_dict(self) -> dict:
        return {
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "metrics_history": self.metrics_history,
        }

    def load_state_dict(self, d: dict):
        self.opt_actor.load_state_dict(d["opt_actor"])
        self.opt_critic.load_state_dict(d["opt_critic"])
        self.metrics_history = d.get("metrics_history", [])
