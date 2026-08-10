"""
run_baseline_final.py
======================
PPO Baseline (vanilla) — VERSI FINAL untuk acuan resmi RQ2.

Konfigurasi final setelah seluruh proses kalibrasi:
  - delta_f    = 0.008  (dikalibrasi empiris: mean_cost_max~0.011 > 0.008)
  - lr_actor   = 2e-4   (clip_frac sehat <0.25, KL <0.015)
  - lr_critic  = 5e-4
  - clip_eps   = 0.15
  - n_epochs   = 5
  - ent_coef   = 0.001  (entropy stabil, tidak runaway)
  - batch_size = 256
  - n_iter     = 1000   (setara dengan run_hltf.py)

Jalankan dari root folder proyek:
    python training/run_baseline_final.py
"""

import sys, json, time, torch, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from train.train import Trainer, TrainConfig
from agents.ppo import PPOBaselineConfig

DELTA_F_FINAL = 0.008


def main():
    print("=" * 65)
    print("  PPO Baseline FINAL — Acuan resmi RQ2")
    print("  BAZNAS Kota Bandung (30 kecamatan, T=52 pekan)")
    print(f"  delta_f = {DELTA_F_FINAL}")
    print("=" * 65)

    train_cfg = TrainConfig(
        run_name="baseline_ppo_vanila",
        seed=42,
        n_iterations=1000,
        n_rollout=52,
        scenario=2,
        hidden_dim=256,
        n_layers=2,
        save_every=100,
        log_every=20,
        output_dir="outputs",
        xlsx_path="Data_BPS_Kota_Bandung.xlsx",
        delta_f=DELTA_F_FINAL,
    )

    ppo_cfg = PPOBaselineConfig.from_env_params(
        type("P", (), {"delta_f": DELTA_F_FINAL})(),
        lr_actor=2e-4,
        lr_critic=5e-4,
        clip_eps=0.15,
        n_epochs=5,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        ent_coef=0.001,
        max_grad_norm=0.5,
        normalize_adv=True,
    )

    trainer = Trainer(cfg=train_cfg, ppo_cfg=ppo_cfg, mode="baseline")
    log = trainer.train()

    rewards = [e["cumulative_reward"] for e in log]
    lbr_devs = [e["mean_delta_lbr"] for e in log]
    viol_rates = [e["violation_rate"] for e in log]

    print("\n" + "=" * 65)
    print("  Ringkasan Hasil Baseline Final")
    print("=" * 65)
    print(f"  Reward   (100 iter terakhir) : {np.mean(rewards[-100:]):.5f}")
    print(f"  Delta LBR(100 iter terakhir) : {np.mean(lbr_devs[-100:]):.5f}")
    print(f"  Viol rate(100 iter terakhir) : {np.mean(viol_rates[-100:])*100:.1f}%")
    print(f"  delta_f                      : {DELTA_F_FINAL}")
    print("=" * 65)


if __name__ == "__main__":
    main()
