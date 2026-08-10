"""
Sensitivity Tuning HLTF-CMDP
Fokus:
- delta_f fairness threshold
- lr_lambda Lagrangian update

Tujuan:
Mencari konfigurasi dengan:
1. violation rate rendah
2. fairness deviation rendah
3. reward tetap stabil
"""

import json
import numpy as np
from pathlib import Path

from train.train import Trainer, TrainConfig
from agents.ppo_lagrangian import PPOLagConfig

# ============================================================
# SEARCH SPACE
# ============================================================

EXPERIMENTS = [
    {
        "name": "delta_015",
        "delta_f": 0.015,
        "lr_lambda": 0.5,
    },
    {
        "name": "delta_017",
        "delta_f": 0.017,
        "lr_lambda": 0.5,
    },
    {
        "name": "delta_020",
        "delta_f": 0.020,
        "lr_lambda": 0.5,
    },
    {
        "name": "delta_025",
        "delta_f": 0.025,
        "lr_lambda": 0.5,
    },
    {
        "name": "delta_017_lambda01",
        "delta_f": 0.017,
        "lr_lambda": 0.1,
    },
]


# ============================================================
# TRAINING FUNCTION
# ============================================================


def run_experiment(exp):

    print("\n")
    print("=" * 70)

    print("RUNNING:", exp["name"])

    print("delta_f:", exp["delta_f"], "lr_lambda:", exp["lr_lambda"])

    print("=" * 70)

    train_cfg = TrainConfig(
        run_name=exp["name"],
        seed=42,
        # gunakan lebih panjang
        # agar policy stabil
        n_iterations=1000,
        n_rollout=52,
        scenario=0,
        hidden_dim=256,
        n_layers=2,
        save_every=200,
        log_every=50,
        output_dir="outputs/tuning_delta_f",
        xlsx_path="Data_BPS_Kota_Bandung.xlsx",
        delta_f=exp["delta_f"],
    )

    ppo_cfg = PPOLagConfig(
        # PPO tetap
        lr_actor=2e-4,
        lr_critic=5e-4,
        clip_eps=0.15,
        n_epochs=5,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        # sudah terbukti lebih stabil
        ent_coef=0.001,
        max_grad_norm=0.5,
        normalize_adv=True,
        # tuning
        lr_lambda=exp["lr_lambda"],
        lambda_init=0.0,
        lambda_max=10.0,
        delta_f=exp["delta_f"],
    )

    trainer = Trainer(cfg=train_cfg, ppo_cfg=ppo_cfg, mode="hltf")

    log = trainer.train()

    # ====================================================
    # Evaluasi 100 iterasi terakhir
    # ====================================================

    last = log[-100:]

    reward = np.mean([x["cumulative_reward"] for x in last])

    fairness = np.mean([x["mean_delta_lbr"] for x in last])

    violation = np.mean([x["violation_rate"] for x in last])

    lambda_final = log[-1]["lambda"]

    result = {
        "experiment": exp["name"],
        "delta_f": exp["delta_f"],
        "lr_lambda": exp["lr_lambda"],
        "reward": float(reward),
        "fairness": float(fairness),
        "violation_rate": float(violation),
        "lambda_final": float(lambda_final),
    }

    print("\nRESULT")

    print(result)

    return result


# ============================================================
# MAIN
# ============================================================


def main():

    results = []

    for exp in EXPERIMENTS:

        result = run_experiment(exp)

        results.append(result)

    Path("outputs/tuning_delta_f").mkdir(exist_ok=True)

    with open("outputs/tuning_delta_f/result.json", "w") as f:

        json.dump(results, f, indent=4)

    print("\n")
    print("=" * 70)

    print("FINAL RESULT")

    for r in results:
        print(r)


if __name__ == "__main__":

    main()
