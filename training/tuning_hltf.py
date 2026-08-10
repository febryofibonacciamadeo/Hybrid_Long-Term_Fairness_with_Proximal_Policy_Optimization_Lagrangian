"""
Automatic Hyperparameter Tuning HLTF-CMDP

Parameter:
- delta_f
- lr_lambda

Evaluasi:
- reward
- fairness deviation
- violation rate
- lambda

"""

import json
import numpy as np
from pathlib import Path

from train.train import Trainer, TrainConfig
from agents.ppo_lagrangian import PPOLagConfig

# ======================================================
# SEARCH SPACE
# ======================================================

SEARCH_SPACE = [
    {"name": "HLTF_default", "delta_f": 0.008, "lr_lambda": 0.1},
    {"name": "HLTF_relaxed", "delta_f": 0.012, "lr_lambda": 0.1},
    {"name": "HLTF_relaxed_fast_lambda", "delta_f": 0.012, "lr_lambda": 0.5},
    {"name": "HLTF_medium", "delta_f": 0.015, "lr_lambda": 0.5},
    {"name": "HLTF_strong_constraint", "delta_f": 0.008, "lr_lambda": 1.0},
]


# ======================================================
# TRAIN ONE EXPERIMENT
# ======================================================


def run_experiment(param):

    print("\n")
    print("=" * 70)

    print("RUN:", param["name"])

    print("delta_f:", param["delta_f"], "lr_lambda:", param["lr_lambda"])

    print("=" * 70)

    train_cfg = TrainConfig(
        run_name=param["name"],
        seed=42,
        # tuning cepat
        n_iterations=300,
        n_rollout=52,
        scenario=0,
        hidden_dim=256,
        n_layers=2,
        save_every=100,
        log_every=50,
        output_dir="outputs/tuning",
        xlsx_path="Data_BPS_Kota_Bandung.xlsx",
        delta_f=param["delta_f"],
    )

    ppo_cfg = PPOLagConfig(
        lr_actor=2e-4,
        lr_critic=5e-4,
        clip_eps=0.15,
        n_epochs=5,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        # tetap
        ent_coef=0.001,
        max_grad_norm=0.5,
        normalize_adv=True,
        # tuning
        lr_lambda=param["lr_lambda"],
        lambda_init=0.0,
        lambda_max=10.0,
        delta_f=param["delta_f"],
    )

    trainer = Trainer(cfg=train_cfg, ppo_cfg=ppo_cfg, mode="hltf")

    log = trainer.train()

    # ==============================
    # ambil hasil terakhir
    # ==============================

    reward = np.mean([x["cumulative_reward"] for x in log[-50:]])

    fairness = np.mean([x["mean_delta_lbr"] for x in log[-50:]])

    violation = np.mean([x["violation_rate"] for x in log[-50:]])

    lambda_final = log[-1]["lambda"]

    result = {
        "experiment": param["name"],
        "delta_f": param["delta_f"],
        "lr_lambda": param["lr_lambda"],
        "reward": float(reward),
        "fairness": float(fairness),
        "violation_rate": float(violation),
        "lambda": float(lambda_final),
    }

    print(result)

    return result


# ======================================================
# MAIN
# ======================================================


def main():

    results = []

    for param in SEARCH_SPACE:

        result = run_experiment(param)

        results.append(result)

    with open("outputs/tuning_result.json", "w") as f:

        json.dump(results, f, indent=4)

    print("\nSELESAI")
    print(results)


if __name__ == "__main__":

    main()
