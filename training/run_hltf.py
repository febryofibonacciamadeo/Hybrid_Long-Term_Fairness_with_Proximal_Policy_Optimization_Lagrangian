import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from train.train import Trainer, TrainConfig
from agents.ppo_lagrangian import PPOLagConfig

DELTA_F_FINAL = 0.020


def main():
    print("=" * 65)
    print("  PPO-Lagrangian HLTF — VERSI FINAL")
    print("  BAZNAS Kota Bandung (30 kecamatan, T=52 pekan)")
    print(f"  delta_f = {DELTA_F_FINAL}")
    print("=" * 65)

    train_cfg = TrainConfig(
        run_name="hltf",
        seed=42,
        n_iterations=1000,
        n_rollout=52,
        scenario=0,
        hidden_dim=256,
        n_layers=2,
        save_every=100,
        log_every=20,
        output_dir="outputs",
        xlsx_path="Data_BPS_Kota_Bandung.xlsx",
        delta_f=DELTA_F_FINAL,
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
        ent_coef=0.001,
        max_grad_norm=0.5,
        normalize_adv=True,
        # Lagrangian
        lr_lambda=0.5,
        lambda_init=0.0,
        lambda_max=10.0,
        delta_f=0.015,
    )

    trainer = Trainer(cfg=train_cfg, ppo_cfg=ppo_cfg, mode="hltf")

    # Sanity check sebelum training dimulai
    assert abs(trainer.params.delta_f - DELTA_F_FINAL) < 1e-9, (
        f"delta_f tidak sinkron! params={trainer.params.delta_f} "
        f"vs DELTA_F_FINAL={DELTA_F_FINAL}"
    )
    print(f"  [OK] delta_f = {trainer.params.delta_f} (tersinkron)\n")

    log = trainer.train()

    rewards = [e["cumulative_reward"] for e in log]
    lbr_devs = [e["mean_delta_lbr"] for e in log]
    viol_rates = [e["violation_rate"] for e in log]
    lambdas = [e["lambda"] for e in log]

    print("\n" + "=" * 65)
    print("  Ringkasan Hasil HLTF Final")
    print("=" * 65)
    print(f"  Reward   (100 iter terakhir) : {np.mean(rewards[-100:]):.5f}")
    print(f"  Delta LBR(100 iter terakhir) : {np.mean(lbr_devs[-100:]):.5f}")
    print(f"  Viol rate(100 iter terakhir) : {np.mean(viol_rates[-100:])*100:.1f}%")
    print(f"  Lambda akhir                 : {lambdas[-1]:.4f}")
    print(f"  delta_f                      : {DELTA_F_FINAL}")
    if lambdas[-1] == 0.0:
        print("\n  [!] PERINGATAN: lambda tidak pernah aktif.")
        print(
            "      Pastikan delta_f < mean_cost_max dan ppo_lagrangian.py versi final."
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
