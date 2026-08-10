"""
evaluate_hltf.py — Evaluasi batch model HLTF/PPO-Lagrangian (v2)
Lokasi disarankan : <root>/src/evaluate_hltf.py  (sejajar dengan app.py)
Jalankan          : python src/evaluate_hltf.py   (dari root project)

v2: TEST_SEEDS sekarang dibaca OTOMATIS dari outputs/<run_name>/test_seeds.json
yang dihasilkan train.py (v2) -- tidak perlu diisi manual lagi, dan dijamin
seed yang sama persis dengan yang disisihkan (tidak pernah dilihat) saat training.

Dua kebutuhan yang dijawab script ini:

  1. TEST-SPLIT EVALUATION (poin #5)
     Menjalankan kebijakan pada seed-seed di test_seeds.json (tidak pernah
     dipakai saat training), lalu merata-ratakan reward, delta LBR, dan
     constraint violation rate -- satu episode penuh (T periode) per seed.

  2. GENERALIZATION TEST (poin #6)
     Menjalankan kebijakan yang SAMA pada skenario yang tidak pernah
     dilihat model (scenario id lain dari yang dipakai training), untuk
     menunjukkan model tetap bekerja pada kondisi baru.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for p in (PROJECT_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from agents.network import ActorCritic
from env.param_builder import ParamBuilder
from env.zakat_env import ZakatEnv

# ═══════════════════════════════════════════════════════════════════════════
# KONFIGURASI
# ═══════════════════════════════════════════════════════════════════════════

RUN_NAME = "hltf"  # harus sama dengan cfg.run_name yang dipakai saat training
CKPT_PATH = str(PROJECT_ROOT / "outputs" / RUN_NAME / "checkpoints" / "model_final.pt")
TEST_SEEDS_PATH = str(PROJECT_ROOT / "outputs" / RUN_NAME / "test_seeds.json")
XLSX_PATH = str(PROJECT_ROOT / "Data_BPS_Kota_Bandung.xlsx")
DEVICE = "cpu"

# --- poin #6: skenario/data baru yang belum pernah dilihat model ---
# Cek scenario apa yang dipakai saat training (cfg.scenario di train.py).
# Kalau training pakai scenario=0 (Normal), maka 1 (Ramadan) atau 2 (Shock)
# valid sebagai "data baru" tanpa perlu skenario tambahan.
NEW_SCENARIO_ID = 2
NEW_SCENARIO_SEEDS = [901, 902, 903, 904, 905]

DETERMINISTIC = True


# ═══════════════════════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════════════════════


def load_system(ckpt_path: str, xlsx_path: str, scenario: int, device: str):
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {ckpt}")

    try:
        checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt, map_location=device)

    if not isinstance(checkpoint, dict) or "ac_state_dict" not in checkpoint:
        raise KeyError("Checkpoint tidak valid — pastikan disimpan oleh Trainer.")

    train_cfg = checkpoint.get("config", {}) or {}
    ppo_cfg = checkpoint.get("ppo_config", {}) or {}
    delta_f = float(train_cfg.get("delta_f", ppo_cfg.get("delta_f", 0.10)))
    hidden_dim = int(train_cfg.get("hidden_dim", 256))
    n_layers = int(train_cfg.get("n_layers", 2))

    builder = ParamBuilder(xlsx_path=xlsx_path, delta_f=delta_f)
    params = builder.build(verbose=False)
    env = ZakatEnv(params=params, scenario=scenario)

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    model = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        device=device,
    )
    model.load_state_dict(checkpoint["ac_state_dict"], strict=True)
    model.eval()
    return model, env, params, delta_f, int(train_cfg.get("scenario", 0))


def load_test_seeds(path: str) -> list[int]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} tidak ditemukan. File ini dibuat otomatis oleh train.py (v2) "
            "saat training dijalankan. Pastikan Anda memakai train.py versi baru "
            "dan sudah melatih ulang model."
        )
    with open(p) as f:
        data = json.load(f)
    return data["test_seeds"]


# ═══════════════════════════════════════════════════════════════════════════
# EVALUASI SATU EPISODE PENUH
# ═══════════════════════════════════════════════════════════════════════════


def run_full_episode(model, env, params, seed: int, deterministic: bool) -> dict:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    delta_lbr_hist = []
    violations = 0
    n_steps = 0

    for _ in range(int(params.T)):
        with torch.inference_mode():
            raw_action, _, _ = model.get_action(obs, deterministic=deterministic)
        raw_action = np.asarray(raw_action, dtype=np.float64)
        obs, reward, terminated, truncated, info = env.step(raw_action)

        total_reward += float(reward)
        delta_lbr = float(info["delta_fair"])
        delta_lbr_hist.append(delta_lbr)
        if delta_lbr > params.delta_f:
            violations += 1
        n_steps += 1

        if terminated or truncated:
            break

    return {
        "seed": seed,
        "episode_reward": total_reward,
        "mean_delta_lbr": (
            float(np.mean(delta_lbr_hist)) if delta_lbr_hist else float("nan")
        ),
        "final_delta_lbr": delta_lbr_hist[-1] if delta_lbr_hist else float("nan"),
        "violation_rate": violations / n_steps if n_steps else float("nan"),
        "n_steps": n_steps,
    }


def run_batch(
    model, env, params, seeds: list[int], deterministic: bool
) -> pd.DataFrame:
    rows = [run_full_episode(model, env, params, s, deterministic) for s in seeds]
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str) -> dict:
    return {
        "kelompok": label,
        "n_episode": len(df),
        "reward_mean": df["episode_reward"].mean(),
        "reward_std": df["episode_reward"].std(),
        "delta_lbr_mean": df["mean_delta_lbr"].mean(),
        "delta_lbr_std": df["mean_delta_lbr"].std(),
        "violation_rate_mean": df["violation_rate"].mean(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main():
    print(f"Memuat model dari: {CKPT_PATH}")
    test_seeds = load_test_seeds(TEST_SEEDS_PATH)
    print(f"Memuat {len(test_seeds)} test seed dari: {TEST_SEEDS_PATH}")

    # intip scenario training dari checkpoint config dulu, sebelum load penuh
    _peek = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    train_scenario = int(_peek.get("config", {}).get("scenario", 0))
    del _peek

    model, env, params, delta_f, train_scenario = load_system(
        CKPT_PATH, XLSX_PATH, scenario=train_scenario, device=DEVICE
    )
    print(
        f"delta_f = {delta_f}, T = {params.T}, G = {params.G}, "
        f"scenario training = {train_scenario}"
    )

    print(
        f"\n[Poin #5] Menjalankan {len(test_seeds)} episode test "
        f"(scenario={train_scenario}, seed dari test_seeds.json)..."
    )
    df_test = run_batch(model, env, params, test_seeds, DETERMINISTIC)
    summary_test = summarize(df_test, "Test-split (held-out)")

    print(
        f"\n[Poin #6] Menjalankan {len(NEW_SCENARIO_SEEDS)} episode pada skenario baru "
        f"(scenario={NEW_SCENARIO_ID})..."
    )
    model2, env2, params2, _, _ = load_system(
        CKPT_PATH, XLSX_PATH, NEW_SCENARIO_ID, DEVICE
    )
    df_new = run_batch(model2, env2, params2, NEW_SCENARIO_SEEDS, DETERMINISTIC)
    summary_new = summarize(df_new, f"Skenario baru (id={NEW_SCENARIO_ID})")

    result = pd.DataFrame([summary_test, summary_new])
    print("\n=== RINGKASAN HASIL ===")
    print(result.to_string(index=False))

    out_dir = PROJECT_ROOT / "outputs" / RUN_NAME / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_test.to_csv(out_dir / "test_split_per_episode.csv", index=False)
    df_new.to_csv(out_dir / "new_scenario_per_episode.csv", index=False)
    result.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    print(f"\nDetail per-episode & ringkasan tersimpan di: {out_dir}")


if __name__ == "__main__":
    main()
