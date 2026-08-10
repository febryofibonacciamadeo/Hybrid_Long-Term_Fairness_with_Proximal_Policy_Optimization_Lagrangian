from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ============================================================
# IMPORT MODUL PROYEK
# ============================================================

# Mendukung dua pola struktur:
# 1. python -m src.inference
# 2. python src/inference.py
try:
    from src.agents.network import ActorCritic
    from src.env.param_builder import ParamBuilder
    from src.env.zakat_env import ZakatEnv
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    sys.path.insert(0, str(CURRENT_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))

    from agents.network import ActorCritic
    from env.param_builder import ParamBuilder
    from env.zakat_env import ZakatEnv


# ============================================================
# ARGUMEN
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inference checkpoint HLTF/PPO untuk menghasilkan rekomendasi "
            "alokasi dana zakat per kecamatan."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path checkpoint .pt hasil training.",
    )
    parser.add_argument(
        "--xlsx",
        default="Data_BPS_Kota_Bandung.xlsx",
        help="Path data Excel BPS/BAZNAS.",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0=normal, 1=Ramadan, 2=shock.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1001,
        help="Seed kondisi awal lingkungan.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Jumlah keputusan berurutan. Maksimum mengikuti horizon environment.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Jumlah kecamatan dengan alokasi terbesar yang ditampilkan.",
    )
    parser.add_argument(
        "--output",
        default="outputs/inference/hltf",
        help="Folder penyimpanan hasil inference.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Perangkat inference.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sampling policy. Default memakai mean policy deterministik.",
    )
    return parser.parse_args()


# ============================================================
# UTILITAS
# ============================================================


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA dipilih, tetapi CUDA tidak tersedia.")

    return requested


def load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint harus berupa dictionary.")

    if "ac_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint tidak memiliki key 'ac_state_dict'. "
            "Gunakan checkpoint yang disimpan oleh Trainer."
        )

    return checkpoint


def rupiah(value: float) -> str:
    return "Rp{:,.0f}".format(float(value)).replace(",", ".")


def safe_float(value: Any) -> float:
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# LOAD MODEL DAN ENVIRONMENT
# ============================================================


def build_system(
    checkpoint_path: Path,
    xlsx_path: Path,
    scenario: int,
    device: str,
):
    checkpoint = load_checkpoint(checkpoint_path, device)

    train_cfg = checkpoint.get("config", {}) or {}
    ppo_cfg = checkpoint.get("ppo_config", {}) or {}

    delta_f = float(
        train_cfg.get(
            "delta_f",
            ppo_cfg.get("delta_f", 0.10),
        )
    )
    hidden_dim = int(train_cfg.get("hidden_dim", 256))
    n_layers = int(train_cfg.get("n_layers", 2))

    builder = ParamBuilder(
        xlsx_path=str(xlsx_path),
        delta_f=delta_f,
    )
    params = builder.build(verbose=False)

    env = ZakatEnv(
        params=params,
        scenario=scenario,
    )

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    model = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        device=device,
    )
    model.load_state_dict(
        checkpoint["ac_state_dict"],
        strict=True,
    )
    model.eval()

    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(checkpoint.get("iteration", -1)),
        "checkpoint_mode": str(checkpoint.get("mode", "unknown")),
        "delta_f": delta_f,
        "scenario": scenario,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "device": device,
    }

    return model, env, params, metadata


# ============================================================
# INFERENCE
# ============================================================


def run_inference(
    model: ActorCritic,
    env: ZakatEnv,
    params: Any,
    seed: int,
    steps: int,
    deterministic: bool,
    top_k: int,
):
    obs, _ = env.reset(seed=seed)

    decision_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []

    terminated = False
    truncated = False

    max_steps = min(int(steps), int(params.T))

    for step_index in range(max_steps):
        if terminated or truncated:
            break

        # Kondisi yang benar-benar menjadi input keputusan periode t.
        budget_t = float(env._B)
        kebutuhan_t = np.asarray(env._d, dtype=np.float64).copy()
        mustahik_t = np.asarray(env._n, dtype=np.float64).copy()
        eligibility_t = np.asarray(env._e, dtype=np.float64).copy()
        lbr_before = np.asarray(
            env.fairness.compute_lbr(),
            dtype=np.float64,
        ).copy()

        with torch.inference_mode():
            raw_action, log_prob, entropy = model.get_action(
                obs,
                deterministic=deterministic,
            )

        raw_action = np.asarray(raw_action, dtype=np.float64)

        next_obs, reward, terminated, truncated, info = env.step(raw_action)

        allocation = np.asarray(
            info["allocation"],
            dtype=np.float64,
        )
        benefit = np.asarray(
            info["benefit"],
            dtype=np.float64,
        )
        lbr_after = np.asarray(
            info["lbr"],
            dtype=np.float64,
        )

        delta_lbr = float(info["delta_fair"])
        fairness_cost = float(info["cost_fairness"])

        # Verifikasi memakai anggaran dan eligibility periode t.
        feasibility = env.projector.verify_feasibility(
            x=allocation,
            B_t=budget_t,
            e=eligibility_t,
            u_max=params.u_max,
        )

        total_allocation = float(allocation.sum())
        unallocated = max(0.0, budget_t - total_allocation)
        budget_ratio = total_allocation / budget_t if budget_t > params.eps else 0.0

        decision = {
            "step": step_index + 1,
            "seed": seed,
            "budget": budget_t,
            "total_allocation": total_allocation,
            "unallocated_budget": unallocated,
            "budget_utilization": budget_ratio,
            "reward": float(reward),
            "delta_lbr": delta_lbr,
            "delta_f": float(params.delta_f),
            "fairness_violation": int(delta_lbr > params.delta_f),
            "fairness_cost": fairness_cost,
            "feasible": int(bool(feasibility["feasible"])),
            "budget_respected": int(bool(feasibility["budget_respected"])),
            "cap_respected": int(bool(feasibility["cap_respected"])),
            "eligibility_ok": int(bool(feasibility["eligibility_ok"])),
            "non_negative": int(bool(feasibility["non_negative"])),
            "log_prob": safe_float(log_prob),
            "entropy": safe_float(entropy),
        }
        decision_rows.append(decision)

        for index, wilayah in enumerate(params.wilayah):
            share = (
                float(allocation[index] / total_allocation)
                if total_allocation > params.eps
                else 0.0
            )

            allocation_rows.append(
                {
                    "step": step_index + 1,
                    "seed": seed,
                    "wilayah": wilayah,
                    "budget": budget_t,
                    "raw_action": float(raw_action[index]),
                    "allocation": float(allocation[index]),
                    "allocation_share": share,
                    "kebutuhan": float(kebutuhan_t[index]),
                    "mustahik_weight": float(mustahik_t[index]),
                    "eligible": int(eligibility_t[index] > 0.5),
                    "upper_cap": float(params.u_max[index]),
                    "benefit": float(benefit[index]),
                    "lbr_before": float(lbr_before[index]),
                    "lbr_after": float(lbr_after[index]),
                }
            )

        # Tampilkan keputusan periode ini.
        order = np.argsort(allocation)[::-1]
        shown = min(max(1, top_k), len(order))

        print("\n" + "=" * 92)
        print(f"INFERENCE PERIODE {step_index + 1}")
        print("=" * 92)
        print(f"Anggaran tersedia       : {rupiah(budget_t)}")
        print(f"Total rekomendasi       : {rupiah(total_allocation)}")
        print(f"Dana belum dialokasikan : {rupiah(unallocated)}")
        print(f"Utilisasi anggaran      : {budget_ratio * 100:.2f}%")
        print(f"Reward                  : {float(reward):.8f}")
        print(
            f"Delta LBR               : {delta_lbr:.8f} "
            f"(batas {params.delta_f:.8f})"
        )
        print(
            "Status fairness         : "
            + ("MELANGGAR" if delta_lbr > params.delta_f else "MEMENUHI")
        )
        print(
            "Status hard constraint  : "
            + ("FEASIBLE" if feasibility["feasible"] else "TIDAK FEASIBLE")
        )

        print("-" * 92)
        print(
            f"{'No':>3}  {'Kecamatan':<24} "
            f"{'Alokasi':>18} {'Proporsi':>11} "
            f"{'Kebutuhan':>11} {'LBR':>11}"
        )
        print("-" * 92)

        for rank, index in enumerate(order[:shown], start=1):
            print(
                f"{rank:>3}  "
                f"{str(params.wilayah[index]):<24} "
                f"{rupiah(allocation[index]):>18} "
                f"{allocation[index] / max(total_allocation, params.eps) * 100:>10.2f}% "
                f"{kebutuhan_t[index]:>11.6f} "
                f"{lbr_after[index]:>11.6f}"
            )

        obs = next_obs

    return decision_rows, allocation_rows


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    args = parse_args()

    if args.steps < 1:
        raise ValueError("--steps harus minimal 1.")

    if args.top_k < 1:
        raise ValueError("--top-k harus minimal 1.")

    device = resolve_device(args.device)
    deterministic = not args.stochastic

    model, env, params, metadata = build_system(
        checkpoint_path=Path(args.checkpoint),
        xlsx_path=Path(args.xlsx),
        scenario=args.scenario,
        device=device,
    )

    print("=" * 78)
    print("INFERENCE MODEL HLTF/PPO")
    print("=" * 78)
    print(f"Checkpoint        : {metadata['checkpoint']}")
    print(f"Iterasi           : {metadata['checkpoint_iteration']}")
    print(f"Mode              : {metadata['checkpoint_mode']}")
    print(
        f"Policy            : " f"{'deterministik' if deterministic else 'stokastik'}"
    )
    print(f"Device            : {device}")
    print(f"Skenario          : {args.scenario}")
    print(f"Seed              : {args.seed}")
    print(f"Jumlah keputusan  : {args.steps}")
    print(f"Jumlah kecamatan  : {params.G}")
    print(f"Delta fairness    : {params.delta_f}")
    print("=" * 78)

    decision_rows, allocation_rows = run_inference(
        model=model,
        env=env,
        params=params,
        seed=args.seed,
        steps=args.steps,
        deterministic=deterministic,
        top_k=args.top_k,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "inference_decisions.csv",
        decision_rows,
    )
    write_csv(
        output_dir / "inference_allocations.csv",
        allocation_rows,
    )

    summary = {
        **metadata,
        "policy": ("deterministic" if deterministic else "stochastic"),
        "seed": args.seed,
        "requested_steps": args.steps,
        "completed_steps": len(decision_rows),
        "outputs": {
            "decisions_csv": str(output_dir / "inference_decisions.csv"),
            "allocations_csv": str(output_dir / "inference_allocations.csv"),
        },
    }

    with (output_dir / "inference_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 78)
    print("INFERENCE SELESAI")
    print("=" * 78)
    print(f"Keputusan tersimpan : {output_dir / 'inference_decisions.csv'}")
    print(f"Alokasi tersimpan   : {output_dir / 'inference_allocations.csv'}")
    print(f"Metadata tersimpan  : {output_dir / 'inference_summary.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
