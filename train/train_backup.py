"""
train.py (revisi)
==================
Main training loop untuk PPO-Lagrangian (HLTF) dan PPO Baseline.

PERBAIKAN dari versi sebelumnya:
  - TrainConfig sekarang punya field `delta_f`, diteruskan ke ParamBuilder
    (sebelumnya delta_f TIDAK diteruskan sama sekali -> selalu default 0.10,
    membuat violation_rate selalu 0% terlepas dari nilai ppo_cfg.delta_f).
  - Trainer memaksa SATU SUMBER KEBENARAN: delta_f yang benar-benar dipakai
    ZakatEnv (lewat params.delta_f) otomatis disinkronkan ke ppo_cfg.delta_f
    setelah environment dibangun -- jadi tidak mungkin lagi ada dua delta_f
    yang berbeda secara diam-diam, bahkan jika ppo_cfg dibuat manual
    (bukan lewat from_env_params()).
  - ppo_cfg ikut disimpan ke checkpoint untuk reproduktifitas.

Alur training per iterasi:
  1. Rollout T langkah dari ZakatEnv
  2. Bootstrap nilai akhir + hitung GAE
  3. Update kebijakan (PPO-Lagrangian atau PPO vanilla)
  4. Log metrik ke file dan terminal
  5. Simpan checkpoint setiap N iterasi
"""

import os
import sys
import time
import json
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.param_builder import ParamBuilder
from env.zakat_env import ZakatEnv
from agents.network import ActorCritic
from agents.buffer import RolloutBuffer
from agents.ppo_lagrangian import PPOLagrangian, PPOLagConfig
from agents.ppo import PPOBaseline, PPOBaselineConfig

# ─────────────────────────────────────────────────────────────
# Konfigurasi training
# ─────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    # Identitas run
    run_name: str = "hltf"  # "hltf" atau "baseline"
    seed: int = 42

    # Durasi training
    n_iterations: int = 500
    n_rollout: int = 52
    scenario: int = 0  # 0=normal, 1=Ramadan, 2=shock

    # Arsitektur jaringan
    hidden_dim: int = 256
    n_layers: int = 2

    # Checkpoint dan logging
    save_every: int = 50
    log_every: int = 10
    output_dir: str = "outputs"

    # Path data
    xlsx_path: str = "Data_BPS_Kota_Bandung.xlsx"

    # ── BARU: delta_f sekarang bagian resmi TrainConfig ──
    # Ini SATU-SATUNYA sumber kebenaran delta_f untuk seluruh run
    # (diteruskan ke ParamBuilder, lalu disinkronkan paksa ke ppo_cfg).
    delta_f: float = 0.10


# ─────────────────────────────────────────────────────────────
# Utilitas
# ─────────────────────────────────────────────────────────────


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dirs(cfg: TrainConfig) -> Path:
    out = Path(cfg.output_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    return out


def collect_rollout(
    env: ZakatEnv,
    ac: ActorCritic,
    buffer: RolloutBuffer,
    device: str,
    n_steps: int,
    obs_init: np.ndarray,
) -> tuple:
    buffer.reset()
    obs = obs_init
    ep_info = {}
    total_r = 0.0
    total_c = 0.0

    for _ in range(n_steps):
        action, log_prob, _ = ac.get_action(obs)

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            v_r, v_c = ac.get_values(obs_t)

        next_obs, reward, done, _, info = env.step(action)
        cost = info["cost_fairness"]

        buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            cost=cost,
            done=done,
            log_prob=log_prob,
            value_r=float(v_r.item()),
            value_c=float(v_c.item()),
        )

        total_r += reward
        total_c += cost
        obs = next_obs

        if done:
            ep_info = info.get("episode", {})
            ep_info["cumulative_reward"] = total_r
            ep_info["cumulative_cost"] = total_c
            obs, _ = env.reset()

    return obs, ep_info


def compute_bootstrap(
    ac: ActorCritic,
    buffer: RolloutBuffer,
    obs_last: np.ndarray,
    device: str,
):
    obs_t = torch.FloatTensor(obs_last).unsqueeze(0).to(device)
    with torch.no_grad():
        last_vr, last_vc = ac.get_values(obs_t)
    buffer.compute_returns_and_advantages(
        last_value_r=float(last_vr.item()),
        last_value_c=float(last_vc.item()),
    )


# ─────────────────────────────────────────────────────────────
# Trainer utama
# ─────────────────────────────────────────────────────────────


class Trainer:
    """
    Trainer generik yang mendukung PPO-Lagrangian (HLTF) dan PPO Baseline.

    PENTING (perbaikan): delta_f SELALU diambil dari cfg.delta_f (TrainConfig),
    diteruskan ke ParamBuilder, lalu ppo_cfg.delta_f dipaksa ikut sama.
    Ini berarti cfg.delta_f adalah satu-satunya tempat yang perlu diubah
    untuk mengganti ambang fairness -- tidak peduli apakah ppo_cfg dibuat
    manual atau lewat from_env_params().
    """

    def __init__(
        self,
        cfg: TrainConfig,
        ppo_cfg: object,  # PPOLagConfig atau PPOBaselineConfig
        mode: str = "hltf",  # "hltf" atau "baseline"
    ):
        self.cfg = cfg
        self.ppo_cfg = ppo_cfg
        self.mode = mode

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n  Device : {self.device}")

        set_seed(cfg.seed)
        self.out_dir = make_dirs(cfg)

        # ── Parameter environment dari BPS — delta_f DITERUSKAN ──
        builder = ParamBuilder(xlsx_path=cfg.xlsx_path, delta_f=cfg.delta_f)
        self.params = builder.build(verbose=False)
        self.obs_dim = 4 * self.params.G + 2
        self.act_dim = self.params.G

        # ── Sinkronisasi paksa: ppo_cfg.delta_f HARUS sama dengan
        #    params.delta_f yang benar-benar dipakai ZakatEnv ──
        if hasattr(ppo_cfg, "delta_f") and ppo_cfg.delta_f != self.params.delta_f:
            print(
                f"  [sinkronisasi delta_f] ppo_cfg.delta_f={ppo_cfg.delta_f} "
                f"!= params.delta_f={self.params.delta_f} "
                f"-> ppo_cfg diselaraskan ke params (satu sumber kebenaran)."
            )
            ppo_cfg.delta_f = self.params.delta_f

        print(
            f"  G={self.params.G} kecamatan | "
            f"obs_dim={self.obs_dim} | act_dim={self.act_dim} | "
            f"delta_f={self.params.delta_f}"
        )

        self.env = ZakatEnv(self.params, scenario=cfg.scenario)

        self.ac = ActorCritic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
            device=self.device,
        )
        p = self.ac.count_parameters()
        print(
            f"  Parameter: {p['total']:,} total "
            f"(actor={p['actor']:,}, criticR={p['criticR']:,}, criticC={p['criticC']:,})"
        )

        if mode == "hltf":
            self.agent = PPOLagrangian(self.ac, ppo_cfg, self.device)
        else:
            self.agent = PPOBaseline(self.ac, ppo_cfg, self.device)

        self.buffer = RolloutBuffer(
            buffer_size=cfg.n_rollout,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            device=self.device,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
        )

        self.log: list = []
        self.best_reward = -np.inf

    # ── Training loop ─────────────────────────────────────────

    def train(self):
        print(
            f"\n  Mulai training [{self.mode.upper()}] "
            f"— {self.cfg.n_iterations} iterasi — delta_f={self.params.delta_f}\n"
        )
        print(
            f"  {'Iter':>6} {'Reward':>10} {'DeltaLBR':>10} "
            f"{'ViolRate':>10} {'Lambda':>8} {'Loss_P':>10} {'Time':>7}"
        )
        print("  " + "─" * 70)

        obs, _ = self.env.reset(seed=self.cfg.seed)
        t_start = time.time()

        for it in range(1, self.cfg.n_iterations + 1):
            t_iter = time.time()

            obs, ep_info = collect_rollout(
                env=self.env,
                ac=self.ac,
                buffer=self.buffer,
                device=self.device,
                n_steps=self.cfg.n_rollout,
                obs_init=obs,
            )

            compute_bootstrap(self.ac, self.buffer, obs, self.device)
            metrics = self.agent.update(self.buffer)

            stats = self.buffer.stats()
            lbr_dev = ep_info.get("mean_delta_lbr", stats["mean_cost"])
            viol_rate = ep_info.get("violation_rate", 0.0)
            cum_r = ep_info.get(
                "cumulative_reward", stats["mean_reward"] * self.cfg.n_rollout
            )
            lam = metrics.get("lagrangian/lambda", 0.0)
            loss_p = metrics.get("loss/policy", 0.0)
            elapsed = time.time() - t_iter

            entry = {
                "iteration": it,
                "cumulative_reward": cum_r,
                "mean_delta_lbr": lbr_dev,
                "violation_rate": viol_rate,
                "lambda": lam,
                "mean_cost": stats["mean_cost"],
                "elapsed_sec": elapsed,
                **metrics,
            }
            self.log.append(entry)

            if it % self.cfg.log_every == 0 or it == 1:
                print(
                    f"  {it:>6} {cum_r:>10.4f} {lbr_dev:>10.5f} "
                    f"{viol_rate:>10.3f} {lam:>8.4f} "
                    f"{loss_p:>10.5f} {elapsed:>6.2f}s"
                )

            if it % self.cfg.save_every == 0:
                self._save_checkpoint(it)

            if cum_r > self.best_reward:
                self.best_reward = cum_r
                self._save_checkpoint(it, tag="best")

        total_time = time.time() - t_start
        print(f"\n  Training selesai — {total_time/60:.1f} menit")
        print(f"  Best reward: {self.best_reward:.4f}")

        self._save_log()
        self._save_checkpoint(self.cfg.n_iterations, tag="final")

        return self.log

    # ── Checkpoint ───────────────────────────────────────────

    def _save_checkpoint(self, iteration: int, tag: str = ""):
        suffix = f"_{tag}" if tag else f"_iter{iteration}"
        path = self.out_dir / "checkpoints" / f"model{suffix}.pt"
        torch.save(
            {
                "iteration": iteration,
                "ac_state_dict": self.ac.state_dict(),
                "agent_state": self.agent.state_dict(),
                "best_reward": self.best_reward,
                "config": asdict(self.cfg),
                # BARU: ppo_cfg ikut disimpan untuk reproduktifitas penuh
                "ppo_config": asdict(self.ppo_cfg),
                "mode": self.mode,
            },
            path,
        )

    def _save_log(self):
        path = self.out_dir / "training_log.json"
        with open(path, "w") as f:
            json.dump(self.log, f, indent=2)
        print(f"  Log disimpan → {path}")
