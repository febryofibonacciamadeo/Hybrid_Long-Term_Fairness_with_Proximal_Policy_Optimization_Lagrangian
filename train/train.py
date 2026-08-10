"""
train.py (revisi v2 — dengan train-test split eksplisit)
==========================================================
Main training loop untuk PPO-Lagrangian (HLTF) dan PPO Baseline.

PERBAIKAN dari versi sebelumnya (v1 -> v2):
  - v1 hanya men-seed env SEKALI di awal training (env.reset(seed=cfg.seed)).
    Episode berikutnya di tengah training pakai env.reset() TANPA seed,
    sehingga tidak ada train/test split yang benar-benar terkontrol.
  - v2 membuat POOL SEED TETAP di awal run, dibagi menjadi:
      * train_seeds  -> HANYA seed ini yang boleh dipakai collect_rollout()
      * test_seeds   -> TIDAK PERNAH dipakai selama training sama sekali,
                        disimpan ke outputs/<run_name>/test_seeds.json
                        agar evaluate_hltf.py bisa memakai seed yang SAMA
                        PERSIS untuk evaluasi held-out (poin #5 revisi).
  - (Perbaikan delta_f dari revisi sebelumnya tetap dipertahankan.)

Alur training per iterasi (tidak berubah):
  1. Rollout T langkah dari ZakatEnv (sekarang di-reset dengan seed dari
     train_seeds, bukan acak tak terkendali)
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

    # delta_f -- satu-satunya sumber kebenaran untuk seluruh run
    delta_f: float = 0.10

    # ── BARU: train-test split ──
    # test_split=0.2 -> selain n_iterations episode training, akan dibuat
    # ekstra episode (20% dari n_iterations) yang seed-nya disisihkan
    # khusus untuk testing dan TIDAK PERNAH dipakai collect_rollout().
    test_split: float = 0.2
    split_seed: int = 12345  # seed KHUSUS untuk generate pool train/test,
    # terpisah dari cfg.seed (yang dipakai untuk inisialisasi bobot &
    # RNG training lain) agar pool episode reproducible independen.


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


def build_episode_seed_pool(cfg: TrainConfig) -> tuple[list, list]:
    """
    Membuat pool seed episode yang TETAP (reproducible lewat split_seed),
    lalu membaginya menjadi train_seeds dan test_seeds yang SALING LEPAS
    (disjoint) -- tidak ada satu seed pun yang muncul di kedua sisi.

    n_train = cfg.n_iterations (satu seed unik per iterasi training,
              karena n_rollout == T -> satu iterasi == satu episode penuh)
    n_test  = round(n_train * test_split / (1 - test_split))
              (mis. test_split=0.2 -> n_test = 25% dari n_train,
              yang secara proporsi keseluruhan pool == 20% test : 80% train)
    """
    rng = np.random.default_rng(cfg.split_seed)
    n_train = cfg.n_iterations
    n_test = max(1, int(round(n_train * cfg.test_split / (1 - cfg.test_split))))

    pool = rng.choice(10_000_000, size=n_train + n_test, replace=False)
    train_seeds = pool[:n_train].tolist()
    test_seeds = pool[n_train:].tolist()
    return train_seeds, test_seeds


def collect_rollout(
    env: ZakatEnv,
    ac: ActorCritic,
    buffer: RolloutBuffer,
    device: str,
    n_steps: int,
    obs_init: np.ndarray,
    next_reset_seed: Optional[int] = None,
) -> tuple:
    """
    Sama seperti sebelumnya, TAPI: ketika episode selesai (done) di tengah
    rollout, reset environment sekarang WAJIB memakai next_reset_seed
    (diambil dari train_seeds), bukan env.reset() tanpa argumen.
    """
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
            # ── PERBAIKAN: reset WAJIB pakai seed dari train_seeds ──
            if next_reset_seed is not None:
                obs, _ = env.reset(seed=next_reset_seed)
            else:
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

        # ── BARU: bangun & simpan pool train/test seed ──
        self.train_seeds, self.test_seeds = build_episode_seed_pool(cfg)
        with open(self.out_dir / "test_seeds.json", "w") as f:
            json.dump(
                {
                    "test_seeds": self.test_seeds,
                    "train_seeds_count": len(self.train_seeds),
                    "test_split": cfg.test_split,
                    "split_seed": cfg.split_seed,
                    "scenario": cfg.scenario,
                },
                f,
                indent=2,
            )
        print(
            f"  Episode pool: {len(self.train_seeds)} train, "
            f"{len(self.test_seeds)} test (disjoint, split_seed={cfg.split_seed})"
        )
        print(f"  test_seeds.json disimpan -> {self.out_dir / 'test_seeds.json'}")

        # ── Parameter environment dari BPS ──
        builder = ParamBuilder(xlsx_path=cfg.xlsx_path, delta_f=cfg.delta_f)
        self.params = builder.build(verbose=False)
        self.obs_dim = 4 * self.params.G + 2
        self.act_dim = self.params.G

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

        # ── episode ke-0 (sebelum iterasi pertama) pakai train_seeds[0] ──
        episode_idx = 0
        obs, _ = self.env.reset(seed=self.train_seeds[episode_idx])
        episode_idx += 1
        t_start = time.time()

        for it in range(1, self.cfg.n_iterations + 1):
            t_iter = time.time()

            # seed untuk episode BERIKUTNYA (kalau episode saat ini selesai
            # di tengah rollout ini) -- selalu dari train_seeds, tidak pernah
            # test_seeds, dan tidak pernah kehabisan karena n_train ==
            # n_iterations (lihat build_episode_seed_pool)
            next_seed = (
                self.train_seeds[episode_idx]
                if episode_idx < len(self.train_seeds)
                else None
            )

            obs, ep_info = collect_rollout(
                env=self.env,
                ac=self.ac,
                buffer=self.buffer,
                device=self.device,
                n_steps=self.cfg.n_rollout,
                obs_init=obs,
                next_reset_seed=next_seed,
            )
            if next_seed is not None:
                episode_idx += 1

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
        print(
            f"  {len(self.test_seeds)} episode test TIDAK PERNAH dipakai "
            f"selama training (lihat {self.out_dir / 'test_seeds.json'})"
        )

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
