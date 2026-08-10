"""
zakat_env.py (revisi)
======================
Perbaikan dari versi sebelumnya:

BUG 2 (cold start LBR): Di awal episode, h_cum=0, n_cum=0 sehingga
LBR_g=0 semua wilayah dan Delta_fair=0 selama ~10-15 langkah awal.
Sekitar 20-30% langkah per rollout (T=52) menghasilkan cost=0 palsu,
merendahkan mean_cost yang diterima update_lambda() secara artifisial.
PERBAIKAN: warm_start_lbr() mengisi h_cum dan n_cum awal dengan nilai
estimasi proporsional dari parameter lingkungan, sehingga LBR sudah
"matang" sejak langkah pertama tiap episode.

BUG 3 (skala n_t dalam state): n_t dari _sample_bobot() skala ribuan
(jumlah mustahik) sedangkan elemen state lain skala (0,1]. Perbedaan
skala ini menyulitkan MLP mempelajari representasi state yang seimbang.
PERBAIKAN: n_t dinormalisasi terhadap n_mean.max() dalam _get_obs().
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple

from env.param_builder import EnvParams
from env.fairness import FairnessTracker
from env.projection import ActionProjector


class ZakatEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        params: EnvParams,
        scenario: int = 0,
        gamma: float = 0.99,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.p = params
        self.scenario = scenario
        self.gamma = gamma
        self.render_mode = render_mode

        self.fairness = FairnessTracker(params.G, params.delta_f, params.eps)
        self.projector = ActionProjector(params.G, params.eps)

        obs_dim = 1 + 4 * params.G + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(params.G,),
            dtype=np.float32,
        )

        # PERBAIKAN BUG 3: simpan normalisasi n untuk dipakai di _get_obs
        self._n_max = float(params.n_mean.max()) + params.eps

        self._t: int = 0
        self._B: float = 0.0
        self._d: np.ndarray = np.zeros(params.G)
        self._n: np.ndarray = np.zeros(params.G)
        self._e: np.ndarray = np.ones(params.G)

        self._episode_rewards: list = []
        self._episode_costs: list = []
        self._episode_actions: list = []

    # ── Warm start LBR (PERBAIKAN BUG 2) ─────────────────────

    def _warm_start_lbr(self):
        """
        Isi h_cum dan n_cum awal dengan estimasi proporsional dari
        parameter lingkungan -- diskalakan agar merepresentasikan
        kondisi 'setelah beberapa periode distribusi', bukan kondisi
        'baru mulai dari nol' yang menghasilkan Delta_fair=0 palsu.

        Menggunakan alokasi proporsional terhadap bobot w_g (sesuai
        kebijakan rata-rata) selama T_warm=10 langkah virtual.
        """
        T_warm = 10
        x_virtual = self.p.w * self.p.B_mean  # alokasi proporsional

        # Normalisasi sama seperti di _compute_reward
        x_norm = x_virtual / (self.p.B_mean + self.p.eps)
        benefit_per_step = self.p.alpha * x_norm

        self.fairness.h_cum = benefit_per_step * T_warm
        self.fairness.n_cum = self.p.n_mean * T_warm

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._t = 0
        self.fairness.reset()

        # PERBAIKAN BUG 2: warm start LBR setelah reset
        self._warm_start_lbr()

        self._episode_rewards.clear()
        self._episode_costs.clear()
        self._episode_actions.clear()

        self._B = self._sample_budget()
        self._d = self._sample_kebutuhan()
        self._n = self._sample_bobot()
        self._e = self._compute_eligibility(self._d)

        obs = self._get_obs()
        info = {"scenario": self.scenario, "t": self._t}
        return obs, info

    def step(
        self,
        raw_action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:

        x = self.projector.project(
            z=raw_action,
            B_t=self._B,
            e=self._e,
            u_max=self.p.u_max,
        )

        reward, benefit = self._compute_reward(x)
        cost_fair = self.fairness.compute_fairness_cost(benefit, self._n)
        fairness_info = self.fairness.update(benefit, self._n)

        self._t += 1
        self._B = self._sample_budget()
        self._d = self._transition_kebutuhan(self._d)
        self._n = self._sample_bobot()
        self._e = self._compute_eligibility(self._d)

        obs = self._get_obs()
        done = self._t >= self.p.T

        self._episode_rewards.append(reward)
        self._episode_costs.append(cost_fair)
        self._episode_actions.append(x.copy())

        info = {
            "cost_fairness": cost_fair,
            "lbr": fairness_info["lbr"].tolist(),
            "delta_fair": fairness_info["delta_fair"],
            "budget": self._B,
            "allocation": x.tolist(),
            "benefit": benefit.tolist(),
            "t": self._t,
            "feasible": self.projector.verify_feasibility(
                x, self._B, self._e, self.p.u_max
            )["feasible"],
        }

        if done:
            info["episode"] = self.fairness.episode_summary()
            info["episode"]["cumulative_reward"] = float(sum(self._episode_rewards))

        return obs, reward, done, False, info

    def _get_obs(self) -> np.ndarray:
        """
        s_t = (B_t, d_t, n_t, LBR_t, e_t, z_t)  dim = 4G + 2

        PERBAIKAN BUG 3: n_t dinormalisasi terhadap n_mean.max()
        agar skala konsisten dengan elemen state lain yang bernilai (0,1].
        """
        lbr = self.fairness.compute_lbr()
        obs = np.concatenate(
            [
                [self._B / (self.p.B_mean + self.p.eps)],
                self._d,
                self._n / self._n_max,  # PERBAIKAN: normalisasi n_t
                lbr,
                self._e,
                [float(self.scenario)],
            ]
        )
        return obs.astype(np.float32)

    def _compute_reward(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        x_norm = x / (self.p.B_mean + self.p.eps)
        ratio = self.p.alpha * x_norm / (self._d + self.p.eps)
        reward = float(np.sum(self.p.w * np.log1p(ratio)))
        benefit = self.p.alpha * x_norm
        return reward, benefit

    def _sample_budget(self) -> float:
        m_t = self.p.seasonal_factors[self._t % self.p.T]
        B_t = m_t * self.np_random.lognormal(
            mean=self.p.mu_B,
            sigma=np.sqrt(self.p.sigma2_LN),
        )
        if self.scenario == 2:
            B_t *= 0.40
        return float(B_t)

    def _sample_kebutuhan(self) -> np.ndarray:
        d = self.np_random.normal(loc=self.p.d_mean, scale=self.p.d_std)
        return np.clip(d, self.p.d_min, 1.0).astype(np.float64)

    def _transition_kebutuhan(self, d_prev: np.ndarray) -> np.ndarray:
        noise = self.np_random.normal(loc=0.0, scale=self.p.d_std, size=self.p.G)
        d_new = self.p.rho_d * d_prev + (1 - self.p.rho_d) * self.p.d_mean + noise
        return np.clip(d_new, self.p.d_min, 1.0).astype(np.float64)

    def _sample_bobot(self) -> np.ndarray:
        xi = self.np_random.uniform(-0.05, 0.05, size=self.p.G)
        n_raw = self.p.n_mean * (1 + xi)
        return np.clip(n_raw, self.p.eps, None).astype(np.float64)

    def _compute_eligibility(self, d: np.ndarray) -> np.ndarray:
        return (d >= self.p.d_min).astype(np.float64)

    def render(self):
        if self.render_mode == "human":
            lbr = self.fairness.compute_lbr()
            delta = self.fairness.compute_delta_fair(lbr)
            print(
                f"t={self._t:3d} | B={self._B/1e6:6.1f}M | "
                f"Δ_fair={delta:.4f} | LBR=[{lbr.min():.3f}..{lbr.max():.3f}]"
            )
