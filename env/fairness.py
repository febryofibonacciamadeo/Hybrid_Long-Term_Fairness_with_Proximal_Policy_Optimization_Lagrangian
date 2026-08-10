"""
fairness.py (revisi)
=====================
Perbaikan dari versi sebelumnya:

BUG 4 (skala n_weight untuk LBR): n_weight skala ribuan (jumlah mustahik)
sedangkan benefit skala sangat kecil (alpha*x_norm, order 1e-4 s.d. 1e-3).
LBR_g = sum(benefit)/sum(n_weight) menghasilkan nilai sangat kecil (1e-7)
dan SANGAT sensitif terhadap heterogenitas n_g antarwilayah -- wilayah
dengan n_g besar secara otomatis dapat LBR kecil, BUKAN karena kebijakan
tidak adil tapi karena skala penyebut yang berbeda-beda.
PERBAIKAN: n_weight dinormalisasi menjadi proporsi (sum=1) sebelum
dimasukkan ke akumulator LBR -- ini membuat LBR mengukur manfaat per
unit mustahik yang sebanding antarwilayah, bukan rasio angka absolut.
"""

import numpy as np


class FairnessTracker:
    """
    Melacak dan menghitung metrik fairness jangka panjang
    berbasis Long-Term Benefit Rate (LBR) antarwilayah.

    Persamaan yang diimplementasikan:
    - (3.34) LBR_g(t) = Σ b_{g,τ} / (Σ n_{g,τ} + ε)
    - (3.35) Δ_fair(t) = max_g LBR_g(t) − min_g LBR_g(t)
    - (3.36) c_t^fair  = max(0, Δ_fair(t) − δ_f)
    - (3.37) J_C(π)    = E[Σ γ^{t-1} c_t^fair]
    """

    def __init__(self, G: int, delta_f: float = 0.10, eps: float = 1e-8):
        self.G = G
        self.delta_f = delta_f
        self.eps = eps

        self.h_cum = np.zeros(G, dtype=np.float64)
        self.n_cum = np.zeros(G, dtype=np.float64)

        self.lbr_history: list = []
        self.delta_history: list = []
        self.cost_history: list = []

    def reset(self):
        self.h_cum[:] = 0.0
        self.n_cum[:] = 0.0
        self.lbr_history.clear()
        self.delta_history.clear()
        self.cost_history.clear()

    # ── Normalisasi n_weight (PERBAIKAN BUG 4) ───────────────

    @staticmethod
    def _normalize_n(n_weight: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Normalisasi n_weight menjadi proporsi (sum=1) sebelum dimasukkan
        ke akumulator LBR. Ini memastikan LBR_g mengukur manfaat per
        unit mustahik yang SEBANDING antarwilayah, bukan rasio absolut
        yang sensitif terhadap heterogenitas skala populasi mustahik.
        """
        total = n_weight.sum()
        if total < eps:
            return np.ones_like(n_weight) / len(n_weight)
        return n_weight / total

    # ── Pers. 3.34 ───────────────────────────────────────────

    def compute_lbr(
        self, h_cum: np.ndarray = None, n_cum: np.ndarray = None
    ) -> np.ndarray:
        """LBR_g(t) = Σ b_{g,τ} / (Σ n_{g,τ} + ε)"""
        h = h_cum if h_cum is not None else self.h_cum
        n = n_cum if n_cum is not None else self.n_cum
        return h / (n + self.eps)

    # ── Pers. 3.35 ───────────────────────────────────────────

    def compute_delta_fair(self, lbr: np.ndarray = None) -> float:
        """Δ_fair(t) = max_g LBR_g(t) − min_g LBR_g(t)"""
        if lbr is None:
            lbr = self.compute_lbr()
        return float(lbr.max() - lbr.min())

    # ── Pers. 3.36 ───────────────────────────────────────────

    def compute_fairness_cost(
        self,
        benefit: np.ndarray,
        n_weight: np.ndarray,
    ) -> float:
        """
        c_t^fair = max(0, Δ_fair(t) − δ_f)

        PERBAIKAN BUG 4: n_weight dinormalisasi sebelum proyeksi LBR.
        """
        n_norm = self._normalize_n(n_weight, self.eps)  # PERBAIKAN
        h_proj = self.h_cum + benefit
        n_proj = self.n_cum + n_norm                     # PERBAIKAN

        lbr = self.compute_lbr(h_proj, n_proj)
        delta_fair = self.compute_delta_fair(lbr)
        cost_fair = max(0.0, delta_fair - self.delta_f)

        return float(cost_fair)

    # ── Pembaruan akumulator ──────────────────────────────────

    def update(self, benefit: np.ndarray, n_weight: np.ndarray) -> dict:
        """
        PERBAIKAN BUG 4: n_weight dinormalisasi sebelum masuk akumulator.
        """
        n_norm = self._normalize_n(n_weight, self.eps)  # PERBAIKAN

        cost_fair = self.compute_fairness_cost(benefit, n_weight)

        # Update akumulator dengan n ternormalisasi
        self.h_cum += benefit
        self.n_cum += n_norm                             # PERBAIKAN

        lbr = self.compute_lbr()
        delta_fair = self.compute_delta_fair(lbr)

        self.lbr_history.append(lbr.copy())
        self.delta_history.append(delta_fair)
        self.cost_history.append(cost_fair)

        return {
            "lbr": lbr,
            "delta_fair": delta_fair,
            "cost_fair": cost_fair,
            "lbr_max": float(lbr.max()),
            "lbr_min": float(lbr.min()),
        }

    # ── Pers. 3.37 ───────────────────────────────────────────

    def compute_JC(self, gamma: float = 0.99) -> float:
        if not self.cost_history:
            return 0.0
        costs = np.array(self.cost_history)
        gammas = gamma ** np.arange(len(costs))
        return float((gammas * costs).sum())

    def compute_mean_delta(self) -> float:
        if not self.delta_history:
            return 0.0
        return float(np.mean(self.delta_history))

    def episode_summary(self) -> dict:
        return {
            "mean_delta_lbr": self.compute_mean_delta(),
            "final_lbr": self.compute_lbr().tolist(),
            "final_delta": self.compute_delta_fair(),
            "total_cost_fair": float(np.sum(self.cost_history)),
            "constraint_violations": int(np.sum(np.array(self.cost_history) > 0)),
            "violation_rate": (
                float(np.mean(np.array(self.cost_history) > 0))
                if self.cost_history else 0.0
            ),
        }
