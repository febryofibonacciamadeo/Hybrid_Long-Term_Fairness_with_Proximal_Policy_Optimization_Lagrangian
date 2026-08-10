"""
projection.py
=============
Implementasi Action Projection untuk memastikan alokasi zakat x_t
berada dalam himpunan feasible F_t ⊆ R_+^G.

Mengimplementasikan dua langkah proyeksi (Persamaan 2.34–2.36 BAB II):

  Langkah 1 — Clip & masking eligibility (Pers. 2.34):
    x̃_g = max(0, z_g) · e_{g,t}
    Aksi mentah z_g dari aktor diclip ke non-negatif dan dikali indikator
    kelayakan e_{g,t} ∈ {0,1} agar wilayah tidak layak mendapat nol alokasi.

  Langkah 2 — Cap batas atas per wilayah (Pers. 2.35):
    x̃_g = min(x̃_g, u_{g,t})
    Alokasi setiap wilayah dibatasi oleh kapasitas maksimum u_{g,t}.

  Langkah 3 — Proyeksi proporsional ke simplex anggaran (Pers. 2.36):
    x_g = x̃_g · B_t / Σ x̃_g    jika Σ x̃_g > ε
    x_g = 0                       jika tidak ada wilayah layak

    Seluruh anggaran B_t didistribusikan proporsional terhadap preferensi
    aktor yang sudah dimask dan dicap. Proyeksi ini menjamin:
      (a) Σ_g x_{g,t} = B_t  (anggaran habis terpakai penuh)
      (b) x_{g,t} ≥ 0        untuk semua g
      (c) x_{g,t} ≤ u_{g,t}  untuk semua g (batas kapasitas)
      (d) x_{g,t} = 0        untuk g dengan e_{g,t} = 0 (tidak layak)

Catatan implementasi:
  - Projector tidak mempelajari parameter; ia adalah fungsi deterministik
    yang dijalankan di setiap langkah env.step() SEBELUM reward dihitung.
  - Karena proyeksi selalu menjamin feasibility, tidak diperlukan
    invalid action masking (yang hanya berlaku untuk ruang aksi diskret).
  - u_max dalam unit ABSOLUT (Rupiah), bukan proporsi.
"""

import numpy as np


class ActionProjector:
    """
    Memproyeksikan aksi mentah z_t ∈ [-1, 1]^G (output aktor)
    ke himpunan alokasi feasible F_t ⊆ R_+^G.

    Properti jaminan setelah proyeksi:
      1. x_g ≥ 0                         untuk semua g
      2. Σ_g x_g = B_t                   (anggaran habis penuh)
      3. x_g ≤ e_{g,t} · u_{g,t}         (batas kapasitas & kelayakan)
    """

    def __init__(self, G: int, eps: float = 1e-8):
        """
        Args:
            G   : jumlah wilayah distribusi
            eps : konstanta stabilitas numerik (mencegah pembagian nol)
        """
        self.G = G
        self.eps = eps

    # ─────────────────────────────────────────────────────────────
    # Proyeksi utama (Pers. 2.34–2.36)
    # ─────────────────────────────────────────────────────────────

    def project(
        self,
        z: np.ndarray,
        B_t: float,
        e: np.ndarray,
        u_max: np.ndarray,
    ) -> np.ndarray:
        """
        Proyeksikan aksi mentah z ke alokasi feasible x ∈ F_t.

        Args:
            z     : aksi mentah dari aktor, shape (G,), rentang bebas
            B_t   : anggaran periode t (Rupiah), skalar
            e     : indikator kelayakan e_{g,t} ∈ {0,1}, shape (G,)
            u_max : batas atas alokasi per wilayah (Rupiah), shape (G,)

        Returns:
            x     : alokasi feasible (Rupiah), shape (G,)
                    menjamin: x ≥ 0, Σx = B_t, x ≤ e·u_max
        """
        # ── Langkah 1: Clip non-negatif + masking eligibility (Pers. 2.34) ──
        # max(0, z_g) membuang preferensi negatif aktor,
        # dikali e_{g,t} memastikan wilayah tidak layak selalu nol.
        x_tilde = np.maximum(0.0, z) * e             # shape (G,)

        # ── Langkah 2: Cap batas atas per wilayah (Pers. 2.35) ──
        # Clip ke u_max yang sudah disesuaikan kelayakan (e · u_max).
        effective_cap = e * u_max                     # nol untuk g tidak layak
        x_tilde = np.minimum(x_tilde, effective_cap)  # shape (G,)

        # ── Langkah 3: Proyeksi proporsional ke simplex anggaran (Pers. 2.36) ──
        total = x_tilde.sum()

        if total < self.eps:
            # Tidak ada wilayah layak yang mendapat preferensi positif.
            # Fallback: distribusikan merata ke seluruh wilayah LAYAK,
            # sambil tetap menghormati batas atas masing-masing.
            eligible = e > 0
            if eligible.sum() == 0:
                # Kasus degenerate: semua wilayah tidak layak (tidak seharusnya
                # terjadi jika param_builder dikalibrasi dengan benar).
                return np.zeros(self.G, dtype=np.float64)

            # Alokasi merata ke wilayah layak, diclip ke u_max
            per_wilayah = B_t / eligible.sum()
            x = np.where(eligible, np.minimum(per_wilayah, effective_cap), 0.0)

            # Distribusikan sisa anggaran yang belum terserap (jika ada cap aktif)
            x = self._redistribute_residual(x, B_t, effective_cap, eligible)
        else:
            # Proyeksi proporsional standar (Pers. 2.36)
            x = x_tilde * (B_t / total)

            # Penegakan batas atas setelah scaling
            # (scaling bisa membuat beberapa x_g > u_max walau sudah dicap sebelumnya,
            # jika u_max satu wilayah jauh lebih kecil dari proporsi anggarannya)
            x = self._enforce_cap_and_redistribute(x, B_t, effective_cap)

        return x.astype(np.float64)

    # ─────────────────────────────────────────────────────────────
    # Helper: redistribusi sisa anggaran setelah penegakan cap
    # ─────────────────────────────────────────────────────────────

    def _enforce_cap_and_redistribute(
        self,
        x: np.ndarray,
        B_t: float,
        effective_cap: np.ndarray,
        max_rounds: int = 10,
    ) -> np.ndarray:
        """
        Iteratif: clip wilayah yang melebihi cap, redistribut sisa ke yang lain.
        Konvergen dalam beberapa iterasi karena setiap ronde paling tidak satu
        wilayah "terkunci" di batas atasnya.
        """
        x = x.copy()
        for _ in range(max_rounds):
            over = x > effective_cap + self.eps
            if not over.any():
                break

            # Kunci wilayah yang melebihi cap
            excess = (x - effective_cap)[over].sum()
            x = np.where(over, effective_cap, x)

            # Redistribut excess ke wilayah yang masih bisa menerima
            under = (effective_cap - x > self.eps) & (effective_cap > self.eps)
            if not under.any():
                break

            headroom = (effective_cap - x)[under].sum()
            if headroom < self.eps:
                break

            # Proporsi headroom
            factor = min(1.0, excess / headroom)
            x[under] += (effective_cap - x)[under] * factor

        # Pastikan total tepat B_t (koreksi floating point kecil)
        total_now = x.sum()
        if total_now > self.eps:
            x = x * (B_t / total_now)

        return x

    def _redistribute_residual(
        self,
        x: np.ndarray,
        B_t: float,
        effective_cap: np.ndarray,
        eligible: np.ndarray,
    ) -> np.ndarray:
        """Redistribusi sisa anggaran dari fallback alokasi merata."""
        residual = B_t - x.sum()
        if residual < self.eps:
            return x

        under = eligible & (x < effective_cap - self.eps)
        headroom = (effective_cap - x)[under].sum()
        if headroom < self.eps:
            return x

        factor = min(1.0, residual / headroom)
        x_new = x.copy()
        x_new[under] += (effective_cap - x)[under] * factor
        return x_new

    # ─────────────────────────────────────────────────────────────
    # Verifikasi feasibility (untuk debugging & logging)
    # ─────────────────────────────────────────────────────────────

    def verify_feasibility(
        self,
        x: np.ndarray,
        B_t: float,
        e: np.ndarray,
        u_max: np.ndarray,
        tol: float = 1e-4,
    ) -> dict:
        """
        Periksa apakah alokasi x memenuhi semua syarat feasibility F_t.

        Args:
            x     : alokasi yang akan diperiksa, shape (G,)
            B_t   : anggaran periode t
            e     : indikator kelayakan, shape (G,)
            u_max : batas atas alokasi, shape (G,)
            tol   : toleransi floating point untuk pengecekan kesetaraan

        Returns:
            dict berisi:
                feasible          : True jika semua kondisi terpenuhi
                non_negative      : True jika x_g ≥ 0 semua
                budget_respected  : True jika |Σx - B_t| ≤ tol
                cap_respected     : True jika x_g ≤ e_g·u_max_g + tol semua
                eligibility_ok    : True jika x_g = 0 untuk semua g tidak layak
                violations        : dict detail pelanggaran yang ditemukan
        """
        effective_cap = e * u_max
        violations = {}

        # (a) Non-negativitas
        neg_mask = x < -tol
        non_negative = not neg_mask.any()
        if not non_negative:
            violations["negative_alloc"] = {
                "wilayah": np.where(neg_mask)[0].tolist(),
                "values": x[neg_mask].tolist(),
            }

        # (b) Anggaran
        budget_diff = abs(x.sum() - B_t)
        budget_respected = budget_diff <= tol * max(1.0, B_t)
        if not budget_respected:
            violations["budget"] = {
                "sum_x": float(x.sum()),
                "B_t": float(B_t),
                "diff": float(budget_diff),
            }

        # (c) Batas atas
        over_mask = x > effective_cap + tol
        cap_respected = not over_mask.any()
        if not cap_respected:
            violations["cap_exceeded"] = {
                "wilayah": np.where(over_mask)[0].tolist(),
                "x_values": x[over_mask].tolist(),
                "caps": effective_cap[over_mask].tolist(),
            }

        # (d) Kelayakan (wilayah tidak layak harus nol)
        ineligible = e < 0.5
        ineligible_nonzero = ineligible & (x > tol)
        eligibility_ok = not ineligible_nonzero.any()
        if not eligibility_ok:
            violations["ineligible_alloc"] = {
                "wilayah": np.where(ineligible_nonzero)[0].tolist(),
                "values": x[ineligible_nonzero].tolist(),
            }

        feasible = non_negative and budget_respected and cap_respected and eligibility_ok

        return {
            "feasible": feasible,
            "non_negative": non_negative,
            "budget_respected": budget_respected,
            "cap_respected": cap_respected,
            "eligibility_ok": eligibility_ok,
            "budget_utilization": float(x.sum() / (B_t + self.eps)),
            "violations": violations,
        }

    # ─────────────────────────────────────────────────────────────
    # Representasi ringkas untuk debugging
    # ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"ActionProjector(G={self.G}, eps={self.eps})"
