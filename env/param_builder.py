"""
param_builder.py
================
Penurunan parameter lingkungan simulasi ZakatEnv dari data resmi BPS
dan BAZNAS Kota Bandung — Data_BPS_Kota_Bandung.xlsx (versi terbaru).

═══════════════════════════════════════════════════════════════════════
SUMBER DATA
═══════════════════════════════════════════════════════════════════════
[1] BPS Kota Bandung, sheet "Parameter RL" — 30 kecamatan, penduduk
    2024, dan penduduk miskin (proporsional dari P0 kota = 3.99%,
    total miskin 101.100 jiwa dari 2.528.163 jiwa). Tahun data: 2024.

[2] BPS Kota Bandung, sheet "Jumlah Populasi" — penduduk per
    kecamatan time-series 2018–2026.

[3] BPS Kota Bandung, sheet "Indeks Kemiskinan" — jumlah penduduk
    miskin Kota Bandung 2013–2025 dalam ribuan jiwa.

[4] BAZNAS Kota Bandung, sheet "Pengumpulan" — total pengumpulan
    dan penyaluran ZIS BAZNAS Kota Bandung 2019–2024 (data aktual,
    cakupan: BAZNAS Kota Bandung, bukan seluruh LAZ).

═══════════════════════════════════════════════════════════════════════
CATATAN METODOLOGI (wajib dicantumkan di BAB III skripsi)
═══════════════════════════════════════════════════════════════════════
1.  TINGKAT KEMISKINAN PER KECAMATAN (ip_g):
    BPS hanya menyediakan tingkat kemiskinan agregat kota P0 = 3.99%.
    Data ip_g per kecamatan tidak tersedia secara publik. Penelitian
    ini mensintesis ip_g menggunakan tiga proksi berbasis data populasi
    time-series 2018–2025 [2]:
      (a) Ukuran absolut penduduk miskin per kecamatan  (bobot 0.45)
      (b) Volatilitas pertumbuhan penduduk 2018–2025     (bobot 0.35)
      (c) Laju pertumbuhan penduduk                      (bobot 0.20)
    Hasil ip_g diskalakan agar rata-rata tertimbang = P0 = 3.99%.
    Rentang ip_g: [P0×0.60, P0×1.40].

2.  JUMLAH MUSTAHIK PER KECAMATAN:
    Data mustahik per kecamatan belum tersedia di sheet "Parameter RL".
    Disintesis proporsional dari penduduk miskin [1] dikalikan rasio
    penyaluran/pengumpulan aktual BAZNAS [4] sebagai faktor kalibrasi.

3.  DATA ZIS 2019–2024 [4]:
    Seluruh 6 tahun data aktual BAZNAS Kota Bandung digunakan untuk
    menghitung B̄ dan σ_B. Pola stabil (Rp 21–26 M per tahun)
    mencerminkan kapasitas penghimpunan BAZNAS Kota Bandung yang
    konsisten. σ_B dihitung dari deviasi standar data aktual.

4.  PARAMETER DINAMIKA (rho_d = 0.80, kappa = 0.10):
    Ditetapkan sebagai asumsi desain karena data time-series kebutuhan
    mustahik per kecamatan tidak tersedia secara publik.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# ═════════════════════════════════════════════════════════════
# Data class output
# ═════════════════════════════════════════════════════════════


@dataclass
class EnvParams:
    """Kumpulan parameter lingkungan simulasi ZakatEnv."""

    # Dimensi
    G: int = 30
    T: int = 52

    # Dana zakat (Pers. 3.1–3.6)
    B_mean: float = 0.0
    B_std: float = 0.0
    mu_B: float = 0.0
    sigma2_LN: float = 0.0
    seasonal_factors: np.ndarray = field(default_factory=lambda: np.ones(52))

    # Kebutuhan mustahik (Pers. 3.7–3.10)
    d_mean: np.ndarray = field(default_factory=lambda: np.zeros(30))
    d_std: np.ndarray = field(default_factory=lambda: np.zeros(30))
    rho_d: float = 0.80
    n_mean: np.ndarray = field(default_factory=lambda: np.zeros(30))

    # Bobot dan efektivitas (Pers. 3.11–3.13)
    w: np.ndarray = field(default_factory=lambda: np.zeros(30))
    alpha: np.ndarray = field(default_factory=lambda: np.zeros(30))
    u_max: np.ndarray = field(default_factory=lambda: np.zeros(30))

    # Kelayakan dan fairness
    d_min: float = 0.05
    delta_f: float = 0.10
    eps: float = 1e-8

    # Asumsi desain
    rho_capacity: float = 2.0
    kappa: float = 0.10

    # Metadata dokumentasi
    wilayah: list = field(default_factory=list)
    ip_per_kec: np.ndarray = field(default_factory=lambda: np.zeros(30))
    penduduk_miskin: np.ndarray = field(default_factory=lambda: np.zeros(30))
    penduduk: np.ndarray = field(default_factory=lambda: np.zeros(30))
    rasio_penyaluran: float = 0.0


# ═════════════════════════════════════════════════════════════
# ParamBuilder
# ═════════════════════════════════════════════════════════════


class ParamBuilder:
    """
    Membangun EnvParams dari file Data_BPS_Kota_Bandung.xlsx.

    Gunakan:
        builder = ParamBuilder(xlsx_path="path/ke/file.xlsx")
        params  = builder.build(verbose=True)
    """

    # Konstanta aktual dari BPS 2024 [1]
    TOTAL_PENDUDUK_2024 = 2_528_163
    TOTAL_MISKIN_2024 = 101_100
    P0_KOTA = TOTAL_MISKIN_2024 / TOTAL_PENDUDUK_2024  # 3.999%

    def __init__(
        self,
        xlsx_path: str = "Data_BPS_Kota_Bandung.xlsx",
        T: int = 52,
        delta_f: float = 0.10,
        kappa: float = 0.10,
        rho_d: float = 0.80,
        rho_capacity: float = 2.0,
        d_min: float = 0.05,
        eps: float = 1e-8,
    ):
        self.xlsx_path = xlsx_path
        self.T = T
        self.delta_f = delta_f
        self.kappa = kappa
        self.rho_d = rho_d
        self.rho_capacity = rho_capacity
        self.d_min = d_min
        self.eps = eps
        self._load_data()

    # ── Pembacaan data ────────────────────────────────────────

    def _load_data(self):
        """Baca empat sheet dari file Excel."""

        # [1] Parameter RL — penduduk & miskin per kecamatan 2024
        df = pd.read_excel(
            self.xlsx_path, sheet_name="Parameter RL Tahun 2024", header=None
        )
        rl = df.iloc[2:33, 1:8].copy()
        rl.columns = [
            "kecamatan",
            "penduduk",
            "proporsi",
            "penduduk_miskin",
            "pengumpulan_jt",
            "penyaluran",
            "jumlah_mustahik",
        ]
        for c in ["penduduk", "proporsi", "penduduk_miskin"]:
            rl[c] = pd.to_numeric(rl[c], errors="coerce")
        self.df_rl = (
            rl[rl["kecamatan"] != "Kota Bandung"]
            .dropna(subset=["kecamatan", "penduduk"])
            .reset_index(drop=True)
        )

        # [2] Jumlah Populasi — time-series 2018–2025
        df2 = pd.read_excel(self.xlsx_path, sheet_name="Jumlah Populasi", header=None)
        pop = df2.iloc[2:34, 1:11].copy()
        pop.columns = [
            "kecamatan",
            2018,
            2019,
            2020,
            2021,
            2022,
            2023,
            2024,
            2025,
            2026,
        ]
        for yr in [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
            pop[yr] = pd.to_numeric(pop[yr], errors="coerce")
        self.df_pop = (
            pop[pop["kecamatan"] != "Kota Bandung"]
            .dropna(subset=["kecamatan"])
            .reset_index(drop=True)
        )

        # [3] Indeks Kemiskinan — jumlah miskin kota 2013–2025 (ribu jiwa)
        df3 = pd.read_excel(self.xlsx_path, sheet_name="Indeks Kemiskinan", header=None)
        ik = df3.iloc[1:, 2:4].copy()
        ik.columns = ["tahun", "jumlah_ribu"]
        ik["tahun"] = pd.to_numeric(ik["tahun"], errors="coerce")
        ik["jumlah_ribu"] = pd.to_numeric(ik["jumlah_ribu"], errors="coerce")
        self.df_ik = ik.dropna().set_index("tahun")["jumlah_ribu"]

        # [4] Pengumpulan — ZIS aktual BAZNAS Kota Bandung 2019–2024
        df4 = pd.read_excel(self.xlsx_path, sheet_name="Pengumpulan", header=None)
        zis = df4.iloc[2:9, 1:4].copy()
        zis.columns = ["tahun", "pengumpulan", "penyaluran"]
        zis["tahun"] = pd.to_numeric(zis["tahun"], errors="coerce")
        zis["pengumpulan"] = pd.to_numeric(zis["pengumpulan"], errors="coerce")
        zis["penyaluran"] = pd.to_numeric(zis["penyaluran"], errors="coerce")
        self.df_zis = zis.dropna(subset=["tahun", "pengumpulan"]).reset_index(drop=True)

        self.wilayah = self.df_rl["kecamatan"].tolist()
        self.G = len(self.wilayah)

    # ── a.1: Sintesis IP per kecamatan ────────────────────────

    def _derive_ip_per_kecamatan(self) -> np.ndarray:
        """
        Sintesis ip_g dari tiga proksi berbasis data populasi [1][2].
        P0 kota = 3.999% (BPS 2024 [1]).

        Proksi:
          f1 = penduduk miskin absolut per kecamatan     (bobot 0.45)
          f2 = volatilitas populasi 2018–2025            (bobot 0.35)
          f3 = laju pertumbuhan penduduk 2018–2025       (bobot 0.20)

        Hasil diskalakan agar Σ(ip_g × penduduk_g) / Σ penduduk_g = P0.
        """
        P0 = self.P0_KOTA

        # Kumpulkan time-series populasi per kecamatan
        pop_series = np.zeros((self.G, 8), dtype=np.float64)
        for i, kec in enumerate(self.wilayah):
            row = self.df_pop[self.df_pop["kecamatan"] == kec]
            if len(row):
                pop_series[i] = [
                    row[yr].values[0]
                    for yr in [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
                ]

        pop_2018 = pop_series[:, 0]
        pop_2025 = pop_series[:, 7]

        # Faktor 1: ukuran penduduk miskin absolut [1]
        pm_abs = self.df_rl["penduduk_miskin"].values.astype(np.float64)
        f1 = pm_abs / (pm_abs.mean() + self.eps)

        # Faktor 2: volatilitas populasi 2018–2025 [2]
        vol = pop_series.std(axis=1) / (pop_series.mean(axis=1) + self.eps)
        f2 = (vol - vol.min()) / (vol.max() - vol.min() + self.eps)

        # Faktor 3: laju pertumbuhan penduduk 2018–2025 [2]
        gr = (pop_2025 - pop_2018) / (pop_2018 + self.eps)
        f3 = (gr - gr.min()) / (gr.max() - gr.min() + self.eps)

        # Skor gabungan
        skor = 0.45 * f1 + 0.35 * f2 + 0.20 * f3

        # Skalakan ke [0.60, 1.40] × P0
        skor_norm = 0.60 + 0.80 * (
            (skor - skor.min()) / (skor.max() - skor.min() + self.eps)
        )
        ip_g = P0 * skor_norm

        # Koreksi agar rata-rata tertimbang = P0
        wt = self.df_rl["penduduk"].values.astype(np.float64)
        wt = wt / wt.sum()
        ip_g = ip_g * (P0 / (np.sum(ip_g * wt) + self.eps))

        return ip_g

    # ── a.2: Parameter dana — dari data aktual [4] ────────────

    def _derive_budget_params(self):
        """
        B̄ dan σ_B dari data pengumpulan ZIS aktual 2019–2024 [4].

        B̄    = Σ ZIS_y / (n_tahun × T)     Pers. 3.1
        σ_B   = std(ZIS_y / T, ddof=1)      Pers. 3.2
        μ_B   = log(B̄² / √(B̄² + σ_B²))   Pers. 3.4
        σ²_LN = log(1 + (σ_B/B̄)²)         Pers. 3.5

        Rasio penyaluran/pengumpulan rata-rata digunakan
        untuk kalibrasi bobot mustahik yang terlayani.
        """
        pengumpulan = self.df_zis["pengumpulan"].values.astype(np.float64)
        penyaluran = self.df_zis["penyaluran"].values.astype(np.float64)
        n_tahun = len(pengumpulan)

        # Pers. 3.1
        B_mean = float(pengumpulan.sum() / (n_tahun * self.T))

        # Pers. 3.2
        B_std = float(np.std(pengumpulan / self.T, ddof=1))

        # Pers. 3.4–3.5
        mu_B = np.log(B_mean**2 / np.sqrt(B_mean**2 + B_std**2))
        sigma2_LN = np.log(1 + (B_std / B_mean) ** 2)

        # Rasio penyaluran rata-rata
        rasio = float(np.mean(penyaluran / (pengumpulan + self.eps)))

        return B_mean, B_std, mu_B, sigma2_LN, rasio

    def _build_seasonal_factors(self) -> np.ndarray:
        """
        Faktor musiman m_t (Pers. 3.6).
        Dikalibrasi dari pola zakat fitrah nasional [LPZN 2024]:
          Pekan 24–26 (Ramadan): m_t = 3.5
          Pekan 27   (Syawal):   m_t = 1.8
          Lainnya:               m_t = 1.0
        """
        m = np.ones(self.T, dtype=np.float64)
        for w in [24, 25, 26]:
            if w < self.T:
                m[w] = 3.5
        if 27 < self.T:
            m[27] = 1.8
        return m

    # ── a.3: Kebutuhan mustahik (Pers. 3.7–3.10) ──────────────

    def _derive_need_params(self, ip_g: np.ndarray, rasio: float):
        """
        d̄_g = (ip_g × pm_g) / max(ip × pm)    Pers. 3.7
        n̄_g = (pm_g × rasio) / Σ(pm × rasio)  — Pers. 3.10
              (rasio penyaluran aktual [4] sebagai proxy mustahik
               yang realistis terlayani per periode)
        σ_{d,g} = κ × d̄_g                      Pers. 3.8 (asumsi)
        """
        pm_g = self.df_rl["penduduk_miskin"].values.astype(np.float64)

        # Pers. 3.7
        raw = ip_g * pm_g
        d_mean = raw / (raw.max() + self.eps)

        # Pers. 3.8
        d_std = self.kappa * d_mean

        # Bobot mustahik dikalibrasi rasio penyaluran aktual
        n_raw = pm_g * rasio
        n_mean = n_raw / (n_raw.sum() + self.eps)

        return d_mean, d_std, n_mean, pm_g

    # ── a.4: Bobot dan efektivitas (Pers. 3.11–3.13) ──────────

    def _derive_weight_params(
        self,
        d_mean: np.ndarray,
        n_mean: np.ndarray,
        B_mean: float,
    ):
        """
        w_g   = (d̄_g × n̄_g) / Σ(d̄ × n̄)          Pers. 3.11
        α_g   = (1/d̄_g) / Σ(1/d̄)                  Pers. 3.12
        u_max = ρ × n̄_g × d̄_g × B̄ / Σ(n̄×d̄)      Pers. 3.13
        """
        raw_w = d_mean * n_mean
        w = raw_w / (raw_w.sum() + self.eps)

        inv_d = 1.0 / (d_mean + self.eps)
        alpha = inv_d / (inv_d.sum() + self.eps)

        denom = (n_mean * d_mean).sum() + self.eps
        u_max = self.rho_capacity * n_mean * d_mean * B_mean / denom

        return w, alpha, u_max

    # ── a.5: Validasi konsistensi (Pers. 3.15–3.17) ───────────

    def _validate(self, p: EnvParams, beta=0.5, zeta=0.05) -> list:
        warns = []

        # C3.15 — Budget adequacy
        min_need = beta * (p.w * p.d_mean * p.n_mean).sum()
        if p.B_mean < min_need:
            warns.append(
                f"[C3.15] Budget adequacy: " f"B̄={p.B_mean:,.0f} < {min_need:,.0f}"
            )

        # C3.16 — Upper bound reachability
        if p.u_max.sum() < p.B_mean:
            warns.append(
                f"[C3.16] Upper bound: "
                f"Σu_g={p.u_max.sum():,.0f} < B̄={p.B_mean:,.0f}"
            )

        # C3.17 — Non-trivialitas fairness
        unif = p.n_mean * p.B_mean / p.G
        dU = float(unif.max() - unif.min())
        if dU <= p.delta_f + zeta:
            warns.append(
                f"[C3.17] Fairness mungkin trivial: "
                f"Δ_unif={dU:.4f} ≤ {p.delta_f + zeta:.4f}"
            )

        return warns

    # ── Build utama ───────────────────────────────────────────

    def build(self, verbose: bool = True) -> EnvParams:
        """Bangun seluruh parameter dan kembalikan EnvParams."""

        # 1. Sintesis IP per kecamatan
        ip_g = self._derive_ip_per_kecamatan()

        # 2. Parameter dana dari data aktual [4]
        B_mean, B_std, mu_B, s2, rasio = self._derive_budget_params()
        seasonal = self._build_seasonal_factors()

        # 3. Kebutuhan mustahik
        d_mean, d_std, n_mean, pm_g = self._derive_need_params(ip_g, rasio)

        # 4. Bobot dan efektivitas
        w, alpha, u_max = self._derive_weight_params(d_mean, n_mean, B_mean)

        # 5. Penduduk per kecamatan 2024
        penduduk = self.df_rl["penduduk"].values.astype(np.float64)

        # 6. Susun EnvParams
        params = EnvParams(
            G=self.G,
            T=self.T,
            B_mean=B_mean,
            B_std=B_std,
            mu_B=mu_B,
            sigma2_LN=s2,
            seasonal_factors=seasonal,
            d_mean=d_mean,
            d_std=d_std,
            rho_d=self.rho_d,
            n_mean=n_mean,
            w=w,
            alpha=alpha,
            u_max=u_max,
            d_min=self.d_min,
            delta_f=self.delta_f,
            eps=self.eps,
            rho_capacity=self.rho_capacity,
            kappa=self.kappa,
            wilayah=self.wilayah,
            ip_per_kec=ip_g,
            penduduk_miskin=pm_g,
            penduduk=penduduk,
            rasio_penyaluran=rasio,
        )

        # 7. Validasi
        warns = self._validate(params)

        # 8. Cetak ringkasan
        if verbose:
            self._print_summary(params, ip_g, pm_g, rasio, warns)
            self._print_all_params(params)

        return params

    # ── Cetak ringkasan ───────────────────────────────────────

    def _print_summary(self, p, ip_g, pm_g, rasio, warns):
        W = 82
        print("=" * W)
        print("  ParamBuilder — ZakatEnv")
        print("  Sumber: BPS Kota Bandung 2024 [1][2][3] + BAZNAS 2019–2024 [4]")
        print("=" * W)

        # Tabel kecamatan
        print(
            f"\n  {'No':<4}{'Kecamatan':<22}{'Penduduk':>11}"
            f"{'Miskin':>10}{'IP (%)':>9}{'d̄_g':>8}{'w_g':>9}"
        )
        print("  " + "─" * 75)
        for i, kec in enumerate(self.wilayah):
            print(
                f"  {i+1:<4}{kec:<22}"
                f"{p.penduduk[i]:>11,.0f}"
                f"{pm_g[i]:>10,.1f}"
                f"{ip_g[i]*100:>8.3f}%"
                f"{p.d_mean[i]:>8.4f}"
                f"{p.w[i]:>9.4f}"
            )
        print("  " + "─" * 75)
        total_pm = pm_g.sum()
        print(
            f"  {'Total / Kota Bandung':<22}"
            f"{p.penduduk.sum():>11,.0f}"
            f"{total_pm:>10,.1f}"
            f"{self.P0_KOTA*100:>8.3f}%"
        )

        # Ringkasan IP sintesis
        print(
            f"\n  P0 kota aktual (BPS 2024) : {self.P0_KOTA*100:.3f}%"
            f"  ({self.TOTAL_MISKIN_2024:,} / {self.TOTAL_PENDUDUK_2024:,} jiwa)"
        )
        print(
            f"  IP sintesis range         : "
            f"[{ip_g.min()*100:.3f}%, {ip_g.max()*100:.3f}%]"
        )
        print(
            f"  IP tertinggi              : "
            f"{self.wilayah[ip_g.argmax()]}  ({ip_g.max()*100:.3f}%)"
        )
        print(
            f"  IP terendah               : "
            f"{self.wilayah[ip_g.argmin()]}  ({ip_g.min()*100:.3f}%)"
        )

        # Data ZIS aktual
        print(f"\n  {'─'*65}")
        print(f"  Data ZIS aktual BAZNAS Kota Bandung [4] — 2019–2024:")
        print(
            f"  {'Tahun':<8}{'Pengumpulan (Rp)':>22}"
            f"{'Penyaluran (Rp)':>22}{'Rasio':>8}"
        )
        print(f"  {'─'*63}")
        for _, r in self.df_zis.iterrows():
            rs = r["penyaluran"] / r["pengumpulan"] * 100
            print(
                f"  {int(r['tahun']):<8}"
                f"{r['pengumpulan']:>22,.0f}"
                f"{r['penyaluran']:>22,.0f}"
                f"{rs:>7.1f}%"
            )
        print(f"  {'─'*63}")

        # Parameter turunan
        print(f"\n  B̄  (per pekan)   : Rp {p.B_mean:>15,.0f}  ← data aktual [4]")
        print(f"  σ_B              : Rp {p.B_std:>15,.0f}  ← data aktual [4]")
        print(f"  μ_B              : {p.mu_B:.6f}")
        print(f"  σ²_LN            : {p.sigma2_LN:.6f}")
        print(f"  Rasio penyaluran : {rasio*100:.2f}%  ← rata-rata aktual [4]")
        print(f"  δ_f              : {p.delta_f}")
        print(f"  G × T            : {p.G} kecamatan × {p.T} pekan")

        # Validasi
        print(f"\n  {'─'*65}")
        if warns:
            print(f"  ⚠  {len(warns)} peringatan konsistensi:")
            for wm in warns:
                print(f"     {wm}")
        else:
            print("  ✓  Semua kondisi konsistensi (Pers. 3.15–3.17) terpenuhi.")
        print("=" * W)

        # ── Cetak seluruh parameter environment ─────────────────────

    def _print_all_params(self, p: EnvParams):
        """Tampilkan seluruh parameter yang digunakan oleh ZakatEnv."""

        W = 82
        print("\n" + "=" * W)
        print("  SELURUH PARAMETER ENVIRONMENT (HASIL SINTESIS)")
        print("=" * W)

        # Parameter skalar
        scalar_params = {
            "G": p.G,
            "T": p.T,
            "B_mean": p.B_mean,
            "B_std": p.B_std,
            "mu_B": p.mu_B,
            "sigma2_LN": p.sigma2_LN,
            "rho_d": p.rho_d,
            "delta_f": p.delta_f,
            "d_min": p.d_min,
            "rho_capacity": p.rho_capacity,
            "kappa": p.kappa,
            "rasio_penyaluran": p.rasio_penyaluran,
        }

        print("\n[Parameter Skalar]")
        for k, v in scalar_params.items():
            print(f"  {k:<20}: {v}")

        # Parameter vektor per wilayah
        print("\n[Parameter Per Wilayah]")
        print(
            f"  {'Kecamatan':<22}"
            f"{'d_mean':>10}{'d_std':>10}{'n_mean':>10}"
            f"{'w':>10}{'alpha':>10}{'u_max':>15}"
        )
        print("  " + "─" * 87)

        for i, kec in enumerate(p.wilayah):
            print(
                f"  {kec:<22}"
                f"{p.d_mean[i]:>10.4f}"
                f"{p.d_std[i]:>10.4f}"
                f"{p.n_mean[i]:>10.4f}"
                f"{p.w[i]:>10.4f}"
                f"{p.alpha[i]:>10.4f}"
                f"{p.u_max[i]:>15,.0f}"
            )

        # Seasonal factors
        print("\n[Seasonal Factors]")
        print(np.array2string(p.seasonal_factors, precision=2, separator=", "))

        print("=" * W)


# ═════════════════════════════════════════════════════════════
# Alias kompatibilitas
# ═════════════════════════════════════════════════════════════


def get_default_baznas_data():
    return {}


def get_default_zis_tahunan():
    return [23_896_828_346]


if __name__ == "__main__":
    import sys

    xlsx = sys.argv[1] if len(sys.argv) > 1 else "Data_BPS_Kota_Bandung.xlsx"
    ParamBuilder(xlsx_path=xlsx).build(verbose=True)
