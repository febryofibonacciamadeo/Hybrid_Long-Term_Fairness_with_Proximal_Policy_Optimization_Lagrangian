"""
di mana:
    x_proporsional_{g,t} = B_t * (d_{g,t} * n_{g,t}) / Σ_g(d_{g,t} * n_{g,t})
        → alokasi sebanding kebutuhan dan jumlah mustahik (paling umum
          dipraktikkan BAZNAS: wilayah dengan kebutuhan/populasi miskin
          lebih besar mendapat alokasi lebih besar)

    x_merata_{g,t} = B_t / G
        → alokasi rata sama besar ke semua kecamatan (praktik pemerataan
          yang juga umum demi keadilan administratif)

Kebijakan ini TIDAK belajar dari pengalaman — bobot 70:30 ditetapkan
di awal dan tidak berubah sepanjang simulasi, persis seperti praktik
distribusi BAZNAS yang mengikuti pedoman tetap (SOP), bukan optimasi
adaptif berbasis algoritma.
"""

import sys, json, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.param_builder import ParamBuilder
from env.zakat_env import ZakatEnv

# ─────────────────────────────────────────────────────────────
# Kebijakan deterministik BAZNAS
# ─────────────────────────────────────────────────────────────


class BAZNASPolicy:
    """
    Kebijakan distribusi tetap yang merepresentasikan praktik
    riil BAZNAS: campuran proporsional kebutuhan dan pemerataan.

    Tidak ada parameter yang dipelajari — w_proporsional dan
    w_merata ditetapkan sebagai konstanta sejak awal.
    """

    def __init__(self, w_proporsional: float = 0.7, w_merata: float = 0.3):
        assert (
            abs(w_proporsional + w_merata - 1.0) < 1e-6
        ), "Bobot proporsional + merata harus = 1.0"
        self.w_prop = w_proporsional
        self.w_flat = w_merata

    def get_action(
        self,
        B_t: float,
        d_t: np.ndarray,
        n_t: np.ndarray,
        e_t: np.ndarray,
        G: int,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Hitung alokasi x_t berdasarkan kebijakan campuran tetap.

        Args:
            B_t : anggaran tersedia periode t
            d_t : tingkat kebutuhan per wilayah
            n_t : bobot/jumlah mustahik per wilayah
            e_t : indikator kelayakan per wilayah
            G   : jumlah wilayah
        Returns:
            x_t : alokasi dana per wilayah, shape (G,)
        """
        # Hanya wilayah layak yang dipertimbangkan
        kebutuhan_layak = d_t * n_t * e_t

        # Komponen 1: alokasi proporsional kebutuhan × mustahik
        total_kebutuhan = kebutuhan_layak.sum() + eps
        x_proporsional = B_t * (kebutuhan_layak / total_kebutuhan)

        # Komponen 2: alokasi merata ke wilayah yang layak
        n_layak = max(int(e_t.sum()), 1)
        x_merata = np.where(e_t == 1, B_t / n_layak, 0.0)

        # Kombinasi 70:30
        x_t = self.w_prop * x_proporsional + self.w_flat * x_merata

        # Pastikan tidak melebihi anggaran (koreksi pembulatan)
        total_x = x_t.sum()
        if total_x > B_t + eps:
            x_t = x_t * B_t / (total_x + eps)

        return x_t.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Runner simulasi
# ─────────────────────────────────────────────────────────────


def run_baznas_baseline(
    xlsx_path: str,
    n_iterations: int = 1000,
    n_rollout: int = 52,
    scenario: int = 0,
    seed: int = 42,
    w_proporsional: float = 0.7,
    w_merata: float = 0.3,
    output_dir: str = "outputs",
):
    """
    Jalankan simulasi kebijakan BAZNAS aktual selama n_iterations
    episode, tanpa pembelajaran apapun.
    """
    print("=" * 65)
    print("  Simulasi Kebijakan AKTUAL BAZNAS (non-RL, deterministik)")
    print(
        f"  Kebijakan: {w_proporsional*100:.0f}% proporsional + "
        f"{w_merata*100:.0f}% pemerataan"
    )
    print("=" * 65)

    np.random.seed(seed)

    builder = ParamBuilder(xlsx_path=xlsx_path)
    params = builder.build(verbose=False)

    env = ZakatEnv(params, scenario=scenario)
    policy = BAZNASPolicy(w_proporsional, w_merata)

    out_dir = Path(output_dir) / "baznas_actual"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  G          : {params.G} kecamatan")
    print(f"  T          : {params.T} pekan")
    print(f"  Iterasi    : {n_iterations}")
    print(f"  Skenario   : {scenario}")
    print(
        f"\n  {'Iter':>6} {'Reward':>10} {'ΔLBRmean':>10} "
        f"{'MeanCost':>10} {'ViolRate':>10}"
    )
    print("  " + "─" * 52)

    log = []
    t_total = time.time()
    obs, _ = env.reset(seed=seed)

    for it in range(1, n_iterations + 1):
        obs, _ = env.reset(seed=seed + it)  # variasi stokastik antar episode

        total_reward = 0.0
        total_cost = 0.0
        n_violations = 0

        for step in range(n_rollout):
            # Ekstrak komponen state untuk kebijakan deterministik
            G = params.G
            B_t = obs[0] * (params.B_mean + params.eps)  # denormalisasi
            d_t = obs[1 : G + 1]
            n_t = obs[G + 1 : 2 * G + 1]
            e_t = obs[3 * G + 1 : 4 * G + 1]

            # Hitung aksi dari kebijakan tetap
            x_t = policy.get_action(B_t, d_t, n_t, e_t, G, params.eps)

            # Normalisasi ke rentang action_space [-1, 1] yang diharapkan
            # ZakatEnv akan memproyeksikan ulang via action_projection,
            # sehingga kita kirim x_t langsung sebagai "raw action" yang
            # sudah mendekati skala yang benar (positif, proporsional)
            raw_action = x_t / (B_t + params.eps)  # skala ke [0,1] relatif anggaran

            obs, reward, done, _, info = env.step(raw_action)
            total_reward += reward
            total_cost += info["cost_fairness"]
            if info["cost_fairness"] > 0:
                n_violations += 1

            if done:
                break

        ep_info = info.get("episode", {})
        viol_rate = n_violations / n_rollout
        lbr_dev = ep_info.get("mean_delta_lbr", total_cost / n_rollout)

        entry = {
            "iteration": it,
            "cumulative_reward": total_reward,
            "mean_delta_lbr": lbr_dev,
            "violation_rate": viol_rate,
            "mean_cost": total_cost / n_rollout,
        }
        log.append(entry)

        if it % 50 == 0 or it == 1:
            print(
                f"  {it:>6} {total_reward:>10.4f} {lbr_dev:>10.5f} "
                f"{total_cost/n_rollout:>10.5f} {viol_rate:>10.3f}"
            )

    total_time = time.time() - t_total

    # Simpan log
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Ringkasan
    rewards = [e["cumulative_reward"] for e in log]
    lbr_devs = [e["mean_delta_lbr"] for e in log]
    viol_rates = [e["violation_rate"] for e in log]
    costs = [e["mean_cost"] for e in log]

    print("\n" + "=" * 65)
    print(f"  Simulasi selesai — {total_time/60:.1f} menit")
    print("=" * 65)
    print(f"  Reward kumulatif")
    print(f"    Mean (semua iterasi) : {np.mean(rewards):.4f}")
    print(f"    Std                  : {np.std(rewards):.4f}")
    print(f"  Mean delta LBR")
    print(f"    Mean (semua iterasi) : {np.mean(lbr_devs):.5f}")
    print(f"    Std                  : {np.std(lbr_devs):.5f}")
    print(f"  Violation rate")
    print(f"    Mean (semua iterasi) : {np.mean(viol_rates)*100:.1f}%")
    print(f"  Mean cost")
    print(f"    Mean (semua iterasi) : {np.mean(costs):.5f}")
    print(f"\n  Log disimpan → {out_dir/'training_log.json'}")
    print("=" * 65)
    print("\n  Catatan: Hasil ini TIDAK menunjukkan 'pembelajaran' —")
    print("  nilai relatif konstan di sekitar mean karena kebijakan")
    print("  bersifat tetap. Variasi antar-iterasi murni berasal dari")
    print("  stokastisitas lingkungan (fluktuasi dana & kebutuhan).")

    return log


if __name__ == "__main__":
    xlsx = sys.argv[1] if len(sys.argv) > 1 else "Data_BPS_Kota_Bandung.xlsx"
    run_baznas_baseline(
        xlsx_path=xlsx,
        n_iterations=1000,
        n_rollout=52,
        scenario=2,
        seed=42,
        w_proporsional=0.7,
        w_merata=0.3,
        output_dir="outputs",
    )
