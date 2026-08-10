"""
app.py — Streamlit UI untuk Inference Model HLTF/PPO
Lokasi  : src/app.py
Jalankan: streamlit run src/app.py   (dari root project)
       atau: cd src && streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import time
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

# ── path resolution ───────────────────────────────────────────────────────────
# app.py ada di  : <root>/src/app.py
# agents & env   : <root>/agents/  dan  <root>/env/
# Excel default  : <root>/Data_BPS_Kota_Bandung.xlsx
#
# Path(__file__).parent       → <root>/src
# Path(__file__).parent.parent → <root>   ← PROJECT_ROOT yang benar
SRC_DIR = Path(__file__).resolve().parent  # .../src
PROJECT_ROOT = SRC_DIR.parent  # root project

# Tambahkan root ke sys.path agar `agents.*` dan `env.*` bisa diimpor
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from agents.network import ActorCritic
    from env.param_builder import ParamBuilder
    from env.zakat_env import ZakatEnv
except ImportError as e:
    st.error(
        f"❌ Gagal mengimpor modul proyek: **{e}**\n\n"
        "Pastikan Anda menjalankan dari root project:\n\n"
        "```\nstreamlit run src/app.py\n```"
    )
    st.stop()


# ── helpers ───────────────────────────────────────────────────────────────────


def rupiah(v: float) -> str:
    return "Rp {:,.0f}".format(float(v)).replace(",", ".")


def resolve_device(req: str) -> str:
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA dipilih tetapi tidak tersedia.")
    return req


def safe_float(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


# ── load model (cached) ───────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_system(ckpt_path: str, xlsx_path: str, scenario: int, device_req: str):
    device = resolve_device(device_req)
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

    meta = {
        "iterasi": int(checkpoint.get("iteration", -1)),
        "mode": str(checkpoint.get("mode", "unknown")),
        "delta_f": delta_f,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "device": device,
        "G": params.G,
        "T": params.T,
    }
    return model, env, params, meta


# ── inference satu periode ────────────────────────────────────────────────────


def run_one_step(model, env, params, seed: int, deterministic: bool):
    obs, _ = env.reset(seed=seed)

    budget_t = float(env._B)
    kebutuhan_t = np.asarray(env._d, dtype=np.float64).copy()
    mustahik_t = np.asarray(env._n, dtype=np.float64).copy()
    eligibility_t = np.asarray(env._e, dtype=np.float64).copy()
    lbr_before = np.asarray(env.fairness.compute_lbr(), dtype=np.float64).copy()

    with torch.inference_mode():
        raw_action, log_prob, entropy = model.get_action(
            obs, deterministic=deterministic
        )

    raw_action = np.asarray(raw_action, dtype=np.float64)
    _, reward, _, _, info = env.step(raw_action)

    allocation = np.asarray(info["allocation"], dtype=np.float64)
    benefit = np.asarray(info["benefit"], dtype=np.float64)
    lbr_after = np.asarray(info["lbr"], dtype=np.float64)
    delta_lbr = float(info["delta_fair"])
    fairness_cost = float(info["cost_fairness"])

    feasibility = env.projector.verify_feasibility(
        x=allocation,
        B_t=budget_t,
        e=eligibility_t,
        u_max=params.u_max,
    )

    total_alloc = float(allocation.sum())
    unallocated = max(0.0, budget_t - total_alloc)
    budget_ratio = total_alloc / budget_t if budget_t > params.eps else 0.0

    decision = {
        "budget": budget_t,
        "total_allocation": total_alloc,
        "unallocated": unallocated,
        "budget_utilization": budget_ratio,
        "reward": float(reward),
        "delta_lbr": delta_lbr,
        "delta_f": float(params.delta_f),
        "fairness_ok": delta_lbr <= params.delta_f,
        "fairness_cost": fairness_cost,
        "feasible": bool(feasibility["feasible"]),
        "budget_respected": bool(feasibility["budget_respected"]),
        "cap_respected": bool(feasibility["cap_respected"]),
        "eligibility_ok": bool(feasibility["eligibility_ok"]),
        "log_prob": safe_float(log_prob),
        "entropy": safe_float(entropy),
    }

    rows = []
    for i, wilayah in enumerate(params.wilayah):
        share = float(allocation[i] / total_alloc) if total_alloc > params.eps else 0.0
        rows.append(
            {
                "Kecamatan": str(wilayah),
                "Alokasi (Rp)": float(allocation[i]),
                "Proporsi (%)": round(share * 100, 4),
                "Kebutuhan": float(kebutuhan_t[i]),
                "Mustahik Weight": float(mustahik_t[i]),
                "Eligible": bool(eligibility_t[i] > 0.5),
                "Cap (Rp)": float(params.u_max[i]),
                "Benefit": float(benefit[i]),
                "LBR Sebelum": float(lbr_before[i]),
                "LBR Sesudah": float(lbr_after[i]),
                "Raw Action": float(raw_action[i]),
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("Alokasi (Rp)", ascending=False)
        .reset_index(drop=True)
    )
    df.index += 1

    return decision, df


# ── chart helpers ─────────────────────────────────────────────────────────────


def chart_bar(df: pd.DataFrame) -> go.Figure:
    top = df.head(20).copy()
    fig = px.bar(
        top,
        x="Alokasi (Rp)",
        y="Kecamatan",
        orientation="h",
        color="Proporsi (%)",
        color_continuous_scale=["#1D4ED8", "#38BDF8"],
        text=top["Alokasi (Rp)"].apply(lambda v: rupiah(v)),
        labels={"Alokasi (Rp)": "Alokasi (Rp)", "Proporsi (%)": "Proporsi (%)"},
    )
    fig.update_traces(textposition="outside", textfont_size=11)
    fig.update_layout(
        yaxis=dict(autorange="reversed", tickfont_size=11),
        xaxis=dict(tickformat=",.0f"),
        coloraxis_showscale=False,
        margin=dict(l=0, r=20, t=10, b=10),
        height=480,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def chart_lbr(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="LBR Sebelum",
            x=df["Kecamatan"],
            y=df["LBR Sebelum"],
            marker_color="#94A3B8",
        )
    )
    fig.add_trace(
        go.Bar(
            name="LBR Sesudah",
            x=df["Kecamatan"],
            y=df["LBR Sesudah"],
            marker_color="#3B82F6",
        )
    )
    fig.update_layout(
        barmode="group",
        xaxis=dict(tickangle=-45, tickfont_size=10),
        margin=dict(l=0, r=0, t=10, b=0),
        height=360,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def chart_pie(df: pd.DataFrame) -> go.Figure:
    top10 = df.head(10).copy()
    lain = df.iloc[10:]["Alokasi (Rp)"].sum()
    labels = list(top10["Kecamatan"]) + (["Kecamatan lain"] if lain > 0 else [])
    values = list(top10["Alokasi (Rp)"]) + ([lain] if lain > 0 else [])
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            textinfo="label+percent",
            textfont_size=11,
        )
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=10),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── export helpers ────────────────────────────────────────────────────────────


def to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8-sig")


def to_excel(df: pd.DataFrame, decision: dict) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Alokasi per Kecamatan", index=True)
        pd.DataFrame([decision]).T.rename(columns={0: "Nilai"}).to_excel(
            writer, sheet_name="Ringkasan Keputusan"
        )
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="HLTF Automatic Decision Support System",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* font & base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* sidebar */
section[data-testid="stSidebar"] { background: #0F172A; }
section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb],
section[data-testid="stSidebar"] .stNumberInput input {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    color: #F1F5F9 !important;
    border-radius: 6px;
}
section[data-testid="stSidebar"] label { color: #94A3B8 !important; font-size: 12px; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #F1F5F9 !important; }

/* metric cards */
[data-testid="metric-container"] {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 600 !important; }
[data-testid="stMetricLabel"] { font-size: 0.72rem !important; color: #64748B !important; text-transform: uppercase; letter-spacing: .06em; }

/* badge helper */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 600;
}
.badge-ok   { background:#DCFCE7; color:#166534; }
.badge-warn { background:#FEF9C3; color:#854D0E; }
.badge-err  { background:#FEE2E2; color:#991B1B; }

/* section header */
.sec-header {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #94A3B8;
    margin: 1.5rem 0 .6rem;
    padding-bottom: .3rem;
    border-bottom: 1px solid #E2E8F0;
}

/* dataframe tweaks */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("HLTF Decision System")
    st.markdown(
        "<small style='color:#64748B'>BAZNAS Kota Bandung · PPO-Lagrangian</small>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("#### Model")
    ckpt_input = st.text_input(
        "Checkpoint (.pt)",
        value=str(PROJECT_ROOT / "outputs" / "" / "checkpoints" / "hltf_best.pt"),
        help="Path absolut atau relatif terhadap root project",
    )
    xlsx_input = st.text_input(
        "Data Excel BPS/BAZNAS",
        value=str(PROJECT_ROOT / "Data_BPS_Kota_Bandung.xlsx"),
    )

    st.markdown("#### Skenario")
    scenario_label = st.radio(
        "Kondisi distribusi",
        options=["0 — Normal", "1 — Ramadan", "2 — Shock"],
        index=0,
        label_visibility="collapsed",
    )
    scenario = int(scenario_label[0])

    st.markdown("#### Parameter")
    seed = st.number_input("Seed lingkungan", value=1001, min_value=0, step=1)
    policy_opt = st.selectbox(
        "Policy", ["Deterministik (mean)", "Stokastik (sampling)"]
    )
    device_opt = st.selectbox("Device", ["auto", "cpu", "cuda"])
    deterministic = "Deterministik" in policy_opt

    st.divider()
    run_btn = st.button(
        "▶  Jalankan Inference", use_container_width=True, type="primary"
    )


# ── main area header ──────────────────────────────────────────────────────────
st.markdown("# Inference Alokasi Dana Zakat")
st.markdown(
    "<p style='color:#64748B;margin-top:-.5rem;margin-bottom:1.5rem'>"
    "Rekomendasi distribusi per kecamatan berbasis model PPO-Lagrangian (HLTF)</p>",
    unsafe_allow_html=True,
)

# ── state init ────────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None

# ── run inference ─────────────────────────────────────────────────────────────
if run_btn:
    st.session_state.result = None
    with st.spinner("Memuat model dan menjalankan inference…"):
        try:
            t0 = time.perf_counter()
            model, env, params, meta = load_system(
                ckpt_path=ckpt_input,
                xlsx_path=xlsx_input,
                scenario=scenario,
                device_req=device_opt,
            )
            decision, df = run_one_step(model, env, params, int(seed), deterministic)
            elapsed = time.perf_counter() - t0
            st.session_state.result = {
                "decision": decision,
                "df": df,
                "meta": meta,
                "elapsed": elapsed,
                "scenario": scenario,
                "seed": int(seed),
            }
        except FileNotFoundError as e:
            st.error(f"**File tidak ditemukan:** {e}")
        except KeyError as e:
            st.error(f"**Checkpoint tidak valid:** {e}")
        except Exception as e:
            st.error(f"**Error:** {e}")

# ── display results ───────────────────────────────────────────────────────────
if st.session_state.result:
    res = st.session_state.result
    dec = res["decision"]
    df = res["df"]
    meta = res["meta"]
    elapsed = res["elapsed"]

    # ── checkpoint info strip ──────────────────────────────────────────────
    with st.expander("ℹ️  Info checkpoint & model", expanded=False):
        ci1, ci2, ci3, ci4, ci5 = st.columns(5)
        ci1.metric("Iterasi", meta["iterasi"])
        ci2.metric("Mode", meta["mode"])
        ci3.metric("Hidden dim", meta["hidden_dim"])
        ci4.metric("Layers", meta["n_layers"])
        ci5.metric("Device", meta["device"])
        st.caption(
            f"Waktu inference: {elapsed:.3f} detik · G={meta['G']} kecamatan · T={meta['T']} bulan"
        )

    # ── status badges ──────────────────────────────────────────────────────
    st.markdown(
        "<div class='sec-header'>Status Keputusan</div>", unsafe_allow_html=True
    )

    fair_badge = (
        "<span class='badge badge-ok'>✓ Fairness terpenuhi</span>"
        if dec["fairness_ok"]
        else "<span class='badge badge-warn'>⚠ Pelanggaran fairness</span>"
    )
    feas_badge = (
        "<span class='badge badge-ok'>✓ Feasible</span>"
        if dec["feasible"]
        else "<span class='badge badge-err'>✗ Tidak feasible</span>"
    )
    bud_badge = (
        "<span class='badge badge-ok'>✓ Anggaran terpenuhi</span>"
        if dec["budget_respected"]
        else "<span class='badge badge-err'>✗ Anggaran dilanggar</span>"
    )
    cap_badge = (
        "<span class='badge badge-ok'>✓ Cap terpenuhi</span>"
        if dec["cap_respected"]
        else "<span class='badge badge-err'>✗ Cap dilanggar</span>"
    )
    el_badge = (
        "<span class='badge badge-ok'>✓ Eligibility OK</span>"
        if dec["eligibility_ok"]
        else "<span class='badge badge-err'>✗ Eligibility dilanggar</span>"
    )

    st.markdown(
        f"{fair_badge} &nbsp; {feas_badge} &nbsp; {bud_badge} &nbsp; {cap_badge} &nbsp; {el_badge}",
        unsafe_allow_html=True,
    )

    # ── key metrics ────────────────────────────────────────────────────────
    st.markdown(
        "<div class='sec-header'>Ringkasan Keputusan</div>", unsafe_allow_html=True
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Anggaran tersedia", rupiah(dec["budget"]))
    m2.metric("Total dialokasikan", rupiah(dec["total_allocation"]))
    m3.metric("Dana tersisa", rupiah(dec["unallocated"]))
    m4.metric("Utilisasi anggaran", f"{dec['budget_utilization']*100:.2f}%")
    m5.metric("Reward (utilitas)", f"{dec['reward']:.6f}")

    lbr_delta_color = "normal" if dec["fairness_ok"] else "inverse"
    m6.metric(
        "Δ LBR",
        f"{dec['delta_lbr']:.6f}",
        delta=f"threshold {dec['delta_f']:.4f}",
        delta_color=lbr_delta_color,
    )

    st.caption(
        f"Log-prob: {dec['log_prob']:.6f} · Entropy: {dec['entropy']:.6f} · "
        f"Fairness cost: {dec['fairness_cost']:.6f} · "
        f"Skenario: {['Normal','Ramadan','Shock'][res['scenario']]} · "
        f"Seed: {res['seed']}"
    )

    # ── charts ─────────────────────────────────────────────────────────────
    st.markdown("<div class='sec-header'>Visualisasi</div>", unsafe_allow_html=True)

    tab_bar, tab_lbr, tab_pie = st.tabs(
        [
            "📊 Alokasi per kecamatan",
            "📈 LBR sebelum vs sesudah",
            "🥧 Distribusi proporsi",
        ]
    )

    with tab_bar:
        st.caption("20 kecamatan dengan alokasi tertinggi")
        st.plotly_chart(chart_bar(df), use_container_width=True)

    with tab_lbr:
        st.caption("Long-Term Benefit Rate seluruh 30 kecamatan")
        st.plotly_chart(chart_lbr(df), use_container_width=True)

    with tab_pie:
        st.caption("Proporsi alokasi: 10 kecamatan teratas + sisanya")
        st.plotly_chart(chart_pie(df), use_container_width=True)

    # ── detail table ───────────────────────────────────────────────────────
    st.markdown(
        "<div class='sec-header'>Detail Alokasi per Kecamatan</div>",
        unsafe_allow_html=True,
    )

    display_df = df.copy()
    display_df["Alokasi (Rp)"] = display_df["Alokasi (Rp)"].apply(rupiah)
    display_df["Cap (Rp)"] = display_df["Cap (Rp)"].apply(rupiah)
    display_df["Proporsi (%)"] = display_df["Proporsi (%)"].map("{:.4f}%".format)
    display_df["LBR Sebelum"] = display_df["LBR Sebelum"].map("{:.6f}".format)
    display_df["LBR Sesudah"] = display_df["LBR Sesudah"].map("{:.6f}".format)
    display_df["Benefit"] = display_df["Benefit"].map("{:.6f}".format)
    display_df["Eligible"] = display_df["Eligible"].map({True: "✓", False: "✗"})

    cols_show = [
        "Kecamatan",
        "Alokasi (Rp)",
        "Proporsi (%)",
        "LBR Sebelum",
        "LBR Sesudah",
        "Benefit",
        "Eligible",
        "Cap (Rp)",
    ]
    st.dataframe(display_df[cols_show], use_container_width=True, height=460)

    # ── export ─────────────────────────────────────────────────────────────
    st.markdown("<div class='sec-header'>Ekspor Hasil</div>", unsafe_allow_html=True)

    ex1, ex2, ex3 = st.columns(3)

    with ex1:
        st.download_button(
            label="⬇ Unduh CSV (alokasi)",
            data=to_csv(df),
            file_name="inference_alokasi.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ex2:
        st.download_button(
            label="⬇ Unduh Excel (lengkap)",
            data=to_excel(df, dec),
            file_name="inference_hltf.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with ex3:
        summary_json = json.dumps(
            {
                **res["meta"],
                "decision": dec,
                "seed": res["seed"],
                "scenario": res["scenario"],
                "elapsed_sec": round(res["elapsed"], 4),
            },
            indent=2,
            ensure_ascii=False,
        )
        st.download_button(
            label="⬇ Unduh JSON (metadata)",
            data=summary_json.encode("utf-8"),
            file_name="inference_summary.json",
            mime="application/json",
            use_container_width=True,
        )

else:
    # ── empty state ────────────────────────────────────────────────────────
    st.markdown(
        """
    <div style='text-align:center;padding:4rem 2rem;color:#94A3B8'>
        <div style='font-size:1.1rem;font-weight:600;color:#475569;margin-bottom:.5rem'>
            Belum ada hasil
        </div>
        <div style='font-size:.875rem;line-height:1.7'>
            Isi path checkpoint dan data Excel di sidebar kiri,<br>
            pilih skenario dan parameter, lalu klik <strong>Jalankan</strong>.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
