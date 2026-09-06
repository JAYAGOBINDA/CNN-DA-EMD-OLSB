"""
Research & Evaluation — Streamlit UI.

Renders the 🧪 Research & Evaluation page with 4 tabs:
  1. 📊 Payload Capacity
  2. 📈 Statistical Testing
  3. 🔒 Security Analysis
  4. 🕵️ Steganalysis

All heavy computation is delegated to the research/ backend modules.
Session state is used to cache results between widget interactions.
"""

import io
import datetime
import numpy as np
import streamlit as st
from PIL import Image

# ── Backend imports ───────────────────────────────────────────────────────────
from research.payload_capacity import (
    run_payload_capacity_experiment,
    save_capacity_results,
    get_capacity_zip,
)
from research.statistical_testing import (
    run_statistical_experiment,
    generate_statistical_figures,
    save_statistical_results,
    get_statistical_zip,
    ALL_MODEL_NAMES,
    SUPPORTED_METRICS,
)
from research.security_analysis import (
    run_security_analysis,
    generate_stego_for_security,
    save_security_results,
    get_security_zip,
)
from research.steganalysis import (
    generate_stego_dataset,
    run_steganalysis,
    save_steganalysis_results,
    get_steganalysis_zip,
    pair_cover_and_stego,
)

# ── Shared styling helpers ────────────────────────────────────────────────────

_RESEARCH_CSS = """
<style>
.research-card {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(99, 102, 241, 0.22);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 24px -8px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(12px);
}

.research-card h4 {
    margin: 0 0 0.5rem 0;
    font-size: 1.08rem;
    font-weight: 700;
    color: #e2e8f0;
}

.research-card p, .research-card li {
    color: #94a3b8;
    font-size: 0.92rem;
    line-height: 1.55;
    margin: 0;
}

.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: #fff;
    font-weight: 800;
    font-size: 0.82rem;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    margin-right: 0.6rem;
    flex-shrink: 0;
}

.step-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.6rem;
}

.step-text {
    font-weight: 600;
    font-size: 0.95rem;
    color: #c7d2fe;
}

.status-pill {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.status-ready {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.4);
    color: #4ade80;
}

.status-pending {
    background: rgba(251, 146, 60, 0.12);
    border: 1px solid rgba(251, 146, 60, 0.3);
    color: #fb923c;
}

.metric-highlight {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(168, 85, 247, 0.12));
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    padding: 0.6rem 0.9rem;
    text-align: center;
}

.metric-highlight .value {
    font-size: 1.35rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-highlight .label {
    font-size: 0.75rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
</style>
"""


def _load_uploaded_images(files):
    """Return list of (np.ndarray uint8 RGB, name) for each Streamlit UploadedFile."""
    imgs, names = [], []
    for f in files:
        try:
            pil = Image.open(f).convert("RGB")
            imgs.append(np.array(pil, dtype=np.uint8))
            names.append(f.name)
        except Exception as e:
            st.warning(f"Could not load {f.name}: {e}")
    return imgs, names


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def _progress_factory(bar, text_el):
    """Return a progress_callback compatible with all research backends."""
    def _cb(step, total, msg):
        pct = int(step / total * 100) if total > 0 else 100
        bar.progress(min(pct, 100))
        text_el.markdown(
            f"**Step {step}/{total}** — {msg}"
        )
    return _cb


def _render_env_info():
    """Safely render environment/version info — never crashes."""
    import sys
    rows = [("Python", sys.version.split()[0])]

    for pkg_name, import_name in [
        ("PyTorch", "torch"),
        ("NumPy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("SciPy", "scipy"),
        ("Streamlit", "streamlit"),
    ]:
        try:
            mod = __import__(import_name)
            rows.append((pkg_name, getattr(mod, "__version__", "installed")))
        except ImportError:
            rows.append((pkg_name, "❌ Not installed"))

    table = "| Component | Version |\n|---|---|\n"
    for name, ver in rows:
        table += f"| {name} | `{ver}` |\n"
    table += f"| Random seed (patches) | `42` |\n"
    table += f"| Payload seed | Deterministic (image_idx × 1000 + BPP × 10000) |\n"
    table += f"| Timestamp | `{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}` |\n"
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PAYLOAD CAPACITY
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_payload_capacity():
    st.markdown("""
    <div class="research-card">
        <h4>📊 Payload Capacity Experiment</h4>
        <p>Embeds payloads of increasing size into each uploaded cover image using
        <b>CNN-DA-EMD-OLSB</b> and measures PSNR, SSIM, MSE, BER and recovery accuracy
        at each BPP level. All metrics are computed from actual embedding — no simulated values.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload Cover Images:", type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True, key="pc_upload"
        )
    with col2:
        bpp_options = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        bpp_sel = st.multiselect(
            "BPP Levels to Test:", options=bpp_options,
            default=[0.001, 0.01, 0.05, 0.1], key="pc_bpp"
        )
        password = st.text_input("Embedding Password:", value="Pass123!",
                                 type="password", key="pc_pass")

    st.markdown("---")
    run_btn = st.button("🚀 Run Payload Capacity Experiment", type="primary", key="pc_run")

    if run_btn:
        if not uploaded:
            st.error("Please upload at least one cover image.")
        elif not bpp_sel:
            st.error("Select at least one BPP level.")
        else:
            imgs, names = _load_uploaded_images(uploaded)
            if not imgs:
                st.error("No valid images loaded.")
                return

            prog_bar  = st.progress(0)
            prog_text = st.empty()
            cb = _progress_factory(prog_bar, prog_text)

            with st.spinner("Running payload capacity experiment …"):
                try:
                    results_df, stats_df, figs = run_payload_capacity_experiment(
                        images=imgs, image_names=names,
                        bpp_levels=sorted(bpp_sel),
                        password=password,
                        progress_callback=cb,
                    )
                    out_dir = save_capacity_results(results_df, stats_df, figs)
                    st.session_state["pc_results_df"] = results_df
                    st.session_state["pc_stats_df"]   = stats_df
                    st.session_state["pc_figs"]       = figs
                    st.session_state["pc_zip"]        = get_capacity_zip(results_df, stats_df, figs)
                    st.session_state["pc_out_dir"]    = out_dir
                    prog_bar.progress(100)
                    prog_text.markdown("✅ **Experiment complete!**")
                except Exception as exc:
                    st.error(f"Experiment failed: {exc}")
                    import traceback; st.code(traceback.format_exc())
                    return

    # ── Results display ───────────────────────────────────────────────────────
    if "pc_results_df" in st.session_state:
        results_df = st.session_state["pc_results_df"]
        stats_df   = st.session_state["pc_stats_df"]
        figs       = st.session_state["pc_figs"]

        st.markdown("#### 📋 Per-Image Results")
        display_cols = [c for c in ["image_id","image_hw","target_bpp","actual_bpp",
            "payload_bits","payload_bytes","psnr","ssim","mse","ber",
            "payload_recovery_%","cover_recovery_%","embed_time_s","status"]
            if c in results_df.columns]
        st.dataframe(results_df[display_cols], use_container_width=True)

        st.markdown("#### 📊 Summary Statistics per BPP Level")
        st.dataframe(stats_df, use_container_width=True)

        st.markdown("#### 📈 BPP vs Quality Metrics")
        if figs:
            cols = st.columns(min(len(figs), 2))
            for i, fig in enumerate(figs):
                cols[i % 2].pyplot(fig)
        else:
            st.warning("No successful embeddings to plot.")

        # Download
        st.markdown("#### 💾 Download Results")
        d1, d2, d3 = st.columns(3)
        d1.download_button("📥 Download Results CSV",
                           data=results_df.to_csv(index=False).encode(),
                           file_name="payload_capacity_results.csv", mime="text/csv",
                           key="dl_pc_csv")
        d2.download_button("📥 Download Stats CSV",
                           data=stats_df.to_csv(index=False).encode(),
                           file_name="payload_capacity_stats.csv", mime="text/csv",
                           key="dl_pc_stats")
        if "pc_zip" in st.session_state:
            d3.download_button("📦 Download All (ZIP)",
                               data=st.session_state["pc_zip"],
                               file_name="payload_capacity_results.zip",
                               mime="application/zip", key="dl_pc_zip")
        if "pc_out_dir" in st.session_state:
            st.caption(f"💾 Saved to: `{st.session_state['pc_out_dir']}`")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — STATISTICAL TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_statistical_testing():
    st.markdown("""
    <div class="research-card">
        <h4>📈 Statistical Testing — Friedman + Nemenyi + Kendall's W</h4>
        <p>Compares the <b>6 benchmark models</b> using a Friedman χ² test (non-parametric,
        matched across the same test images). If the Friedman test is significant
        (p &lt; 0.05) a Nemenyi post-hoc pairwise comparison is performed.
        <b>Kendall's W</b> is reported as the effect size.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload Test Images (≥ 3 recommended):",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True, key="st_upload"
        )
        metric = st.selectbox("Quality Metric:", SUPPORTED_METRICS,
                              index=0, key="st_metric")
    with col2:
        model_sel = st.multiselect(
            "Select Models to Compare:",
            options=ALL_MODEL_NAMES,
            default=ALL_MODEL_NAMES, key="st_models"
        )
        password  = st.text_input("Benchmark Password:", value="Pass123!",
                                  type="password", key="st_pass")
        payload_size = st.selectbox("Test Payload Size:",
                                    ["256 bytes", "512 bytes", "1 KB", "2 KB"],
                                    key="st_payload")

    payload_map = {"256 bytes": "A"*256, "512 bytes": "A"*512,
                   "1 KB": "B"*1024, "2 KB": "C"*2048}

    run_btn = st.button("🔬 Run Statistical Testing", type="primary", key="st_run")

    if run_btn:
        if not uploaded:
            st.error("Upload at least 3 test images.")
        elif len(model_sel) < 2:
            st.error("Select at least 2 models.")
        else:
            imgs, names = _load_uploaded_images(uploaded)
            if len(imgs) < 2:
                st.error("Need at least 2 loadable images."); return

            prog_bar  = st.progress(0)
            prog_text = st.empty()
            cb = _progress_factory(prog_bar, prog_text)

            with st.spinner("Running benchmark on all model × image combinations …"):
                try:
                    res = run_statistical_experiment(
                        images=imgs, image_names=names,
                        model_names=model_sel,
                        metric=metric,
                        password=password,
                        payload_str=payload_map[payload_size],
                        progress_callback=cb,
                    )
                    figs = generate_statistical_figures(res)
                    out_dir = save_statistical_results(res, figs)
                    st.session_state["st_result"]  = res
                    st.session_state["st_figs"]    = figs
                    st.session_state["st_zip"]     = get_statistical_zip(res, figs)
                    st.session_state["st_out_dir"] = out_dir
                    prog_bar.progress(100)
                    prog_text.markdown("✅ **Statistical testing complete!**")
                except Exception as exc:
                    st.error(f"Statistical test failed: {exc}")
                    import traceback; st.code(traceback.format_exc()); return

    # ── Results display ───────────────────────────────────────────────────────
    if "st_result" in st.session_state:
        res  = st.session_state["st_result"]
        figs = st.session_state["st_figs"]

        if res.get("warning"):
            st.warning(res["warning"])

        st.markdown("#### 📊 Observation Matrix")
        st.caption(f"Rows = test images   |   Columns = models   |   Metric: **{res['metric']}**")
        st.dataframe(res["observation_matrix"], use_container_width=True)

        if res["friedman_result"]:
            fr = res["friedman_result"]
            sig_icon = "✅" if fr["significant"] else "❌"
            st.markdown("#### 🔬 Friedman Test Result")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("χ² Statistic", fr["chi2_statistic"])
            m2.metric("Degrees of Freedom", fr["degrees_of_freedom"])
            m3.metric("p-value", fr["p_value"])
            m4.metric("α (threshold)", fr["alpha"])
            st.markdown(f"**Decision:** {sig_icon} {fr['decision']}")

        if res["effect_size"] is not None:
            eff = res["effect_size"]
            strength = "Small (<0.1)" if eff < 0.1 else "Medium (0.1-0.3)" if eff < 0.3 else "Large (>0.3)"
            st.markdown(
                f"**Effect Size — Kendall's W:** `{eff}` — {strength}  \n"
                "_Kendall's W = Friedman χ² / (n · (k−1)); ranges from 0 (no agreement) to 1 (perfect agreement)_"
            )

        if res["ranks_df"] is not None:
            st.markdown("#### 🏆 Average Ranks (lower = better for PSNR/SSIM)")
            st.dataframe(res["ranks_df"], use_container_width=True)

        if res["nemenyi_result"] is not None:
            st.markdown("#### 🔗 Nemenyi Post-Hoc Pairwise Comparison")
            st.dataframe(res["nemenyi_result"], use_container_width=True)
        elif res["friedman_result"] and not res["friedman_result"]["significant"]:
            st.info("Nemenyi post-hoc not applicable — Friedman test was not significant.")

        if figs:
            st.markdown("#### 📈 Rank Visualisation")
            for fig in figs:
                st.pyplot(fig)

        # Downloads
        st.markdown("#### 💾 Download Statistical Results")
        d1, d2 = st.columns(2)
        if res["observation_matrix"] is not None:
            d1.download_button("📥 Download Observation Matrix CSV",
                               data=res["observation_matrix"].to_csv().encode(),
                               file_name="observation_matrix.csv", mime="text/csv",
                               key="dl_st_obs")
        if "st_zip" in st.session_state:
            d2.download_button("📦 Download All (ZIP)",
                               data=st.session_state["st_zip"],
                               file_name="statistical_testing_results.zip",
                               mime="application/zip", key="dl_st_zip")
        if "st_out_dir" in st.session_state:
            st.caption(f"💾 Saved to: `{st.session_state['st_out_dir']}`")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SECURITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_security_analysis():
    st.markdown("""
    <div class="research-card">
        <h4>🔒 Security Analysis</h4>
        <p>Performs 6 independent security analyses comparing <b>cover vs stego images</b>:
        Histogram · Entropy · Pixel Correlation · RS Analysis · SPA · Chi-Square PoV.
        Stego images can be generated automatically or uploaded manually.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        cover_files = st.file_uploader(
            "Upload Cover Images:", type=["png","jpg","jpeg","bmp"],
            accept_multiple_files=True, key="sec_covers"
        )
        stego_mode = st.radio(
            "Stego Image Source:",
            ["Auto-generate using CNN-DA-EMD-OLSB", "Upload existing stego images"],
            key="sec_mode"
        )
    with col2:
        if stego_mode == "Upload existing stego images":
            stego_files = st.file_uploader(
                "Upload Stego Images (same order as covers):",
                type=["png","jpg","jpeg","bmp"],
                accept_multiple_files=True, key="sec_stegos"
            )
        else:
            stego_files = None
        password = st.text_input("Embedding Password:", value="Pass123!",
                                 type="password", key="sec_pass")
        payload_bpp = st.slider("Payload BPP for generation:", 0.01, 0.1, 0.05, 0.005,
                                 key="sec_bpp")

    run_btn = st.button("🔍 Run Security Analysis", type="primary", key="sec_run")

    if run_btn:
        if not cover_files:
            st.error("Upload at least one cover image."); return

        covers, c_names = _load_uploaded_images(cover_files)
        if not covers:
            st.error("No valid cover images."); return

        prog_bar  = st.progress(0)
        prog_text = st.empty()

        if stego_mode == "Auto-generate using CNN-DA-EMD-OLSB":
            cb = _progress_factory(prog_bar, prog_text)
            with st.spinner("Generating stego images …"):
                ok_covers, ok_stegos, ok_names = generate_stego_for_security(
                    cover_images=covers, cover_names=c_names,
                    password=password, payload_bpp=payload_bpp,
                    progress_callback=cb,
                )
            if not ok_stegos:
                st.error("Stego generation failed for all images."); return
        else:
            if not stego_files:
                st.error("Upload stego images or switch to auto-generation."); return
            stegos, s_names = _load_uploaded_images(stego_files)
            n = min(len(covers), len(stegos))
            ok_covers, ok_stegos, ok_names = covers[:n], stegos[:n], c_names[:n]

        cb2 = _progress_factory(prog_bar, prog_text)
        with st.spinner("Running security analyses …"):
            try:
                sec_result = run_security_analysis(
                    cover_images=ok_covers,
                    stego_images=ok_stegos,
                    image_names=ok_names,
                    progress_callback=cb2,
                )
                out_dir = save_security_results(sec_result)
                st.session_state["sec_result"]  = sec_result
                st.session_state["sec_zip"]     = get_security_zip(sec_result)
                st.session_state["sec_out_dir"] = out_dir
                prog_bar.progress(100)
                prog_text.markdown("✅ **Security analysis complete!**")
            except Exception as exc:
                st.error(f"Security analysis failed: {exc}")
                import traceback; st.code(traceback.format_exc()); return

    # ── Results display ───────────────────────────────────────────────────────
    if "sec_result" in st.session_state:
        res = st.session_state["sec_result"]

        htab1, htab2, htab3, htab4, htab5, htab6 = st.tabs([
            "🖼️ Histogram", "🔢 Entropy", "📐 Correlation",
            "🔴 RS Analysis", "🔵 SPA", "📊 Chi-Square"
        ])

        with htab1:
            st.markdown("**RGB Histograms — Cover vs Stego**")
            st.caption("Similar histograms suggest low perceptual distortion but do not alone prove undetectability.")
            for fig in res.get("histogram_figures", []):
                st.pyplot(fig)

        with htab2:
            st.markdown("**Shannon Entropy — Cover vs Stego**")
            df = res.get("entropy_df")
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)
                # Summary
                summary = df.groupby(["image_type","channel"])["entropy"].agg(["mean","std"]).round(4)
                st.markdown("**Summary (mean ± std)**")
                st.dataframe(summary, use_container_width=True)

        with htab3:
            st.markdown("**Pixel Correlation — Horizontal / Vertical / Diagonal**")
            df = res.get("correlation_df")
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)
            fig = res.get("correlation_figure")
            if fig:
                st.pyplot(fig)

        with htab4:
            st.markdown("**RS Analysis** (Fridrich et al. 2001)")
            st.caption(
                "R, S = Regular/Singular group rates under positive flip (+1 LSB).  "
                "RM, SM = rates under negative flip.  "
                "`p_estimate` ≈ fraction of pixels carrying hidden data."
            )
            df = res.get("rs_df")
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)

        with htab5:
            st.markdown("**Sample Pair Analysis** (Dumitrescu et al. 2003)")
            st.caption(
                "E, O = even→odd and odd→even adjacent pixel pair counts.  "
                "`p_estimate` = normalised parity asymmetry |E−O| / (E+O).  "
                "Clean images have E ≈ O."
            )
            df = res.get("spa_df")
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)

        with htab6:
            st.markdown("**Chi-Square PoV Test** (Westfeld & Pfitzmann 2000)")
            st.caption(
                "Tests whether adjacent histogram bins (2i, 2i+1) have equal counts.  "
                "p > 0.05 → bins statistically uniform → LSB steganography suspected.  "
                "p < 0.05 → natural image, non-uniform LSB histogram."
            )
            df = res.get("chi_df")
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)

        # Downloads
        st.markdown("#### 💾 Download Security Results")
        d1, d2 = st.columns(2)
        if "sec_zip" in st.session_state:
            d1.download_button("📦 Download All (ZIP)",
                               data=st.session_state["sec_zip"],
                               file_name="security_analysis_results.zip",
                               mime="application/zip", key="dl_sec_zip")
        if "sec_out_dir" in st.session_state:
            st.caption(f"💾 Saved to: `{st.session_state['sec_out_dir']}`")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — STEGANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_steganalysis():
    st.markdown("""
    <div class="research-card">
        <h4>🕵️ Steganalysis — Cover vs. Stego Binary Classification</h4>
        <p>Evaluates whether a dedicated deep-learning classifier can distinguish between original <b>Cover images</b>
        and their corresponding <b>Stego images</b> generated via <b>CNN-DA-EMD-OLSB</b>.<br>
        <b>Strict Zero Data Leakage:</b> Dataset splitting is executed strictly by <b>Image Pair</b>.
        A cover image and its corresponding stego counterpart are NEVER separated into different splits.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Workflow Mode Selection ───────────────────────────────────────────────
    mode = st.radio(
        "Workflow Mode:",
        [
            "● Use Existing Cover + Stego Pairs (Recommended)",
            "○ Generate Stego from Cover (Optional Secondary Mode)",
        ],
        index=0,
        key="sa_workflow_mode",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PRIMARY WORKFLOW: USE EXISTING COVER + STEGO PAIRS
    # ═══════════════════════════════════════════════════════════════════════════
    if "Existing" in mode:
        st.markdown("### 📁 Input Data: Cover & Stego Pairs")
        c_col1, c_col2 = st.columns(2)

        with c_col1:
            st.markdown("#### **STEP 1:** 📁 Upload Cover Images")
            cover_files = st.file_uploader(
                "Select original Cover images:",
                type=["png", "jpg", "jpeg", "bmp"],
                accept_multiple_files=True,
                key="sa_direct_covers",
            )

        with c_col2:
            st.markdown("#### **STEP 2:** 📁 Upload Corresponding Stego Images")
            stego_files = st.file_uploader(
                "Select corresponding Stego images (e.g. image_stego.png):",
                type=["png", "jpg", "jpeg", "bmp"],
                accept_multiple_files=True,
                key="sa_direct_stegos",
            )

        # ── STEP 3: Validate & Pair Images ────────────────────────────────────
        st.markdown("---")
        st.markdown("### **STEP 3:** 🔗 Validate & Pair Images")

        paired_data = []
        is_pairing_ready = False

        if cover_files and stego_files:
            cov_imgs, cov_names = _load_uploaded_images(cover_files)
            stg_imgs, stg_names = _load_uploaded_images(stego_files)

            pairing_res = pair_cover_and_stego(cov_imgs, cov_names, stg_imgs, stg_names)
            paired_data = pairing_res["pairs"]
            is_pairing_ready = pairing_res["is_valid"]

            # Display pair count summary
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            p_col1.metric("Cover Images", pairing_res["n_covers"])
            p_col2.metric("Stego Images", pairing_res["n_stegos"])
            p_col3.metric("Valid Matched Pairs", pairing_res["n_pairs"])
            status_text = "✅ Ready" if is_pairing_ready else "❌ Need ≥ 2 Pairs"
            p_col4.metric("Pairing Status", status_text)

            if pairing_res.get("warning"):
                st.warning(pairing_res["warning"])

            if pairing_res.get("mismatches"):
                for mm in pairing_res["mismatches"]:
                    st.error(f"⚠️ {mm}")

            # Expandable inspection table
            if paired_data:
                with st.expander(f"🔍 Inspect Matched Image Pairs ({len(paired_data)} pairs)", expanded=False):
                    table_rows = []
                    for p in paired_data:
                        table_rows.append({
                            "Pair ID": p["pair_id"],
                            "Cover Image": p["cover_name"],
                            "Stego Image": p["stego_name"],
                            "Dimensions (H×W×C)": f"{p['shape'][0]}×{p['shape'][1]}×{p['shape'][2]}",
                            "Match Strategy": p["match_type"],
                        })
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
        else:
            st.info("Upload both Cover images (Step 1) and Stego images (Step 2) above to automatically pair them.")

    # ═══════════════════════════════════════════════════════════════════════════
    # OPTIONAL SECONDARY WORKFLOW: GENERATE STEGO FROM COVER
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("### 🔧 Secondary Workflow: Generate Stego from Covers")
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            cover_files_gen = st.file_uploader(
                "Upload Cover Images for Embedding:",
                type=["png", "jpg", "jpeg", "bmp"],
                accept_multiple_files=True,
                key="sa_gen_covers",
            )
            pwd_gen = st.text_input("Embedding Password:", value="Pass123!", type="password", key="sa_gen_pwd")
        with g_col2:
            payload_bpp_gen = st.slider("Payload BPP:", 0.01, 0.10, 0.05, 0.005, key="sa_gen_bpp")
            gen_now = st.button("🚀 Embed & Generate Stego Dataset", type="secondary", key="sa_gen_btn")

        if gen_now:
            if not cover_files_gen:
                st.error("Please upload cover images first.")
            else:
                c_imgs, c_names = _load_uploaded_images(cover_files_gen)
                prog_bar = st.progress(0)
                prog_text = st.empty()
                cb = _progress_factory(prog_bar, prog_text)
                with st.spinner("Generating Stego images using CNN-DA-EMD-OLSB..."):
                    c_out, s_out, n_out = generate_stego_dataset(
                        cover_images=c_imgs, cover_names=c_names,
                        password=pwd_gen, payload_bpp=payload_bpp_gen,
                        progress_callback=cb,
                    )
                    st.session_state["sa_gen_c"] = c_out
                    st.session_state["sa_gen_s"] = s_out
                    st.session_state["sa_gen_n"] = n_out
                    prog_bar.progress(100)
                    prog_text.markdown("✅ **Stego dataset generated successfully!**")

        paired_data = []
        is_pairing_ready = False
        if "sa_gen_c" in st.session_state:
            c_list = st.session_state["sa_gen_c"]
            s_list = st.session_state["sa_gen_s"]
            n_list = st.session_state["sa_gen_n"]
            for idx, (c, s, n) in enumerate(zip(c_list, s_list, n_list)):
                if s is not None:
                    paired_data.append({
                        "pair_id": len(paired_data) + 1,
                        "cover_name": n,
                        "stego_name": f"{os.path.splitext(n)[0]}_stego.png",
                        "cover_img": c,
                        "stego_img": s,
                        "shape": c.shape,
                        "match_type": "Auto-generated",
                    })
            is_pairing_ready = len(paired_data) >= 2
            st.success(f"✅ Generated and paired **{len(paired_data)}** valid Cover/Stego pairs.")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4: CONFIGURE CLASSIFIER
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### **STEP 4:** ⚙️ Configure Steganalysis Classifier")

    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        n_epochs = st.slider("Training Epochs:", min_value=5, max_value=50, value=15, step=1, key="sa_epochs_cfg")
        batch_size = st.selectbox("Batch Size:", [16, 32, 64], index=1, key="sa_batch_cfg")
        lr_val = st.selectbox(
            "Learning Rate (AdamW):",
            [0.0001, 0.0005, 0.001, 0.002],
            index=1,
            key="sa_lr_cfg",
        )

    with cfg_col2:
        st.markdown("**Dataset Split (Enforced Strictly by Image Pair):**")
        train_r = st.slider("Training Pairs Ratio:", 0.50, 0.85, 0.70, 0.05, key="sa_train_r_cfg")
        val_r = st.slider("Validation Pairs Ratio:", 0.00, 0.30, 0.15, 0.05, key="sa_val_r_cfg")
        test_r = max(0.0, 1.0 - train_r - val_r)

        # Dynamic preview of pair counts
        if paired_data:
            n_tot = len(paired_data)
            n_tr = max(1, int(round(n_tot * train_r)))
            n_va = max(1, int(round(n_tot * val_r))) if n_tot >= 3 and val_r > 0 else 0
            n_te = max(1, n_tot - n_tr - n_va)
            st.caption(
                f"📊 **Split Preview:** Total: **{n_tot}** pairs → "
                f"Train: **{n_tr}** pairs | Val: **{n_va}** pairs | Unseen Test: **{n_te}** pairs"
            )
            st.caption("🔒 *Zero Data Leakage: Cover and Stego from the same pair never appear in different splits.*")
        else:
            st.caption(f"Test Pairs Ratio: **{test_r:.2f}**")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5: TRAIN & EVALUATE
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### **STEP 5:** 🧠 Train & Evaluate Steganalysis Classifier")

    train_btn = st.button(
        "🧠 Train & Evaluate Classifier",
        type="primary",
        key="sa_exec_train_btn",
        disabled=not is_pairing_ready,
    )

    if train_btn:
        if not paired_data or len(paired_data) < 2:
            st.error("At least 2 valid Cover-Stego pairs are required before training.")
        else:
            prog_bar = st.progress(0)
            prog_text = st.empty()
            cb = _progress_factory(prog_bar, prog_text)

            with st.spinner("Executing SRM-CNN Steganalysis Training & Evaluation..."):
                try:
                    sa_res = run_steganalysis(
                        paired_data=paired_data,
                        train_ratio=train_r,
                        val_ratio=val_r,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        lr=lr_val,
                        progress_callback=cb,
                    )

                    if "error" in sa_res:
                        st.error(sa_res["error"])
                        return

                    out_dir = save_steganalysis_results(sa_res)
                    st.session_state["sa_result"] = sa_res
                    st.session_state["sa_zip"] = get_steganalysis_zip(sa_res)
                    st.session_state["sa_out_dir"] = out_dir

                    prog_bar.progress(100)
                    prog_text.markdown("✅ **Steganalysis training and test evaluation complete!**")
                except Exception as exc:
                    st.error(f"Execution failed: {exc}")
                    import traceback
                    st.code(traceback.format_exc())
                    return

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 6: VIEW RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    if "sa_result" in st.session_state:
        st.markdown("---")
        st.markdown("### **STEP 6:** 📊 Steganalysis Results Dashboard")
        res = st.session_state["sa_result"]
        metrics = res["metrics"]
        split_info = res["split_info"]

        # ── COLLAPSE DETECTION WARNING BANNER ─────────────────────────────────
        if metrics.get("is_collapsed"):
            st.error(f"""
            ### ⚠️ Classifier Collapse Detected
            {metrics.get('collapse_warning')}
            """)
        else:
            st.success("✅ **Balanced Prediction:** The classifier predicted both Cover and Stego classes appropriately.")

        # ── CLASS DISTRIBUTION COMPARISON ─────────────────────────────────────
        st.markdown("#### ⚖️ Test Set Class Distribution: Actual vs. Predicted")
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        d_col1.metric("Actual Cover Patches", metrics["actual_cover_count"])
        d_col2.metric("Actual Stego Patches", metrics["actual_stego_count"])
        d_col3.metric("Predicted Cover Patches", metrics["pred_cover_count"])
        d_col4.metric("Predicted Stego Patches", metrics["pred_stego_count"])

        # ── TOP LEVEL TEST PERFORMANCE METRICS ────────────────────────────────
        st.markdown("#### 🎯 Unseen Test Set Metrics")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        m_col2.metric("Precision", f"{metrics['precision']:.4f}")
        m_col3.metric("Recall (Sensitivity)", f"{metrics['recall']:.4f}")
        m_col4.metric("F1-Score", f"{metrics['f1_score']:.4f}")
        m_col5.metric("ROC-AUC", f"{metrics['roc_auc']}")
        st.caption(f"Target for undetectable stego: **Accuracy ≈ 0.5000**, **ROC-AUC ≈ 0.5000** (Random Guess). Model: `{metrics['classifier']}`")

        # ── VISUALIZATIONS ────────────────────────────────────────────────────
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            if res.get("confusion_matrix_figure"):
                st.markdown("#### 📊 Confusion Matrix (Test Set)")
                st.pyplot(res["confusion_matrix_figure"])

        with fig_col2:
            if res.get("roc_curve_figure"):
                st.markdown("#### 📈 ROC Curve (Test Set)")
                st.pyplot(res["roc_curve_figure"])

        if res.get("training_history_figure"):
            st.markdown("#### 📉 Training & Validation Learning Curves")
            st.pyplot(res["training_history_figure"])

        # ── DIAGNOSTIC EXPANDERS ──────────────────────────────────────────────
        with st.expander("📋 Image-Level Test Predictions Table", expanded=False):
            if "predictions_df" in res and not res["predictions_df"].empty:
                st.dataframe(res["predictions_df"], use_container_width=True)
            else:
                st.info("No test predictions available.")

        with st.expander("🔬 Diagnostic Confusion Matrices (Train / Validation)", expanded=False):
            diag_c1, diag_c2 = st.columns(2)
            with diag_c1:
                if res.get("confusion_matrix_train") is not None:
                    st.markdown("**Training Set Confusion Matrix:**")
                    st.write(res["confusion_matrix_train"])
            with diag_c2:
                if res.get("confusion_matrix_val") is not None:
                    st.markdown("**Validation Set Confusion Matrix:**")
                    st.write(res["confusion_matrix_val"])
                else:
                    st.info("Validation set was 0 pairs (small dataset).")

        with st.expander("📄 Full Markdown Research Report Preview", expanded=False):
            st.markdown(res.get("report_md", "No report generated."))

        # ── SCIENTIFIC RESEARCH INTERPRETATION ────────────────────────────────
        st.markdown("#### 📝 Scientific Interpretation")
        acc_score = metrics["accuracy"]
        if metrics.get("is_collapsed"):
            st.info(
                "**Interpretation:** The classifier predicted all samples as a single class. "
                "This indicates an optimization collapse under the current sample size or learning rate. "
                "The result is inconclusive and should not be cited as empirical evidence for or against detectability."
            )
        elif acc_score > 0.65:
            st.info(
                f"**Interpretation:** The classifier achieved an accuracy of **{acc_score*100:.1f}%**, substantially "
                "above chance level (50%) on the unseen test set. This indicates that the CNN-DA-EMD-OLSB stego images "
                "contain detectable statistical artifacts that allow spatial steganalysis models to distinguish them from clean covers."
            )
        else:
            st.info(
                f"**Interpretation:** The classifier achieved an accuracy of **{acc_score*100:.1f}%**, performing close "
                "to chance level (50%) while predicting both classes. Under this specific classifier architecture, patch size, "
                "and dataset, the stego images exhibited low detectability. *(Note: 50% accuracy under one classifier does not prove universal security against all conceivable steganalysis techniques).*"
            )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 7: DOWNLOAD RESULTS
        # ═══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### **STEP 7:** ⬇️ Download Experiment Artifacts")

        d_col1, d_col2, d_col3, d_col4 = st.columns(4)

        # Metrics CSV
        d_col1.download_button(
            "📥 Metrics CSV",
            data=pd.DataFrame([metrics]).to_csv(index=False).encode("utf-8"),
            file_name="metrics.csv",
            mime="text/csv",
            key="dl_sa_metrics_csv",
        )

        # Predictions CSV
        if "predictions_df" in res and not res["predictions_df"].empty:
            d_col2.download_button(
                "📥 Predictions CSV",
                data=res["predictions_df"].to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                key="dl_sa_preds_csv",
            )

        # Full Report MD
        if "report_md" in res:
            d_col3.download_button(
                "📥 Full Report (MD)",
                data=res["report_md"].encode("utf-8"),
                file_name="report.md",
                mime="text/markdown",
                key="dl_sa_report_md",
            )

        # ZIP Package
        if "sa_zip" in st.session_state:
            d_col4.download_button(
                "📦 Download All (ZIP)",
                data=st.session_state["sa_zip"],
                file_name="steganalysis_results.zip",
                mime="application/zip",
                key="dl_sa_complete_zip",
            )

        # Plot downloads
        pd_col1, pd_col2, pd_col3 = st.columns(3)
        if res.get("confusion_matrix_figure"):
            pd_col1.download_button(
                "📥 Confusion Matrix PNG",
                data=_fig_to_bytes(res["confusion_matrix_figure"]),
                file_name="confusion_matrix.png",
                mime="image/png",
                key="dl_sa_cm_png",
            )
        if res.get("roc_curve_figure"):
            pd_col2.download_button(
                "📥 ROC Curve PNG",
                data=_fig_to_bytes(res["roc_curve_figure"]),
                file_name="roc_curve.png",
                mime="image/png",
                key="dl_sa_roc_png",
            )
        if res.get("training_history_figure"):
            pd_col3.download_button(
                "📥 Training Curves PNG",
                data=_fig_to_bytes(res["training_history_figure"]),
                file_name="training_curves.png",
                mime="image/png",
                key="dl_sa_curves_png",
            )

        if "sa_out_dir" in st.session_state:
            st.caption(f"💾 Results permanently saved to disk at: `{st.session_state['sa_out_dir']}`")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT (called from app.py)
# ═══════════════════════════════════════════════════════════════════════════════

def render_research_evaluation_page():
    """Render the complete 🧪 Research & Evaluation page."""

    # Inject premium CSS
    st.markdown(_RESEARCH_CSS, unsafe_allow_html=True)

    # ── Hero header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(15,23,42,0.94),rgba(30,27,75,0.94));
    border:1px solid rgba(99,102,241,0.28);border-radius:20px;padding:1.6rem 2rem;margin-bottom:1.4rem;
    box-shadow:0 16px 40px -12px rgba(0,0,0,0.45);backdrop-filter:blur(14px);">
        <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.6rem;">
            <span style="font-size:1.8rem;">🧪</span>
            <div style="color:#c084fc;font-weight:700;letter-spacing:1.5px;font-size:0.78rem;
            text-transform:uppercase;">Research Module</div>
        </div>
        <div style="font-size:1.75rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#c084fc,#f472b6);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.5rem;">
            Research &amp; Evaluation Suite
        </div>
        <div style="color:#94a3b8;font-size:0.97rem;line-height:1.6;">
            Four independent experiment modules for systematic evaluation of CNN-DA-EMD-OLSB.<br>
            All results come from <b>actually running the model</b> — no simulated or fabricated data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Experiment configuration info ────────────────────────────────────────
    with st.expander("ℹ️ Experiment Configuration & Reproducibility", expanded=False):
        st.markdown(_render_env_info())

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Payload Capacity",
        "📈 Statistical Testing",
        "🔒 Security Analysis",
        "🕵️ Steganalysis",
    ])

    with tab1:
        _tab_payload_capacity()
    with tab2:
        _tab_statistical_testing()
    with tab3:
        _tab_security_analysis()
    with tab4:
        _tab_steganalysis()
