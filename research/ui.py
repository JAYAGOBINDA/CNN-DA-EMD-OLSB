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
)

# ── Shared styling helpers ────────────────────────────────────────────────────
_CARD = """
<div style="background:rgba(15,23,42,0.7);border:1px solid rgba(99,102,241,0.25);
border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:0.8rem;">
{}</div>"""

_BADGE_OK   = "🟢"
_BADGE_FAIL = "🔴"
_BADGE_WARN = "🟡"

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
            f"**Processing {step}/{total}** — {msg}"
        )
    return _cb


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PAYLOAD CAPACITY
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_payload_capacity():
    st.markdown("#### 📊 Payload Capacity Experiment")
    st.info(
        "Embeds payloads of increasing size into each uploaded cover image using "
        "**CNN-DA-EMD-OLSB** and measures PSNR, SSIM, MSE, BER and recovery accuracy "
        "at each BPP level.  All metrics are computed from actual embedding — no simulated values."
    )

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
                    prog_text.markdown("✅ Experiment complete!")
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
    st.markdown("#### 📈 Statistical Testing — Friedman + Nemenyi + Kendall's W")
    st.info(
        "Compares the **6 benchmark models** using a Friedman χ² test (non-parametric, "
        "matched across the same test images).  If the Friedman test is significant "
        "(p < 0.05) a Nemenyi post-hoc pairwise comparison is performed.  "
        "**Kendall's W** is reported as the effect size."
    )

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
                    prog_text.markdown("✅ Statistical testing complete!")
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
    st.markdown("#### 🔒 Security Analysis")
    st.info(
        "Performs 6 independent security analyses comparing **cover vs stego images**: "
        "Histogram · Entropy · Pixel Correlation · RS Analysis · SPA · Chi-Square PoV.  "
        "Stego images can be generated automatically or uploaded manually."
    )

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
                prog_text.markdown("✅ Security analysis complete!")
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
    st.markdown("#### 🕵️ Steganalysis — Cover vs Stego Classification")
    st.info(
        "Trains a **SRM-inspired CNN** (PyTorch) to classify 64×64 image patches as "
        "cover (class 0) or stego (class 1).  "
        "**Data leakage prevention**: each original cover image and its stego counterpart "
        "are always placed in the **same** split (train / val / test)."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        cover_files = st.file_uploader(
            "Upload Cover Images (≥ 3 recommended):",
            type=["png","jpg","jpeg","bmp"],
            accept_multiple_files=True, key="sa_covers"
        )
        password = st.text_input("Embedding Password:", value="Pass123!",
                                 type="password", key="sa_pass")
        payload_bpp = st.slider("Payload BPP for stego generation:", 0.01, 0.1, 0.05, 0.005,
                                key="sa_bpp")
    with col2:
        n_epochs   = st.slider("Training Epochs:", 5, 30, 15, 1, key="sa_epochs")
        batch_size = st.selectbox("Batch Size:", [16, 32, 64], index=1, key="sa_batch")
        st.markdown("**Split Configuration (by image):**")
        train_r = st.slider("Train ratio:", 0.5, 0.8, 0.70, 0.05, key="sa_train_r")
        val_r   = st.slider("Val ratio:",   0.0, 0.3, 0.15, 0.05, key="sa_val_r")
        st.caption(f"Test ratio: {1.0 - train_r - val_r:.2f}")

    gen_btn   = st.button("🔧 Generate Stego Dataset", key="sa_gen")
    train_btn = st.button("🧠 Train & Evaluate Classifier", type="primary", key="sa_train")

    # ── Stego generation ─────────────────────────────────────────────────────
    if gen_btn:
        if not cover_files:
            st.error("Upload cover images first."); return
        covers, c_names = _load_uploaded_images(cover_files)
        if not covers:
            st.error("No valid images loaded."); return

        prog_bar  = st.progress(0)
        prog_text = st.empty()
        cb = _progress_factory(prog_bar, prog_text)

        with st.spinner("Embedding payloads to create stego dataset …"):
            try:
                cov_out, stg_out, nm_out = generate_stego_dataset(
                    cover_images=covers, cover_names=c_names,
                    password=password, payload_bpp=payload_bpp,
                    progress_callback=cb,
                )
                n_ok = sum(1 for s in stg_out if s is not None)
                st.session_state["sa_covers"] = cov_out
                st.session_state["sa_stegos"] = stg_out
                st.session_state["sa_names"]  = nm_out
                prog_bar.progress(100)
                prog_text.markdown(
                    f"✅ Generated stego for **{n_ok}/{len(covers)}** images.  "
                    f"{len(covers)-n_ok} failed (payload too large for image)."
                )
            except Exception as exc:
                st.error(f"Stego generation failed: {exc}")
                import traceback; st.code(traceback.format_exc())

    if "sa_covers" in st.session_state:
        n_pairs = sum(1 for s in st.session_state["sa_stegos"] if s is not None)
        st.success(f"✅ Stego dataset ready: **{n_pairs}** valid image pairs.")

        # Preview sample
        with st.expander("Preview stego sample"):
            cov_list = st.session_state["sa_covers"]
            stg_list = st.session_state["sa_stegos"]
            nm_list  = st.session_state["sa_names"]
            valid_pairs = [(c,s,n) for c,s,n in zip(cov_list,stg_list,nm_list) if s is not None]
            if valid_pairs:
                c_ex, s_ex, n_ex = valid_pairs[0]
                pc1, pc2 = st.columns(2)
                pc1.image(c_ex, caption=f"Cover: {n_ex}", use_container_width=True)
                pc2.image(s_ex, caption=f"Stego S1: {n_ex}", use_container_width=True)

    # ── Training ─────────────────────────────────────────────────────────────
    if train_btn:
        if "sa_covers" not in st.session_state:
            st.error("Generate the stego dataset first."); return

        prog_bar  = st.progress(0)
        prog_text = st.empty()
        cb = _progress_factory(prog_bar, prog_text)

        with st.spinner("Training steganalysis CNN … (this may take several minutes)"):
            try:
                sa_result = run_steganalysis(
                    cover_images=st.session_state["sa_covers"],
                    stego_images=st.session_state["sa_stegos"],
                    cover_names=st.session_state["sa_names"],
                    train_ratio=train_r,
                    val_ratio=val_r,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    progress_callback=cb,
                )
                if "error" in sa_result:
                    st.error(sa_result["error"]); return

                out_dir = save_steganalysis_results(sa_result)
                st.session_state["sa_result"]  = sa_result
                st.session_state["sa_zip"]     = get_steganalysis_zip(sa_result)
                st.session_state["sa_out_dir"] = out_dir
                prog_bar.progress(100)
                prog_text.markdown("✅ Steganalysis classifier trained and evaluated!")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                import traceback; st.code(traceback.format_exc())

    # ── Results display ───────────────────────────────────────────────────────
    if "sa_result" in st.session_state:
        res = st.session_state["sa_result"]

        # Split info
        if "split_info" in res:
            si = res["split_info"]
            st.markdown("#### 📋 Dataset Split")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Train Pairs",   si.get("train_pairs", "?"))
            sc2.metric("Val Pairs",     si.get("val_pairs",   "?"))
            sc3.metric("Test Pairs",    si.get("test_pairs",  "?"))
            sc4.metric("Train Patches", si.get("train_patches","?"))
            st.caption(f"**Split method:** {si.get('split_method','')}")

        # Metrics
        if "metrics" in res:
            st.markdown("#### 🎯 Test Set Evaluation Metrics")
            m = res["metrics"]
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Accuracy",  m.get("accuracy",  "?"))
            mc2.metric("Precision", m.get("precision", "?"))
            mc3.metric("Recall",    m.get("recall",    "?"))
            mc4.metric("F1-Score",  m.get("f1_score",  "?"))
            mc5.metric("ROC-AUC",   m.get("roc_auc",   "?"))
            st.caption(f"Classifier: {m.get('classifier','')} | Device: {m.get('device','')}")

        fig_col1, fig_col2 = st.columns(2)
        if res.get("confusion_matrix_figure"):
            fig_col1.markdown("#### 📊 Confusion Matrix")
            fig_col1.pyplot(res["confusion_matrix_figure"])
        if res.get("roc_curve_figure"):
            fig_col2.markdown("#### 📈 ROC Curve")
            fig_col2.pyplot(res["roc_curve_figure"])

        if res.get("training_history_figure"):
            st.markdown("#### 📉 Training History")
            st.pyplot(res["training_history_figure"])

        if "training_history" in res and not res["training_history"].empty:
            with st.expander("Training history table"):
                st.dataframe(res["training_history"], use_container_width=True)

        # Downloads
        st.markdown("#### 💾 Download Steganalysis Results")
        d1, d2, d3 = st.columns(3)
        if "metrics" in res:
            import pandas as pd
            d1.download_button("📥 Metrics CSV",
                               data=pd.DataFrame([res["metrics"]]).to_csv(index=False).encode(),
                               file_name="steganalysis_metrics.csv", mime="text/csv",
                               key="dl_sa_metrics")
        if res.get("confusion_matrix_figure"):
            d2.download_button("📥 Confusion Matrix PNG",
                               data=_fig_to_bytes(res["confusion_matrix_figure"]),
                               file_name="confusion_matrix.png", mime="image/png",
                               key="dl_sa_cm")
        if "sa_zip" in st.session_state:
            d3.download_button("📦 Download All (ZIP)",
                               data=st.session_state["sa_zip"],
                               file_name="steganalysis_results.zip",
                               mime="application/zip", key="dl_sa_zip")
        if "sa_out_dir" in st.session_state:
            st.caption(f"💾 Saved to: `{st.session_state['sa_out_dir']}`")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT (called from app.py)
# ═══════════════════════════════════════════════════════════════════════════════

def render_research_evaluation_page():
    """Render the complete 🧪 Research & Evaluation page."""
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(15,23,42,0.92),rgba(30,27,75,0.92));
    border:1px solid rgba(99,102,241,0.28);border-radius:18px;padding:1.4rem 1.6rem;margin-bottom:1.2rem;">
        <div style="color:#c084fc;font-weight:700;letter-spacing:1.5px;font-size:0.78rem;text-transform:uppercase;margin-bottom:0.4rem;">
            🧪 Research Module
        </div>
        <div style="font-size:1.65rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#c084fc,#f472b6);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.5rem;">
            Research &amp; Evaluation Suite
        </div>
        <div style="color:#94a3b8;font-size:0.97rem;line-height:1.55;">
            Four independent experiment modules for systematic evaluation of CNN-DA-EMD-OLSB.<br>
            All results come from <b>actually running the model</b> — no simulated or fabricated data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Experiment configuration info ────────────────────────────────────────
    with st.expander("ℹ️ Experiment Configuration & Reproducibility", expanded=False):
        import sys, torch, numpy, sklearn
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Python | `{sys.version.split()[0]}` |
| PyTorch | `{torch.__version__}` |
| NumPy | `{numpy.__version__}` |
| scikit-learn | `{sklearn.__version__}` |
| Random seed (patches) | `42` |
| Payload seed | Deterministic (image_idx × 1000 + BPP × 10000) |
| Timestamp | `{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}` |
""")

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
