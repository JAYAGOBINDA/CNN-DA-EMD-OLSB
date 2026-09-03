"""
Streamlit Web Application: 6-Model Comparative Steganography Research Platform
Compares Literature Baseline Models (MPEH-RDH, MCSH-RDH, CNN-RDH Predictor, SRDNN-Stego, EMD-OLSB RDH)
against Proposed System: CNN-DA-EMD-OLSB
(CNN-Guided Distortion-Aware Adaptive EMD-OLSB Framework for Reversible Data Hiding in RGB Images)
Fully supports embedding & extracting Text Messages, Secret Images, and Binary/Document Files
with zero-error automatic resolution & compression optimization.
Proposed model uses dual-stego output (S1, S2) for exact cover recovery.
"""

import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import io
import os
import zlib
import matplotlib.pyplot as plt

from benchmark.runner import BenchmarkRunner
from benchmark.metrics import calculate_psnr, calculate_ssim, compute_mse, calculate_wpsnr
from core.cnn_da_emd_olsb import embed_cnn_da_emd_olsb, extract_cnn_da_emd_olsb
from experiments.attacks import run_attack_suite, evaluate_attack_robustness
from utils.image_utils import optimize_secret_image

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="CNN-DA-EMD-OLSB | Reversible Data Hiding Research Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling & Premium Glassmorphism UI Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 27, 75, 0.95) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(16px);
    }
    
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
        border: 1px solid rgba(168, 85, 247, 0.4);
        color: #c084fc;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 1rem;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        max-width: 850px;
        margin: 0 auto;
        line-height: 1.6;
    }

    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 1.8rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px -3px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 10px 25px -5px rgba(124, 58, 237, 0.6) !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 50%, #3b82f6 100%) !important;
        box-shadow: 0 6px 20px -4px rgba(236, 72, 153, 0.5) !important;
    }

    .feature-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3);
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 0.8rem 1rem;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# Hero Header Banner
st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">🔬 Academic Steganography Research Platform</div>
        <div class="hero-title">CNN-DA-EMD-OLSB Research Suite</div>
        <div class="hero-subtitle">
            CNN-Guided Distortion-Aware Adaptive EMD-OLSB Framework for Reversible Data Hiding in RGB Images<br>
            Benchmarking 5 Literature Baselines (MPEH-RDH, MCSH-RDH, CNN-RDH, SRDNN-Stego, EMD-OLSB) vs
            <b>Proposed System: CNN-DA-EMD-OLSB (Dual-Stego RDH)</b>
        </div>
    </div>
""", unsafe_allow_html=True)

# Instantiate Benchmark Runner
runner = BenchmarkRunner()

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Select Module:",
    [
        "🏠 Home / Architecture",
        "📥 Embed Payload (Proposed)",
        "📤 Extract Payload (Proposed)",
        "📊 Research Model Benchmark (6 Models)",
        "🛡️ Robustness Attack Suite",
        "🧪 Research & Evaluation",
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ CNN-DA-EMD-OLSB Parameters")
param_alpha = st.sidebar.slider("Sobel Gradient Weight (α)", 0.0, 1.0, 0.5, 0.05)
param_beta  = st.sidebar.slider("Local Variance Weight (β)", 0.0, 1.0, 0.5, 0.05)
param_gamma = st.sidebar.slider("CNN Blend Factor (γ)", 0.0, 1.0, 0.6, 0.05,
    help="Blend between CNN distortion map (γ=1.0) and analytic Sobel+Var map (γ=0.0)")
param_t1    = st.sidebar.slider("Smooth / Moderate Threshold (T1)", 0.1, 0.5, 0.33, 0.01)
param_t2    = st.sidebar.slider("Moderate / High Threshold (T2)", 0.51, 0.9, 0.66, 0.01)

# Diagnostic verification badge for DistortionCNN weights
cnn_adapter = runner.adapters.get('CNN-DA-EMD-OLSB')
if cnn_adapter and getattr(cnn_adapter.model, '_cnn_trained', False):
    st.sidebar.success("🧠 Loaded trained DistortionCNN from `models/distortion_cnn.pth`")
else:
    st.sidebar.error("❌ Trained DistortionCNN (`models/distortion_cnn.pth`) Not Active")

import base64

def load_image(uploaded_file) -> np.ndarray:
    """Load any uploaded image as uint8 RGB numpy array."""
    image = Image.open(uploaded_file).convert('RGB')
    return np.array(image, dtype=np.uint8)

def image_to_bytes(img_np: np.ndarray) -> bytes:
    """Convert H×W×3 uint8 RGB array to lossless PNG bytes for download."""
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    img_pil = Image.fromarray(img_np)
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

def get_image_download_link(img_np: np.ndarray, filename: str = "stego_image.png", label: str = "💾 Direct Download Stego Image (.png)") -> str:
    """Generate a base64 Data-URI HTML download button that forces Chrome/Windows to save as a valid .png image file with exact filename."""
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    img_pil = Image.fromarray(img_np)
    img_pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    html = f'''
    <a href="data:image/png;base64,{b64}" download="{filename}" target="_blank" style="
        display: inline-block;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: #ffffff !important;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.75rem 1.8rem;
        border-radius: 12px;
        text-decoration: none !important;
        box-shadow: 0 4px 15px -3px rgba(16, 185, 129, 0.4);
        transition: all 0.3s ease;
        margin: 0.5rem 0;
    ">
        {label}
    </a>
    '''
    return html


# PAGE 1: HOME / ARCHITECTURE
if page == "🏠 Home / Architecture":
    st.markdown("### 🎯 Literature Review to Proposed Model Mapping")

    # ---- Architecture overview ----
    st.markdown("#### 🧠 Proposed System: CNN-DA-EMD-OLSB Architecture")
    arch_col1, arch_col2 = st.columns([1, 1])
    with arch_col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📐 4-Phase Pipeline</h4>
            <ol>
                <li><b>Phase 1 — CNN Distortion Maps</b><br>
                    Multi-scale DistortionCNN extracts per-channel sensitivity maps
                    D_r(x,y), D_g(x,y), D_b(x,y) ∈ [0,1] for each color channel.
                </li>
                <li><b>Phase 2 — Adaptive Capacity Assignment</b><br>
                    Each pixel classified into 3 tiers:<br>
                    • Class 0 (smooth, D < T1): EMD embedding<br>
                    • Class 1 (moderate, T1 ≤ D < T2): EMD embedding<br>
                    • Class 2 (textured, D ≥ T2): OLSB embedding
                </li>
                <li><b>Phase 3 — RGB EMD+OLSB Embedding</b><br>
                    R-G coupled via EMD mod-5: f(r,g) = (r + 2g) mod 5.<br>
                    Blue channel: independent adaptive OLSB (3 bits for class 2).
                </li>
                <li><b>Phase 4 — Dual-Stego Reversibility</b><br>
                    Two stego images S1 &amp; S2 produced. Cover recovered exactly:
                    p_orig = round((S1 + S2) / 2). AES-256-GCM payload authentication.
                </li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    with arch_col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Literature Baseline vs Proposed Design Targets</h4>
            <table style="width:100%;border-collapse:collapse;font-size:0.9rem">
                <tr style="border-bottom:1px solid #6366f1">
                    <th align="left">Metric</th>
                    <th>EMD-OLSB (Literature Baseline)</th>
                    <th>CNN-DA-EMD-OLSB (Design Target)</th>
                </tr>
                <tr><td>PSNR</td><td>~38-42 dB (Target)</td><td style="color:#4ade80">~42-48 dB (Target)</td></tr>
                <tr><td>SSIM</td><td>~0.95 (Target)</td><td style="color:#4ade80">&gt;0.97 (Target)</td></tr>
                <tr><td>BPP (capacity)</td><td>~0.5 (Target)</td><td style="color:#4ade80">1.2 – 2.5 (Target)</td></tr>
                <tr><td>BER</td><td>0.0 (Target)</td><td style="color:#4ade80">0.0 (Target)</td></tr>
                <tr><td>Reversibility</td><td>Dual avg</td><td style="color:#4ade80">Dual avg + AES auth</td></tr>
                <tr><td>CNN guidance</td><td>❌ None</td><td style="color:#4ade80">✅ DistortionCNN</td></tr>
                <tr><td>RGB-adaptive</td><td>❌ Flat</td><td style="color:#4ade80">✅ Per-channel D map</td></tr>
                <tr><td>Encryption</td><td>❌ None</td><td style="color:#4ade80">✅ AES-256-GCM</td></tr>
                <tr><td colspan="3" style="font-size:0.75rem;color:#94a3b8;padding-top:8px"><em>Note: Values above represent published literature baseline metrics and design targets (illustrative). All experimental results displayed in other modules are computed dynamically on the actual test images.</em></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 6-Model Comparative Experimental Protocol")
    protocol_df = runner.get_compatibility_protocol_table()
    st.dataframe(protocol_df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧩 Model Descriptions & Papers Mapping")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("""
        <div class="feature-card">
            <h4>1. MPEH-RDH (Paper 1)</h4>
            <p>Multidirectional Prediction Error Histogram RDH using 8-neighbor fluctuation &amp; 4 directional predictor orientations (H, V, D1, D2).</p>
            <h4>2. MCSH-RDH (Paper 2)</h4>
            <p>Multi-Channel Synchronized Histogram RDH for color images with Green-channel inter-color prediction and adaptive channel variance allocation.</p>
            <h4>3. CNN-RDH Predictor (Paper 3)</h4>
            <p>PyTorch CNN pixel predictor network trained with Adam optimizer. Embeds payload into prediction difference G(i,j) = P(i,j) - C(i,j) histogram.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="feature-card">
            <h4>4. SRDNN-Stego (Paper 4)</h4>
            <p>Super-Resolution Deep Neural Network Multi-Image Steganography with 3D Lorenz Chaotic Map Permutation &amp; ECC Key Security.</p>
            <h4>5. EMD-OLSB RDH (Paper 5)</h4>
            <p>Dual-Image Reversible Data Hiding using Exploiting Modification Direction (EMD) mod-5 function with dual-image averaging for cover recovery.</p>
            <h4>6. CNN-DA-EMD-OLSB (Proposed)</h4>
            <p><b>Our proposed system.</b> CNN-Guided Distortion-Aware Adaptive EMD+OLSB with per-channel DistortionCNN maps, R-G EMD coupling, Blue OLSB, AES-256-GCM encryption, and dual-stego reversibility.</p>
        </div>
        """, unsafe_allow_html=True)


# PAGE 2: EMBED PAYLOAD (PROPOSED)
elif page == "📥 Embed Payload (Proposed)":
    st.markdown("### 📥 Embed Secret Data using CNN-DA-EMD-OLSB (Proposed System)")
    st.info(
        "💡 **Proposed System**: CNN-Guided Distortion-Aware Adaptive EMD-OLSB.  \n"
        "Produces **two stego images (S1, S2)** for exact cover recovery via dual-image averaging.  \n"
        "Hiding options: Text Message, Secret Image, or Document / Binary File."
    )

    col1, col2 = st.columns(2)

    cover_rgb = None
    with col1:
        uploaded_cover = st.file_uploader("1. Upload Cover Image:", type=["png", "jpg", "jpeg"])
        password = st.text_input("2. Enter Encryption Password:", type="password", value="Pass123!")

        if uploaded_cover:
            cover_rgb = load_image(uploaded_cover)
            h, w, c = cover_rgb.shape
            from core.cnn_da_emd_olsb import _get_cap_maps, compute_capacity
            upper_c = (cover_rgb & 0xF8).astype(np.uint8)
            cnn_m = runner.adapters['CNN-DA-EMD-OLSB'].model._cnn_model
            cls_r, cls_g, cls_b = _get_cap_maps(upper_c, param_alpha, param_beta, param_gamma, param_t1, param_t2, model=cnn_m)
            cap_info = compute_capacity(cls_r, cls_g, cls_b, upper_c)
            usable_cap_bytes = cap_info['usable_capacity_bytes']
            theo_cap_bytes = cap_info['theoretical_capacity_bytes']
            max_capacity_bytes = max(64, usable_cap_bytes - 128)
            st.image(cover_rgb, caption=f"Cover Image ({w}x{h})", use_container_width=True)
            st.caption(f"Usable Capacity: **{usable_cap_bytes:,} bytes** ({cap_info['usable_capacity_bits']:,} bits) | Theoretical: {theo_cap_bytes:,} bytes")

    with col2:
        payload_option = st.radio(
            "3. Select Payload Type to Hide:",
            ["📝 Text Message", "🖼️ Secret Image", "📄 Document / Binary File"]
        )

        secret_bytes = None
        payload_type = 0

        if payload_option == "📝 Text Message":
            secret_text = st.text_area("Enter Secret Text Message:", "Confidential research data 2026")
            if secret_text:
                raw_b = secret_text.encode('utf-8')
                max_bytes = max_capacity_bytes if cover_rgb is not None else 500000
                if len(raw_b) > max_bytes:
                    secret_bytes = raw_b[:max_bytes]
                    st.warning(f"⚠️ Text payload trimmed to image capacity: {len(secret_bytes):,} bytes")
                else:
                    secret_bytes = raw_b
                    st.caption(f"Payload size: {len(secret_bytes):,} bytes")
                payload_type = 0

        elif payload_option == "🖼️ Secret Image":
            uploaded_secret_img = st.file_uploader("Upload Secret Image to Hide:", type=["png", "jpg", "jpeg", "bmp"])
            if uploaded_secret_img and cover_rgb is not None:
                secret_np = load_image(uploaded_secret_img)
                max_bytes = max_capacity_bytes if cover_rgb is not None else 500000
                opt_bytes, opt_w, opt_h = optimize_secret_image(secret_np, max_bytes)
                secret_bytes = opt_bytes
                payload_type = 1
                st.image(secret_np, caption=f"Secret Image ({secret_np.shape[1]}x{secret_np.shape[0]})", width=200)
                st.success(f"Optimized to {opt_w}x{opt_h} — {len(secret_bytes):,} bytes")

        elif payload_option == "📄 Document / Binary File":
            uploaded_file = st.file_uploader("Upload Document/File to Hide:", type=["pdf", "zip", "txt", "docx", "bin", "dat"])
            if uploaded_file and cover_rgb is not None:
                raw_bytes = uploaded_file.getvalue()
                max_bytes = max_capacity_bytes if cover_rgb is not None else 500000
                compressed_bytes = zlib.compress(raw_bytes, level=9)
                if len(compressed_bytes) <= max_bytes:
                    secret_bytes = compressed_bytes
                    st.info(f"File compressed: {len(secret_bytes):,} bytes")
                else:
                    secret_bytes = compressed_bytes[:max_bytes]
                    st.warning(f"File trimmed to capacity: {len(secret_bytes):,} bytes")
                payload_type = 2

    if cover_rgb is not None and secret_bytes is not None:
        if st.button("🚀 Embed Secret Payload (CNN-DA-EMD-OLSB)", type="primary"):
            with st.spinner("Running CNN distortion maps + adaptive dual-stego embedding..."):
                try:
                    cnn_model = runner.adapters['CNN-DA-EMD-OLSB'].model._cnn_model
                    stego_dual, stats = embed_cnn_da_emd_olsb(
                        cover_rgb=cover_rgb,
                        secret_data=secret_bytes,
                        password=password,
                        alpha=param_alpha,
                        beta=param_beta,
                        gamma=param_gamma,
                        t1=param_t1,
                        t2=param_t2,
                        payload_type=payload_type,
                        model=cnn_model
                    )
                    stego1_rgb, stego2_rgb = stego_dual
                    st.session_state['stego_result'] = {
                        'cover_rgb': cover_rgb,
                        'stego1_rgb': stego1_rgb,
                        'stego2_rgb': stego2_rgb,
                        'stats': stats,
                        'stego1_bytes': image_to_bytes(stego1_rgb),
                        'stego2_bytes': image_to_bytes(stego2_rgb)
                    }
                    st.success("✅ Payload encrypted (AES-256-GCM) and embedded into dual stego images (S1, S2)!")
                except Exception as e:
                    import traceback
                    st.error(f"Embedding Error: {str(e)}")
                    st.code(traceback.format_exc())

    if 'stego_result' in st.session_state and st.session_state['stego_result'] is not None:
        res = st.session_state['stego_result']
        c_rgb = res['cover_rgb']
        s1_rgb = res['stego1_rgb']
        s2_rgb = res['stego2_rgb']
        stats = res['stats']

        psnr1 = round(calculate_psnr(c_rgb, s1_rgb), 2)
        psnr2 = round(calculate_psnr(c_rgb, s2_rgb), 2)
        ssim1 = round(calculate_ssim(c_rgb, s1_rgb), 4)
        ssim2 = round(calculate_ssim(c_rgb, s2_rgb), 4)
        mse1  = round(compute_mse(c_rgb, s1_rgb), 4)
        bpp_val   = stats['bpp']
        max_cap   = stats['max_capacity_bits']

        # ---- Metrics row ----
        st.markdown("#### 📊 Embedding Quality Metrics (Computed on Real Images)")
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("PSNR (S1)", f"{psnr1} dB")
        m2.metric("PSNR (S2)", f"{psnr2} dB")
        m3.metric("SSIM (S1)", f"{ssim1}")
        m4.metric("SSIM (S2)", f"{ssim2}")
        m5.metric("Raw BPP", f"{stats['raw_bpp']}")
        m6.metric("Embedded BPP", f"{stats['embedded_bpp']}")
        m7.metric("Usable Cap", f"{stats['usable_capacity_bits']:,} b")

        # ---- Image comparison ----
        st.markdown("#### 🖼️ Dual Stego Images — Preview & Download")
        
        st.error("⚠️ **CRITICAL: DO NOT right-click to save images!** Browsers compress right-clicked images, destroying pixel-level steganography. **Use the Download buttons below** to save lossless PNGs.")
        
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.image(c_rgb, caption="Original Cover", use_container_width=True, output_format="PNG")
        ic2.image(s1_rgb, caption="Stego Image S1", use_container_width=True, output_format="PNG")
        ic3.image(s2_rgb, caption="Stego Image S2", use_container_width=True, output_format="PNG")

        diff = np.abs(c_rgb.astype(np.int16) - s1_rgb.astype(np.int16)) * 10
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        ic4.image(diff, caption="Difference Map S1 (×10)", use_container_width=True, output_format="PNG")

        # ---- Download buttons ----
        st.markdown("#### 💾 Download Stego Images (PNG — Lossless)")

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.markdown(get_image_download_link(s1_rgb, filename="stego_s1.png", label="💾 Direct Download S1 (stego_s1.png)"), unsafe_allow_html=True)
            st.download_button(
                label="💾 Download Stego S1 via Streamlit",
                data=res['stego1_bytes'],
                file_name="stego_s1.png",
                mime="image/png",
                key="download_stego_s1_png"
            )
        with dl_col2:
            st.markdown(get_image_download_link(s2_rgb, filename="stego_s2.png", label="💾 Direct Download S2 (stego_s2.png)"), unsafe_allow_html=True)
            st.download_button(
                label="💾 Download Stego S2 via Streamlit",
                data=res['stego2_bytes'],
                file_name="stego_s2.png",
                mime="image/png",
                key="download_stego_s2_png"
            )

        st.info(
            f"Raw Secret: {stats['raw_payload_bits']:,} bits ({stats['raw_payload_bytes']:,} bytes) | "
            f"Embedded Bitstream: {stats['embedded_bitstream_bits']:,} bits ({stats['embedded_bitstream_bytes']:,} bytes) | "
            f"Raw Payload BPP: {stats['raw_bpp']} | Embedded Bitstream BPP: {stats['embedded_bpp']} | "
            f"Usable Capacity: {stats['usable_capacity_bits']:,} bits ({stats['usable_capacity_bytes']:,} bytes) | "
            f"Theoretical Capacity: {stats['theoretical_capacity_bits']:,} bits | "
            f"Capacity Utilization: {stats['capacity_utilization_%']}%"
        )


# PAGE 3: EXTRACT PAYLOAD (PROPOSED)
elif page == "📤 Extract Payload (Proposed)":
    st.markdown("### 📤 Extract Secret Payload using CNN-DA-EMD-OLSB (Proposed System)")
    st.info(
        "📌 Upload **both Stego Images (S1 and S2, PNG)** produced during embedding to extract the payload and recover the cover image."
    )

    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        uploaded_stego_s1 = st.file_uploader("1. Upload Stego Image S1 (PNG):", type=["png", "bmp"], key="extract_s1")
    with col_e2:
        uploaded_stego_s2 = st.file_uploader("2. Upload Stego Image S2 (PNG):", type=["png", "bmp"], key="extract_s2")
    with col_e3:
        ext_password = st.text_input("3. Enter Encryption Password:", type="password", value="Pass123!")

    if uploaded_stego_s1 and uploaded_stego_s2 and ext_password:
        stego1_rgb = load_image(uploaded_stego_s1)
        stego2_rgb = load_image(uploaded_stego_s2)
        
        col_p1, col_p2 = st.columns(2)
        col_p1.image(stego1_rgb, caption=f"Stego S1 ({stego1_rgb.shape[1]}x{stego1_rgb.shape[0]})", width=300)
        col_p2.image(stego2_rgb, caption=f"Stego S2 ({stego2_rgb.shape[1]}x{stego2_rgb.shape[0]})", width=300)

        if st.button("🔓 Extract & Decrypt Secret Payload (CNN-DA-EMD-OLSB)", type="primary"):
            with st.spinner("Extracting bitstream and recovering cover using CNN-DA-EMD-OLSB dual-stego..."):
                try:
                    cnn_model = runner.adapters['CNN-DA-EMD-OLSB'].model._cnn_model
                    extracted_bytes, recovered_cover, meta = extract_cnn_da_emd_olsb(
                        stego_dual=(stego1_rgb, stego2_rgb),
                        password=ext_password,
                        alpha=param_alpha,
                        beta=param_beta,
                        gamma=param_gamma,
                        t1=param_t1,
                        t2=param_t2,
                        model=cnn_model
                    )
                    st.session_state['extract_result'] = {
                        'extracted_bytes': extracted_bytes,
                        'recovered_cover': recovered_cover,
                        'recovered_cover_bytes': image_to_bytes(recovered_cover),
                        'meta': meta
                    }
                    st.success("✅ Payload Successfully Extracted & Decrypted! Cover Image Recovered via Dual-Image Averaging.")
                except Exception as e:
                    st.error(f"❌ Extraction Failed: {str(e)}. Ensure uploaded files are valid PNG stego images with correct password.")

    if 'extract_result' in st.session_state and st.session_state['extract_result'] is not None:
        ext_res = st.session_state['extract_result']
        extracted_bytes = ext_res['extracted_bytes']
        recovered_cover = ext_res['recovered_cover']
        recovered_cover_bytes = ext_res['recovered_cover_bytes']
        meta = ext_res['meta']

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📋 Header Metadata")
            st.json(meta)
            st.markdown("#### 🗺️ Recovered Cover Image")
            st.image(recovered_cover, caption="Recovered Cover Image (via dual-image averaging)", use_container_width=True)
            st.markdown(get_image_download_link(recovered_cover, filename="recovered_cover.png", label="💾 Direct Download Recovered Cover (.png)"), unsafe_allow_html=True)
            st.download_button(
                label="💾 Download Recovered Cover (PNG)",
                data=recovered_cover_bytes,
                file_name="recovered_cover.png",
                mime="image/png",
                key="download_recovered_cover_png"
            )

        with col2:
            st.markdown("#### 🔓 Decrypted Output")

            is_image = False
            try:
                from PIL import Image as PILImage
                extracted_img = PILImage.open(io.BytesIO(extracted_bytes))
                st.image(extracted_img, caption="Decrypted Secret Image", width=300)
                img_buf = io.BytesIO()
                extracted_img.save(img_buf, format="PNG")
                st.markdown(get_image_download_link(np.array(extracted_img), filename="extracted_secret_image.png", label="💾 Direct Download Secret Image (.png)"), unsafe_allow_html=True)
                st.download_button(
                    label="💾 Download Extracted Secret Image (PNG)",
                    data=img_buf.getvalue(),
                    file_name="extracted_secret_image.png",
                    mime="image/png",
                    key="download_extracted_secret_png"
                )
                is_image = True
            except Exception:
                is_image = False

            if not is_image:
                try:
                    text_content = extracted_bytes.decode('utf-8')
                    st.text_area("Decrypted Text Message:", text_content, height=150)
                    st.download_button(
                        label="💾 Download Decrypted Text (.txt)",
                        data=text_content.encode('utf-8'),
                        file_name="decrypted_text.txt",
                        mime="text/plain",
                        key="download_decrypted_txt"
                    )
                except Exception:
                    st.info("📄 Binary file detected.")
                    st.download_button(
                        label="💾 Download Decrypted Binary File (.bin)",
                        data=extracted_bytes,
                        file_name="extracted_file.bin",
                        mime="application/octet-stream",
                        key="download_decrypted_bin"
                    )


# PAGE 4: RESEARCH MODEL BENCHMARK (6 MODELS)
elif page == "📊 Research Model Benchmark (6 Models)":
    st.markdown("### 📊 Research Model Benchmark Suite")
    st.info("⚡ Compares 5 literature-derived baseline models against **Proposed CNN-DA-EMD-OLSB** on identical cover image and payload conditions.")

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        bench_cover = st.file_uploader("Upload Cover Image for Benchmark:", type=["png", "jpg", "jpeg"])
        model_choice = st.selectbox(
            "Select Model to Benchmark:",
            ["Run All Models", "MPEH-RDH", "MCSH-RDH", "CNN-RDH Predictor", "SRDNN-Stego", "EMD-OLSB RDH", "CNN-DA-EMD-OLSB"]
        )

    with b_col2:
        bench_payload_mode = st.radio("Payload Level:", ["Low (256 B)", "Medium (1 KB)", "High (4 KB)"])
        bench_pass = st.text_input("Benchmark Password:", type="password", value="Pass123!")

    if bench_payload_mode == "Low (256 B)":
        bench_payload = "A" * 256
    elif bench_payload_mode == "Medium (1 KB)":
        bench_payload = "B" * 1024
    else:
        bench_payload = "C" * 4096

    if bench_cover:
        cover_rgb = load_image(bench_cover)

        if st.button("🚀 Execute Research Model Benchmark", type="primary"):
            with st.spinner("Executing benchmark across selected research model(s)..."):
                if model_choice == "Run All Models":
                    df_results = runner.run_all_models(cover_rgb, bench_payload, password=bench_pass)
                else:
                    res = runner.run_single_model(model_choice, cover_rgb, bench_payload, password=bench_pass)
                    df_results = pd.DataFrame([res])

                st.subheader("📋 6-Model Numerical Comparison Table")
                display_cols = ['Model', 'PSNR_dB', 'SSIM', 'MSE', 'wPSNR_dB', 'BPP', 'BER', 'Payload_Recovery_Acc_%', 'Carrier_Recovery_Acc_%', 'Embed_Time_s', 'Extract_Time_s']
                st.dataframe(df_results[display_cols], use_container_width=True)

                # Export CSV
                csv_bytes = df_results[display_cols].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Benchmark Results CSV",
                    data=csv_bytes,
                    file_name="6_model_benchmark_results.csv",
                    mime="text/csv"
                )

                # Display Stego & Carrier Recovery Images
                st.subheader("🖼️ Stego & Carrier Recovery Inspection")
                cols = st.columns(len(df_results))
                for idx, row in df_results.iterrows():
                    col = cols[idx]
                    model_name = row['Model']
                    stego_out = row['Stego_Output']
                    rec_cover = row['Carrier_Cover'] if 'Carrier_Cover' in row else row.get('Recovered_Cover', None)

                    with col:
                        if isinstance(stego_out, tuple):  # Dual Stego (EMD-OLSB RDH & CNN-DA-EMD-OLSB)
                            s1, s2 = stego_out
                            st.image(s1, caption=f"{model_name}\nStego Image S1", use_container_width=True)
                            st.image(s2, caption=f"{model_name}\nStego Image S2", use_container_width=True)
                        else:
                            st.image(stego_out, caption=f"{model_name}\nStego Photo", use_container_width=True)

                        if rec_cover is not None:
                            st.image(rec_cover, caption=f"Recovered Carrier\n(Accuracy: {row['Carrier_Recovery_Acc_%']}%)", use_container_width=True)

                # Research Plots
                st.subheader("📊 Publication Quality Comparison Graphs")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                colors = ['#8b5cf6' if m == 'CNN-DA-EMD-OLSB' else '#3b82f6' for m in df_results['Model']]

                ax1.bar(df_results['Model'], df_results['PSNR_dB'], color=colors, width=0.5)
                ax1.set_title("PSNR (dB) Comparison Across Models", fontsize=11, fontweight='bold')
                ax1.set_ylabel("PSNR (dB)")
                ax1.tick_params(axis='x', rotation=30)

                ax2.bar(df_results['Model'], df_results['SSIM'], color=colors, width=0.5)
                ax2.set_title("SSIM Index Comparison Across Models", fontsize=11, fontweight='bold')
                ax2.set_ylabel("SSIM Index")
                ax2.tick_params(axis='x', rotation=30)

                plt.tight_layout()
                st.pyplot(fig)


# PAGE 5: ROBUSTNESS ATTACK SUITE
elif page == "🛡️ Robustness Attack Suite":
    st.markdown("### 🛡️ Robustness & Bit Error Analysis Suite")
    st.info("📌 Upload **both Stego Images (S1, S2)** from the CNN-DA-EMD-OLSB embedding. Attacks are applied to S1 only; S2 remains clean for cover recovery reference.")
    
    att_col1, att_col2 = st.columns(2)
    with att_col1:
        attack_s1_file = st.file_uploader("Upload Stego Image S1:", type=["png", "bmp"], key="attack_s1")
    with att_col2:
        attack_s2_file = st.file_uploader("Upload Stego Image S2:", type=["png", "bmp"], key="attack_s2")
    
    attack_pass = st.text_input("Password used during embedding:", type="password", value="Pass123!")

    if attack_s1_file and attack_s2_file and attack_pass:
        stego_s1 = load_image(attack_s1_file)
        stego_s2 = load_image(attack_s2_file)
        
        if st.button("⚔️ Execute Robustness Attack Analysis", type="primary"):
            with st.spinner("Simulating spatial & lossy attacks on S1..."):
                attacks_dict = run_attack_suite(stego_s1)
                
                attack_summary = []
                for attack_name, attacked_s1 in attacks_dict.items():
                    res = evaluate_attack_robustness(
                        clean_stego_s1=stego_s1,
                        attacked_stego_s1=attacked_s1,
                        clean_stego_s2=stego_s2,
                        password=attack_pass
                    )
                    attack_summary.append({
                        'Attack Type': attack_name,
                        'PSNR (dB)': res['PSNR_dB'],
                        'SSIM': res['SSIM'],
                        'BER': res['BER'],
                        'Bit Recovery Acc (%)': res['Bit_Recovery_Acc_%'],
                        'Status': res['GCM_Payload_Status']
                    })

                df_att = pd.DataFrame(attack_summary)
                st.dataframe(df_att, use_container_width=True)

                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.bar(df_att['Attack Type'], df_att['BER'], color='#ef4444', width=0.5)
                ax.set_title("Bit Error Rate (BER) per Attack Type")
                ax.set_ylabel("BER (0.0 to 1.0)")
                ax.tick_params(axis='x', rotation=30)
                plt.tight_layout()
                st.pyplot(fig)


# PAGE 6: RESEARCH & EVALUATION
elif page == "🧪 Research & Evaluation":
    from research.ui import render_research_evaluation_page
    render_research_evaluation_page()
