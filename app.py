import os
import hashlib
import tempfile
import pandas as pd
import streamlit as st
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    find_model_file,
    load_unet_checkpoint,
    load_xgboost_model,
    load_calibrated_survival_model,
    load_dicom_series_from_zip,
    generate_mask_from_model,
    adjust_mask_margins,
    extract_radiomics_features,
    compute_tumor_anatomy,
    compute_segmentation_metrics,
    predict_survival_and_explain,
    build_3d_tumor_figure,
    build_shap_plotly_figure,
    build_risk_gauge_figure,
    build_survival_curve_plotly_figure,
    generate_clinical_report_html,
    align_longitudinal_masks,
    compute_longitudinal_progression,
    build_3d_longitudinal_diff_figure,
    compute_gtv_distance_map,
    compute_radiotherapy_plan,
    build_ctv_tradeoff_figure,
    build_ctv_breakdown_figure,
)
from ui_components import (
    load_app_css,
    render_sidebar_brand,
    render_hero_showcase,
    render_feature_card,
    modern_kpi_html,
    render_prognosis_card,
    render_compartments_banner,
    render_impression_box,
    render_clinical_notice,
    build_biomarker_reference_table,
    build_radiological_impression,
)

st.set_page_config(
    page_title="NeuroXAI • AI Clinical Diagnostic Station",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load global clinical dark workstation theme from external stylesheet
load_app_css()


# ── Navigation State ───────────────────────────────────────────────────────────
NAV_PAGES = [
    "Clinical Dashboard",
    "Multi-Planar & 3D Lab",
    "Clinical Report & Biomarkers",
    "Longitudinal Tracking",
]

if "active_nav" not in st.session_state:
    st.session_state.active_nav = NAV_PAGES[0]


# ── Cached Model & Scan Processors ─────────────────────────────────────────────
@st.cache_resource
def get_models():
    unet_path = find_model_file("best_unet_brats.pth", fallback_ext=".pth")
    unet_model, unet_meta, _ = load_unet_checkpoint(unet_path)
    xgb_model = load_xgboost_model()
    cph_model = load_calibrated_survival_model()
    return unet_model, unet_meta, xgb_model, cph_model


def _load_scan_from_bytes(raw_bytes, filename):
    """Loads scan from bytes: supports 3D NIfTI (.nii, .nii.gz) and hospital DICOM ZIP archives."""
    digest = hashlib.md5(raw_bytes).hexdigest()
    lower = filename.lower()

    if lower.endswith(".zip"):
        vol, zooms, meta, err = load_dicom_series_from_zip(raw_bytes, filename)
        if err or vol is None:
            raise ValueError(err or "Unable to extract DICOM ZIP")
        return vol, zooms, digest, meta

    suffix = ".nii.gz" if lower.endswith(".gz") else ".nii"
    temp_path = os.path.join(tempfile.gettempdir(), f"cdss_{digest}{suffix}")
    if not os.path.exists(temp_path):
        with open(temp_path, "wb") as f:
            f.write(raw_bytes)
    img = nib.load(temp_path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, zooms, digest, {}


@st.cache_data(show_spinner=False)
def process_patient_scan(
    primary_bytes,
    primary_name,
    extra_modalities,
    gt_bytes,
    gt_name,
    mask_mode,
    margin_adjust_mm,
    _unet_model,
    _unet_meta,
):
    primary_vol, zooms, primary_hash, dicom_meta = _load_scan_from_bytes(primary_bytes, primary_name)
    key_parts = [primary_hash, mask_mode, f"m_{margin_adjust_mm}"]

    volumes = [primary_vol]
    for b, n in extra_modalities:
        v, _, h, _ = _load_scan_from_bytes(b, n)
        if v.shape == primary_vol.shape:
            volumes.append(v)
            key_parts.append(h)

    gt_mask = None
    if gt_bytes is not None:
        gt_mask, _, gt_hash, _ = _load_scan_from_bytes(gt_bytes, gt_name)
        key_parts.append(gt_hash)

    if gt_mask is not None and mask_mode == "Override with Ground-Truth":
        raw_mask = gt_mask
        seg_source = "Ground-Truth Mask"
        val_metrics = None
    else:
        raw_mask, seg_source = generate_mask_from_model(volumes, _unet_model, _unet_meta)
        val_metrics = (
            compute_segmentation_metrics(raw_mask, gt_mask)
            if (gt_mask is not None and gt_mask.shape == raw_mask.shape)
            else None
        )

    # Human-in-the-Loop 3D margin refinement
    if abs(margin_adjust_mm) >= 0.1:
        active_mask = adjust_mask_margins(raw_mask, margin_adjust_mm, zooms)
        seg_source += f" • Refined ({margin_adjust_mm:+.1f}mm)"
    else:
        active_mask = raw_mask

    features_df = extract_radiomics_features(primary_vol, zooms, active_mask)
    anatomy = compute_tumor_anatomy(primary_vol, active_mask, zooms)
    cache_key = "_".join(key_parts)
    return primary_vol, active_mask, gt_mask, zooms, seg_source, features_df, anatomy, val_metrics, cache_key, dicom_meta


@st.cache_data(show_spinner=False)
def get_cached_3d_figure(cache_key, _img_data, _mask_data, zooms, shell_opacity, tumor_color, camera_preset="iso"):
    return build_3d_tumor_figure(_img_data, _mask_data, zooms, shell_opacity, tumor_color, camera_preset=camera_preset)


@st.cache_data(show_spinner=False)
def get_cached_gtv_distance_map(cache_key, _img_data, _mask_data, zooms):
    return compute_gtv_distance_map(_img_data, _mask_data, zooms)


@st.cache_data(show_spinner=False)
def process_followup_scan(followup_bytes, followup_name, gt_bytes, gt_name, _unet_model, _unet_meta):
    vol, zooms, f_hash, meta = _load_scan_from_bytes(followup_bytes, followup_name)
    gt_mask = None
    if gt_bytes is not None:
        try:
            gt_mask, _, _, _ = _load_scan_from_bytes(gt_bytes, gt_name)
        except Exception:
            gt_mask = None

    if gt_mask is not None and gt_mask.shape == vol.shape:
        raw_mask = gt_mask.astype(np.uint8)
    else:
        raw_mask, _ = generate_mask_from_model([vol], _unet_model, _unet_meta)

    anatomy = compute_tumor_anatomy(vol, raw_mask, zooms)
    return vol, raw_mask, zooms, anatomy, f_hash, meta


@st.cache_data(show_spinner=False)
def get_cached_longitudinal_diff_figure(
    cache_key,
    _vol_t0,
    _unchanged_mask,
    _regressed_mask,
    _new_growth_mask,
    voxel_dims,
    shell_opacity,
    show_unchanged,
    show_regressed,
    show_growth,
):
    return build_3d_longitudinal_diff_figure(
        _vol_t0,
        _unchanged_mask,
        _regressed_mask,
        _new_growth_mask,
        voxel_dims,
        shell_opacity=shell_opacity,
        show_unchanged=show_unchanged,
        show_regressed=show_regressed,
        show_growth=show_growth,
    )


def render_pacs_slice_figure(
    slice_2d,
    mask_2d=None,
    aspect=1.0,
    show_mask=True,
    opacity=0.5,
    contour_only=False,
    plane="axial",
    window_width=None,
    window_center=None,
    color_mode="Single Highlight (Red)",
    patient_id="",
    slice_idx=0,
    max_slices=0,
    slice_thickness=1.0,
    pixel_spacing=1.0,
    ctv_2d=None,
    show_ctv=False,
):
    """Renders hospital-grade PACS viewport with true DICOM corner HUD, orientation tags, and physical scale bar."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=105)
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")

    # Fast Vectorized Window / Level Contrast
    if window_width is not None and window_center is not None and window_width > 0:
        low = window_center - (window_width / 2.0)
        slice_disp = np.clip((slice_2d - low) / window_width, 0.0, 1.0)
    else:
        nz = slice_2d > 0
        p99 = np.percentile(slice_2d[nz], 99) if np.any(nz) else 1.0
        slice_disp = np.clip(slice_2d / max(p99, 1e-4), 0.0, 1.0)

    ax.imshow(slice_disp, cmap="gray", aspect=aspect)

    # Multi-Compartment or Single-Color Tumour Mask Overlay
    if show_mask and mask_2d is not None and np.any(mask_2d > 0):
        if contour_only:
            ax.contour(mask_2d > 0, levels=[0.5], colors=["#FF3B30"], linewidths=1.8)
        elif color_mode == "BraTS Multi-Compartment" and int(np.max(mask_2d)) > 1:
            cmap_custom = plt.matplotlib.colors.ListedColormap(
                ["#00000000", "#06B6D4", "#FACC15", "#EF4444"]
            )
            bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
            norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap_custom.N)
            overlay = np.ma.masked_where(mask_2d == 0, mask_2d)
            ax.imshow(overlay, cmap=cmap_custom, norm=norm, alpha=opacity, aspect=aspect)
        else:
            overlay = np.ma.masked_where(mask_2d == 0, mask_2d > 0)
            ax.imshow(overlay, cmap="autumn", alpha=opacity, aspect=aspect)

    # Radiotherapy CTV Safety Margin Contour (dashed amber ring clipped to brain)
    if show_ctv and ctv_2d is not None and np.any(ctv_2d > 0):
        ax.contour(ctv_2d > 0, levels=[0.5], colors=["#F59E0B"], linewidths=1.7, linestyles="--")

    # Radiological Orientation Markers
    tag_props = dict(
        fontsize=9,
        fontweight="bold",
        color="#2DD4BF",
        bbox=dict(boxstyle="circle,pad=0.25", facecolor="#020617", alpha=0.75, edgecolor="#14B8A6", linewidth=0.5),
    )
    if plane == "axial":
        ax.text(0.5, 0.96, "A", transform=ax.transAxes, ha="center", va="top", **tag_props)
        ax.text(0.5, 0.04, "P", transform=ax.transAxes, ha="center", va="bottom", **tag_props)
        ax.text(0.04, 0.5, "R", transform=ax.transAxes, ha="left", va="center", **tag_props)
        ax.text(0.96, 0.5, "L", transform=ax.transAxes, ha="right", va="center", **tag_props)
    elif plane == "coronal":
        ax.text(0.5, 0.96, "S", transform=ax.transAxes, ha="center", va="top", **tag_props)
        ax.text(0.5, 0.04, "I", transform=ax.transAxes, ha="center", va="bottom", **tag_props)
        ax.text(0.04, 0.5, "R", transform=ax.transAxes, ha="left", va="center", **tag_props)
        ax.text(0.96, 0.5, "L", transform=ax.transAxes, ha="right", va="center", **tag_props)
    elif plane == "sagittal":
        ax.text(0.5, 0.96, "S", transform=ax.transAxes, ha="center", va="top", **tag_props)
        ax.text(0.5, 0.04, "I", transform=ax.transAxes, ha="center", va="bottom", **tag_props)
        ax.text(0.04, 0.5, "A", transform=ax.transAxes, ha="left", va="center", **tag_props)
        ax.text(0.96, 0.5, "P", transform=ax.transAxes, ha="right", va="center", **tag_props)

    # Compact Corner Slice Readout
    hud_font = dict(fontsize=8, fontfamily="monospace", color="#2DD4BF", fontweight="bold")
    plane_prefix = "Z" if plane == "axial" else ("Y" if plane == "coronal" else "X")
    ax.text(0.03, 0.96, f"{plane_prefix}: {slice_idx+1}/{max_slices}", transform=ax.transAxes, ha="left", va="top", **hud_font)

    # Corner 30mm Physical Scale Bar
    if pixel_spacing > 0:
        scale_len_px = 30.0 / max(pixel_spacing, 0.1)
        h, w = slice_2d.shape[:2]
        x_end = w - 16
        x_start = max(10, x_end - scale_len_px)
        y_pos = h - 22
        ax.plot([x_start, x_end], [y_pos, y_pos], color="#14B8A6", linewidth=2.0)
        ax.text((x_start + x_end) / 2.0, y_pos - 4, "30 mm", color="#2DD4BF", fontsize=6.8, ha="center", va="bottom", fontfamily="monospace", fontweight="bold")

    ax.axis("off")
    plt.tight_layout(pad=0.04)
    return fig


def render_longitudinal_diff_slice_figure(
    mri_2d,
    unchanged_2d,
    regressed_2d,
    new_growth_2d,
    slice_idx=0,
    max_slices=1,
    pixel_spacing=1.0,
    aspect=1.0,
):
    """Renders Follow-Up (T1) axial MRI with high-contrast RGBA progression/regression overlay."""
    fig, ax = plt.subplots(figsize=(5.6, 5.6), facecolor="#060911")
    ax.set_facecolor("#060911")

    img = np.asarray(mri_2d, dtype=np.float32)
    nz = img[img > 0]
    vmax = float(np.percentile(nz, 99.2)) if len(nz) > 0 else 1.0
    vmin = float(np.percentile(nz, 1.0)) if len(nz) > 0 else 0.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, aspect=aspect, interpolation="bilinear", zorder=1)

    h, w = img.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    u_bin = np.asarray(unchanged_2d) > 0
    r_bin = np.asarray(regressed_2d) > 0
    g_bin = np.asarray(new_growth_2d) > 0

    # High-contrast translucent overlays (float32 0.0–1.0)
    # Persisting / Unchanged tumour: Surgical Cyan-Blue (visible even on bright FLAIR)
    overlay[u_bin] = [0.22, 0.74, 0.97, 0.45]
    # Regressed / Shrunk tumour (present at T0, gone at T1): Emerald Green
    overlay[r_bin] = [0.06, 0.82, 0.48, 0.55]
    # New Growth / Progression (new at T1): Vivid Crimson Red
    overlay[g_bin] = [0.96, 0.25, 0.37, 0.62]

    ax.imshow(overlay, aspect=aspect, interpolation="nearest", zorder=2)

    # Crisp boundary contours for instant visual separation
    if np.any(u_bin):
        ax.contour(u_bin.astype(np.float32), levels=[0.5], colors=["#38BDF8"], linewidths=1.2, zorder=3)
    if np.any(r_bin):
        ax.contour(r_bin.astype(np.float32), levels=[0.5], colors=["#10B981"], linewidths=1.5, linestyles="--", zorder=4)
    if np.any(g_bin):
        ax.contour(g_bin.astype(np.float32), levels=[0.5], colors=["#FB7185"], linewidths=1.6, zorder=5)

    # Radiological orientation & slice metadata
    ax.text(0.03, 0.96, f"Z: {slice_idx + 1}/{max_slices}", transform=ax.transAxes,
            color="#2DD4BF", fontsize=9, fontweight="bold", va="top", family="monospace", zorder=6)
    ax.text(0.50, 0.96, "A", transform=ax.transAxes, color="#2DD4BF", fontsize=9, ha="center", va="top",
            bbox=dict(boxstyle="circle,pad=0.2", fc="#0B1120", ec="#2DD4BF", lw=0.8), zorder=6)
    ax.text(0.50, 0.04, "P", transform=ax.transAxes, color="#2DD4BF", fontsize=9, ha="center", va="bottom",
            bbox=dict(boxstyle="circle,pad=0.2", fc="#0B1120", ec="#2DD4BF", lw=0.8), zorder=6)
    ax.text(0.04, 0.50, "R", transform=ax.transAxes, color="#2DD4BF", fontsize=9, ha="left", va="center",
            bbox=dict(boxstyle="circle,pad=0.2", fc="#0B1120", ec="#2DD4BF", lw=0.8), zorder=6)
    ax.text(0.96, 0.50, "L", transform=ax.transAxes, color="#2DD4BF", fontsize=9, ha="right", va="center",
            bbox=dict(boxstyle="circle,pad=0.2", fc="#0B1120", ec="#2DD4BF", lw=0.8), zorder=6)

    # Physical 30 mm scale bar
    if pixel_spacing and pixel_spacing > 0:
        bar_px = 30.0 / float(pixel_spacing)
        x_end = w * 0.93
        x_start = max(w * 0.60, x_end - bar_px)
        y_pos = h * 0.93
        ax.plot([x_start, x_end], [y_pos, y_pos], color="#2DD4BF", lw=2.0, zorder=6)
        ax.text((x_start + x_end) / 2.0, y_pos - h * 0.025, "30 mm", color="#2DD4BF",
                fontsize=8, ha="center", va="bottom", family="monospace", zorder=6)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.2)
    return fig


# ── Load Models & Render Sidebar Inputs ────────────────────────────────────────
unet_model, unet_meta, xgb_model, cph_model = get_models()

with st.sidebar:
    render_sidebar_brand()

    uploaded_mri = st.file_uploader(
        "Primary MRI Scan",
        type=["nii", "gz", "zip"],
        help="Upload 3D FLAIR MRI (.nii, .nii.gz, or .zip DICOM archive)",
    )

    # Allow clearing demo if user uploads real scan
    if uploaded_mri is not None and "demo_active" in st.session_state:
        del st.session_state["demo_active"]

    uploaded_gt = st.file_uploader(
        "Ground-Truth Mask (Optional)",
        type=["nii", "gz"],
        help="Optional BraTS _seg.nii / .nii.gz mask for Dice & IoU benchmarking",
    )

    mask_mode = "Benchmark U-Net (Dice / IoU)"
    if uploaded_gt is not None:
        mask_mode = st.radio(
            "Ground-Truth Mode",
            ["Benchmark U-Net (Dice / IoU)", "Override with Ground-Truth"],
            help="Benchmark model accuracy against Ground Truth or view Ground Truth directly.",
        )

    st.divider()

    patient_id_input = st.text_input("Patient Identifier", value="", placeholder="Auto from scan metadata")
    patient_age = st.slider("Patient Age (Years)", min_value=1, max_value=100, value=55)

    # 🎯 Human-in-the-Loop Margin Adjustment
    margin_adjust_mm = 0.0
    with st.expander("🎯 Clinician Margin Refinement (HITL)", expanded=False):
        st.caption("Human-in-the-loop: dilate or erode 3D boundaries in millimeters before locking radiomics")
        margin_adjust_mm = st.slider(
            "Boundary Refinement (mm)",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.5,
            help="Erode (-mm) to trim false-positive edges or dilate (+mm) to capture infiltrative edema.",
        )

    with st.expander("Optional: Multi-Modal BraTS Scans", expanded=False):
        t1_file = st.file_uploader("T1 Scan", type=["nii", "gz", "zip"], key="t1")
        t1ce_file = st.file_uploader("T1ce Scan", type=["nii", "gz", "zip"], key="t1ce")
        t2_file = st.file_uploader("T2 Scan", type=["nii", "gz", "zip"], key="t2")


# ── Active Case Resolution (Uploaded Scan or One-Click Demo) ───────────────────
active_scan_bytes = None
active_scan_name = None

if uploaded_mri is not None:
    active_scan_bytes = uploaded_mri.getvalue()
    active_scan_name = uploaded_mri.name
elif st.session_state.get("demo_active", False) and "demo_scan_bytes" in st.session_state:
    active_scan_bytes = st.session_state["demo_scan_bytes"]
    active_scan_name = st.session_state["demo_scan_name"]


# ── Top Navigation Bar (Large Button Cards) ────────────────────────────────────
with st.container(key="top_nav_container"):
    col1, col2, col3, col4 = st.columns(4, gap="small")
    with col1:
        is_dash = "Clinical Dashboard" in st.session_state.active_nav
        if st.button(
            "Clinical Dashboard",
            key="nav_btn_dash",
            use_container_width=True,
            type="primary" if is_dash else "secondary",
        ):
            st.session_state.active_nav = "Clinical Dashboard"
            st.rerun()

    with col2:
        is_lab = "Multi-Planar" in st.session_state.active_nav
        if st.button(
            "Multi-Planar & 3D Lab",
            key="nav_btn_lab",
            use_container_width=True,
            type="primary" if is_lab else "secondary",
        ):
            st.session_state.active_nav = "Multi-Planar & 3D Lab"
            st.rerun()

    with col3:
        is_rep = "Clinical Report" in st.session_state.active_nav
        if st.button(
            "Clinical Report & Biomarkers",
            key="nav_btn_rep",
            use_container_width=True,
            type="primary" if is_rep else "secondary",
        ):
            st.session_state.active_nav = "Clinical Report & Biomarkers"
            st.rerun()

    with col4:
        is_long = "Longitudinal Tracking" in st.session_state.active_nav
        if st.button(
            "Longitudinal Tracking",
            key="nav_btn_long",
            use_container_width=True,
            type="primary" if is_long else "secondary",
        ):
            st.session_state.active_nav = "Longitudinal Tracking"
            st.rerun()

page = st.session_state.active_nav


# ── Showcase Hero State When No Scan Uploaded ──────────────────────────────────
if active_scan_bytes is None:
    if "Longitudinal Tracking" in page:
        st.markdown(
            """
            <div class="hero-showcase-container" style="padding: 28px 32px; margin-top: 10px;">
              <div class="hero-badge-pill">Longitudinal Neuro-Oncology Surveillance</div>
              <h2 style="color:#F8FAFC;font-size:1.45rem;font-weight:700;margin:12px 0 8px 0;">
                Baseline (T0) MRI Scan Required
              </h2>
              <p style="color:#94A3B8;font-size:0.92rem;line-height:1.65;margin:0;max-width:880px;">
                Please upload a <b>Primary MRI Scan (Baseline T0)</b> in the left sidebar to activate automated 3D longitudinal subtraction and RANO volumetric progression tracking.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        render_hero_showcase()

    c1, c2, c3 = st.columns(3)
    with c1:
        render_feature_card(
            icon="🔬",
            icon_bg="rgba(20,184,166,0.15)",
            icon_color="#2DD4BF",
            title="1. Volumetrics & HITL",
            description="Sub-millimeter multi-compartment BraTS delineation (ET, TC, ED) with 3D connected components and live clinician boundary adjustment.",
        )
    with c2:
        render_feature_card(
            icon="📈",
            icon_bg="rgba(129,140,248,0.15)",
            icon_color="#818CF8",
            title="2. Calibrated Survival",
            description="Continuous patient survival probability curve S(t) calibrated via Cox Proportional Hazards with median days and 6m/12m/24m horizons.",
        )
    with c3:
        render_feature_card(
            icon="🩺",
            icon_bg="rgba(52,211,153,0.15)",
            icon_color="#34D399",
            title="3. PACS & Tri-Planar Lab",
            description="Hospital PACS viewport with clinical presets, synchronized tri-planar views, and interactive 360° WebGL 3D tumor surface meshes.",
        )
    st.stop()
    raise SystemExit(0)


# ── Process Uploaded Scan (Cached Across All Pages) ────────────────────────────
extra_mods = tuple(
    (f.getvalue(), f.name) for f in (t1_file, t1ce_file, t2_file) if f is not None
)
gt_bytes = uploaded_gt.getvalue() if uploaded_gt is not None else None
gt_name = uploaded_gt.name if uploaded_gt is not None else None

try:
    with st.spinner("Analysing 3D MRI volume & computing IBSI radiomic biomarkers..."):
        img_data, mask_data, gt_mask, zooms, seg_source, features_df, anatomy, val_metrics, cache_key, dicom_meta = (
            process_patient_scan(
                active_scan_bytes,
                active_scan_name,
                extra_mods,
                gt_bytes,
                gt_name,
                mask_mode,
                margin_adjust_mm,
                unet_model,
                unet_meta,
            )
        )
except Exception as e:
    st.error(f"Error processing scan file: {str(e)}")
    st.stop()

# Auto-derive Patient ID from user input, DICOM header, or scan filename
if patient_id_input.strip():
    display_pid = patient_id_input.strip()
elif dicom_meta.get("patient_id"):
    display_pid = dicom_meta["patient_id"]
else:
    display_pid = active_scan_name.split(".")[0].upper()

pred = predict_survival_and_explain(
    features_df=features_df,
    age=patient_age,
    xgb_model=xgb_model,
    cph_model=cph_model,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: CLINICAL DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if "Clinical Dashboard" in page:
    rt_margin_key = f"rt_margin_{cache_key}"
    if rt_margin_key not in st.session_state:
        st.session_state[rt_margin_key] = 15

    dist_map_mm, brain_mask = get_cached_gtv_distance_map(cache_key, img_data, mask_data, zooms)
    rt_plan = compute_radiotherapy_plan(
        dist_map_mm,
        brain_mask,
        mask_data,
        zooms,
        margin_mm=st.session_state[rt_margin_key],
    )

    badge_class = f"badge-{pred['risk_tier']}"
    short_grade = (
        "High-Grade (HGG)"
        if "HGG" in pred["tumor_class"]
        else ("Low-Grade (LGG)" if "LGG" in pred["tumor_class"] else "No Lesion")
    )
    seg_sub = "U-Net ResNet34" if "U-Net" in seg_source else seg_source

    ctv_display = f"{rt_plan['ctv_cm3']} cm³" if rt_plan["has_tumor"] else "N/A"
    ctv_sub = f"GTV + {int(rt_plan['margin_mm'])} mm Margin" if rt_plan["has_tumor"] else "No lesion detected"
    spared_display = f"{rt_plan['healthy_spared_pct']}%" if rt_plan["has_tumor"] else "100.0%"
    spared_sub = f"{rt_plan['healthy_spared_cm3']} cm³ untouched" if rt_plan["has_tumor"] else "Full brain spared"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(modern_kpi_html("Pathology Grade", short_grade, seg_sub, accent="cyan"), unsafe_allow_html=True)
    k2.markdown(modern_kpi_html("GTV (Visible Tumour)", f"{anatomy['volume_cm3']} cm³", f"{int(features_df['MeshVolume'].values[0]):,} mm³ (IBSI)", accent="emerald"), unsafe_allow_html=True)
    k3.markdown(modern_kpi_html("Brain Occupancy", f"{anatomy['brain_occupancy_pct']}%", anatomy["hemisphere"], accent="amber"), unsafe_allow_html=True)
    k4.markdown(modern_kpi_html("CTV Target Volume", ctv_display, ctv_sub, accent="rose"), unsafe_allow_html=True)
    k5.markdown(modern_kpi_html("Healthy Brain Spared", spared_display, spared_sub, accent="emerald"), unsafe_allow_html=True)

    # BraTS Multi-Compartment Volumetric Proportional Banner
    render_compartments_banner(anatomy)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.12, 1.0], gap="large")

    with col_left:
        max_z = img_data.shape[2] - 1
        default_z = int(anatomy["centroid"][2]) if anatomy["has_tumor"] else max_z // 2

        if f"slice_z_{cache_key}" not in st.session_state:
            st.session_state[f"slice_z_{cache_key}"] = default_z

        col_head1, col_head2 = st.columns([1.5, 1.0])
        with col_head1:
            st.markdown("#### Axial MRI Inspector")
        with col_head2:
            with st.popover("Display Settings", use_container_width=True):
                wl_mode = st.selectbox(
                    "PACS Window/Level Preset",
                    ["Brain Window (Standard)", "High-Contrast Lesion", "Full Dynamic Range"],
                    index=0,
                    key="dash_wl",
                )
                overlay_opts = (
                    ["BraTS Compartments", "Whole Tumour Fill", "Contour Outline", "Off"]
                    if anatomy.get("has_subcompartments", False)
                    else ["Whole Tumour Fill", "Contour Outline", "Off"]
                )
                overlay_mode = st.selectbox(
                    "Overlay Mode",
                    overlay_opts,
                    index=0,
                    key="dash_overlay_mode",
                )
                mask_opacity = st.slider(
                    "Mask Opacity",
                    0.1,
                    0.9,
                    0.5,
                    0.05,
                    key="dash_op",
                    disabled=(overlay_mode in ["Off", "Contour Outline"]),
                )
                show_ctv_ring = st.checkbox("Show CTV Radiation Margin (Dashed)", value=True, key="dash_show_ctv")

        show_mask = overlay_mode != "Off"
        contour_only = overlay_mode == "Contour Outline"
        overlay_color_mode = "BraTS Multi-Compartment" if overlay_mode == "BraTS Compartments" else "Single Highlight (Red)"

        p99 = float(np.percentile(img_data[img_data > 0], 99)) if np.any(img_data > 0) else 250.0
        if wl_mode == "Brain Window (Standard)":
            ww, wc = p99 * 0.70, p99 * 0.35
        elif wl_mode == "High-Contrast Lesion":
            ww, wc = p99 * 0.45, p99 * 0.55
        else:
            ww, wc = None, None

        slice_z = st.slider(
            "Axial Z-Slice",
            0,
            max_z,
            value=st.session_state[f"slice_z_{cache_key}"],
            key=f"slider_z_{cache_key}",
            on_change=lambda: st.session_state.update({f"slice_z_{cache_key}": st.session_state[f"slider_z_{cache_key}"]}),
        )

        ax_img = np.rot90(img_data[:, :, slice_z])
        ax_msk = np.rot90(mask_data[:, :, slice_z]) if mask_data.shape == img_data.shape else None
        fig_2d = render_pacs_slice_figure(
            ax_img,
            ax_msk,
            aspect=zooms[1] / max(zooms[0], 1e-5),
            show_mask=show_mask,
            opacity=mask_opacity,
            contour_only=contour_only,
            plane="axial",
            window_width=ww,
            window_center=wc,
            color_mode=overlay_color_mode,
            patient_id=display_pid,
            slice_idx=slice_z,
            max_slices=max_z + 1,
            slice_thickness=zooms[2],
            pixel_spacing=zooms[0],
            ctv_2d=np.rot90(rt_plan["ctv_mask"][:, :, slice_z]) if rt_plan["has_tumor"] else None,
            show_ctv=show_ctv_ring,
        )
        st.pyplot(fig_2d, use_container_width=True)
        plt.close(fig_2d)

        if val_metrics:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            st.markdown("#### Ground-Truth Accuracy Benchmark")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Dice Score", f"{val_metrics['dice']:.3f}")
            m2.metric("IoU (Jaccard)", f"{val_metrics['iou']:.3f}")
            m3.metric("Precision", f"{val_metrics['precision']:.3f}")
            m4.metric("Recall", f"{val_metrics['recall']:.3f}")

    with col_right:
        if rt_plan["has_tumor"]:
            st.markdown(
                """
                <div style="margin-bottom: 6px;">
                  <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; line-height: 1.3;">
                    Radiotherapy Target Volume Planning (GTV → CTV)
                  </div>
                  <div style="font-size: 0.80rem; color: #94A3B8; margin-top: 2px;">
                    Brain-clipped 3D geometric margin expansion • Standard glioma protocol: 10–20 mm
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.slider(
                "Clinical Target Margin (mm)",
                min_value=5,
                max_value=25,
                step=1,
                key=rt_margin_key,
                help="Expands visible GTV outward in 3D millimeters while clipping strictly inside brain tissue.",
            )

            st.plotly_chart(
                build_ctv_tradeoff_figure(
                    rt_plan["tradeoff_df"],
                    rt_plan["margin_mm"],
                    rt_plan["ctv_cm3"],
                    rt_plan["healthy_spared_pct"],
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            st.markdown(
                f"""
                <div style="margin: 10px 0 4px 0;">
                  <div style="font-size: 0.96rem; font-weight: 700; color: #F8FAFC;">
                    3D Target Volume Breakdown ({int(rt_plan['margin_mm'])} mm Margin)
                  </div>
                  <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 2px;">
                    Healthy Parenchyma Exposed: <b style="color:#F59E0B">{rt_plan['healthy_exposed_pct']}%</b>
                    &nbsp;•&nbsp; Healthy Brain Spared: <b style="color:#10B981">{rt_plan['healthy_spared_pct']}%</b>
                    &nbsp;•&nbsp; Dashed amber ring on MRI marks CTV boundary
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.plotly_chart(
                build_ctv_breakdown_figure(rt_plan),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.success("No abnormal tumour mass detected on this MRI volume (`GTV = 0 mm³`, `100%` healthy brain spared).")

        with st.expander("Complete 3D IBSI Radiomic Feature Profile", expanded=False):
            bio_df = pred["features"].T.rename(columns={0: "Measured Value"})
            st.dataframe(bio_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: MULTI-PLANAR (MPR) & 3D LAB
# ═══════════════════════════════════════════════════════════════════════════════
elif "Multi-Planar" in page:
    st.markdown("#### 🧊 Interactive 3D WebGL Tumour & Brain Surface Reconstruction")

    ctrl1, ctrl2, ctrl3 = st.columns([1.1, 1.0, 1.2])
    shell_opacity = ctrl1.slider("Brain Surface Opacity", 0.0, 0.35, 0.14, 0.02)
    tumor_color = ctrl2.selectbox(
        "Tumour Mesh Tint",
        ["#ff3b30", "#f59e0b", "#ec4899", "#22d3ee"],
        format_func=lambda c: {
            "#ff3b30": "Clinical Crimson",
            "#f59e0b": "Amber Gold",
            "#ec4899": "High-Contrast Magenta",
            "#22d3ee": "Surgical Cyan",
        }[c],
    )
    ctrl3.markdown(
        f"**Bounding Box:** `{anatomy['dims_mm'][0]} × {anatomy['dims_mm'][1]} × {anatomy['dims_mm'][2]} mm`  \n"
        f"**Centroid (X, Y, Z):** `{anatomy['centroid']}` • **{anatomy['hemisphere']}**"
    )

    # Quick Camera Angle Preset Buttons
    c_cam1, c_cam2, c_cam3, c_cam4, _ = st.columns([1, 1, 1, 1, 1.5])
    if "cam_preset" not in st.session_state:
        st.session_state.cam_preset = "iso"

    if c_cam1.button("🌐 3D Isometric", use_container_width=True):
        st.session_state.cam_preset = "iso"
        st.rerun()
    if c_cam2.button("🔝 Axial (Top)", use_container_width=True):
        st.session_state.cam_preset = "top"
        st.rerun()
    if c_cam3.button("👁️ Coronal (Front)", use_container_width=True):
        st.session_state.cam_preset = "front"
        st.rerun()
    if c_cam4.button("👤 Sagittal (Side)", use_container_width=True):
        st.session_state.cam_preset = "side"
        st.rerun()

    with st.spinner("Rendering 3D Marching Cubes mesh..."):
        fig_3d = get_cached_3d_figure(f"{cache_key}_stereo_v2", img_data, mask_data, zooms, shell_opacity, tumor_color, camera_preset=st.session_state.cam_preset)
    st.plotly_chart(fig_3d, use_container_width=True)
    st.caption("Left-drag to orbit 360° • Scroll to zoom • Right-drag to pan • Double-click to reset viewport")

    st.divider()
    st.markdown("#### 📐 Synchronized Multi-Planar Orthogonal Reconstruction (Tri-Planar MPR)")

    cx, cy, cz = anatomy["centroid"]

    # Re-center button
    if st.button("🎯 Re-center All 3 Planes to Tumour Centroid", use_container_width=False):
        st.session_state[f"mpr_z_{cache_key}"] = cz
        st.session_state[f"mpr_y_{cache_key}"] = cy
        st.session_state[f"mpr_x_{cache_key}"] = cx
        st.rerun()

    s1, s2, s3 = st.columns(3)
    idx_z = s1.slider("Axial Plane (Z)", 0, img_data.shape[2] - 1, cz, key=f"mpr_z_{cache_key}")
    idx_y = s2.slider("Coronal Plane (Y)", 0, img_data.shape[1] - 1, cy, key=f"mpr_y_{cache_key}")
    idx_x = s3.slider("Sagittal Plane (X)", 0, img_data.shape[0] - 1, cx, key=f"mpr_x_{cache_key}")

    m_col1, m_col2 = st.columns(2)
    mpr_mask_on = m_col1.checkbox("Overlay Tumour Mask on All Three Planes", value=True, key="mpr_overlay")
    mpr_color_mode = m_col2.selectbox(
        "MPR Colour Mode",
        ["BraTS Multi-Compartment", "Single Highlight (Red)"] if anatomy.get("has_subcompartments", False) else ["Single Highlight (Red)"],
        index=0,
        key="mpr_colmode",
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        f_ax = render_pacs_slice_figure(
            np.rot90(img_data[:, :, idx_z]),
            np.rot90(mask_data[:, :, idx_z]),
            aspect=zooms[1] / max(zooms[0], 1e-5),
            show_mask=mpr_mask_on,
            plane="axial",
            color_mode=mpr_color_mode,
            patient_id=display_pid,
            slice_idx=idx_z,
            max_slices=img_data.shape[2],
            slice_thickness=zooms[2],
            pixel_spacing=zooms[0],
        )
        st.pyplot(f_ax, use_container_width=True)
        plt.close(f_ax)

    with p2:
        f_cor = render_pacs_slice_figure(
            np.rot90(img_data[:, idx_y, :]),
            np.rot90(mask_data[:, idx_y, :]),
            aspect=zooms[2] / max(zooms[0], 1e-5),
            show_mask=mpr_mask_on,
            plane="coronal",
            color_mode=mpr_color_mode,
            patient_id=display_pid,
            slice_idx=idx_y,
            max_slices=img_data.shape[1],
            slice_thickness=zooms[1],
            pixel_spacing=zooms[0],
        )
        st.pyplot(f_cor, use_container_width=True)
        plt.close(f_cor)

    with p3:
        f_sag = render_pacs_slice_figure(
            np.rot90(img_data[idx_x, :, :]),
            np.rot90(mask_data[idx_x, :, :]),
            aspect=zooms[2] / max(zooms[1], 1e-5),
            show_mask=mpr_mask_on,
            plane="sagittal",
            color_mode=mpr_color_mode,
            patient_id=display_pid,
            slice_idx=idx_x,
            max_slices=img_data.shape[0],
            slice_thickness=zooms[0],
            pixel_spacing=zooms[1],
        )
        st.pyplot(f_sag, use_container_width=True)
        plt.close(f_sag)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: CLINICAL REPORT & BIOMARKERS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Clinical Report" in page:
    col_bio, col_rep = st.columns([1.25, 0.85], gap="large")

    with col_bio:
        st.markdown("#### 📝 Automated Radiological Impression")
        impression_html = build_radiological_impression(display_pid, patient_age, anatomy, pred)
        render_impression_box(impression_html)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown("#### 🔬 Quantitative 3D IBSI Radiomic Biomarker Profile")
        ref_df = build_biomarker_reference_table(pred["features"])
        st.dataframe(ref_df, use_container_width=True, hide_index=True, height=390)

        if val_metrics:
            st.markdown("#### 🎯 Ground-Truth Benchmark Metrics")
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Dice Coefficient", f"{val_metrics['dice']:.4f}")
            v2.metric("IoU (Jaccard)", f"{val_metrics['iou']:.4f}")
            v3.metric("Precision", f"{val_metrics['precision']:.4f}")
            v4.metric("Recall / Sensitivity", f"{val_metrics['recall']:.4f}")

    with col_rep:
        st.markdown("#### 📄 Clinical Documentation & Export")
        st.caption("Generate a structured clinical summary (open in browser and press `Ctrl + P` to print to PDF) or export raw radiomics as CSV.")

        clinician_notes = st.text_area(
            "Attending Radiologist / Clinical Notes",
            placeholder="Enter clinical correlation, surgical candidacy, or follow-up imaging notes...",
            height=130,
        )

        html_report = generate_clinical_report_html(
            patient_id=display_pid,
            age=patient_age,
            scan_name=active_scan_name,
            anatomy=anatomy,
            pred=pred,
            notes=clinician_notes,
        )
        csv_data = pred["features"].to_csv(index=False).encode("utf-8")

        d1, d2 = st.columns(2)
        d1.download_button(
            label="📄 Download Clinical Report",
            data=html_report,
            file_name=f"{display_pid}_NeuroXAI_Report.html",
            mime="text/html",
            use_container_width=True,
            type="primary",
        )
        d2.download_button(
            label="📊 Export Radiomics CSV",
            data=csv_data,
            file_name=f"{display_pid}_Radiomics.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.divider()
        st.markdown("#### 🗂️ Case Demographics & Scan Metadata")
        surv_line = f"`~{pred['days']} Days ({pred['months']} Months)`" if pred["days"] is not None else "`N/A`"
        st.markdown(
            f"""
            * **Patient ID:** `{display_pid}` ({patient_age} yrs)
            * **Active File:** `{active_scan_name}`
            * **Voxel Spacing:** `{zooms[0]:.2f} × {zooms[1]:.2f} × {zooms[2]:.2f} mm`
            * **Segmentation Engine:** {seg_source}
            * **Predicted Grade:** {pred['tumor_class']}
            * **Location & Size:** {anatomy['hemisphere']} (`{anatomy['dims_mm'][0]} × {anatomy['dims_mm'][1]} × {anatomy['dims_mm'][2]} mm`)
            * **Tumour Burden:** `{anatomy['volume_cm3']} cm³` (`{anatomy['brain_occupancy_pct']}%` of brain)
            * **Prognosis Tier:** {pred['risk_category']} — {surv_line}
            * **Relative Hazard:** `{pred.get('hazard_ratio', 1.0):.2f}×` cohort baseline (`ΣSHAP: {pred.get('net_shap', 0.0):+.3f}`)
            """
        )

        render_clinical_notice()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: LONGITUDINAL PROGRESSION TRACKING (T0 vs. T1 3D Growth/Regression Diff)
# ═══════════════════════════════════════════════════════════════════════════════
elif "Longitudinal Tracking" in page:
    # ── Scan Pairing Bar ──
    st.markdown("#### Longitudinal Scan Pairing & Spatial Alignment")

    col_p1, col_p2, col_p3, col_p4 = st.columns([1.6, 1.8, 1.0, 1.0], gap="medium")
    with col_p1:
        st.markdown(f"**Baseline Scan (T0):** `{active_scan_name}`")
        st.caption(f"3D Volume: {anatomy['volume_cm3']} cm³ • {anatomy['hemisphere']}")

    with col_p2:
        followup_file = st.file_uploader(
            "Follow-Up Scan (T1)",
            type=["nii", "gz", "zip"],
            help="Upload Follow-Up FLAIR MRI (.nii, .nii.gz, or DICOM .zip archive)",
            key="longitudinal_t1_uploader",
        )
        followup_gt_file = st.file_uploader(
            "Optional Follow-Up Mask (Ground-Truth)",
            type=["nii", "gz"],
            help="Upload optional expert ground-truth mask for Follow-Up (T1)",
            key="longitudinal_t1_gt_uploader",
        )

    with col_p3:
        interval_days = st.number_input(
            "Scan Interval (Days)",
            min_value=1,
            max_value=3650,
            value=90,
            step=15,
            key="longitudinal_interval_days",
        )

    with col_p4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        swap_t0_t1 = st.toggle(
            "Swap T0 ↔ T1",
            value=False,
            key="longitudinal_swap_toggle",
            help="Inverts baseline and follow-up scan roles to test reverse progression/regression dynamics.",
        )

    if followup_file is None:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="hero-showcase-container" style="padding: 28px 32px; margin-top: 10px;">
              <div class="hero-badge-pill">Longitudinal Neuro-Oncology Surveillance</div>
              <h2 style="color:#F8FAFC;font-size:1.45rem;font-weight:700;margin:12px 0 8px 0;">
                Baseline (T0) vs. Follow-Up (T1) Volumetric Subtraction
              </h2>
              <p style="color:#94A3B8;font-size:0.92rem;line-height:1.65;margin:0 0 18px 0;max-width:920px;">
                NeuroXAI longitudinal tracking pairs your baseline MRI against a post-resection, chemo-radiation, 
                or surveillance follow-up scan. The pipeline performs automated spatial voxel grid alignment, 
                RANO-aligned 2D perpendicular cross-sectional product tracking, volumetric growth velocity calculation, 
                and 3D voxel-wise tissue subtraction.
              </p>
              <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:16px;">
                <div style="background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:16px;">
                  <div style="font-weight:700;color:#2DD4BF;font-size:0.92rem;margin-bottom:6px;">1. Volumetric Delta & Velocity</div>
                  <div style="color:#94A3B8;font-size:0.82rem;line-height:1.5;">Calculates absolute volume shift (cm³) and normalized monthly growth velocity (cm³/mo) across patient scan intervals.</div>
                </div>
                <div style="background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:16px;">
                  <div style="font-weight:700;color:#2DD4BF;font-size:0.92rem;margin-bottom:6px;">2. RANO Criteria Stratification</div>
                  <div style="color:#94A3B8;font-size:0.82rem;line-height:1.5;">Evaluates 2D cross-sectional product change (d1 × d2 in mm²) to categorize Progression, Stable Disease, or Regression.</div>
                </div>
                <div style="background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:16px;">
                  <div style="font-weight:700;color:#2DD4BF;font-size:0.92rem;margin-bottom:6px;">3. 3D Growth / Regression Diff</div>
                  <div style="color:#94A3B8;font-size:0.82rem;line-height:1.5;">Isolates persistent core (grey), regressed tissue (green), and active new growth (red) in synchronized 3D and 2D PACS views.</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Processing Follow-Up (T1) volume and computing 3D longitudinal subtraction..."):
            try:
                t1_gt_bytes = followup_gt_file.getvalue() if followup_gt_file is not None else None
                t1_gt_name = followup_gt_file.name if followup_gt_file is not None else None
                t1_vol, t1_mask, t1_zooms, t1_anatomy, t1_hash, t1_meta = process_followup_scan(
                    followup_file.getvalue(),
                    followup_file.name,
                    t1_gt_bytes,
                    t1_gt_name,
                    unet_model,
                    unet_meta,
                )
            except Exception as ex:
                st.error(f"Error loading Follow-Up scan: {str(ex)}")
                st.stop()

        if swap_t0_t1:
            base_vol, base_mask, base_zooms, base_name = t1_vol, t1_mask, t1_zooms, followup_file.name
            fu_vol, fu_mask, fu_zooms, fu_name = img_data, mask_data, zooms, active_scan_name
        else:
            base_vol, base_mask, base_zooms, base_name = img_data, mask_data, zooms, active_scan_name
            fu_vol, fu_mask, fu_zooms, fu_name = t1_vol, t1_mask, t1_zooms, followup_file.name

        prog = compute_longitudinal_progression(
            base_vol, base_mask, base_zooms,
            fu_vol, fu_mask, fu_zooms,
            interval_days=interval_days,
        )

        if prog["was_resampled"]:
            st.info(
                "Follow-Up (T1) volume was resampled to match Baseline (T0) voxel space for voxel-wise spatial subtraction. "
                "Pre-registered longitudinal series provide maximum boundary accuracy."
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── 5 Equal-Height KPI Cards ──
        tier_to_accent = {
            "high": "rose",
            "low": "emerald",
            "moderate": "amber",
            "healthy": "emerald",
        }
        resp_accent = tier_to_accent.get(prog["response_tier"], "cyan")
        vol_accent = "rose" if prog["delta_vol_cm3"] > 0 else "emerald"
        rano_accent = "rose" if prog["delta_rano_pct"] > 0 else "emerald"

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(
                f"""
                <div class="glass-kpi-card kpi-accent-{resp_accent}">
                  <div>
                    <div class="glass-kpi-label">Response Status</div>
                    <div style="margin-top:6px;"><span class="badge-{prog['response_tier']}">{prog['response_status']}</span></div>
                  </div>
                  <div class="glass-kpi-sub">RANO Volumetric Criteria</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            vol_sign = "+" if prog["delta_vol_cm3"] > 0 else ""
            st.markdown(
                modern_kpi_html(
                    "Volume Change",
                    f"{vol_sign}{prog['delta_vol_cm3']} cm³",
                    f"T0: {prog['vol_t0_cm3']} → T1: {prog['vol_t1_cm3']} cm³",
                    accent=vol_accent,
                ),
                unsafe_allow_html=True,
            )
        with k3:
            rano_sign = "+" if prog["delta_rano_pct"] > 0 else ""
            st.markdown(
                modern_kpi_html(
                    "RANO 2D Product",
                    f"{rano_sign}{prog['delta_rano_pct']}%",
                    f"T0: {prog['rano_prod_t0']} → T1: {prog['rano_prod_t1']} mm²",
                    accent=rano_accent,
                ),
                unsafe_allow_html=True,
            )
        with k4:
            growth_sign = "+" if prog["growth_rate_cm3_per_month"] > 0 else ""
            st.markdown(
                modern_kpi_html(
                    "Growth Velocity",
                    f"{growth_sign}{prog['growth_rate_cm3_per_month']} cm³/mo",
                    f"Interval: {interval_days} Days",
                    accent="cyan",
                ),
                unsafe_allow_html=True,
            )
        with k5:
            st.markdown(
                modern_kpi_html(
                    "3D Tissue Shift",
                    f"+{prog['new_growth_vol_cm3']} cm³",
                    f"New: {prog['new_growth_vol_cm3']} • Regr: {prog['regressed_vol_cm3']} cm³",
                    accent="cyan",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── 3D Interactive Growth/Regression Mesh (Full Width) ──
        st.markdown("#### 3D Longitudinal Growth & Regression Spatial Model")
        ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1.6, 1.0, 1.0, 1.0])
        with ctrl_c1:
            shell_op = st.slider("Brain Shell Opacity", 0.0, 0.35, 0.10, 0.02, key="long_shell_op")
        with ctrl_c2:
            show_unchanged = st.checkbox("Unchanged (Grey)", value=True, key="long_show_unchanged")
        with ctrl_c3:
            show_regressed = st.checkbox("Regressed (Green)", value=True, key="long_show_regressed")
        with ctrl_c4:
            show_growth = st.checkbox("New Growth (Red)", value=True, key="long_show_growth")

        cache_key_3d = f"long3d_{cache_key}_{t1_hash}_{swap_t0_t1}_{shell_op}_{show_unchanged}_{show_regressed}_{show_growth}"
        fig_3d = get_cached_longitudinal_diff_figure(
            cache_key_3d,
            base_vol,
            prog["unchanged_mask"],
            prog["regressed_mask"],
            prog["new_growth_mask"],
            base_zooms,
            shell_op,
            show_unchanged,
            show_regressed,
            show_growth,
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Synced Side-by-Side Axial Comparison (2 Columns) ──
        st.markdown("#### Synced Axial Slice Progression Inspector")

        # Compact status line confirming segmented volumes for both scans
        st.markdown(
            f"""
            <div style="background:rgba(15,23,42,0.65);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:8px 14px;margin-bottom:12px;font-size:0.86rem;color:#E2E8F0;display:flex;gap:18px;align-items:center;flex-wrap:wrap;">
              <span><b style="color:#38BDF8;">Baseline (T0) Mask:</b> {prog['vol_t0_cm3']:.2f} cm³</span>
              <span style="color:#64748B;">•</span>
              <span><b style="color:#2DD4BF;">Follow-Up (T1) Mask:</b> {prog['vol_t1_cm3']:.2f} cm³</span>
              <span style="color:#64748B;">•</span>
              <span><b style="color:#F59E0B;">Volumetric Delta:</b> {prog['delta_vol_cm3']:+.2f} cm³ ({prog['delta_vol_pct']:+.1f}%)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        max_z = base_vol.shape[2]
        default_z = int(prog["anatomy_t0"]["centroid"][2]) if prog["vol_t0_cm3"] > 0 else (int(prog["anatomy_t1"]["centroid"][2]) if prog["vol_t1_cm3"] > 0 else max_z // 2)
        default_z = max(0, min(default_z, max_z - 1))

        col_ctrl1, col_ctrl2 = st.columns([1.35, 1.15], gap="medium")
        with col_ctrl1:
            axial_idx = st.slider("Axial Z-Slice", min_value=0, max_value=max_z - 1, value=default_z, key="long_axial_slider")
        with col_ctrl2:
            t1_view_mode = st.radio(
                "Follow-Up View Mode",
                [
                    "Growth & Regression Overlay (T0 vs T1)",
                    "Follow-Up Segmentation Only (T1 Mask)",
                ],
                index=0,
                key="long_view_mode",
            )

        col_t0, col_t1 = st.columns(2, gap="medium")
        with col_t0:
            st.markdown(f"<div style='font-size:0.90rem;font-weight:700;color:#94A3B8;margin-bottom:6px;'>Baseline (T0): <span style='color:#F8FAFC;'>{base_name}</span></div>", unsafe_allow_html=True)
            f_t0 = render_pacs_slice_figure(
                np.rot90(base_vol[:, :, axial_idx]),
                np.rot90(base_mask[:, :, axial_idx]),
                aspect=base_zooms[1] / max(base_zooms[0], 1e-5),
                slice_idx=axial_idx,
                max_slices=max_z,
                plane="axial",
                pixel_spacing=base_zooms[0],
                show_mask=True,
                color_mode="BraTS Multi-Compartment",
                patient_id=display_pid,
            )
            st.pyplot(f_t0, use_container_width=True)
            plt.close(f_t0)
            st.caption(f"T0 Baseline Burden: {prog['vol_t0_cm3']} cm³ • RANO 2D Product: {prog['rano_prod_t0']} mm²")

        with col_t1:
            if t1_view_mode == "Growth & Regression Overlay (T0 vs T1)":
                st.markdown(f"<div style='font-size:0.90rem;font-weight:700;color:#94A3B8;margin-bottom:6px;'>Follow-Up (T1) — Growth & Regression Overlay: <span style='color:#F8FAFC;'>{fu_name}</span></div>", unsafe_allow_html=True)
                f_t1 = render_longitudinal_diff_slice_figure(
                    np.rot90(prog["aligned_vol_t1"][:, :, axial_idx]),
                    np.rot90(prog["unchanged_mask"][:, :, axial_idx]),
                    np.rot90(prog["regressed_mask"][:, :, axial_idx]),
                    np.rot90(prog["new_growth_mask"][:, :, axial_idx]),
                    slice_idx=axial_idx,
                    max_slices=max_z,
                    pixel_spacing=base_zooms[0],
                    aspect=base_zooms[1] / max(base_zooms[0], 1e-5),
                )
                st.pyplot(f_t1, use_container_width=True)
                plt.close(f_t1)
                st.caption(f"T1 Follow-Up Burden: {prog['vol_t1_cm3']} cm³ • RANO 2D Product: {prog['rano_prod_t1']} mm²")
            else:
                st.markdown(f"<div style='font-size:0.90rem;font-weight:700;color:#94A3B8;margin-bottom:6px;'>Follow-Up (T1) — Tumour Segmentation: <span style='color:#F8FAFC;'>{fu_name}</span></div>", unsafe_allow_html=True)
                f_t1 = render_pacs_slice_figure(
                    np.rot90(prog["aligned_vol_t1"][:, :, axial_idx]),
                    np.rot90(prog["aligned_mask_t1"][:, :, axial_idx]),
                    aspect=base_zooms[1] / max(base_zooms[0], 1e-5),
                    slice_idx=axial_idx,
                    max_slices=max_z,
                    plane="axial",
                    pixel_spacing=base_zooms[0],
                    show_mask=True,
                    color_mode="BraTS Multi-Compartment",
                    patient_id=display_pid,
                )
                st.pyplot(f_t1, use_container_width=True)
                plt.close(f_t1)
                st.caption(f"T1 Segmented Mask Volume: {prog['vol_t1_cm3']} cm³ • RANO 2D Product: {prog['rano_prod_t1']} mm²")

        # Dynamic Legend based on view mode
        if t1_view_mode == "Growth & Regression Overlay (T0 vs T1)":
            st.markdown(
                """
                <div style="display:flex;gap:24px;justify-content:center;align-items:center;margin:8px 0 16px 0;font-size:0.82rem;font-weight:600;color:#94A3B8;">
                  <span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:12px;height:12px;border-radius:3px;background:#38BDF8;display:inline-block;"></span> Unchanged Core</span>
                  <span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:12px;height:12px;border-radius:3px;background:#10B981;display:inline-block;"></span> Regressed / Resected</span>
                  <span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:12px;height:12px;border-radius:3px;background:#FB7185;display:inline-block;"></span> New Growth / Progression</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="display:flex;gap:24px;justify-content:center;align-items:center;margin:8px 0 16px 0;font-size:0.82rem;font-weight:600;color:#94A3B8;">
                  <span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:12px;height:12px;border-radius:3px;background:#FF4B4B;display:inline-block;"></span> Enhancing Tumour (ET)</span>
                  <span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:12px;height:12px;border-radius:3px;background:#00F0FF;display:inline-block;"></span> Tumour Core (TC)</span>
                  <span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:12px;height:12px;border-radius:3px;background:#FACC15;display:inline-block;"></span> Whole Tumour / Edema (WT)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ── CSV Export ──
        export_rows = [
            {"Metric": "Patient Identifier", "Baseline (T0)": display_pid, "Follow-Up (T1)": display_pid, "Delta": "--", "Change (%)": "--"},
            {"Metric": "Scan Filename", "Baseline (T0)": base_name, "Follow-Up (T1)": fu_name, "Delta": "--", "Change (%)": "--"},
            {"Metric": "Tumour Burden (cm³)", "Baseline (T0)": f"{prog['vol_t0_cm3']:.2f}", "Follow-Up (T1)": f"{prog['vol_t1_cm3']:.2f}", "Delta": f"{prog['delta_vol_cm3']:+.2f}", "Change (%)": f"{prog['delta_vol_pct']:+.1f}%"},
            {"Metric": "RANO 2D Product (mm²)", "Baseline (T0)": f"{prog['rano_prod_t0']:.1f}", "Follow-Up (T1)": f"{prog['rano_prod_t1']:.1f}", "Delta": f"{prog['delta_rano_prod']:+.1f}", "Change (%)": f"{prog['delta_rano_pct']:+.1f}%"},
            {"Metric": "RANO Diameter 1 (mm)", "Baseline (T0)": f"{prog['d1_t0']:.1f}", "Follow-Up (T1)": f"{prog['d1_t1']:.1f}", "Delta": f"{prog['d1_t1'] - prog['d1_t0']:+.1f}", "Change (%)": "--"},
            {"Metric": "RANO Diameter 2 (mm)", "Baseline (T0)": f"{prog['d2_t0']:.1f}", "Follow-Up (T1)": f"{prog['d2_t1']:.1f}", "Delta": f"{prog['d2_t1'] - prog['d2_t0']:+.1f}", "Change (%)": "--"},
            {"Metric": "Growth Velocity (cm³/month)", "Baseline (T0)": "--", "Follow-Up (T1)": "--", "Delta": f"{prog['growth_rate_cm3_per_month']:+.2f}", "Change (%)": "--"},
            {"Metric": "Surveillance Interval (Days)", "Baseline (T0)": "--", "Follow-Up (T1)": "--", "Delta": str(interval_days), "Change (%)": "--"},
            {"Metric": "Unchanged Core Volume (cm³)", "Baseline (T0)": "--", "Follow-Up (T1)": "--", "Delta": f"{prog['unchanged_vol_cm3']:.2f}", "Change (%)": "--"},
            {"Metric": "Regressed Volume (cm³)", "Baseline (T0)": "--", "Follow-Up (T1)": "--", "Delta": f"{prog['regressed_vol_cm3']:.2f}", "Change (%)": "--"},
            {"Metric": "New Growth Volume (cm³)", "Baseline (T0)": "--", "Follow-Up (T1)": "--", "Delta": f"{prog['new_growth_vol_cm3']:.2f}", "Change (%)": "--"},
            {"Metric": "RANO Volumetric Response Tier", "Baseline (T0)": "--", "Follow-Up (T1)": "--", "Delta": prog["response_status"], "Change (%)": "--"},
        ]
        long_df = pd.DataFrame(export_rows)
        csv_long_bytes = long_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Export Longitudinal Summary (.csv)",
            data=csv_long_bytes,
            file_name=f"{display_pid}_Longitudinal_Summary.csv",
            mime="text/csv",
            key="download_long_csv_btn",
            type="primary",
        )
