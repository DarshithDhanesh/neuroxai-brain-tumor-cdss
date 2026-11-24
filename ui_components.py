"""
NeuroXAI UI Components & Unified HTML Template Management Module.
Loads all HTML markup from a single consolidated templates.html file
and injects styles from static/css/styles.css.
"""

import os
import re
from functools import lru_cache
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_FILE = os.path.join(BASE_DIR, "templates.html")
STYLES_FILE = os.path.join(BASE_DIR, "static", "css", "styles.css")


@lru_cache(maxsize=1)
def _load_all_templates() -> dict:
    """Reads templates.html once and extracts all <template id="..."> blocks into a dictionary."""
    if not os.path.exists(TEMPLATES_FILE):
        raise FileNotFoundError(f"Unified templates file not found: {TEMPLATES_FILE}")
    with open(TEMPLATES_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = r'<template\s+id=["\']([^"\']+)["\']\s*>(.*?)</template>'
    matches = re.findall(pattern, content, flags=re.DOTALL)
    return {tpl_id.strip(): tpl_content.strip() for tpl_id, tpl_content in matches}


def get_template(template_id: str) -> str:
    """Retrieves an HTML template string by its ID from the unified templates.html."""
    templates = _load_all_templates()
    if template_id not in templates:
        raise KeyError(f"Template '{template_id}' not found in {TEMPLATES_FILE}")
    return templates[template_id]


@lru_cache(maxsize=1)
def _load_css() -> str:
    """Reads and caches the global stylesheet from static/css/styles.css."""
    if not os.path.exists(STYLES_FILE):
        raise FileNotFoundError(f"Stylesheet not found: {STYLES_FILE}")
    with open(STYLES_FILE, "r", encoding="utf-8") as f:
        return f.read()


def load_app_css() -> None:
    """Injects the global clinical dark workstation theme CSS into the Streamlit app."""
    css_content = _load_css()
    st.markdown(f"<style>\n{css_content}\n</style>", unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    """Renders the minimalistic NeuroXAI brand header in the sidebar."""
    html = get_template("sidebar_brand")
    st.markdown(html, unsafe_allow_html=True)


def render_hero_showcase() -> None:
    """Renders the hero banner displayed before a scan is uploaded."""
    html = get_template("hero_showcase")
    st.markdown(html, unsafe_allow_html=True)


def render_feature_card(icon: str, icon_bg: str, icon_color: str, title: str, description: str) -> None:
    """Renders an onboarding feature card with an icon and description."""
    tpl = get_template("feature_card")
    card_html = tpl.format(
        icon=icon,
        icon_bg=icon_bg,
        icon_color=icon_color,
        title=title,
        description=description,
    )
    st.markdown(card_html, unsafe_allow_html=True)


def modern_kpi_html(label: str, value: str, sub: str = "", accent: str = "cyan") -> str:
    """Generates the HTML markup for a floating glass KPI metric card."""
    accent_class = f"kpi-accent-{accent}" if accent != "cyan" else ""
    tpl = get_template("kpi_card")
    return tpl.format(
        accent_class=accent_class,
        label=label,
        value=value,
        sub=sub,
    )


def render_prognosis_card(risk_category: str, badge_class: str, risk_score: str) -> str:
    """Generates the HTML markup for the prognosis stratification KPI card."""
    tpl = get_template("prognosis_card")
    return tpl.format(
        risk_category=risk_category,
        badge_class=badge_class,
        risk_score=risk_score,
    )


def render_compartments_banner(anatomy: dict) -> None:
    """Renders the single-line BraTS multi-compartment volumetric banner."""
    if not anatomy.get("has_subcompartments", False):
        return

    wt_val = max(anatomy.get("wt_vol_cm3", 0.0), 1e-4)
    et_pct = anatomy.get("et_ratio_pct", round((anatomy.get("et_vol_cm3", 0.0) / wt_val) * 100.0, 1))
    tc_pct = anatomy.get("tc_ratio_pct", round((anatomy.get("tc_vol_cm3", 0.0) / wt_val) * 100.0, 1))
    ed_pct = anatomy.get("ed_ratio_pct", round((anatomy.get("ed_vol_cm3", 0.0) / wt_val) * 100.0, 1))

    tpl = get_template("compartments_banner")
    banner_html = tpl.format(
        et_vol=anatomy.get("et_vol_cm3", 0.0),
        et_pct=et_pct,
        tc_vol=anatomy.get("tc_vol_cm3", 0.0),
        tc_pct=tc_pct,
        ed_vol=anatomy.get("ed_vol_cm3", 0.0),
        ed_pct=ed_pct,
    )
    st.markdown(banner_html, unsafe_allow_html=True)


def render_impression_box(content: str) -> None:
    """Renders the clinical radiological impression enclosed in a glass container."""
    tpl = get_template("impression_box")
    box_html = tpl.format(content=content)
    st.markdown(box_html, unsafe_allow_html=True)


def render_clinical_notice() -> None:
    """Renders the academic & clinical decision-support advisory notice."""
    html = get_template("clinical_notice")
    st.markdown(html, unsafe_allow_html=True)


def build_biomarker_reference_table(features_df: pd.DataFrame) -> pd.DataFrame:
    """Pairs measured radiomic values with units and clinical meanings."""
    meta = {
        "MeshVolume": ("mm³", "Total 3D segmented tumour burden; primary driver of mass effect."),
        "SurfaceArea": ("mm²", "Outer 3D boundary area of the lesion; elevated in infiltrative growth."),
        "Sphericity": ("0 – 1 Index", "Morphological compactness (1.0 = sphere); lower values suggest irregular margins."),
        "Energy": ("Intensity²", "Sum of squared voxel intensities inside the lesion ROI."),
        "Entropy": ("Bits", "First-order intensity randomness reflecting necrotic/oedema heterogeneity."),
        "MeanIntensity": ("Signal Units", "Mean FLAIR signal intensity within the segmented tumour mass."),
        "GLCM_Contrast": ("Gradient²", "Local voxel-to-voxel brightness contrast across tumour tissue."),
        "GLCM_Correlation": ("0 – 1 Index", "Spatial uniformity of grey-level transitions inside the lesion."),
        "GLCM_Entropy": ("Bits", "Spatial texture disorder of internal tumour microstructure."),
        "Patient_Age": ("Years", "Independent clinical prognostic covariate for glioma survival."),
    }
    rows = []
    for col in features_df.columns:
        unit, desc = meta.get(col, ("—", ""))
        val = features_df[col].values[0]
        rows.append({
            "Biomarker": col,
            "Measured Value": f"{val:,.4f}" if isinstance(val, float) and val < 10 else f"{val:,.2f}",
            "Unit": unit,
            "Clinical Significance": desc,
        })
    return pd.DataFrame(rows)


def build_radiological_impression(patient_id: str, age: int, anatomy: dict, pred: dict) -> str:
    """Creates a structured clinical impression strictly from extracted measurements."""
    if pred["days"] is None or not anatomy.get("has_tumor", False):
        return (
            f"Automated 3D volumetric screening for patient <b>{patient_id}</b> (Age: {age} yrs) detected "
            f"<b>no abnormal hyperintense tumour mass</b> (<code>MeshVolume = 0 mm³</code>). "
            f"Clinical correlation with multi-sequence radiology review is advised."
        )

    sph = float(pred["features"]["Sphericity"].values[0])
    morphology = "irregular, infiltrative margins" if sph < 0.55 else "relatively compact, circumscribed margins"
    imp = pred["importances"]
    pos_imp = imp[imp > 0]
    neg_imp = imp[imp < 0]
    if len(pos_imp) > 0 and len(neg_imp) > 0:
        shap_sentence = (
            f"TreeSHAP attribution (ΣSHAP = <code>{pred.get('net_shap', 0.0):+.3f}</code>) identifies "
            f"<b>{pos_imp.idxmax()}</b> as the primary risk-elevating driver and "
            f"<b>{neg_imp.idxmin()}</b> as the leading protective factor."
        )
    elif len(pos_imp) > 0:
        shap_sentence = (
            f"TreeSHAP attribution (ΣSHAP = <code>{pred.get('net_shap', 0.0):+.3f}</code>) shows exclusively "
            f"risk-elevating drivers, led by <b>{pos_imp.idxmax()}</b>."
        )
    elif len(neg_imp) > 0:
        shap_sentence = (
            f"TreeSHAP attribution (ΣSHAP = <code>{pred.get('net_shap', 0.0):+.3f}</code>) shows predominantly "
            f"protective factors, led by <b>{neg_imp.idxmin()}</b>."
        )
    else:
        shap_sentence = "TreeSHAP attribution shows neutral net risk impact relative to the cohort baseline."

    subcomp_str = ""
    if anatomy.get("has_subcompartments", False):
        subcomp_str = (
            f"Sub-compartment volumetrics resolve: Enhancing Tumour (ET) = <b>{anatomy['et_vol_cm3']} cm³</b>, "
            f"Tumour Core (TC) = <b>{anatomy['tc_vol_cm3']} cm³</b>, and Peritumoral Oedema (ED) = <b>{anatomy['ed_vol_cm3']} cm³</b>. "
        )

    hr_val = pred.get("hazard_ratio", 1.0)
    surv_cal = (
        f"Unified Cox Proportional Hazards modeling (Relative Hazard: <b>{hr_val:.2f}×</b> cohort baseline) projects median overall survival at "
        f"<b>~{pred['days']} days ({pred['months']} months)</b> with a 6-month survival probability of <b>{pred['prob_6m']}%</b>, "
        f"12-month probability of <b>{pred['prob_12m']}%</b> (12m mortality risk: <b>{pred['risk_score']}%</b>), and 24-month probability of <b>{pred['prob_24m']}%</b>."
        if pred.get("prob_6m") is not None
        else f"Projected overall survival: <b>~{pred['days']} days / {pred['months']} months</b>."
    )

    return (
        f"Automated 3D MRI analysis for patient <b>{patient_id}</b> (Age: {age} yrs) localizes a primary lesion in the "
        f"<b>{anatomy['hemisphere']}</b> measuring <b>{anatomy['dims_mm'][0]} × {anatomy['dims_mm'][1]} × {anatomy['dims_mm'][2]} mm</b>, "
        f"with a 3D volume of <b>{anatomy['volume_cm3']} cm³</b> ({anatomy['brain_occupancy_pct']}% of total brain volume). "
        f"{subcomp_str}"
        f"Shape radiomics indicate <b>{morphology}</b> (Sphericity: <code>{sph:.4f}</code>), consistent with "
        f"<b>{pred['tumor_class']}</b>. {surv_cal} "
        f"{shap_sentence}"
    )
