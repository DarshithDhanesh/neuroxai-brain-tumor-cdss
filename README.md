# NeuroXAI — 3D Brain Tumour Clinical Decision Support System (CDSS)

NeuroXAI is a local, offline-capable 3D Brain Tumour Clinical Decision Support System built for neuro-oncology research. It integrates multi-modal MRI segmentation (ResNet34-UNet), anatomically constrained radiotherapy target planning (`GTV → CTV`), stereotactic 3D reconstruction, IBSI-aligned 3D radiomics with native C++ TreeSHAP biomarker attribution, and longitudinal ($T_0 \to T_1$) tumour progression tracking inside a dark-themed PACS workstation.

---

## Core Modules

1. **Clinical Dashboard (`GTV → CTV` Radiotherapy Planner)**
   - Ingests 3D NIfTI (`.nii`/`.nii.gz`) volumes and zipped DICOM series with automatic affine voxel spacing calibration (`dx, dy, dz` in mm).
   - Runs ResNet34-UNet segmentation (`best_unet_brats.pth`) for Enhancing Tumour (`ET`), Tumour Core (`TC`), and Peritumoural Oedema (`ED`).
   - Computes physical-space 3D Euclidean Distance Transform margins (`5–25 mm`) clipped strictly to brain parenchyma, quantifying exact **Healthy Brain Spared (%)** across expansion margins.

2. **Multi-Planar & 3D Stereotactic Lab**
   - Smooths and triangulates 3D brain and tumour meshes via Marching Cubes scaled to physical millimetre coordinates.
   - Renders an interactive Plotly WebGL **Stereotactic 3D Millimetre Graph** (`L–R`, `P–A`, `I–S` axes) alongside synchronised Axial, Coronal, and Sagittal MPR views with radiological orientation markers and a 30 mm scale bar.

3. **Clinical Report & Explainable Biomarkers**
   - Resamples volumes to a `1.0 × 1.0 × 1.0 mm³` isotropic lattice and extracts 9 IBSI-aligned 3D radiomic features (`MeshVolume`, `Sphericity`, `SurfaceArea`, `Energy`, `Entropy`, `MeanIntensity`, `GLCM_Contrast`, `GLCM_Correlation`, `GLCM_Entropy`).
   - Computes exact native C++ `pred_contribs` TreeSHAP biomarker attributions using an XGBoost model (`xgboost_survival.json`) trained on the real BraTS 2020 survival cohort ($N = 235$) for transparent, cohort-relative risk driver ranking.
   - Exports structured, printable HTML clinical reports.

4. **Longitudinal Progression Tracking ($T_0 \to T_1$)**
   - Aligns follow-up ($T_1$) scans to the baseline ($T_0$) spatial lattice and performs voxel-wise 3D subtraction into mutually exclusive compartments: **Persisting Core** ($T_0 \cap T_1$), **Regressed Tissue** ($T_0 \setminus T_1$), and **New Growth** ($T_1 \setminus T_0$).
   - Reports volumetric change ($\Delta V\%$) alongside RANO-aligned 2D bidirectional measurements.

---

## Validation Summary (BraTS 2020)

- **Segmentation Backbone:** U-Net with pretrained ResNet34 encoder trained on BraTS 2020 multi-modal MRI (`T1`, `T1ce`, `T2`, `FLAIR`).
- **Biomarker Attribution Baseline (Official BraTS 2020 Survival Cohort, $N = 235$, 5-Fold Stratified CV):**

| Metric | 5-Fold Stratified CV (Mean ± SD) |
| :--- | :---: |
| **Accuracy** | `59.57% ± 2.69%` |
| **Precision** | `0.5936 ± 0.0266` |
| **Recall** | `0.5688 ± 0.1136` |
| **F1-Score** | `0.5749 ± 0.0728` |
| **ROC-AUC** | `0.6486 ± 0.0582` |

*Because structural MRI morphology and age alone have an inherent biological ceiling without genomic markers (IDH/MGMT) or extent of resection, NeuroXAI intentionally avoids deterministic survival countdowns on the primary dashboard and uses TreeSHAP strictly for transparent biomarker attribution.*

---

## Tech Stack

- **Deep Learning & ML:** PyTorch, `segmentation-models-pytorch`, XGBoost (native C++ TreeSHAP)
- **Medical Imaging & Geometry:** `nibabel`, `pydicom`, `scipy.ndimage`, `scikit-image` (Marching Cubes)
- **Interface & Visualisation:** Streamlit, Plotly WebGL, Matplotlib, HTML/CSS

---

## Repository Structure

```text
neuroxai-brain-tumor-cdss/
├── app.py                  # 4-page Streamlit PACS workstation entry point
├── utils.py                # Ingestion, U-Net inference, EDT CTV, 3D radiomics & TreeSHAP
├── ui_components.py        # Interactive charts, 3D stereotactic plots & UI widgets
├── templates.html          # Printable HTML clinical report template
├── static/                 # Dark-mode medical stylesheet (CSS)
├── models/
│   ├── best_unet_brats.pth     # Trained ResNet34-UNet segmentation weights
│   └── xgboost_survival.json   # Real BraTS 2020 (N=235) XGBoost biomarker model
└── requirements.txt
```

---

## Local Setup

```bash
git clone [https://github.com/DarshithDhanesh/neuroxai-brain-tumor-cdss.git](https://github.com/DarshithDhanesh/neuroxai-brain-tumor-cdss.git)
cd neuroxai-brain-tumor-cdss
python -m venv venv
venv\Scripts\activate       # On macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Disclaimer

NeuroXAI is a research prototype developed for academic and clinical decision support exploration. It is **not** a certified medical device and must not be used for standalone clinical diagnosis or radiotherapy dose delivery.
