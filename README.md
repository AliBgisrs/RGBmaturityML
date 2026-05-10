# 🌾 HMI Precision Agriculture — RGB Crop Maturity Analyzer ML Edition

> **Desktop application for automated crop maturity detection from UAV RGB orthomosaics,  
> with built-in machine learning to predict and explain maturity per genotype.**  
> Developed by **Aliasghar Bazrafkan** | bazrafka@msu.edu

---

## Table of Contents

1. [Overview](#overview)
2. [What's New in the ML Edition](#whats-new-in-the-ml-edition)
3. [Features](#features)
4. [System Requirements](#system-requirements)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Input Requirements](#input-requirements)
8. [GUI Walkthrough](#gui-walkthrough)
9. [Methods Implemented (23 total)](#methods-implemented)
10. [Output Structure](#output-structure)
11. [Field Data Comparison](#field-data-comparison)
12. [ExG Slope Analysis](#exg-slope-analysis)
13. [ML Prediction Module](#ml-prediction-module)
14. [Building the Windows EXE](#building-the-windows-exe)
15. [Troubleshooting](#troubleshooting)
16. [Project Structure](#project-structure)
17. [Citation](#citation)

---

## Overview

The **HMI Precision Agriculture Maturity Analyzer ML Edition** is a standalone desktop tool that processes multi-temporal UAV RGB orthomosaics to automatically estimate the **physiological maturity date** of crop plots. Given a folder of dated flight images and a geospatial vector file with plot boundaries, the tool:

- Clips each plot polygon from every image date
- Computes **23 spectral / colour indices** per plot per date
- Estimates a maturity DAP (Days After Planting) for each index
- Generates **publication-quality diagnostic figures** for every method, every plot, and every trial
- Optionally compares predictions against user-supplied **field observation data**
- **NEW:** Accumulates trial results in a persistent database and trains **Ridge / Random Forest / XGBoost** models to predict maturity from UAV data alone, with per-genotype formulas and a flight planning advisor

The application runs entirely offline as a graphical desktop tool (Tkinter GUI).

---

## What's New in the ML Edition

| Feature | Description |
|---|---|
| **Persistent trial database** | Save each trial's UAV features to a CSV database that grows with every run |
| **Field data join** | Link the database with your field Excel (PlotID × Experiment Name → MTR + genotype Name) |
| **Three ML models** | Ridge (interpretable formula), Random Forest, XGBoost (if installed) |
| **Leave-One-Trial-Out CV** | Unbiased cross-validation; falls back to k-fold when < 3 trials |
| **Per-genotype Ridge formula** | Each genotype gets its own intercept offset: `MTR = base + genotype_offset + β₁×HMI + ...` |
| **Flight planning advisor** | Simulates accuracy vs. N flights used → recommends start DAP, interval, and total flights |
| **Scrollable left panel** | All controls accessible regardless of screen height |
| **ML_Analysis/ output folder** | `cv_report.png`, `feature_importance.png`, `field_vs_predicted.png`, `genotype_formulas.png`, `flight_planning.png`, `feature_correlation.png`, `model.pkl`, `predictions.xlsx`, `feature_matrix.xlsx` |

---

## Features

| Category | Detail |
|---|---|
| **23 spectral indices** | Chromatic, ratio, quadratic, HSV, CIE Lab, and histogram-based methods |
| **Plot-by-plot processing** | Any number of plots; auto-detects PlotID column |
| **Multi-temporal analysis** | Handles 2–20+ flight dates; dates parsed from filenames |
| **Maturity detection** | Absolute threshold + relative 80%-transition fallback |
| **Diagnostic plots** | Time-series, ridgeline distributions, hue histograms, 3-D hue stacks, CIE Lab scatter, chromatic scatter, cover-class bars |
| **Consensus histograms** | Per-plot and per-trial DAP frequency charts |
| **ExG slope analysis** | PCHIP spline regression, decline slope per plot, trial summary |
| **Field data comparison** | Cross-plot (Field Data ± SD vs IQR-trimmed Prediction ± SD) |
| **Green reference band** | User-supplied field range shown on every output figure |
| **Interpretation cards** | Auto-generated formula + biology + reading guide per method |
| **RGB thumbnails** | Vertical strip of clipped plot thumbnails per flight date |
| **SUMMARY.xlsx** | Maturity DAP and calendar date for every plot × method |
| **Trial grouping** | Auto-detects Trial/Experiment/Block columns |
| **ML prediction** | Ridge + RF + XGBoost trained on accumulated trials; LOTO cross-validation |
| **Per-genotype formula** | Interpretable linear formula per genotype from Ridge model |
| **Flight planning advisor** | RMSE vs. N-flights curve → minimum flights for target accuracy |
| **EXE build** | Single-file Windows executable via PyInstaller |

---

## System Requirements

| Component | Minimum |
|---|---|
| OS | Windows 10 / 11 (64-bit) |
| Python | 3.9 – 3.12 |
| RAM | 8 GB (16 GB recommended for large trials) |
| Disk | 2 GB free for dependencies + output |
| GPU | Not required |

### Python dependencies

```
numpy
pandas
geopandas
rasterio
fiona
shapely
pyogrio
opencv-python
matplotlib
scipy
xlsxwriter
openpyxl
scikit-learn       # required for ML module
xgboost            # optional — enables XGBoost model
pyinstaller        # only needed to build the EXE
```

---

## Installation

### Option A — Run from source (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/AliBgisrs/RGB-Crop-Maturity-ML.git
cd RGB-Crop-Maturity-ML

# 2. Create a virtual environment (optional but recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Launch the app
python maturity_app.py
```

### Option B — Quick launch (double-click)

Double-click `run.bat` — it checks for `pyogrio` and starts the app automatically.

### Option C — Build the EXE yourself

```bat
build.bat
```

The executable will be created at `dist\MaturityAnalyzer.exe`.

---

## Quick Start

### Standard pipeline (one trial)

```
1. Open the app:  python maturity_app.py   (or run.bat)
2. Images Directory  → folder containing dated flight subfolders or TIF files
3. Vector Path       → your .shp or .gdb with plot polygons
4. Layer Name        → layer inside the GDB (use ⟳ Detect button)
5. Sowing Date       → MM_DD_YYYY  (e.g. 06_03_2025)
6. Plot ID Field     → column name that uniquely identifies each plot
7. (Optional) Field Maturity DAP Min / Max → shows green band on all figures
8. Click  ▶ Run Pipeline
9. Click  📂 Open Output Folder  when done
```

### ML workflow (multi-trial)

```
After each trial run:
  10. Trial Label  → set a name (auto-filled from Layer Name)
  11. ML Database Dir → choose a persistent folder (same across all runs)
  12. Click  💾 Save Trial to ML Database

After accumulating ≥ 2 trials:
  13. Click  🧠 Load Field Data & Train Models
      → select your field Excel containing MTR and genotype Name
      → ML_Analysis/ folder created with all diagnostic outputs

To predict a new trial:
  14. Run the pipeline for the new trial
  15. Click  🔮 Predict This Trial
      → predictions_{TrialName}.xlsx saved to ML_Analysis/
```

---

## Input Requirements

### Images directory

The tool searches recursively for raster files and groups them by date. Supported naming patterns:

```
Flight_2025-06-03/
  orthomosaic.tif          ← multi-band RGB stack (bands 1,2,3 = R,G,B)

Flight_20250610/
  Red.tif                  ← separate single-band files
  Green.tif
  Blue.tif
```

Date parsing supports separators `-`, `_`, or none: `YYYYMMDD`, `YYYY-MM-DD`, `MM_DD_YYYY`.

> **Important:** At least **2 flight dates** are required.

### Vector layer (plot boundaries)

- Formats: **Shapefile (.shp)** or **ESRI File GDB (.gdb)**
- CRS: any projected or geographic CRS (auto-reprojected to match each raster)
- Required column: a **PlotID** column (name configurable in the GUI)
- Optional column: **Trial / Experiment / Block** — used to group plots for trial-level outputs

### Raster format

- **RGB only** (3-band or 3 separate files)
- Accepted dtypes: `uint8`, `uint16` (typical UAV outputs)
- Bands must map to Red → Green → Blue in order

### Field data Excel (for ML training)

The Excel file must contain at minimum:

| Column | Example | Description |
|---|---|---|
| `PlotID` | 1001 | Matches the PlotID in the vector layer |
| `Experiment Name` | 2501 | Trial identifier (numeric part matched to Layer Name) |
| `Name` | Genotype_A | Genotype / cultivar name |
| `MTR` | 95 | Ground-truth maturity DAP (the ML target) |

Other columns (ENTRY, REP, FLWR, HT, etc.) are ignored.

---

## GUI Walkthrough

```
┌───────────────────────────────────────────────────────────────────┐
│  🌾  HMI Precision Agriculture          v3.0                      │
├───────────────────────────────────────────────────────────────────┤
│  Images Directory   [C:\flights\2025]              [Browse]       │
│  Vector Format      ○ Shapefile  ● File GDB                       │
│  Vector Path        [C:\data\plots.gdb]            [Browse]       │
│  Layer Name (GDB)   [SEVREC2501 ▼]                 [⟳ Detect]    │
│  Sowing Date        [06_03_2025]                                   │
│  Plot ID Field      [PlotID]            (exact or case-insensitive)│
│  Output Directory   [C:\Output]                    [Browse]       │
├───────────────────────────────────────────────────────────────────┤
│  Field Maturity DAP  Min [85]   Max [100]                         │
│  Reference median = 92.5 DAP                                      │
├───────────────────────────────────────────────────────────────────┤
│  [▶  Run Pipeline]                                                │
│  ████████████░░░░ 65 %                                            │
│  [📂  Open Output Folder]                                         │
│  [📊  Re-run Field Comparison]                                    │
├───────────────────────────────────────────────────────────────────┤
│  🤖  ML Maturity Prediction                                        │
│  ML Database Dir    [C:\Output]                    [Browse]       │
│  Trial Label        [SEVREC2501]                                   │
│  [💾  Save Trial to ML Database]                                  │
│  [🧠  Load Field Data & Train Models]                             │
│  [🔮  Predict This Trial]                                         │
└───────────────────────────────────────────────────────────────────┘
```

> **Tip:** The left panel is **scrollable** — use the mouse wheel or the scrollbar to reach the ML section.

---

## Methods Implemented

### Group 1 — Chromatic Coordinates

| ID | Name | Direction | Threshold |
|---|---|---|---|
| `GCC` | Green Chromatic Coordinate | ↓ decrease | 0.36 |
| `RCC` | Red Chromatic Coordinate | ↑ increase | 0.40 |

### Group 2 — Ratio / Linear Indices

| ID | Name | Direction | Threshold |
|---|---|---|---|
| `NGRDI` | Normalized Green-Red Difference Index | ↓ | 0.00 |
| `VARI` | Visible Atmospherically Resistant Index | ↓ | 0.00 |
| `GLI` | Green Leaf Index | ↓ | 0.00 |
| `ExGR` | Excess Green minus Excess Red | ↓ | 0.00 |
| `IKAW` | Kawashima & Nakatani Index | ↑ | 0.05 |
| `NDYI` | Normalized Difference Yellowness Index | ↑ | 0.15 |
| `R_over_G` | Red-to-Green Ratio | ↑ | 1.10 |
| `TGI` | Triangular Greenness Index | ↓ | — (relative) |
| `WI` | Woebbecke Index | ↑ | — (relative) |

### Group 3 — Quadratic Indices

| ID | Name | Direction | Threshold |
|---|---|---|---|
| `MGRVI` | Modified Green-Red Vegetation Index | ↓ | 0.00 |
| `RGBVI` | RGB Vegetation Index (Gruner 2019) | ↓ | 0.15 |

### Group 4 — HSV-Based Indices

| ID | Name | Direction | Threshold |
|---|---|---|---|
| `HMI_MASKED` | Hue Maturity Index (ExG masked) | ↑ | 0.80 |
| `HMI_MAIN` | Hue Maturity Index (all pixels) | ↑ | 0.80 |
| `MPI` | Maturity Progression Index | ↑ | 0.85 |
| `desicc_frac` | Desiccation Fraction | ↑ | 0.50 |
| `green_cover` | Green Cover Fraction | ↓ | 0.20 |

### Group 5 — CIE Lab Colour Space

| ID | Name | Direction | Threshold |
|---|---|---|---|
| `Lab_a` | CIE Lab a* (green→red axis) | ↑ | 5.0 |
| `Lab_b` | CIE Lab b* (blue→yellow axis) | ↑ | 28.0 |
| `Lab_Chroma` | CIE Lab Chroma √(a²+b²) | ↑ | 25.0 |
| `Lab_HueAngle` | CIE Lab Hue Angle atan2(b,a) | ↓ | 90.0 |

### Group 6 — Histogram-Based

| ID | Name | Direction | Threshold |
|---|---|---|---|
| `hist_ratio` | Bhattacharyya Distance Ratio | ↑ | 1.20 |

#### Maturity detection logic

For each method the tool:
1. Computes the index mean per plot per date → time series
2. Applies a 3-point moving average (MA-3)
3. Checks if the absolute threshold is crossed → records the interpolated crossing DAP
4. Falls back to the **relative 80% transition** when no threshold crossing occurs
5. Reports `not detected` if fewer than 2 finite values exist

---

## Output Structure

```
{output_root}/
│
├── SUMMARY.xlsx                  ← Maturity DAP + date for every plot × method
├── Summary_Heatmap.png           ← Plots × Methods heatmap
│
├── 00_RGB_Thumbnails/            ← Vertical RGB thumbnail strips per plot
│
├── 01_GCC/ … 23_hist_ratio/      ← Per-method folders
│   ├── interpretation.png        ← Formula, biology, reading guide
│   ├── GCC_Plot1001.png          ← Time-series per plot
│   ├── dist_ridgeline_Plot1001.png
│   ├── GCC_distribution.png      ← Boxplot across all plots
│   └── stats_GCC.csv
│
├── Comparison/
│   └── Comparison_Plot1001.png   ← All-23-methods panel per plot
│
├── Method_Consensus/
│   └── consensus_Plot1001.png    ← DAP histogram + per-method bar chart
│
├── Trial_Histograms/
│   └── Trial_All_Plots_histogram.png
│
├── ExG_Slope/
│   ├── ExG_slope_Plot1001.png    ← ExG PCHIP regression per plot
│   └── ExG_slope_trial_All_Plots.png
│
├── Field_Comparison/             ← Only when DAP range is provided
│   ├── FieldComp_Plot1001.png
│   └── FieldComp_AllPlots_Summary.png
│
├── TimeSeries/
│   └── ts_1001.csv               ← Raw index time series per plot (input to ML)
│
└── ML_Analysis/                  ← Created by the ML module
    ├── feature_matrix.xlsx       ← Full feature matrix used for training
    ├── feature_correlation.png   ← Feature × feature correlation heatmap
    ├── cv_report.png             ← Scatter + per-fold RMSE bar chart
    ├── feature_importance.png    ← Ridge coefficients + RF importance
    ├── field_vs_predicted.png    ← Field MTR vs predicted scatter by genotype
    ├── genotype_formulas.png     ← Per-genotype Ridge formula cards
    ├── flight_planning.png       ← RMSE vs N-flights advisor
    ├── model.pkl                 ← Saved model bundle (used by Predict button)
    ├── predictions.xlsx          ← Training + CV predictions per plot
    └── ml_summary.txt            ← Text summary of CV results + flight plan
```

Additionally, the **ML database** is stored separately:

```
{ML Database Dir}/
└── ML_Database/
    └── training_data.csv         ← Accumulates one row per plot across all trials
```

---

## Field Data Comparison

When you enter a **Field Maturity DAP range** (Min and Max) before running, the tool automatically:

1. Computes the reference: `mean = (Min + Max) / 2`, `SD = (Max − Min) / 2`
2. Generates a cross-plot for every plot comparing:
   - **Left — Field Data**: mean ± SD (blue)
   - **Right — Prediction**: IQR-trimmed mean ± SD of all 23 method estimates (orange)
   - **Δ arrow**: labelled difference between the two means
3. Saves an **AllPlots Summary** aggregating all method predictions across the trial

The **"📊 Re-run Field Comparison"** button re-generates these plots with a different DAP range without re-running the full pipeline.

A **green shaded band** appears on every time-series, comparison, consensus, and ExG slope figure.

---

## ExG Slope Analysis

The `ExG_Slope/` folder contains **senescence rate figures** based on the Excess Green (ExG) index:

### Per-plot (`ExG_slope_Plot{ID}.png`)
- **PCHIP spline** fit through all observed ExG values
- **Peak detection** on the smoothed curve
- **Slope line** from peak to last observation: `slope = (y_last − y_peak) / (DAP_last − DAP_peak)`
  - More negative slope = faster senescence

### Trial summary (`ExG_slope_trial_{name}.png`)
- **TOP panel**: horizontal bar chart sorted by slope; coloured red (fast) → green (slow)
- **BOTTOM panel**: histogram + KDE distribution of slopes across the trial

---

## ML Prediction Module

### Overview

The ML module (`ml_analysis.py`) builds a supervised model that predicts **maturity DAP from UAV spectral data alone**, using genotype name as the only non-UAV feature. Field data (MTR) is used only as the training target and for cross-validation — never as a predictor at inference time.

### Features used (36 total)

| Group | Features |
|---|---|
| **23 method DAPs** | `GCC_DAP`, `RCC_DAP`, … `hist_ratio_DAP` — from SUMMARY.xlsx |
| **ExG regression** | `exg_peak_dap`, `exg_slope` — PCHIP peak + decline slope |
| **Senescence rates** | `gcc_drop_rate`, `rcc_rise_rate` — linear slope of GCC/RCC series |
| **HMI signal** | `hmi_crossing_dap` — first DAP where HMI_MASKED ≥ 0.80 |
| **Consensus** | `consensus_mean`, `consensus_median`, `consensus_std`, `n_detected` |
| **Flight info** | `n_flights`, `first_flight_dap`, `last_flight_dap`, `flight_span_days` |
| **Categorical** | `Name` (genotype) — OrdinalEncoded, contributes per-genotype offset in Ridge |

### Step-by-step ML workflow

```
Trial 1 run complete
  └─ 💾 Save Trial to ML Database
        → extracts 36 features from SUMMARY.xlsx + TimeSeries/*.csv
        → appends to ML_Database/training_data.csv

Trial 2 run complete
  └─ 💾 Save Trial to ML Database  (previous trial data is kept)

Trial 3+ ...

🧠 Load Field Data & Train Models
  └─ Select field Excel (PlotID + Experiment Name + Name + MTR)
  └─ Joins database with field data on PlotID × Trial
  └─ Trains:
       ▸ Ridge (CV alpha selection, interpretable per-genotype formula)
       ▸ Random Forest (300 trees, sqrt features)
       ▸ XGBoost (300 rounds, lr=0.05) — if installed
  └─ Leave-One-Trial-Out CV (or 5-fold if < 3 trials)
  └─ Saves ML_Analysis/ with all plots and model.pkl

🔮 Predict This Trial  (new season, no field data needed)
  └─ Extracts features from the new trial's output folder
  └─ Applies saved model.pkl
  └─ Saves predictions_{TrialName}.xlsx
```

### Per-genotype Ridge formula

Ridge regression learns a **global formula** plus a **genotype-specific intercept offset**:

```
MTR (Genotype A) = 87.3 + 0 offset
  + 0.41230 × HMI_MASKED_DAP
  + 0.28710 × consensus_median
  − 0.19450 × exg_slope
  + 0.15320 × GCC_DAP
  ...

MTR (Genotype B) = 87.3 + 3.2 offset = 90.5
  (same coefficients, different intercept)
```

This captures the fact that some genotypes mature consistently earlier or later than the global average, while the UAV spectral features still predict the within-genotype variation.

### Flight planning advisor

The advisor answers: **"How many flights do I actually need?"**

For each N from 2 to max:
1. Re-computes features using only the first N flights per plot
2. Predicts MTR with the trained model
3. Measures RMSE against field data

The `flight_planning.png` plot shows RMSE vs. N and highlights the optimal point. The recommendation box reports:

```
✈  Start:    DAP 45
✈  Interval: every 7 days
✈  Flights:  6 total
```

### Cross-validation strategy

| Condition | Strategy |
|---|---|
| ≥ 3 unique trials in database | **Leave-One-Trial-Out (LOTO)** — each trial is held out in turn |
| < 3 trials | **5-fold cross-validation** |

LOTO is the recommended approach because it tests whether the model generalises to an unseen trial (location / year), which is the real deployment scenario.

---

## Building the Windows EXE

```bat
build.bat
```

This upgrades PyInstaller, verifies `pyogrio`, and calls PyInstaller with all required hidden imports. Output: `dist\MaturityAnalyzer.exe`.

> **Note:** `scikit-learn` and `xgboost` must be installed before building if you want the ML module included in the EXE.

> **Tip:** Anti-virus software may flag PyInstaller executables. Add a folder exclusion or use the source version instead.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `pyogrio not available` | `pip install pyogrio --force-reinstall` then restart |
| `0 features loaded` from GDB | Use **Preview** to inspect columns; try exporting to Shapefile |
| All methods show `not detected` | Need ≥ 2 flight dates with valid pixel overlap; check raster/vector CRS alignment |
| Images not found | Filenames must contain a date string (`YYYYMMDD`, `YYYY-MM-DD`) |
| Memory error on large trials | Reduce `MAX_PX` in `analysis.py` (default 3000 pixels per plot/date) |
| EXE crashes silently | Run `python maturity_app.py` from terminal to see the full traceback |
| `Sowing date` format error | Must be `MM_DD_YYYY` — e.g. `06_03_2025` for June 3rd 2025 |
| `scikit-learn` not installed | `pip install scikit-learn` — ML buttons will be greyed out without it |
| ML: `0 rows matched` after join | Verify PlotID values match exactly and the Trial numeric part (e.g. `2501`) appears in both the Layer Name and `Experiment Name` column of the field Excel |
| ML: `Only N matched rows` | Check that field Excel uses the same PlotID numbering as the vector layer |
| ML: LOTO RMSE very high | Normal with few trials — add more trial data or use the 5-fold result |
| Push to GitHub: `401 Unauthorized` | Token expired or copied incorrectly — generate a fresh Classic PAT |
| Push to GitHub: `403 Forbidden` | Token missing `repo` scope — tick the **repo** checkbox when generating |

---

## Project Structure

```
RGB-Crop-Maturity-ML/
├── maturity_app.py       # Tkinter GUI (scrollable, includes ML section)
├── analysis.py           # All 23 spectral indices, plotting, pipeline logic
├── ml_analysis.py        # ML module: feature extraction, training, prediction
├── requirements.txt      # Python dependencies
├── build.bat             # PyInstaller build script (Windows)
├── run.bat               # Quick-launch script
├── push_to_github.bat    # Create GitHub repo + push
└── README.md             # This file
```

---

## Citation

If you use this tool in your research, please cite:

```
Bazrafkan, A. (2025). HMI Precision Agriculture — RGB Crop Maturity Analyzer ML Edition (v3.1).
Michigan State University. https://github.com/AliBgisrs/RGB-Crop-Maturity-ML
```

---

## License

This project is **proprietary and confidential**. Unauthorised copying, distribution, or modification without explicit written permission from the author is prohibited.

© 2025 Aliasghar Bazrafkan — Michigan State University
