# ======================================================================
# ml_analysis.py  -  Machine Learning module for RGB Crop Maturity Analyzer
# Predicts maturity DAP using UAV-derived spectral indices
# Developed by Aliasghar Bazrafkan  |  bazrafka@msu.edu
# ======================================================================
"""
Workflow
--------
1. After EACH trial pipeline run click "Save Trial to ML Database":
      save_trial_to_database(output_root, trial_name, db_path)
      → extracts features from SUMMARY.xlsx + TimeSeries/*.csv
      → appends to a persistent CSV database

2. After accumulating ≥ 2 trials with field data, click "Train & Analyze":
      run_ml_pipeline(field_excel, db_path, out_dir)
      → joins DB with field Excel (MTR ground-truth + Name genotype)
      → trains Ridge + RandomForest [+ XGBoost if installed]
      → Leave-One-Trial-Out CV (or k-fold if < 3 trials)
      → per-genotype Ridge formula
      → flight planning advisor
      → saves all diagnostic plots + model.pkl + predictions.xlsx

3. Click "Predict This Trial" to apply the saved model to any output folder:
      predict_new_trial(output_root, model_path)

Features (UAV-only + Name as categorical)
-----------------------------------------
  - 23 method DAP estimates from SUMMARY.xlsx  (GCC_DAP … hist_ratio_DAP)
  - exg_peak_dap, exg_slope        — from ExGR time series (PCHIP)
  - gcc_drop_rate, rcc_rise_rate   — linear slope of GCC/RCC over season
  - hmi_crossing_dap               — first DAP where HMI_MASKED ≥ 0.80
  - consensus_mean/median/std      — cross-method consensus statistics
  - n_detected, n_flights          — data quality indicators
  - first_flight_dap, last_flight_dap, flight_span_days
  - Name (genotype)                — categorical; Ridge encodes per-genotype offset
"""

import os
import re
import pickle
import shutil
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import PchipInterpolator

# ── scikit-learn ──────────────────────────────────────────────────────────────
try:
    from sklearn.linear_model import RidgeCV, ElasticNetCV
    from sklearn.svm import SVR
    from sklearn.ensemble import (RandomForestRegressor,
                                   GradientBoostingRegressor,
                                   ExtraTreesRegressor)
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import LeaveOneGroupOut, KFold
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ── Optional XGBoost ──────────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────── CONSTANTS ──────────────────────────────────────
DPI            = 300
FONT           = 12
TITLE_FONT     = 13
SMALL          = 9
FLIGHT_TOP_K   = 5    # number of lowest-RMSE points that define the "optimal window"
BG_C       = "#ffffff"
PANEL_C    = "#f8f9fa"
BORDER_C   = "#dee2e6"
LABEL_C    = "#495057"
TEXT_C     = "#212529"

METHOD_NAMES: List[str] = [
    "GCC", "RCC", "NGRDI", "VARI", "GLI", "MGRVI", "RGBVI", "ExGR",
    "IKAW", "NDYI", "R_over_G", "TGI", "WI", "HMI_MASKED", "HMI_MAIN",
    "MPI", "desicc_frac", "green_cover", "Lab_a", "Lab_b", "Lab_Chroma",
    "Lab_HueAngle", "hist_ratio",
]
DAP_COLS: List[str] = [f"{m}_DAP" for m in METHOD_NAMES]   # 23 target columns

HMI_THR = 0.80   # HMI_MASKED threshold for crossing DAP

# Continuous features fed to the ML model
_CONT_FEATURES: List[str] = [
    *DAP_COLS,
    "exg_peak_dap", "exg_slope",
    "gcc_drop_rate", "rcc_rise_rate",
    "hmi_crossing_dap",
    "consensus_mean", "consensus_median", "consensus_std",
    "n_detected", "n_flights",
    "first_flight_dap", "last_flight_dap", "flight_span_days",
]

# Database column order
_DB_META: List[str] = ["PlotID", "TrialName", "output_root", "Name", "SavedAt"]
_DB_COLS: List[str] = _DB_META + _CONT_FEATURES


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _safe_pchip_crossing(dap: np.ndarray, values: np.ndarray,
                          threshold: float,
                          direction: str = "increase") -> Optional[float]:
    """Interpolated DAP where *values* first crosses *threshold*.
    direction: 'increase' (values rises to ≥ threshold)
               'decrease' (values falls to ≤ threshold).
    Returns None if never crossed or fewer than 2 finite points."""
    mask = np.isfinite(values) & np.isfinite(dap)
    dap_m, val_m = dap[mask], values[mask]
    if len(dap_m) < 2:
        return None
    try:
        fn       = PchipInterpolator(dap_m, val_m)
        fine_dap = np.linspace(dap_m[0], dap_m[-1], 500)
        fine_val = fn(fine_dap)
        idx = (np.where(fine_val >= threshold)[0] if direction == "increase"
               else np.where(fine_val <= threshold)[0])
        return float(fine_dap[idx[0]]) if len(idx) else None
    except Exception:
        return None


def _exg_features(dap: np.ndarray, exg: np.ndarray
                   ) -> Tuple[Optional[float], Optional[float]]:
    """Return (peak_dap, decline_slope) from an ExG time series.
    slope = (ExG_last − ExG_peak) / (DAP_last − DAP_peak);
    more negative = faster senescence."""
    mask = np.isfinite(exg) & np.isfinite(dap)
    d, e = dap[mask], exg[mask]
    if len(d) < 3:
        return None, None
    try:
        fn       = PchipInterpolator(d, e)
        fine     = np.linspace(d[0], d[-1], 500)
        fine_e   = fn(fine)
        pk_idx   = int(np.argmax(fine_e))
        pk_dap   = float(fine[pk_idx])
        pk_val   = float(fine_e[pk_idx])
        last_dap = float(fine[-1])
        last_val = float(fine_e[-1])
        if last_dap <= pk_dap:
            return pk_dap, None
        slope = (last_val - pk_val) / (last_dap - pk_dap)
        return pk_dap, float(slope)
    except Exception:
        return None, None


def _linear_rate(dap: np.ndarray, values: np.ndarray) -> Optional[float]:
    """Linear regression slope (change per DAP)."""
    mask = np.isfinite(values) & np.isfinite(dap)
    d, v = dap[mask], values[mask]
    if len(d) < 2:
        return None
    coeffs = np.polyfit(d, v, 1)
    return float(coeffs[0])


def _light_ax(ax):
    """Apply consistent light/white-theme styling to a matplotlib Axes."""
    ax.set_facecolor(BG_C)
    ax.tick_params(colors=LABEL_C, labelsize=SMALL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER_C)
    ax.title.set_color(TEXT_C)
    ax.xaxis.label.set_color(LABEL_C)
    ax.yaxis.label.set_color(LABEL_C)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_features_from_output(output_root: str,
                                  trial_name: str,
                                  log_fn=None) -> pd.DataFrame:
    """
    Extract ML features for all plots from a completed pipeline output folder.

    Reads:
      {output_root}/SUMMARY.xlsx          → 23 method DAP estimates
      {output_root}/TimeSeries/ts_{pid}.csv → per-plot time series

    Returns DataFrame with columns = _DB_COLS (PlotID, TrialName,
    output_root, Name, + all continuous features).
    """
    def _log(m):
        if log_fn:
            log_fn(m)

    summary_path = os.path.join(output_root, "SUMMARY.xlsx")
    ts_dir       = os.path.join(output_root, "TimeSeries")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"SUMMARY.xlsx not found in: {output_root}")

    _log(f"[ML] Reading features from: {output_root}")

    # ── Load SUMMARY.xlsx ─────────────────────────────────────────────────────
    df_sum = pd.read_excel(summary_path, engine="openpyxl")
    df_sum.columns = [str(c).strip() for c in df_sum.columns]

    pid_col = next(
        (c for c in df_sum.columns if c.lower() == "plotid"),
        df_sum.columns[0])

    # Convert DAP columns to numeric
    available_dap = [c for c in DAP_COLS if c in df_sum.columns]
    for c in available_dap:
        df_sum[c] = pd.to_numeric(df_sum[c], errors="coerce")

    rows: List[Dict] = []
    for _, row in df_sum.iterrows():
        pid  = str(row[pid_col]).strip()
        feat: Dict[str, Any] = {
            "PlotID":      pid,
            "TrialName":   trial_name,
            "output_root": output_root,
            "Name":        "",
        }

        # 23 method DAP estimates
        for col in DAP_COLS:
            feat[col] = (float(row[col])
                         if col in row.index and pd.notna(row[col]) else np.nan)

        # Consensus stats across methods
        dap_vals = np.array([feat[c] for c in available_dap])
        finite   = dap_vals[np.isfinite(dap_vals)]
        feat["n_detected"]       = int(len(finite))
        feat["consensus_mean"]   = float(np.nanmean(finite))   if len(finite) > 0 else np.nan
        feat["consensus_median"] = float(np.nanmedian(finite)) if len(finite) > 0 else np.nan
        feat["consensus_std"]    = float(np.nanstd(finite))    if len(finite) > 1 else np.nan

        # ── Time-series features ──────────────────────────────────────────────
        ts_path = os.path.join(ts_dir, f"ts_{pid}.csv")
        if os.path.exists(ts_path):
            try:
                ts = pd.read_csv(ts_path)
                ts.columns = [c.strip() for c in ts.columns]

                dap_c   = next((c for c in ts.columns if c.lower() == "dap"), ts.columns[0])
                dap_arr = pd.to_numeric(ts[dap_c], errors="coerce").values
                ok_dap  = np.isfinite(dap_arr)

                feat["n_flights"]        = int(ok_dap.sum())
                feat["first_flight_dap"] = float(dap_arr[ok_dap][0])  if ok_dap.sum() > 0 else np.nan
                feat["last_flight_dap"]  = float(dap_arr[ok_dap][-1]) if ok_dap.sum() > 0 else np.nan
                feat["flight_span_days"] = (float(feat["last_flight_dap"] - feat["first_flight_dap"])
                                            if ok_dap.sum() > 1 else np.nan)

                # ExG slope (use ExGR column as proxy for Excess Green)
                exg_c = next((c for c in ts.columns if c.upper() in ("EXGR", "EXG")), None)
                if exg_c:
                    ep, es = _exg_features(dap_arr,
                                           pd.to_numeric(ts[exg_c], errors="coerce").values)
                    feat["exg_peak_dap"] = ep if ep is not None else np.nan
                    feat["exg_slope"]    = es if es is not None else np.nan
                else:
                    feat["exg_peak_dap"] = feat["exg_slope"] = np.nan

                # GCC drop rate
                gcc_c = next((c for c in ts.columns if c.upper() == "GCC"), None)
                if gcc_c:
                    rate = _linear_rate(dap_arr,
                                        pd.to_numeric(ts[gcc_c], errors="coerce").values)
                    feat["gcc_drop_rate"] = rate if rate is not None else np.nan
                else:
                    feat["gcc_drop_rate"] = np.nan

                # RCC rise rate
                rcc_c = next((c for c in ts.columns if c.upper() == "RCC"), None)
                if rcc_c:
                    rate = _linear_rate(dap_arr,
                                        pd.to_numeric(ts[rcc_c], errors="coerce").values)
                    feat["rcc_rise_rate"] = rate if rate is not None else np.nan
                else:
                    feat["rcc_rise_rate"] = np.nan

                # HMI_MASKED crossing DAP
                hmi_c = next(
                    (c for c in ts.columns if c.upper() in ("HMI_MASKED", "HMI MASKED")),
                    None)
                if hmi_c:
                    cross = _safe_pchip_crossing(
                        dap_arr,
                        pd.to_numeric(ts[hmi_c], errors="coerce").values,
                        HMI_THR, "increase")
                    feat["hmi_crossing_dap"] = cross if cross is not None else np.nan
                else:
                    feat["hmi_crossing_dap"] = np.nan

            except Exception as ex:
                _log(f"  [ML] Warning ts_{pid}.csv: {ex}")
                for k in ("exg_peak_dap", "exg_slope", "gcc_drop_rate", "rcc_rise_rate",
                          "hmi_crossing_dap", "n_flights", "first_flight_dap",
                          "last_flight_dap", "flight_span_days"):
                    feat.setdefault(k, np.nan)
        else:
            for k in ("exg_peak_dap", "exg_slope", "gcc_drop_rate", "rcc_rise_rate",
                      "hmi_crossing_dap", "n_flights", "first_flight_dap",
                      "last_flight_dap", "flight_span_days"):
                feat[k] = np.nan

        rows.append(feat)

    df_feat = pd.DataFrame(rows)
    for col in _DB_COLS:
        if col not in df_feat.columns:
            df_feat[col] = np.nan

    _log(f"[ML] Extracted {len(df_feat)} plots, "
         f"{df_feat[_CONT_FEATURES].notna().sum().sum()} finite feature values.")
    return df_feat[_DB_COLS]


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — PERSISTENT DATABASE
# ═════════════════════════════════════════════════════════════════════════════

def save_trial_to_database(output_root: str,
                            trial_name: str,
                            db_path: str,
                            log_fn=None) -> str:
    """Extract features from *output_root* and **always append** to the database.

    Every call adds a new batch of rows stamped with ``SavedAt`` (today's date).
    Nothing is ever replaced or deleted, so the database grows with every save
    and captures the full history of all pipeline runs across all trials.

    Returns the db_path.
    """
    def _log(m):
        if log_fn:
            log_fn(m)

    today_str = datetime.now().strftime("%Y-%m-%d")
    df_new    = extract_features_from_output(output_root, trial_name, log_fn=log_fn)
    df_new["SavedAt"] = today_str

    db_dir = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(db_dir, exist_ok=True)

    if os.path.exists(db_path):
        df_old = pd.read_csv(db_path, low_memory=False)
        if "SavedAt" not in df_old.columns:
            df_old["SavedAt"] = "2000-01-01"   # back-fill legacy rows
        df_out = pd.concat([df_old, df_new], ignore_index=True)
        _log(f"[ML] Appended {len(df_new)} rows → database now "
             f"{len(df_out)} rows across "
             f"{df_out['TrialName'].nunique()} trial(s).")
    else:
        df_out = df_new
        _log(f"[ML] New database created: {len(df_out)} rows  (SavedAt={today_str}).")

    # Per-trial row count summary
    for t in sorted(df_out["TrialName"].astype(str).unique()):
        n_t = int((df_out["TrialName"].astype(str) == t).sum())
        _log(f"[ML]   {t}: {n_t} rows total")

    df_out.to_csv(db_path, index=False)
    _log(f"[ML] Saved → {db_path}")
    return db_path


def load_database(db_path: str) -> pd.DataFrame:
    """Load the accumulated trial database CSV."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"ML database not found: {db_path}\n"
                                "Save at least one trial first.")
    return pd.read_csv(db_path, low_memory=False)


def database_summary(db_path: str) -> str:
    """Return a short human-readable summary of the database."""
    if not os.path.exists(db_path):
        return "Database not found."
    df = pd.read_csv(db_path, low_memory=False)
    trials = sorted(df["TrialName"].astype(str).unique().tolist())
    lines = [
        f"Database: {db_path}",
        f"  Total rows : {len(df)}",
        f"  Trials     : {len(trials)}",
    ]
    for t in trials:
        t_df   = df[df["TrialName"].astype(str) == t]
        if "SavedAt" in t_df.columns:
            dates = sorted(t_df["SavedAt"].astype(str).dropna().unique().tolist())
            date_str = ", ".join(dates) if dates else "unknown"
        else:
            date_str = "unknown (legacy)"
        lines.append(f"    {t}: {len(t_df)} rows — saved on: {date_str}")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — JOIN WITH FIELD DATA
# ═════════════════════════════════════════════════════════════════════════════

def join_with_field_data(db_df: pd.DataFrame,
                          field_excel: str,
                          plot_id_col:   str = "PlotID",
                          trial_col:     str = "Experiment Name",
                          genotype_col:  str = "Name",
                          mtr_col:       str = "MTR",
                          log_fn=None) -> pd.DataFrame:
    """
    Inner-join the ML database with the field Excel.

    Field Excel requirements:
      - *plot_id_col*  : plot identifier  (e.g. 'PlotID' or 'Plot')
      - *trial_col*    : trial/experiment (e.g. 'Experiment Name')
      - *genotype_col* : genotype name    (e.g. 'Name')
      - *mtr_col*      : maturity DAP     (e.g. 'MTR')

    Matching rule: PlotID (exact string) AND trial numeric part match.
    Column lookup is fuzzy: exact → case-insensitive → alias table.
    """
    def _log(m):
        if log_fn:
            log_fn(m)

    _log(f"[ML] Loading field data: {field_excel}")
    if not os.path.exists(field_excel):
        raise FileNotFoundError(f"Field data not found: {field_excel}")

    # Support both .xlsx and .csv
    try:
        if field_excel.lower().endswith(".csv"):
            df_f = pd.read_csv(field_excel, low_memory=False)
        else:
            df_f = pd.read_excel(field_excel, engine="openpyxl")
    except Exception:
        try:
            df_f = pd.read_excel(field_excel)
        except Exception:
            df_f = pd.read_csv(field_excel, low_memory=False)

    df_f.columns = [str(c).strip() for c in df_f.columns]
    _log(f"[ML] Field data: {len(df_f)} rows, columns: {list(df_f.columns)}")

    # ── Column aliases: common names that mean the same thing ─────────────────
    _ALIASES: Dict[str, List[str]] = {
        "plotid":          ["plotid", "plot_id", "plot id", "plot", "plotno",
                            "plot_no", "plot number", "plotnumber"],
        "experiment name": ["experiment name", "experimentname", "exp name",
                            "expname", "experiment", "trial", "trial_name",
                            "trialname", "trial name", "exp"],
        "mtr":             ["mtr", "maturity", "mat", "mat_dap", "maturity_dap",
                            "maturity dap"],
        "name":            ["name", "genotype", "variety", "cultivar", "line",
                            "entry_name", "genotypename"],
    }

    def _find(df: pd.DataFrame, target: str) -> Optional[str]:
        """Find *target* column in *df* using exact → case-insensitive → alias lookup."""
        cols_lower = {c.lower(): c for c in df.columns}
        tgt_lower  = target.strip().lower()

        # 1. Exact match
        if target in df.columns:
            return target
        # 2. Case-insensitive exact
        if tgt_lower in cols_lower:
            return cols_lower[tgt_lower]
        # 3. Alias table
        for canonical, aliases in _ALIASES.items():
            if tgt_lower in aliases or tgt_lower == canonical:
                for alias in aliases:
                    if alias in cols_lower:
                        found = cols_lower[alias]
                        _log(f"[ML]   Column '{target}' → matched as '{found}'")
                        return found
        # 4. Prefix/contains fallback for plot-like columns
        if "plot" in tgt_lower:
            matches = [c for c in df.columns if c.lower().startswith("plot")]
            if matches:
                _log(f"[ML]   Column '{target}' → prefix-matched as '{matches[0]}'")
                return matches[0]
        return None

    pid_f  = _find(df_f, plot_id_col)
    mtr_f  = _find(df_f, mtr_col)
    geno_f = _find(df_f, genotype_col)
    tri_f  = _find(df_f, trial_col)

    _log(f"[ML] Column mapping:")
    _log(f"[ML]   Plot ID       : '{plot_id_col}' → '{pid_f}'")
    _log(f"[ML]   Trial/Exp     : '{trial_col}' → '{tri_f}'")
    _log(f"[ML]   Genotype      : '{genotype_col}' → '{geno_f}'")
    _log(f"[ML]   MTR           : '{mtr_col}' → '{mtr_f}'")

    # Hard stop only for the two required columns
    missing = []
    if pid_f is None or pid_f not in df_f.columns:
        missing.append(
            f"Plot ID column '{plot_id_col}' — available: {list(df_f.columns)}")
    if mtr_f is None or mtr_f not in df_f.columns:
        missing.append(
            f"MTR column '{mtr_col}' — available: {list(df_f.columns)}")
    if missing:
        raise ValueError(
            "Cannot find required columns in field data:\n  " +
            "\n  ".join(missing) +
            "\n\nTip: update the 'Field Excel Columns' fields in the ML section "
            "of the app to match your file's actual column names.")

    df_f[mtr_f] = pd.to_numeric(df_f[mtr_f], errors="coerce")

    # ── PlotID normalisation helpers ──────────────────────────────────────────
    def _norm_pid(s: str) -> str:
        """Canonical PlotID: strip whitespace, remove trailing '.0'.
        '101.0' → '101',  ' 101 ' → '101',  'P101' → 'P101'
        """
        s = str(s).strip()
        try:
            return str(int(float(s)))   # '101.0' → '101'
        except (ValueError, TypeError):
            return s.lower()            # non-numeric → lowercase

    def _trial_num(s: str) -> str:
        """Extract first digit sequence for fuzzy trial matching.
        'SEVREC2501' → '2501',  'MRC-2021' → '2021'
        """
        m = re.search(r"\d+", str(s))
        return m.group(0) if m else str(s).strip().lower()

    # ── Build normalised key columns ───────────────────────────────────────────
    db = db_df.copy()
    db["_pid"]       = db["PlotID"].apply(_norm_pid)
    db["_trial_num"] = db["TrialName"].apply(_trial_num)

    df_f["_pid"] = df_f[pid_f].apply(_norm_pid)
    have_trial   = tri_f is not None and tri_f in df_f.columns
    df_f["_trial_num"] = (df_f[tri_f].apply(_trial_num) if have_trial
                          else pd.Series([""] * len(df_f), index=df_f.index))

    # ── Diagnostic log ────────────────────────────────────────────────────────
    _log(f"[ML] DB  PlotID samples   : {sorted(db['_pid'].unique())[:8]}")
    _log(f"[ML] DB  TrialName nums   : {sorted(db['_trial_num'].unique())[:8]}")
    _log(f"[ML] Field PlotID samples : {sorted(df_f['_pid'].unique())[:8]}")
    if have_trial:
        _log(f"[ML] Field trial nums     : {sorted(df_f['_trial_num'].unique())[:8]}")
    else:
        _log(f"[ML] Field trial col      : NOT FOUND — will match on PlotID only")

    # ── Extra columns to pull from field data ─────────────────────────────────
    extra = []
    if geno_f and geno_f in df_f.columns:
        extra.append(geno_f)

    def _try_merge(keys: list, label: str) -> pd.DataFrame:
        """Attempt a merge on the given key columns; return empty DF on failure."""
        slim_keys = list(dict.fromkeys(keys + [mtr_f] + extra))
        slim_keys = [c for c in slim_keys if c in df_f.columns]
        slim = df_f[slim_keys].drop_duplicates(subset=keys).copy()
        merged = db.merge(slim, on=keys, how="inner", suffixes=("", "_fld"))
        merged = merged.rename(columns={mtr_f: "MTR_field"})
        merged["MTR_field"] = pd.to_numeric(merged["MTR_field"], errors="coerce")
        merged = merged.dropna(subset=["MTR_field"])
        if len(merged):
            _log(f"[ML] Join strategy '{label}' succeeded: {len(merged)} rows matched.")
        return merged

    # ── Cascade of four fallback strategies ───────────────────────────────────
    # Strategy 1 (best): normalised PlotID  +  trial number
    # Strategy 2       : normalised PlotID  only  (no trial constraint)
    # Strategy 3       : raw-string PlotID  +  trial number
    # Strategy 4       : raw-string PlotID  only

    df_merged = pd.DataFrame()
    strategies = []
    if have_trial:
        strategies += [
            (["_pid", "_trial_num"], "norm-PlotID + TrialNum"),
            (["_pid"],               "norm-PlotID only"),
        ]
    strategies += [
        (["_pid"],               "norm-PlotID only (no trial col)"),
    ]

    for keys, label in strategies:
        df_merged = _try_merge(keys, label)
        if len(df_merged):
            if "PlotID only" in label:
                _log("[ML] ⚠  Matched on PlotID only (trial number was not used).\n"
                     "[ML]    This is fine for single-trial datasets but may produce\n"
                     "[ML]    wrong matches if multiple trials share the same PlotIDs.\n"
                     "[ML]    Add the 'Experiment Name' column to your Field Excel to\n"
                     "[ML]    enable more precise matching.")
            break

    # ── Post-merge cleanup ────────────────────────────────────────────────────
    if len(df_merged):
        if geno_f and geno_f in df_merged.columns:
            df_merged["Name"] = df_merged[geno_f].astype(str).str.strip()
        elif geno_f and geno_f + "_fld" in df_merged.columns:
            df_merged["Name"] = df_merged[geno_f + "_fld"].astype(str).str.strip()

        drop_extra = [c for c in df_merged.columns
                      if c.startswith("_") or (c.endswith("_fld") and c != "MTR_field")]
        df_merged = df_merged.drop(columns=drop_extra, errors="ignore")

        _log(f"[ML] Final: {len(df_merged)} matched rows across "
             f"{df_merged['TrialName'].nunique()} trial(s).")
        return df_merged

    # ── All strategies failed ─────────────────────────────────────────────────
    _log("[ML] ✘ All join strategies failed.  Detailed diagnosis:")
    _log(f"[ML]   DB PlotID   (all)  : {sorted(db['_pid'].unique())}")
    _log(f"[ML]   Field PlotID (all) : {sorted(df_f['_pid'].unique())}")
    _log(f"[ML]   DB TrialNames      : {sorted(db['_trial_num'].unique())}")
    if have_trial:
        _log(f"[ML]   Field TrialNames   : {sorted(df_f['_trial_num'].unique())}")
    _log("[ML]")
    _log("[ML]   Tips:")
    _log("[ML]   1. PlotID in field Excel must match PlotID in your vector layer")
    _log("[ML]   2. If PlotID column is named differently, update 'Plot ID col' in the app")
    _log("[ML]   3. Check that the correct field Excel is assigned to this dataset")
    raise ValueError(
        "0 rows matched after joining database with field data.\n\n"
        "All four match strategies failed.  Check the Run Log for\n"
        "the full list of PlotID values from both sides.\n\n"
        "Common causes:\n"
        "  • Plot ID column name is different (update 'Plot ID col' in the app)\n"
        "  • PlotID scheme differs (e.g. GIS OID vs field book number)\n"
        "  • Wrong field Excel assigned to this dataset")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — FEATURE MATRIX
# ═════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(df: pd.DataFrame,
                          log_fn=None
                          ) -> Tuple[pd.DataFrame, np.ndarray,
                                     np.ndarray, List[str], List[str]]:
    """Build (X, y, groups, cont_features, genotypes).

    X            — DataFrame with continuous features + optional 'Name' column
    y            — 1-D float array (MTR_field)
    groups       — 1-D str array (TrialName, for LOTO CV)
    cont_features — list of continuous column names in X
    genotypes    — sorted list of unique Name values
    """
    def _log(m):
        if log_fn:
            log_fn(m)

    df = df.copy()
    for col in _CONT_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    y      = df["MTR_field"].values.astype(float)
    groups = df["TrialName"].astype(str).values

    genotypes = (sorted(df["Name"].dropna().unique().tolist())
                 if "Name" in df.columns else [])

    cont_features = [c for c in _CONT_FEATURES if c in df.columns]
    feat_cols     = cont_features + (["Name"] if "Name" in df.columns and genotypes else [])

    X = df[feat_cols].copy()
    if "Name" in X.columns:
        X["Name"] = X["Name"].fillna("Unknown").astype(str)

    nan_cont = X[cont_features].isna().sum().sum()
    _log(f"[ML] Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    _log(f"[ML] NaN values in continuous features: {nan_cont}")
    _log(f"[ML] Trials: {sorted(set(groups))}")
    _log(f"[ML] Genotypes ({len(genotypes)}): "
         f"{genotypes[:8]}{'...' if len(genotypes) > 8 else ''}")
    return X, y, groups, cont_features, genotypes


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — MODEL BUILDING
# ═════════════════════════════════════════════════════════════════════════════

def _build_preprocessor(cont_features: List[str], has_name: bool) -> ColumnTransformer:
    """Median-impute + StandardScale continuous; OrdinalEncode Name."""
    transformers = [
        ("cont", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
        ]), cont_features),
    ]
    if has_name:
        transformers.append(
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1)),
            ]), ["Name"])
        )
    return ColumnTransformer(transformers, remainder="drop")


def train_models(X: pd.DataFrame,
                 y: np.ndarray,
                 cont_features: List[str],
                 log_fn=None) -> Dict[str, Any]:
    """
    Train Ridge (CV alpha tuning) + Random Forest [+ XGBoost].
    Returns a dict {model_name: fitted sklearn Pipeline}.
    """
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is not installed.\n"
            "Run:  pip install scikit-learn\nThen restart the app.")

    def _log(m):
        if log_fn:
            log_fn(m)

    has_name = "Name" in X.columns
    prep     = _build_preprocessor(cont_features, has_name)
    models   = {}

    # ── Ridge (cross-validated alpha) ─────────────────────────────────────────
    _log("[ML] Training Ridge (alpha search 0.01 – 10 000) ...")
    alphas     = np.logspace(-2, 4, 60)
    ridge_pipe = Pipeline([
        ("prep", deepcopy(prep)),
        ("reg",  RidgeCV(alphas=alphas, cv=5)),
    ])
    ridge_pipe.fit(X, y)
    _log(f"[ML]   Ridge best alpha = {ridge_pipe.named_steps['reg'].alpha_:.4f}")
    models["Ridge"] = ridge_pipe

    # ── Random Forest ─────────────────────────────────────────────────────────
    _log("[ML] Training Random Forest (300 trees) ...")
    rf_pipe = Pipeline([
        ("prep", deepcopy(prep)),
        ("reg",  RandomForestRegressor(
            n_estimators=300, max_features="sqrt",
            min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ])
    rf_pipe.fit(X, y)
    models["RandomForest"] = rf_pipe

    # ── XGBoost (optional) ────────────────────────────────────────────────────
    if _HAS_XGB:
        _log("[ML] Training XGBoost ...")
        xgb_pipe = Pipeline([
            ("prep", deepcopy(prep)),
            ("reg",  XGBRegressor(
                n_estimators=300, learning_rate=0.05,
                max_depth=4, subsample=0.8,
                colsample_bytree=0.8, random_state=42,
                verbosity=0, eval_metric="rmse")),
        ])
        xgb_pipe.fit(X, y)
        models["XGBoost"] = xgb_pipe

    # ── ElasticNet (alpha + l1_ratio CV) ─────────────────────────────────────
    _log("[ML] Training ElasticNet (alpha/l1_ratio search) ...")
    en_pipe = Pipeline([
        ("prep", deepcopy(prep)),
        ("reg",  ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
            alphas=np.logspace(-3, 3, 40),
            cv=5, max_iter=5000, random_state=42)),
    ])
    en_pipe.fit(X, y)
    _log(f"[ML]   ElasticNet  α={en_pipe.named_steps['reg'].alpha_:.4f}  "
         f"l1={en_pipe.named_steps['reg'].l1_ratio_:.2f}")
    models["ElasticNet"] = en_pipe

    # ── SVR (RBF kernel) ──────────────────────────────────────────────────────
    _log("[ML] Training SVR (RBF kernel) ...")
    svr_pipe = Pipeline([
        ("prep", deepcopy(prep)),
        ("reg",  SVR(kernel="rbf", C=10.0, epsilon=0.5, gamma="scale")),
    ])
    svr_pipe.fit(X, y)
    models["SVR"] = svr_pipe

    # ── Gradient Boosting ─────────────────────────────────────────────────────
    _log("[ML] Training GradientBoosting (300 trees) ...")
    gb_pipe = Pipeline([
        ("prep", deepcopy(prep)),
        ("reg",  GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, subsample=0.8,
            min_samples_leaf=2, random_state=42)),
    ])
    gb_pipe.fit(X, y)
    models["GradientBoosting"] = gb_pipe

    # ── Extra Trees ───────────────────────────────────────────────────────────
    _log("[ML] Training ExtraTrees (300 trees) ...")
    et_pipe = Pipeline([
        ("prep", deepcopy(prep)),
        ("reg",  ExtraTreesRegressor(
            n_estimators=300, max_features="sqrt",
            min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ])
    et_pipe.fit(X, y)
    models["ExtraTrees"] = et_pipe

    _log(f"[ML] Models ready: {list(models.keys())}")
    return models


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def cross_validate_loto(X: pd.DataFrame,
                         y: np.ndarray,
                         groups: np.ndarray,
                         models: Dict[str, Any],
                         cont_features: List[str],
                         log_fn=None) -> Dict[str, Dict]:
    """
    Leave-One-Trial-Out CV (LOTO) when ≥ 3 unique trials; k-fold otherwise.

    Returns
    -------
    cv_results : dict  {model_name: {y_true, y_pred, rmse, mae, r2, bias,
                                     fold_results, cv_name}}
    """
    def _log(m):
        if log_fn:
            log_fn(m)

    n_trials = len(np.unique(groups))
    if n_trials >= 3:
        splitter   = LeaveOneGroupOut()
        cv_name    = "Leave-One-Trial-Out"
        use_groups = True
        _log(f"[ML] CV strategy: LOTO ({n_trials} trials)")
    else:
        k          = min(5, len(y))
        splitter   = KFold(n_splits=k, shuffle=True, random_state=42)
        cv_name    = f"{k}-Fold"
        use_groups = False
        _log(f"[ML] Only {n_trials} trial(s) — using {k}-fold CV")

    cv_results: Dict[str, Dict] = {}

    for mname, pipe in models.items():
        _log(f"[ML] {mname} CV ...")
        y_pred_cv   = np.full(len(y), np.nan, dtype=float)
        fold_results = []

        split_iter = (splitter.split(X, y, groups)
                      if use_groups else splitter.split(X, y))

        for fi, (tr_idx, te_idx) in enumerate(split_iter):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y[tr_idx],      y[te_idx]

            p_fold = deepcopy(pipe)
            p_fold.fit(X_tr, y_tr)
            y_hat = p_fold.predict(X_te)
            y_pred_cv[te_idx] = y_hat

            f_rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))
            f_mae  = float(mean_absolute_error(y_te, y_hat))
            f_lbl  = (str(np.unique(groups[te_idx])[0])
                      if use_groups else f"Fold {fi + 1}")
            fold_results.append({"fold": f_lbl, "n": len(te_idx),
                                  "rmse": f_rmse, "mae": f_mae})
            _log(f"  {f_lbl:>20s}  n={len(te_idx):3d}  "
                 f"RMSE={f_rmse:.2f}  MAE={f_mae:.2f}")

        ok    = np.isfinite(y_pred_cv)
        rmse  = float(np.sqrt(mean_squared_error(y[ok], y_pred_cv[ok])))
        mae   = float(mean_absolute_error(y[ok], y_pred_cv[ok]))
        r2    = float(r2_score(y[ok], y_pred_cv[ok]))
        bias  = float(np.mean(y_pred_cv[ok] - y[ok]))
        _log(f"[ML] {mname} overall → "
             f"RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}  bias={bias:+.2f}")

        cv_results[mname] = {
            "cv_name":      cv_name,
            "y_true":       y[ok],
            "y_pred":       y_pred_cv[ok],
            "y_pred_full":  y_pred_cv,      # full-length; NaN where not predicted
            "rmse":         rmse,
            "mae":          mae,
            "r2":           r2,
            "bias":         bias,
            "fold_results": fold_results,
        }

    return cv_results


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 7b — MULTI-MODAL STACKING ENSEMBLE
#               Level-1: Ridge + RandomForest
#               Level-2: XGBoost meta-learner trained on OOF level-1 preds
# ═════════════════════════════════════════════════════════════════════════════

def _generate_oof_preds(X: pd.DataFrame,
                         y: np.ndarray,
                         groups: np.ndarray,
                         base_models: Dict[str, Any],
                         base_names: List[str],
                         log_fn=None) -> np.ndarray:
    """Generate Out-of-Fold (OOF) predictions for each named base model.

    Uses the same LOTO / k-fold strategy as cross_validate_loto so predictions
    are unbiased (no test-fold data in training).

    Returns
    -------
    array (n_samples, len(base_names)).  Residual NaN cells are filled with
    the column median before returning.
    """
    def _log(m):
        if log_fn:
            log_fn(m)

    n_trials   = len(np.unique(groups))
    use_groups = n_trials >= 3
    splitter   = (LeaveOneGroupOut() if use_groups
                  else KFold(n_splits=min(5, len(y)),
                             shuffle=True, random_state=42))

    meta_X = np.full((len(y), len(base_names)), np.nan, dtype=float)

    split_iter = (splitter.split(X, y, groups) if use_groups
                  else splitter.split(X, y))

    for tr_idx, te_idx in split_iter:
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr       = y[tr_idx]
        for bi, bname in enumerate(base_names):
            if bname not in base_models:
                continue
            p = deepcopy(base_models[bname])
            p.fit(X_tr, y_tr)
            meta_X[te_idx, bi] = p.predict(X_te)

    # Fill any NaN slots (e.g. from tiny folds) with column median
    for bi in range(meta_X.shape[1]):
        col      = meta_X[:, bi]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            meta_X[nan_mask, bi] = float(np.nanmedian(col))

    _log(f"[ML]   OOF matrix shape: {meta_X.shape}  "
         f"(NaN after fill: {np.isnan(meta_X).sum()})")
    return meta_X


def train_stacking_ensemble(X: pd.DataFrame,
                             y: np.ndarray,
                             groups: np.ndarray,
                             base_models: Dict[str, Any],
                             cont_features: List[str],
                             log_fn=None) -> Dict[str, Any]:
    """
    Train the final two-level stacking ensemble on the full dataset.

    Level-1 base learners : Ridge + RandomForest (already trained)
    Level-2 meta-learner  : XGBoost (or RF fallback) trained on OOF
                            level-1 predictions — no data leakage.

    Returns
    -------
    bundle : dict
        base_models, meta_model, meta_name, base_names, meta_cols
    Returns {} if no base models are available.
    """
    def _log(m):
        if log_fn:
            log_fn(m)

    _ALL_BASE_ORDER = ("Ridge", "ElasticNet", "SVR",
                       "RandomForest", "GradientBoosting", "ExtraTrees", "XGBoost")
    base_names = [k for k in _ALL_BASE_ORDER if k in base_models]
    if not base_names:
        _log("[ML] Stacking: no base models available — skipping.")
        return {}

    _log(f"[ML] Stacking: generating OOF predictions for {base_names} ...")
    meta_X_np = _generate_oof_preds(
        X, y, groups, base_models, base_names, log_fn)

    meta_cols = [f"{n}_pred" for n in base_names]

    # ── Build meta-feature matrix: original X  +  OOF predictions ────────────
    # Using all original features in addition to the base-model predictions
    # gives the meta-learner the full predictive signal rather than just the
    # compressed two-number summary from Ridge/RF.
    oof_df     = pd.DataFrame(meta_X_np, columns=meta_cols)
    meta_X_df  = pd.concat([X.reset_index(drop=True), oof_df], axis=1)
    meta_fcols = list(X.columns) + meta_cols   # column names meta model expects

    _log(f"[ML] Stacking meta-features: {len(X.columns)} original  +  "
         f"{len(meta_cols)} OOF preds  =  {len(meta_fcols)} total.")

    # Meta-learner: XGBoost preferred; RandomForest as graceful fallback
    if _HAS_XGB:
        _log("[ML] Stacking: training XGBoost meta-learner ...")
        meta_model = XGBRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=3, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            verbosity=0, eval_metric="rmse")
        meta_name = "XGBoost"
    else:
        _log("[ML] XGBoost not installed — using RandomForest as meta-learner.")
        meta_model = RandomForestRegressor(
            n_estimators=300, max_features="sqrt",
            min_samples_leaf=2, random_state=42, n_jobs=-1)
        meta_name = "RandomForest"

    meta_model.fit(meta_X_df, y)
    _log(f"[ML] Stacking meta-learner ({meta_name}) ready  "
         f"(trained on {len(meta_X_df)} samples × {len(meta_fcols)} features).")

    return {
        "base_models":    {bn: base_models[bn] for bn in base_names},
        "meta_model":     meta_model,
        "meta_name":      meta_name,
        "base_names":     base_names,
        "meta_cols":      meta_cols,
        "meta_fcols":     meta_fcols,   # full list of meta-model input columns
    }


def cross_validate_stacking(X: pd.DataFrame,
                              y: np.ndarray,
                              groups: np.ndarray,
                              base_models: Dict[str, Any],
                              cont_features: List[str],
                              log_fn=None) -> Dict:
    """
    Proper **nested** cross-validation for the stacking ensemble.

    For each outer test fold T
    ──────────────────────────
    1. Train Ridge + RF on the outer train set.
    2. Run an inner LOTO / k-fold on the outer train set to generate
       inner OOF meta-features (meta_X_train).
    3. Train XGBoost meta-learner on meta_X_train.
    4. Apply outer Ridge + RF to test fold T → meta_X_test.
    5. Apply trained meta to meta_X_test → final stacking prediction.

    The meta-learner never sees the outer test fold during training —
    estimates are unbiased.

    Returns
    -------
    dict with y_true, y_pred (ok-filtered), y_pred_full (full-length,
    NaN where not predicted), rmse, mae, r2, bias, fold_results,
    cv_name, base_names, meta_name
    """
    def _log(m):
        if log_fn:
            log_fn(m)

    _ALL_BASE_ORDER = ("Ridge", "ElasticNet", "SVR",
                       "RandomForest", "GradientBoosting", "ExtraTrees", "XGBoost")
    base_names = [k for k in _ALL_BASE_ORDER if k in base_models]
    if not base_names:
        _log("[ML] Stacking CV: no base models — skipped.")
        return {}

    n_trials   = len(np.unique(groups))
    use_groups = n_trials >= 3
    outer_sp   = (LeaveOneGroupOut() if use_groups
                  else KFold(n_splits=min(5, len(y)),
                             shuffle=True, random_state=42))
    cv_name    = "LOTO-Stacking" if use_groups else f"{min(5,len(y))}-Fold-Stacking"
    meta_label = "XGBoost" if _HAS_XGB else "RandomForest"
    meta_cols  = [f"{n}_pred" for n in base_names]

    _log(f"[ML] Stacking CV ({'+'.join(base_names)} → {meta_label}) ...")

    y_pred_stack = np.full(len(y), np.nan, dtype=float)
    fold_results: List[Dict] = []

    outer_iter = (outer_sp.split(X, y, groups) if use_groups
                  else outer_sp.split(X, y))

    for fi, (tr_idx, te_idx) in enumerate(outer_iter):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        g_tr       = groups[tr_idx]

        # ── Step 1 : train base models on outer train set ─────────────────
        outer_base: Dict[str, Any] = {}
        for bname in base_names:
            p = deepcopy(base_models[bname])
            p.fit(X_tr, y_tr)
            outer_base[bname] = p

        # ── Step 2 : inner OOF on the outer train set ─────────────────────
        n_inner_tri = len(np.unique(g_tr))
        if use_groups and n_inner_tri >= 2:
            inner_sp   = LeaveOneGroupOut()
            inner_iter = list(inner_sp.split(X_tr, y_tr, g_tr))
        else:
            k_in       = max(2, min(5, len(y_tr)))
            inner_sp   = KFold(n_splits=k_in, shuffle=True, random_state=42)
            inner_iter = list(inner_sp.split(X_tr, y_tr))

        inner_meta = np.full((len(y_tr), len(base_names)), np.nan, dtype=float)
        for ti2, te2 in inner_iter:
            X_i_tr, X_i_te = X_tr.iloc[ti2], X_tr.iloc[te2]
            y_i_tr         = y_tr[ti2]
            for bi, bname in enumerate(base_names):
                p = deepcopy(base_models[bname])
                p.fit(X_i_tr, y_i_tr)
                inner_meta[te2, bi] = p.predict(X_i_te)

        for bi in range(inner_meta.shape[1]):
            col   = inner_meta[:, bi]
            nan_m = np.isnan(col)
            if nan_m.any():
                inner_meta[nan_m, bi] = float(np.nanmedian(col))

        # Meta-train: original X_tr  +  inner OOF preds
        oof_tr_df  = pd.DataFrame(inner_meta, columns=meta_cols)
        meta_tr_df = pd.concat(
            [X_tr.reset_index(drop=True), oof_tr_df], axis=1)

        # ── Step 3 : train meta-learner on inner OOF ─────────────────────
        if _HAS_XGB:
            meta_model = XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=3,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                verbosity=0, eval_metric="rmse")
        else:
            meta_model = RandomForestRegressor(
                n_estimators=200, max_features="sqrt",
                min_samples_leaf=2, random_state=42, n_jobs=-1)

        meta_model.fit(meta_tr_df, y_tr)

        # ── Steps 4 & 5 : apply to outer test fold ────────────────────────
        # Meta-test: original X_te  +  outer base-model predictions
        meta_te_np = np.column_stack([
            outer_base[bn].predict(X_te) for bn in base_names
        ])
        oof_te_df  = pd.DataFrame(meta_te_np, columns=meta_cols)
        meta_te_df = pd.concat(
            [X_te.reset_index(drop=True), oof_te_df], axis=1)

        y_hat             = meta_model.predict(meta_te_df)
        y_pred_stack[te_idx] = y_hat

        f_rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))
        f_mae  = float(mean_absolute_error(y_te, y_hat))
        f_lbl  = (str(np.unique(groups[te_idx])[0]) if use_groups
                  else f"Fold {fi + 1}")
        fold_results.append({"fold": f_lbl, "n": len(te_idx),
                              "rmse": f_rmse, "mae": f_mae})
        _log(f"  {f_lbl:>20s}  n={len(te_idx):3d}  "
             f"RMSE={f_rmse:.2f}  MAE={f_mae:.2f}")

    ok   = np.isfinite(y_pred_stack)
    rmse = float(np.sqrt(mean_squared_error(y[ok], y_pred_stack[ok])))
    mae  = float(mean_absolute_error(y[ok], y_pred_stack[ok]))
    r2   = float(r2_score(y[ok], y_pred_stack[ok]))
    bias = float(np.mean(y_pred_stack[ok] - y[ok]))

    _log(f"[ML] Stacking ({'+'.join(base_names)}→{meta_label}) overall → "
         f"RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}  bias={bias:+.2f}")

    return {
        "cv_name":       cv_name,
        "y_true":        y[ok],
        "y_pred":        y_pred_stack[ok],
        "y_pred_full":   y_pred_stack,      # full-length; NaN where not predicted
        "rmse":          rmse,
        "mae":           mae,
        "r2":            r2,
        "bias":          bias,
        "fold_results":  fold_results,
        "base_names":    base_names,
        "meta_name":     meta_label,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — RIDGE FORMULA EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def get_ridge_formula(ridge_pipe,
                       cont_features: List[str],
                       genotypes: List[str],
                       log_fn=None) -> Dict[str, str]:
    """
    Extract per-genotype formula from a fitted Ridge pipeline.

    For each genotype:
      MTR ≈ (global_intercept + genotype_offset)
             + β₁ × feat_1  + β₂ × feat_2  + ...

    The genotype_offset comes from the OrdinalEncoder coefficient (linear).
    Top-10 continuous features by |coefficient| are shown.
    """
    def _log(m):
        if log_fn:
            log_fn(m)

    try:
        reg       = ridge_pipe.named_steps["reg"]
        coef      = reg.coef_
        intercept = float(reg.intercept_)

        n_cont      = len(cont_features)
        cont_coef   = coef[:n_cont]
        has_name    = len(coef) > n_cont

        # Top-10 continuous terms by absolute coefficient
        order  = np.argsort(np.abs(cont_coef))[::-1][:10]
        terms  = [f"  {cont_coef[i]:+.5f} × {cont_features[i]}"
                  for i in order]

        # OrdinalEncoder assigns 0, 1, 2 … in alphabetical order
        name_coef = float(coef[n_cont]) if has_name else 0.0

        # Try to retrieve the actual encoder categories
        try:
            enc_cats = list(
                ridge_pipe.named_steps["prep"]
                .named_transformers_["cat"]
                .named_steps["encode"]
                .categories_[0]
            )
        except Exception:
            enc_cats = sorted(genotypes)

        formulas: Dict[str, str] = {}
        geno_list = genotypes if genotypes else ["(all genotypes)"]
        for gname in geno_list:
            rank   = enc_cats.index(gname) if gname in enc_cats else 0
            offset = name_coef * rank
            adj_ic = intercept + offset
            sign   = "+" if offset >= 0 else "−"
            f_str  = (
                f"Genotype : {gname}\n"
                f"MTR  =  {adj_ic:.2f}   "
                f"[global {intercept:.2f} {sign} {abs(offset):.2f} genotype offset]\n"
                + "\n".join(terms)
            )
            formulas[gname] = f_str

        if not formulas:
            global_str = (
                f"MTR  =  {intercept:.2f}\n" + "\n".join(terms)
            )
            formulas["(global)"] = global_str

        return formulas

    except Exception as ex:
        _log(f"[ML] Warning — Ridge formula extraction failed: {ex}")
        return {"(global)": "Formula extraction failed."}


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — FLIGHT PLANNING ADVISOR
# ═════════════════════════════════════════════════════════════════════════════

def simulate_flight_plans(df: pd.DataFrame,
                           y: np.ndarray,
                           models: Dict[str, Any],
                           cont_features: List[str],
                           stacking_bundle: Optional[Dict] = None,
                           log_fn=None) -> Dict[str, Dict]:
    """
    For each model (including optional stacking ensemble), simulate accuracy
    when using only the FIRST N flights.

    For each N (2 … max_N):
      - Truncate each plot's TimeSeries to first N rows
      - Re-compute ExG slope, GCC/RCC rates, HMI crossing, flight span
      - Predict MTR with the trained model
      - Measure RMSE against field MTR

    Returns {model_name: {n_range, rmse_by_n, best_n, best_rmse, rec_interval}}.
    """
    def _log(m):
        if log_fn:
            log_fn(m)

    if "n_flights" not in df.columns:
        _log("[ML] n_flights not found — skipping flight planning simulation.")
        return {}

    max_n = int(df["n_flights"].max())
    if max_n < 3:
        _log(f"[ML] max_n={max_n} — need ≥ 3 for simulation.")
        return {}

    # Build a dict of TimeSeries CSV paths from stored output_root values
    ts_paths: Dict[str, str] = {}
    for _, row in df.iterrows():
        pid = str(row["PlotID"])
        root = str(row.get("output_root", ""))
        if root:
            p = os.path.join(root, "TimeSeries", f"ts_{pid}.csv")
            if os.path.exists(p):
                ts_paths[pid] = p

    if not ts_paths:
        _log("[ML] No TimeSeries CSV files found — skipping flight planning.")
        return {}

    n_range = list(range(2, max_n + 1))
    _log(f"[ML] Flight planning: testing N = 2 … {max_n}")
    results: Dict[str, Dict] = {}

    for mname, pipe in models.items():
        rmse_by_n: List[float] = []
        dap_axis:  List[float] = []   # mean last_flight_dap for each N

        for N in n_range:
            sim_rows:        List[Dict]  = []
            y_sim:           List[float] = []
            last_dap_vals_n: List[float] = []   # last-flight DAP per plot at this N

            for _, row in df.iterrows():
                pid = str(row["PlotID"])
                mtr = row.get("MTR_field", np.nan)
                if not np.isfinite(mtr) or pid not in ts_paths:
                    continue

                try:
                    ts = pd.read_csv(ts_paths[pid]).head(N)
                    ts.columns = [c.strip() for c in ts.columns]
                    dap_c   = next((c for c in ts.columns if c.lower() == "dap"), ts.columns[0])
                    dap_arr = pd.to_numeric(ts[dap_c], errors="coerce").values
                    ok_dap  = np.isfinite(dap_arr)

                    # Start with stored features, then override flight-dependent ones
                    feat = {c: row.get(c, np.nan) for c in cont_features}
                    feat["n_flights"] = float(N)

                    if ok_dap.sum() > 0:
                        fd = dap_arr[ok_dap]
                        feat["first_flight_dap"] = float(fd[0])
                        feat["last_flight_dap"]  = float(fd[-1])
                        feat["flight_span_days"] = float(fd[-1] - fd[0]) if len(fd) > 1 else 0.0
                        last_dap_vals_n.append(float(fd[-1]))

                    exg_c = next((c for c in ts.columns if c.upper() in ("EXGR","EXG")), None)
                    if exg_c:
                        ep, es = _exg_features(dap_arr,
                                               pd.to_numeric(ts[exg_c], errors="coerce").values)
                        feat["exg_peak_dap"] = ep if ep is not None else np.nan
                        feat["exg_slope"]    = es if es is not None else np.nan

                    hmi_c = next(
                        (c for c in ts.columns if c.upper() in ("HMI_MASKED","HMI MASKED")),
                        None)
                    if hmi_c:
                        cross = _safe_pchip_crossing(
                            dap_arr,
                            pd.to_numeric(ts[hmi_c], errors="coerce").values,
                            HMI_THR, "increase")
                        feat["hmi_crossing_dap"] = cross if cross is not None else np.nan

                    if "Name" in df.columns:
                        feat["Name"] = str(row.get("Name", "Unknown"))

                    sim_rows.append(feat)
                    y_sim.append(float(mtr))
                except Exception:
                    continue

            # Record mean last-flight DAP for this N (independent of model)
            mean_last_dap = (float(np.nanmean(last_dap_vals_n))
                             if last_dap_vals_n else float(N))
            dap_axis.append(mean_last_dap)

            if len(sim_rows) < 3:
                rmse_by_n.append(np.nan)
                continue

            X_sim = pd.DataFrame(sim_rows)
            for c in cont_features:
                if c not in X_sim.columns:
                    X_sim[c] = np.nan
            if "Name" not in X_sim.columns and "Name" in X:
                X_sim["Name"] = "Unknown"

            try:
                y_hat = pipe.predict(X_sim)
                rmse  = float(np.sqrt(mean_squared_error(np.array(y_sim), y_hat)))
            except Exception:
                rmse = np.nan
            rmse_by_n.append(rmse)

        arr        = np.array(rmse_by_n, dtype=float)
        finite_ok  = np.isfinite(arr)
        best_idx   = int(np.nanargmin(arr)) if finite_ok.sum() > 0 else len(arr) - 1
        best_n     = n_range[best_idx]
        best_rmse  = float(arr[best_idx]) if finite_ok.sum() > 0 else np.nan

        # Start DAP = mean of the actual first-flight DAP across all plots
        mean_first = float(df["first_flight_dap"].mean())
        rec_start  = int(round(mean_first)) if np.isfinite(mean_first) else 40

        # DAP of the optimal stopping flight
        best_dap_val = (float(dap_axis[best_idx])
                        if best_idx < len(dap_axis) and np.isfinite(dap_axis[best_idx])
                        else np.nan)

        # Interval = average spacing between consecutive flights up to the optimal stop
        # Formula: (last_flight_DAP − first_flight_DAP) / (N − 1)
        # This guarantees:  rec_start + (best_n−1) × rec_int  ≈  best_dap
        if np.isfinite(best_dap_val) and best_n > 1:
            rec_int = max(1, int(round((best_dap_val - rec_start) / max(best_n - 1, 1))))
        elif best_n == 1:
            rec_int = 0      # single-flight scenario: no interval to report
        else:
            # Fallback: full-season span if DAP info is missing
            mean_span = float(df["flight_span_days"].mean())
            rec_int   = (max(1, int(round(mean_span / max(best_n - 1, 1))))
                         if np.isfinite(mean_span) else 7)

        # ── Optimal flying window: top-K lowest-RMSE scenarios ──────────────
        # The pilot does not need to hit a single exact DAP — any last-flight
        # DAP within the window gives near-optimal accuracy.
        valid_idx = np.where(finite_ok)[0]
        if len(valid_idx) > 0:
            # Indices sorted by RMSE ascending (lowest = best)
            top_k_idx = valid_idx[np.argsort(arr[valid_idx])][:FLIGHT_TOP_K]
            window_daps = sorted(
                float(dap_axis[i]) for i in top_k_idx
                if i < len(dap_axis) and np.isfinite(dap_axis[i])
            )
            window_start_dap = window_daps[0]  if window_daps else np.nan
            window_end_dap   = window_daps[-1] if window_daps else np.nan
        else:
            window_daps, window_start_dap, window_end_dap = [], np.nan, np.nan

        _dap_str = f"DAP {best_dap_val:.0f}" if np.isfinite(best_dap_val) else "unknown"
        _win_str = (f"DAP {window_start_dap:.0f}–{window_end_dap:.0f}"
                    if np.isfinite(window_start_dap) else "unknown")
        _log(f"[ML] {mname}: best N={best_n} (RMSE={best_rmse:.1f} d)  "
             f"start DAP≈{rec_start}  interval≈{rec_int} d  "
             f"last flight≈{_dap_str}  window≈{_win_str}")

        results[mname] = {
            "n_range":          n_range,
            "dap_axis":         dap_axis,
            "rmse_by_n":        rmse_by_n,
            "best_n":           best_n,
            "best_rmse":        best_rmse,
            "rec_interval":     rec_int,
            "rec_start_dap":    rec_start,
            # Optimal window
            "window_start_dap": window_start_dap,
            "window_end_dap":   window_end_dap,
            "window_daps":      window_daps,
            "top_k":            FLIGHT_TOP_K,
        }

    # ── Stack Model simulation ─────────────────────────────────────────────────
    if (stacking_bundle and
            stacking_bundle.get("base_models") and
            stacking_bundle.get("meta_model")):
        _log("[ML] Flight planning: simulating Stack Model ...")
        sb        = stacking_bundle
        bn_list   = sb["base_names"]
        meta_cols = sb["meta_cols"]
        meta_fcols = sb.get("meta_fcols")

        rmse_by_n_s: List[float] = []
        dap_axis_s:  List[float] = []

        for N in n_range:
            sim_rows_s:     List[Dict]  = []
            y_sim_s:        List[float] = []
            last_dap_vals_s: List[float] = []

            for _, row in df.iterrows():
                pid = str(row["PlotID"])
                mtr = row.get("MTR_field", np.nan)
                if not np.isfinite(mtr) or pid not in ts_paths:
                    continue
                try:
                    ts = pd.read_csv(ts_paths[pid]).head(N)
                    ts.columns = [c.strip() for c in ts.columns]
                    dap_c   = next((c for c in ts.columns if c.lower() == "dap"), ts.columns[0])
                    dap_arr = pd.to_numeric(ts[dap_c], errors="coerce").values
                    ok_dap  = np.isfinite(dap_arr)

                    feat = {c: row.get(c, np.nan) for c in cont_features}
                    feat["n_flights"] = float(N)

                    if ok_dap.sum() > 0:
                        fd = dap_arr[ok_dap]
                        feat["first_flight_dap"] = float(fd[0])
                        feat["last_flight_dap"]  = float(fd[-1])
                        feat["flight_span_days"] = float(fd[-1] - fd[0]) if len(fd) > 1 else 0.0
                        last_dap_vals_s.append(float(fd[-1]))

                    exg_c = next((c for c in ts.columns if c.upper() in ("EXGR","EXG")), None)
                    if exg_c:
                        ep, es = _exg_features(
                            dap_arr, pd.to_numeric(ts[exg_c], errors="coerce").values)
                        feat["exg_peak_dap"] = ep if ep is not None else np.nan
                        feat["exg_slope"]    = es if es is not None else np.nan

                    hmi_c = next(
                        (c for c in ts.columns if c.upper() in ("HMI_MASKED","HMI MASKED")), None)
                    if hmi_c:
                        cross = _safe_pchip_crossing(
                            dap_arr,
                            pd.to_numeric(ts[hmi_c], errors="coerce").values,
                            HMI_THR, "increase")
                        feat["hmi_crossing_dap"] = cross if cross is not None else np.nan

                    if "Name" in df.columns:
                        feat["Name"] = str(row.get("Name", "Unknown"))

                    sim_rows_s.append(feat)
                    y_sim_s.append(float(mtr))
                except Exception:
                    continue

            mean_last_dap_s = (float(np.nanmean(last_dap_vals_s))
                               if last_dap_vals_s else float(N))
            dap_axis_s.append(mean_last_dap_s)

            if len(sim_rows_s) < 3:
                rmse_by_n_s.append(np.nan)
                continue

            X_sim_s = pd.DataFrame(sim_rows_s)
            for c in cont_features:
                if c not in X_sim_s.columns:
                    X_sim_s[c] = np.nan
            if "Name" not in X_sim_s.columns and "Name" in df.columns:
                X_sim_s["Name"] = "Unknown"

            try:
                # Build meta-feature matrix: original X + base-model predictions
                base_preds_np = np.column_stack([
                    sb["base_models"][bn].predict(X_sim_s)
                    for bn in bn_list
                    if bn in sb["base_models"]
                ])
                oof_df_s  = pd.DataFrame(base_preds_np, columns=meta_cols)
                meta_df_s = pd.concat(
                    [X_sim_s.reset_index(drop=True), oof_df_s], axis=1)
                if meta_fcols is not None:
                    meta_df_s = meta_df_s.reindex(columns=meta_fcols, fill_value=0)
                y_hat_s = sb["meta_model"].predict(meta_df_s)
                rmse_s  = float(np.sqrt(mean_squared_error(np.array(y_sim_s), y_hat_s)))
            except Exception:
                rmse_s = np.nan
            rmse_by_n_s.append(rmse_s)

        arr_s       = np.array(rmse_by_n_s, dtype=float)
        fin_ok_s    = np.isfinite(arr_s)
        best_idx_s  = int(np.nanargmin(arr_s)) if fin_ok_s.sum() > 0 else len(arr_s) - 1
        best_n_s    = n_range[best_idx_s]
        best_rmse_s = float(arr_s[best_idx_s]) if fin_ok_s.sum() > 0 else np.nan

        mean_first_s = float(df["first_flight_dap"].mean())
        rec_start_s  = int(round(mean_first_s)) if np.isfinite(mean_first_s) else 40
        best_dap_s   = (float(dap_axis_s[best_idx_s])
                        if best_idx_s < len(dap_axis_s) and np.isfinite(dap_axis_s[best_idx_s])
                        else np.nan)

        if np.isfinite(best_dap_s) and best_n_s > 1:
            rec_int_s = max(1, int(round((best_dap_s - rec_start_s) / max(best_n_s - 1, 1))))
        elif best_n_s == 1:
            rec_int_s = 0
        else:
            mean_span_s = float(df["flight_span_days"].mean())
            rec_int_s   = (max(1, int(round(mean_span_s / max(best_n_s - 1, 1))))
                           if np.isfinite(mean_span_s) else 7)

        valid_idx_s = np.where(fin_ok_s)[0]
        if len(valid_idx_s) > 0:
            top_k_idx_s  = valid_idx_s[np.argsort(arr_s[valid_idx_s])][:FLIGHT_TOP_K]
            window_daps_s = sorted(
                float(dap_axis_s[i]) for i in top_k_idx_s
                if i < len(dap_axis_s) and np.isfinite(dap_axis_s[i])
            )
            win_start_s = window_daps_s[0]  if window_daps_s else np.nan
            win_end_s   = window_daps_s[-1] if window_daps_s else np.nan
        else:
            window_daps_s, win_start_s, win_end_s = [], np.nan, np.nan

        _log(f"[ML] Stack Model: best N={best_n_s} (RMSE={best_rmse_s:.1f} d)  "
             f"start DAP≈{rec_start_s}  interval≈{rec_int_s} d  "
             f"last flight≈DAP {best_dap_s:.0f}" if np.isfinite(best_dap_s)
             else f"[ML] Stack Model: best N={best_n_s} (RMSE={best_rmse_s:.1f} d)")

        results["Stack Model"] = {
            "n_range":          n_range,
            "dap_axis":         dap_axis_s,
            "rmse_by_n":        rmse_by_n_s,
            "best_n":           best_n_s,
            "best_rmse":        best_rmse_s,
            "rec_interval":     rec_int_s,
            "rec_start_dap":    rec_start_s,
            "window_start_dap": win_start_s,
            "window_end_dap":   win_end_s,
            "window_daps":      window_daps_s,
            "top_k":            FLIGHT_TOP_K,
        }

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — PREDICT NEW TRIAL
# ═════════════════════════════════════════════════════════════════════════════

def predict_new_trial(output_root: str,
                       model_path: str,
                       trial_name: str = "New_Trial",
                       sowing_date=None,
                       log_fn=None) -> pd.DataFrame:
    """
    Apply a saved model bundle to a new pipeline output folder.

    Returns DataFrame with columns:
      PlotID, Predicted_MTR_DAP, [Predicted_MTR_Date if sowing_date given]
    """
    def _log(m):
        if log_fn:
            log_fn(m)

    _log(f"[ML] Predicting: {output_root}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as fh:
        bundle = pickle.load(fh)

    df_feat       = extract_features_from_output(output_root, trial_name, log_fn=log_fn)
    cont_features = bundle.get("cont_features", _CONT_FEATURES)
    overall_best  = bundle.get("best_model_name", "RandomForest")
    # "Stack Model" is not in bundle["models"]; fall back to best base model
    if overall_best == "Stack Model" or overall_best not in bundle["models"]:
        best_name = bundle.get("best_base_model",
                               next(iter(bundle["models"]), "RandomForest"))
    else:
        best_name = overall_best
    pipe = bundle["models"][best_name]

    feat_cols = cont_features + (["Name"] if "Name" in df_feat.columns else [])
    X_new = df_feat[[c for c in feat_cols if c in df_feat.columns]].copy()
    if "Name" in X_new.columns:
        X_new["Name"] = X_new["Name"].fillna("Unknown").astype(str)

    y_pred = pipe.predict(X_new)

    result = pd.DataFrame({
        "PlotID":            df_feat["PlotID"].values,
        "Predicted_MTR_DAP": np.round(y_pred, 1),
    })

    if sowing_date is not None:
        result["Predicted_MTR_Date"] = result["Predicted_MTR_DAP"].apply(
            lambda d: ((sowing_date + timedelta(days=int(d))).strftime("%Y-%m-%d")
                       if np.isfinite(d) else "N/A"))

    _log(f"[ML] Predicted {len(result)} plots using {best_name}"
         f"{' (via Stack Model)' if overall_best == 'Stack Model' else ''}. "
         f"Mean DAP = {y_pred.mean():.1f} ± {y_pred.std():.1f} days")

    # ── Stacking prediction (if model bundle includes the stacking layer) ─────
    stacking_b = bundle.get("stacking_bundle")
    if stacking_b and stacking_b.get("base_models") and stacking_b.get("meta_model"):
        try:
            _base_preds = np.column_stack([
                stacking_b["base_models"][_bn].predict(X_new)
                for _bn in stacking_b["base_names"]
                if _bn in stacking_b["base_models"]
            ])
            _oof_df  = pd.DataFrame(_base_preds, columns=stacking_b["meta_cols"])
            # meta input = original features + base-model predictions
            _meta_df = pd.concat([X_new.reset_index(drop=True), _oof_df], axis=1)
            # Ensure column order matches what the meta-model was trained on
            _meta_fcols = stacking_b.get("meta_fcols", list(_meta_df.columns))
            _meta_df = _meta_df.reindex(columns=_meta_fcols, fill_value=0)
            y_stack = stacking_b["meta_model"].predict(_meta_df)
            result["Predicted_Stacking_DAP"] = np.round(y_stack, 1)
            if sowing_date is not None:
                result["Predicted_Stacking_Date"] = (
                    result["Predicted_Stacking_DAP"].apply(
                        lambda d: ((sowing_date + timedelta(days=int(d)))
                                   .strftime("%Y-%m-%d")
                                   if np.isfinite(d) else "N/A")))
            _log(f"[ML] Stacking prediction: "
                 f"Mean DAP = {y_stack.mean():.1f} ± {y_stack.std():.1f} days  "
                 f"(meta: {stacking_b.get('meta_name', 'XGBoost')})")
        except Exception as _ex:
            _log(f"[ML] Warning — stacking prediction failed: {_ex}")

    return result


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def _light_fig(nrows=1, ncols=1, figsize=(8, 6), **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    fig.patch.set_facecolor(BG_C)
    return fig, axes


def plot_cv_report(cv_results: Dict[str, Dict], out_path: str):
    """Scatter (predicted vs actual) + per-fold RMSE bar chart, one column per model."""
    n = len(cv_results)
    fig, axes = _light_fig(2, n, figsize=(6 * n, 10),
                           squeeze=False, dpi=DPI)

    for ci, (mname, res) in enumerate(cv_results.items()):
        yt, yp = res["y_true"], res["y_pred"]

        # ── Scatter ────────────────────────────────────────────────────────────
        ax = axes[0, ci];  _light_ax(ax)
        lim = [min(yt.min(), yp.min()) - 5, max(yt.max(), yp.max()) + 5]
        ax.plot(lim, lim, "--", color="#64748b", lw=1.4, zorder=1)
        sc = ax.scatter(yt, yp, c=yp - yt,
                        cmap="RdYlGn_r", vmin=-15, vmax=15, s=36, alpha=0.85, zorder=2)
        plt.colorbar(sc, ax=ax, label="Residual (days)", fraction=0.04)
        ax.set_xlim(lim);  ax.set_ylim(lim)
        ax.set_xlabel("Actual MTR (DAP)")
        ax.set_ylabel("Predicted MTR (DAP)")
        ax.set_title(
            f"{mname}\nRMSE={res['rmse']:.1f} d   R²={res['r2']:.3f}   "
            f"bias={res['bias']:+.1f} d   [{res['cv_name']}]",
            fontsize=TITLE_FONT)

        # ── Per-fold bar ───────────────────────────────────────────────────────
        ax2 = axes[1, ci];  _light_ax(ax2)
        folds  = res["fold_results"]
        labels = [r["fold"] for r in folds]
        rmses  = [r["rmse"]  for r in folds]
        ns_    = [r["n"]     for r in folds]
        bar_c  = cm.RdYlGn_r(np.linspace(0, 1, len(folds)))
        bars   = ax2.barh(labels, rmses, color=bar_c,
                          edgecolor=BORDER_C, linewidth=0.5)
        for bar, n_ in zip(bars, ns_):
            ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                     f"n={n_}", va="center", ha="left", color=LABEL_C, fontsize=7)
        ax2.axvline(res["rmse"], color="#f59e0b", lw=1.5, ls="--",
                    label=f"Overall {res['rmse']:.1f} d")
        ax2.set_xlabel("RMSE (days)")
        ax2.set_title(f"{mname}  —  per-fold RMSE", fontsize=TITLE_FONT - 1)
        ax2.legend(fontsize=8, labelcolor=TEXT_C, facecolor=PANEL_C, edgecolor=BORDER_C)

    fig.suptitle("ML Cross-Validation Report  —  Maturity DAP Prediction",
                 color=TEXT_C, fontsize=TITLE_FONT + 2, y=1.01)
    plt.tight_layout(pad=2)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG_C)
    plt.close(fig)


def plot_feature_importance(models: Dict[str, Any],
                             cont_features: List[str],
                             cv_results: Dict[str, Dict],
                             out_path: str):
    """Feature importance / coefficient bars for all interpretable models.

    • Ridge / ElasticNet  : |coefficient|, green = positive, red = negative
    • Tree-based models   : Gini importance (top-20 features)
    • SVR                 : skipped (no stable per-feature importance)
    """
    # Preserve canonical order; skip SVR (not interpretable without permutation)
    _PLOT_ORDER = ("Ridge", "ElasticNet",
                   "RandomForest", "GradientBoosting", "ExtraTrees", "XGBoost")
    plot_models = {k: models[k] for k in _PLOT_ORDER if k in models}
    n = max(len(plot_models), 1)

    # Use 2-row layout if more than 4 models to keep each panel readable
    if n <= 4:
        n_rows, n_cols = 1, n
    else:
        n_cols = 4
        n_rows = (n + n_cols - 1) // n_cols

    fig, axes = _light_fig(n_rows, n_cols,
                           figsize=(7 * n_cols, 8 * n_rows),
                           squeeze=False, dpi=DPI)

    for ci, (mname, pipe) in enumerate(plot_models.items()):
        ri, col_i = divmod(ci, n_cols)
        ax = axes[ri, col_i];  _light_ax(ax)
        try:
            reg = pipe.named_steps["reg"]
            if mname in ("Ridge", "ElasticNet"):
                raw_coef = reg.coef_[:len(cont_features)]
                imp      = np.abs(raw_coef)
                sign_c   = np.where(raw_coef >= 0, "#198754", "#dc3545")
                ylabel   = "|Coefficient|"
            elif hasattr(reg, "feature_importances_"):
                imp    = reg.feature_importances_[:len(cont_features)]
                sign_c = cm.RdYlGn(imp / (imp.max() + 1e-9))
                ylabel = "Importance"
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        color=TEXT_C, transform=ax.transAxes)
                ax.set_title(mname, fontsize=TITLE_FONT - 1)
                continue

            top_n   = min(20, len(cont_features))
            top_idx = np.argsort(imp)[-top_n:]
            top_lbl = [cont_features[i] for i in top_idx]
            top_val = imp[top_idx]
            top_col = (sign_c[top_idx] if isinstance(sign_c, np.ndarray)
                       else [sign_c[i] for i in top_idx])

            ax.barh(top_lbl, top_val, color=top_col,
                    edgecolor=BORDER_C, linewidth=0.5)
            rmse_tag = (f"RMSE={cv_results[mname]['rmse']:.1f} d  "
                        f"R²={cv_results[mname]['r2']:.3f}"
                        if mname in cv_results else "")
            ax.set_xlabel(f"{ylabel}  [{rmse_tag}]", fontsize=FONT - 1)
            ax.set_title(f"{mname}  —  Top-{top_n} Features",
                         fontsize=TITLE_FONT - 1)
        except Exception as ex:
            ax.text(0.5, 0.5, f"Error:\n{ex}", ha="center", va="center",
                    color="#ef4444", transform=ax.transAxes, fontsize=9)

    # Hide unused tiles in last row
    for ci in range(len(plot_models), n_rows * n_cols):
        ri, col_i = divmod(ci, n_cols)
        axes[ri, col_i].set_visible(False)

    fig.suptitle("Feature Importance  &  Model Coefficients",
                 color=TEXT_C, fontsize=TITLE_FONT + 1, y=1.01)
    plt.tight_layout(pad=2)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG_C)
    plt.close(fig)


def plot_field_vs_predicted(cv_results: Dict[str, Dict],
                             df: pd.DataFrame,
                             out_path: str):
    """Scatter + residual histogram per model, with genotype colouring."""
    n = len(cv_results)
    fig, axes = _light_fig(2, n, figsize=(6 * n, 10),
                           squeeze=False, dpi=DPI)

    genotypes = (sorted(df["Name"].dropna().unique().tolist())
                 if "Name" in df.columns else [])
    cmap_g    = cm.get_cmap("tab20", max(len(genotypes), 1))
    geno_col  = {g: cmap_g(i) for i, g in enumerate(genotypes)}

    for ci, (mname, res) in enumerate(cv_results.items()):
        yt, yp  = res["y_true"], res["y_pred"]
        resid   = yp - yt

        ax_s = axes[0, ci];  _light_ax(ax_s)
        lim  = [min(yt.min(), yp.min()) - 5, max(yt.max(), yp.max()) + 5]
        ax_s.plot(lim, lim, "--", color="#64748b", lw=1.4)

        # Colour by genotype if alignment works
        if len(genotypes) > 0 and len(df) == len(yt):
            names_arr = df["Name"].values
            for g in genotypes:
                mask = names_arr == g
                if mask.sum():
                    ax_s.scatter(yt[mask], yp[mask],
                                 color=geno_col[g], s=36, alpha=0.85,
                                 label=g, zorder=2)
            ncol = max(1, len(genotypes) // 15)
            ax_s.legend(fontsize=6, labelcolor=TEXT_C,
                        facecolor=PANEL_C, edgecolor=BORDER_C,
                        markerscale=0.9, ncol=ncol)
        else:
            ax_s.scatter(yt, yp, c="#4a9fd5", s=36, alpha=0.85, zorder=2)

        ax_s.set_xlim(lim);  ax_s.set_ylim(lim)
        ax_s.set_xlabel("Field MTR (DAP)")
        ax_s.set_ylabel("Predicted MTR (DAP)")
        ax_s.set_title(
            f"{mname}  —  Field vs Predicted\n"
            f"RMSE={res['rmse']:.1f} d   R²={res['r2']:.3f}   "
            f"bias={res['bias']:+.1f} d",
            fontsize=TITLE_FONT - 1)

        ax_r = axes[1, ci];  _light_ax(ax_r)
        bins = min(20, max(5, int(len(resid) / 3)))
        ax_r.hist(resid, bins=bins, color="#4a9fd5",
                  edgecolor=BG_C, alpha=0.85)
        ax_r.axvline(0, color="#22c55e", lw=2, ls="--", label="zero")
        ax_r.axvline(np.mean(resid), color="#f59e0b", lw=1.5, ls=":",
                     label=f"mean={np.mean(resid):+.1f} d")
        ax_r.set_xlabel("Residual (Predicted − Field) [days]")
        ax_r.set_ylabel("Count")
        ax_r.set_title(f"{mname}  —  Residual Distribution",
                       fontsize=TITLE_FONT - 1)
        ax_r.legend(fontsize=8, labelcolor=TEXT_C,
                    facecolor=PANEL_C, edgecolor=BORDER_C)

    fig.suptitle("Field Maturity vs ML Prediction  (Cross-Validated)",
                 color=TEXT_C, fontsize=TITLE_FONT + 2, y=1.01)
    plt.tight_layout(pad=2)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG_C)
    plt.close(fig)


def plot_genotype_formulas(formulas: Dict[str, str],
                            cv_results: Dict[str, Dict],
                            out_path: str):
    """Text cards — one per genotype — showing the per-genotype Ridge formula."""
    n    = max(1, len(formulas))
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = _light_fig(rows, cols,
                           figsize=(cols * 5.8, rows * 4.2),
                           squeeze=False, dpi=DPI)

    ridge_note = ""
    if "Ridge" in cv_results:
        r = cv_results["Ridge"]
        ridge_note = (f"  Ridge CV:  RMSE={r['rmse']:.1f} d  "
                      f"R²={r['r2']:.3f}  bias={r['bias']:+.1f} d")

    for i, (gname, formula_str) in enumerate(formulas.items()):
        r_i, c_i = divmod(i, cols)
        ax = axes[r_i][c_i]
        ax.set_facecolor(PANEL_C)
        ax.axis("off")
        ax.text(0.04, 0.97, gname,
                transform=ax.transAxes, va="top", ha="left",
                fontsize=TITLE_FONT, color="#198754", fontweight="bold")
        ax.text(0.04, 0.80, formula_str,
                transform=ax.transAxes, va="top", ha="left",
                fontsize=7.5, color=TEXT_C, family="monospace",
                linespacing=1.5)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                   transform=ax.transAxes,
                                   fill=False,
                                   edgecolor=BORDER_C, lw=1))

    # Hide unused tiles
    for i in range(n, rows * cols):
        r_i, c_i = divmod(i, cols)
        axes[r_i][c_i].set_visible(False)

    fig.suptitle(f"Per-Genotype Ridge Formula  —  MTR Prediction{ridge_note}",
                 color=TEXT_C, fontsize=TITLE_FONT + 1, y=1.01)
    plt.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG_C)
    plt.close(fig)


def plot_flight_planning(fp_results: Dict[str, Dict],
                          best_model_name: str,
                          out_path: str):
    """Single-panel bell-curve flight schedule figure.

    Shows the flight schedule for *best_model_name* only:
    • Flying starts 2 weeks before predicted maturity DAP.
    • Flights are twice a week (every 3.5 days).
    • Strip colour = chance of maturity (light-blue → dark-blue).
    """
    if not fp_results:
        return

    # Use the requested best model; fall back to the first available entry
    mname = best_model_name if best_model_name in fp_results else next(iter(fp_results))
    res   = fp_results[mname]

    n_range   = res["n_range"]
    dap_axis  = res.get("dap_axis", n_range)
    if len(dap_axis) != len(n_range):
        dap_axis = n_range

    best_n    = res["best_n"]
    best_rmse = res.get("best_rmse", np.nan)
    rec_int   = res.get("rec_interval", 7)
    rec_start = res.get("rec_start_dap", 40)

    best_idx = (n_range.index(best_n) if best_n in n_range else len(n_range) - 1)
    best_dap = float(dap_axis[best_idx]) if best_idx < len(dap_axis) else float(best_n)

    fig, ax2 = _light_fig(1, 1, figsize=(9, 6), squeeze=True, dpi=DPI)
    _light_ax(ax2)

    # ── Flight calendar parameters ─────────────────────────────────────────
    maturity_dap  = best_dap if np.isfinite(best_dap) else rec_start + 30
    fly_start_dap = maturity_dap - 14          # 2 weeks before maturity
    fly_interval  = 3.5                         # twice a week  (7 / 2)

    fly_end_dap = maturity_dap + 14
    fly_dates: List[float] = []
    _d = fly_start_dap
    while _d <= fly_end_dap + fly_interval:
        fly_dates.append(_d)
        _d += fly_interval
    if fly_dates and fly_dates[-1] > fly_end_dap:
        fly_dates = [d for d in fly_dates if d <= fly_end_dap]
    if not fly_dates:
        fly_dates = [fly_start_dap]
    fly_end_actual = float(fly_dates[-1])

    # ── Continuous x range for the smooth bell curve ───────────────────────
    x_lo  = fly_start_dap - 20
    x_hi  = fly_end_actual + 10
    x_c   = np.linspace(x_lo, x_hi, 1000)

    # Bell curve: Gaussian peaking at fly_start_dap (onset of max greenness)
    sigma_bell = (fly_end_actual - fly_start_dap) / 2.2
    if sigma_bell <= 0:
        sigma_bell = 5.0
    bell = np.exp(-0.5 * ((x_c - fly_start_dap) / sigma_bell) ** 2)

    # ── Draw bell curve ────────────────────────────────────────────────────
    ax2.plot(x_c, bell, color=TEXT_C, lw=2.5, zorder=6)

    # ── "No need to fly" shading ───────────────────────────────────────────
    mask_pre = x_c <= fly_start_dap
    ax2.fill_between(x_c[mask_pre], bell[mask_pre],
                      alpha=0.12, color="#adb5bd", zorder=1)
    mid_pre = (x_lo + fly_start_dap) / 2
    ax2.text(mid_pre, 0.28, "no need to fly",
             ha="center", va="center", fontsize=9,
             color="#6c757d", style="italic", zorder=7)

    # ── Colour strips between consecutive flights ──────────────────────────
    blue_cmap = cm.get_cmap("Blues")
    mat_sigma = 5.0   # days

    from math import erf as _erf
    def _ncdf(x):
        return 0.5 * (1 + _erf((x - maturity_dap) / (mat_sigma * 1.4142)))

    for i in range(len(fly_dates) - 1):
        d0, d1   = fly_dates[i], fly_dates[i + 1]
        mid_date = (d0 + d1) / 2
        prob     = float(np.clip(_ncdf(mid_date), 0.0, 1.0))
        color    = blue_cmap(0.20 + 0.75 * prob)

        mask_strip = (x_c >= d0) & (x_c <= d1)
        if mask_strip.sum() > 0:
            ax2.fill_between(x_c[mask_strip], bell[mask_strip],
                              color=color, alpha=0.85, zorder=3)

        bell_at_d0 = float(np.interp(d0, x_c, bell))
        ax2.plot([d0, d0], [0, bell_at_d0],
                 color=blue_cmap(0.20 + 0.75 * float(np.clip(_ncdf(d0), 0, 1))),
                 lw=1.5, zorder=4)

    # Last flight line
    bell_last = float(np.interp(fly_end_actual, x_c, bell))
    ax2.plot([fly_end_actual, fly_end_actual], [0, bell_last],
             color=blue_cmap(0.95), lw=1.5, zorder=4)

    # ── Key vertical markers ───────────────────────────────────────────────
    bell_fs = float(np.interp(fly_start_dap, x_c, bell))
    ax2.plot([fly_start_dap, fly_start_dap], [0, bell_fs],
             color=TEXT_C, lw=2.2, zorder=7)
    ax2.annotate("Start\nFlying",
                 xy=(fly_start_dap, bell_fs),
                 xytext=(fly_start_dap, bell_fs + 0.10),
                 ha="center", fontsize=9, fontweight="bold",
                 color=TEXT_C, zorder=8,
                 arrowprops=dict(arrowstyle="-", color=TEXT_C, lw=0))

    ax2.axvline(maturity_dap, color="#0077b6", lw=2.0, ls="--", zorder=7,
                label=f"Maturity Date  DAP {maturity_dap:.0f}")
    ax2.text(maturity_dap, 1.06, "Maturity Date",
             ha="center", fontsize=9, color="#0077b6",
             transform=ax2.get_xaxis_transform(), zorder=8)

    ax2.axvline(fly_end_actual, color="#8B4513", lw=2.0, ls="--", zorder=7,
                label=f"End of flight  DAP {fly_end_actual:.0f}")
    ax2.text(fly_end_actual, 1.06, "end of flight",
             ha="center", fontsize=9, color="#8B4513",
             transform=ax2.get_xaxis_transform(), zorder=8)

    # ── Colorbar ───────────────────────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=blue_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, fraction=0.025, pad=0.02)
    cbar.set_label("Chance of maturity", fontsize=8, color=LABEL_C)
    cbar.ax.tick_params(labelsize=7, colors=LABEL_C)

    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylim(-0.04, 1.18)
    ax2.set_xlabel("DAP (days after sowing)", fontsize=FONT)
    ax2.set_ylabel("Rel. Freq. of Hue", fontsize=FONT)
    ax2.set_title(
        f"Flight Planning Advisor  —  {mname}\n"
        f"Start: DAP {fly_start_dap:.0f}  ·  twice a week  "
        f"·  End of flight: DAP {fly_end_actual:.0f}",
        fontsize=TITLE_FONT - 1, color=TEXT_C)
    ax2.legend(fontsize=8, labelcolor=TEXT_C,
               facecolor=PANEL_C, edgecolor=BORDER_C, loc="upper right")

    plt.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG_C)
    plt.close(fig)


def plot_stacking_report(stacking_cv: Dict,
                          cv_results: Dict[str, Dict],
                          out_path: str):
    """Scatter + per-fold RMSE bars for every model including the stacking layer.

    Columns: Ridge | RandomForest | [XGBoost if trained] | Stacking
    Row 0  : Predicted vs Actual scatter (residual-coloured).
    Row 1  : Per-fold RMSE horizontal bar chart.
    The Stacking column title is highlighted in green.
    """
    meta_label = stacking_cv.get("meta_name", "XGB")
    bases_str  = "+".join(stacking_cv.get("base_names", []))
    stack_key  = "Stack Model"

    # Canonical model order (all base models + stack)
    _BASE_ORDER = ("Ridge", "ElasticNet", "SVR",
                   "RandomForest", "GradientBoosting", "ExtraTrees", "XGBoost")
    all_results: Dict[str, Dict] = {}
    for k in _BASE_ORDER:
        if k in cv_results:
            all_results[k] = cv_results[k]
    all_results[stack_key] = stacking_cv

    n   = len(all_results)
    fig, axes = _light_fig(2, n, figsize=(6 * n, 10),
                           squeeze=False, dpi=DPI)

    for ci, (mname, res) in enumerate(all_results.items()):
        yt, yp  = res["y_true"], res["y_pred"]
        is_stack = "Stacking" in mname
        title_c  = "#198754" if is_stack else TEXT_C

        # ── Scatter ────────────────────────────────────────────────────────
        ax = axes[0, ci];  _light_ax(ax)
        lim = [min(yt.min(), yp.min()) - 5, max(yt.max(), yp.max()) + 5]
        ax.plot(lim, lim, "--", color="#64748b", lw=1.4, zorder=1)
        sc = ax.scatter(yt, yp, c=yp - yt,
                        cmap="RdYlGn_r", vmin=-15, vmax=15,
                        s=36, alpha=0.85, zorder=2)
        plt.colorbar(sc, ax=ax, label="Residual (days)", fraction=0.04)
        ax.set_xlim(lim);  ax.set_ylim(lim)
        ax.set_xlabel("Actual MTR (DAP)")
        ax.set_ylabel("Predicted MTR (DAP)")
        ax.set_title(
            f"{mname}\nRMSE={res['rmse']:.1f} d   R²={res['r2']:.3f}   "
            f"bias={res['bias']:+.1f} d   [{res['cv_name']}]",
            fontsize=TITLE_FONT - 1, color=title_c)

        # ── Per-fold bar ───────────────────────────────────────────────────
        ax2 = axes[1, ci];  _light_ax(ax2)
        folds  = res["fold_results"]
        labels = [r["fold"] for r in folds]
        rmses  = [r["rmse"]  for r in folds]
        ns_    = [r["n"]     for r in folds]
        bar_c  = cm.RdYlGn_r(np.linspace(0, 1, len(folds)))
        bars   = ax2.barh(labels, rmses, color=bar_c,
                          edgecolor=BORDER_C, linewidth=0.5)
        for bar, n_ in zip(bars, ns_):
            ax2.text(bar.get_width() + 0.2,
                     bar.get_y() + bar.get_height() / 2,
                     f"n={n_}", va="center", ha="left",
                     color=LABEL_C, fontsize=7)
        ax2.axvline(res["rmse"], color="#f59e0b", lw=1.5, ls="--",
                    label=f"Overall {res['rmse']:.1f} d")
        ax2.set_xlabel("RMSE (days)")
        ax2.set_title(f"{mname}  —  per-fold RMSE",
                      fontsize=TITLE_FONT - 2, color=title_c)
        ax2.legend(fontsize=8, labelcolor=TEXT_C,
                   facecolor=PANEL_C, edgecolor=BORDER_C)

    fig.suptitle(
        "Multi-Modal Stacking Ensemble  —  Cross-Validation Report\n"
        f"Base learners: {bases_str}  →  Meta-learner: {meta_label}",
        color=TEXT_C, fontsize=TITLE_FONT + 1, y=1.02)
    plt.tight_layout(pad=2)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG_C)
    plt.close(fig)


def plot_model_comparison(cv_results: Dict[str, Dict],
                           stacking_cv: Dict,
                           out_path: str):
    """Grouped bar chart comparing RMSE, MAE, and R² for all models + stacking.

    The stacking bar is highlighted in green to make any improvement
    (or regression) immediately visible.
    """
    meta_label = stacking_cv.get("meta_name", "XGB")
    bases_str  = "+".join(stacking_cv.get("base_names", []))
    stack_key  = "Stack Model"

    _BASE_ORDER = ("Ridge", "ElasticNet", "SVR",
                   "RandomForest", "GradientBoosting", "ExtraTrees", "XGBoost")
    all_res: Dict[str, Dict] = {}
    for k in _BASE_ORDER:
        if k in cv_results:
            all_res[k] = cv_results[k]
    all_res[stack_key] = stacking_cv

    names  = list(all_res.keys())
    rmses  = [all_res[n]["rmse"] for n in names]
    maes   = [all_res[n]["mae"]  for n in names]
    r2s    = [all_res[n]["r2"]   for n in names]
    colors = ["#198754" if n == "Stack Model" else "#4a9fd5" for n in names]

    fig, axes = _light_fig(1, 3, figsize=(14, 5), squeeze=False, dpi=DPI)
    x = np.arange(len(names))
    w = 0.55

    def _bar_panel(ax, vals, ylabel, title, fmt=".2f"):
        _light_ax(ax)
        span = max(vals) - min(vals) if max(vals) != min(vals) else 1.0
        bars = ax.bar(x, vals, width=w, color=colors,
                      edgecolor=BORDER_C, linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + span * 0.03,
                    f"{v:{fmt}}", ha="center", va="bottom",
                    fontsize=8, color=TEXT_C)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(ylabel, fontsize=FONT)
        ax.set_title(title, fontsize=TITLE_FONT)
        # Subtle green shading behind the Stack Model column
        for xi, nm in enumerate(names):
            if nm == "Stack Model":
                ax.axvspan(xi - 0.45, xi + 0.45,
                           alpha=0.07, color="#198754", zorder=0)

    _bar_panel(axes[0, 0], rmses, "RMSE (days)",
               "RMSE  ↓ lower is better")
    _bar_panel(axes[0, 1], maes,  "MAE (days)",
               "MAE   ↓ lower is better")
    _bar_panel(axes[0, 2], r2s,   "R²",
               "R²    ↑ higher is better", fmt=".3f")

    fig.suptitle(
        "Model Performance Comparison  —  Base Models  vs  Multi-Modal Stacking\n"
        f"({bases_str} → {meta_label} meta-learner)",
        color=TEXT_C, fontsize=TITLE_FONT + 1, y=1.04)
    plt.tight_layout(pad=2)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG_C)
    plt.close(fig)


def plot_trial_comparison(df_j: pd.DataFrame,
                           y:    np.ndarray,
                           groups: np.ndarray,
                           cv_results: Dict[str, Dict],
                           stacking_cv_res: Dict,
                           overall_best: str,
                           out_path: str):
    """Stacked dual horizontal box-plot: Field MTR and ML MTR on separate rows
    for each trial, so they can be compared at a glance.

    Layout per trial (two rows, separated by a small gap):
        [Trial name label]
          upper row  — Field MTR  (yellow)
          lower row  — ML MTR     (red)

    Trials are sorted alphabetically; the first trial appears at the top.
    """
    from matplotlib.patches import Patch

    # ── Per-trial data ────────────────────────────────────────────────────────
    trial_names = sorted(np.unique(groups).tolist(), key=str)
    if len(trial_names) < 2:
        return

    # Best CV predictions (prefer Stack Model)
    _PREF = ["Stack Model"] + list(cv_results.keys())
    pred_key = next((k for k in _PREF
                     if (k == "Stack Model" and stacking_cv_res)
                     or k in cv_results),
                    None)

    def _full_pred(name: str) -> Optional[np.ndarray]:
        if name == "Stack Model" and stacking_cv_res:
            return stacking_cv_res.get("y_pred_full")
        return cv_results.get(name, {}).get("y_pred_full")

    pred_arr = _full_pred(pred_key) if pred_key else None

    field_data: Dict[str, np.ndarray] = {}
    pred_data:  Dict[str, np.ndarray] = {}
    for tn in trial_names:
        mask = np.asarray(groups) == tn
        y_tn = y[mask];  ok = np.isfinite(y_tn)
        if ok.sum() > 0:
            field_data[tn] = y_tn[ok]
        if pred_arr is not None:
            p_tn = pred_arr[mask];  ok_p = np.isfinite(p_tn)
            if ok_p.sum() > 0:
                pred_data[tn] = p_tn[ok_p]

    if not field_data:
        return

    # Display order: alphabetical, first trial at top → reverse so index 0 = top
    display_order = list(reversed(list(field_data.keys())))
    n             = len(display_order)

    # ── Stacked y-positions ───────────────────────────────────────────────────
    # Each trial occupies a 2.6-unit band:
    #   • lower row (ML,    red):    y = base + 0.0
    #   • upper row (Field, yellow): y = base + 1.0
    #   • y-tick label:              y = base + 0.5  (midpoint)
    #   • gap to next trial:                0.6 units
    SLOT = 2.6   # units between consecutive trial band bases

    ml_pos     = [(n - 1 - j) * SLOT + 0.0 for j in range(n)]   # lower
    field_pos  = [(n - 1 - j) * SLOT + 1.0 for j in range(n)]   # upper
    tick_pos   = [(n - 1 - j) * SLOT + 0.5 for j in range(n)]   # label midpoint

    # ── Colour palette ────────────────────────────────────────────────────────
    C_FIELD      = "#fde047";  C_FIELD_EDGE = "#a16207"
    C_ML         = "#ef4444";  C_ML_EDGE    = "#7f1d1d"
    C_MEDIAN     = "#1e293b"
    C_DIVIDER    = "#94a3b8"   # thin line between each trial pair

    # ── Figure ───────────────────────────────────────────────────────────────
    fig_h = max(6, n * SLOT * 0.45 + 2.5)
    fig, ax = _light_fig(1, 1, figsize=(11, fig_h), dpi=DPI)
    _light_ax(ax)

    def _bp(arrays, pos, face, edge, width=0.42):
        ax.boxplot(
            arrays,
            vert         = False,
            positions    = pos,
            widths       = width,
            patch_artist = True,
            manage_ticks = False,
            medianprops  = dict(color=C_MEDIAN, linewidth=2.4),
            boxprops     = dict(facecolor=face, edgecolor=edge, linewidth=1.5),
            whiskerprops = dict(color=edge, linewidth=1.2),
            capprops     = dict(color=edge, linewidth=1.5),
            flierprops   = dict(marker="o", color=edge, markersize=3.5,
                                markerfacecolor=face, alpha=0.55),
        )

    # ── Draw Field MTR (yellow, upper row) ────────────────────────────────────
    _bp([field_data[t] for t in display_order], field_pos, C_FIELD, C_FIELD_EDGE)

    # ── Draw ML MTR (red, lower row) — only trials that have predictions ──────
    if pred_data:
        ml_arrays, ml_p = [], []
        for j, t in enumerate(display_order):
            if t in pred_data:
                ml_arrays.append(pred_data[t])
                ml_p.append(ml_pos[j])
        if ml_arrays:
            _bp(ml_arrays, ml_p, C_ML, C_ML_EDGE)

    # ── Thin divider line between each trial pair ─────────────────────────────
    for j in range(n - 1):
        y_div = (n - 1 - j) * SLOT - 0.35    # just below the ML box
        ax.axhline(y_div, color=C_DIVIDER, linewidth=0.6, linestyle="--",
                   alpha=0.5, zorder=1)

    # ── Axes decoration ───────────────────────────────────────────────────────
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(display_order, fontsize=SMALL + 1)
    ax.set_ylim(ml_pos[-1] - 0.8, field_pos[0] + 0.8)   # [-1] = bottom, [0] = top
    ax.set_xlabel("Maturity  (DAP)", fontsize=FONT)
    pred_label = pred_key if pred_key else "Best Model"
    ax.set_title(
        f"Maturity Comparison Across Trials\n"
        f"(Field MTR vs.  {pred_label}  CV predictions)",
        fontsize=TITLE_FONT, color=TEXT_C, pad=10)
    ax.grid(True, axis="x", alpha=0.3, color=BORDER_C, linewidth=0.8, zorder=0)

    # ── Row labels (Field / ML) on the right margin ───────────────────────────
    x_right = ax.get_xlim()[1] if ax.get_xlim()[1] != 1.0 else 110.0
    for j in range(min(n, 3)):   # annotate first 3 pairs to avoid clutter
        base = (n - 1 - j) * SLOT
        ax.annotate("Field", xy=(1.001, base + 1.0 + 0.15),
                    xycoords=("axes fraction", "data"),
                    fontsize=7, color=C_FIELD_EDGE, va="center", ha="left")
        ax.annotate("ML",    xy=(1.001, base + 0.0 + 0.15),
                    xycoords=("axes fraction", "data"),
                    fontsize=7, color=C_ML_EDGE, va="center", ha="left")

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.legend(
        handles=[
            Patch(facecolor=C_FIELD, edgecolor=C_FIELD_EDGE, label="Field MTR"),
            Patch(facecolor=C_ML,    edgecolor=C_ML_EDGE,    label="ML MTR"),
        ],
        loc="upper right", title="Legend", title_fontsize=FONT,
        fontsize=FONT, frameon=True, framealpha=0.92, edgecolor=BORDER_C,
    )

    plt.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG_C)
    plt.close(fig)


def plot_feature_correlation(X: pd.DataFrame, y: np.ndarray,
                              cont_features: List[str],
                              out_path: str):
    """Heatmap of feature × feature + feature × MTR correlations."""
    df_v = X[cont_features].copy()
    df_v["MTR_field"] = y
    corr = df_v.corr(numeric_only=True)

    sz   = min(24, len(cont_features) + 2)
    fig, ax = plt.subplots(figsize=(sz, sz - 1), dpi=DPI)
    fig.patch.set_facecolor(BG_C)
    ax.set_facecolor(BG_C)

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.025, label="Pearson r")

    labels = list(corr.columns)
    ax.set_xticks(range(len(labels)));  ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7, color=LABEL_C)
    ax.set_yticklabels(labels, fontsize=7, color=LABEL_C)
    ax.set_title("Feature Correlation Matrix  (incl. MTR_field)",
                 color=TEXT_C, fontsize=TITLE_FONT, pad=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG_C)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 11b — CUMULATIVE FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════════════

def plot_cumulative_feature_importance(
        models:          Dict[str, Any],
        X:               pd.DataFrame,
        cont_features:   List[str],
        out_dir:         str,
        stacking_bundle: Optional[Dict] = None,
        log_fn=None):
    """
    Two PNG figures saved into *out_dir*:

    (a) feature_importance_cumulative_base.png
        One cumulative-importance curve per base model that exposes
        feature_importances_ (tree-based) or |coef_| (linear).
        SVR is skipped (no analytic importance).

    (b) feature_importance_cumulative_stack.png
        Cumulative importance of the XGBoost / RF meta-learner over its
        combined input space (original features + OOF base-model predictions).
    """
    def _log(m):
        if log_fn:
            log_fn(m)

    has_name   = "Name" in X.columns
    feat_names = cont_features + (["Name"] if has_name else [])

    # ── helper: extract sorted cumulative importance from a pipeline ──────────
    def _cumulative(pipe):
        try:
            reg = pipe.named_steps["reg"]
            if hasattr(reg, "feature_importances_"):
                raw = np.array(reg.feature_importances_, dtype=float)
            elif hasattr(reg, "coef_"):
                raw = np.abs(np.asarray(reg.coef_, dtype=float).ravel())
            else:
                return None, None
            n      = min(len(raw), len(feat_names))
            raw    = raw[:n]
            fnames = feat_names[:n]
            total  = raw.sum()
            if total == 0:
                return None, None
            norm   = raw / total * 100.0
            order  = np.argsort(norm)[::-1]
            return np.cumsum(norm[order]), [fnames[i] for i in order]
        except Exception as ex:
            _log(f"[ML] Warning — importance extraction: {ex}")
            return None, None

    # colour palette consistent with the rest of the report
    _COLORS = {
        "Ridge":            "#1f77b4",
        "ElasticNet":       "#ff7f0e",
        "SVR":              "#9467bd",
        "RandomForest":     "#2ca02c",
        "GradientBoosting": "#d62728",
        "ExtraTrees":       "#8c564b",
        "XGBoost":          "#e377c2",
    }

    # ── Figure (a) — base models ──────────────────────────────────────────────
    fig_a, ax_a = _light_fig(1, 1, figsize=(10, 6), dpi=DPI)
    _light_ax(ax_a)
    ax_a.grid(True, alpha=0.35, color=BORDER_C, zorder=0)

    plotted = False
    for mname, pipe in models.items():
        cum, _ = _cumulative(pipe)
        if cum is None:
            continue
        x_vals = np.arange(1, len(cum) + 1)
        ax_a.plot(x_vals, cum, lw=2.2,
                  color=_COLORS.get(mname, "#333333"),
                  label=mname, zorder=3)
        plotted = True

    if plotted:
        for thresh, ls, alpha in [(80, "--", 0.65), (95, ":", 0.65)]:
            ax_a.axhline(thresh, color="#6c757d", lw=1.3, ls=ls,
                         alpha=alpha, zorder=2, label=f"{thresh}% threshold")

        ax_a.set_xlabel("Number of Predictors (ranked by importance)", fontsize=FONT)
        ax_a.set_ylabel("Cumulative Importance (%)", fontsize=FONT)
        ax_a.set_title(
            "Sufficient Number of Features — Base Models\n"
            "Cumulative feature importance (sorted descending)",
            fontsize=TITLE_FONT, color=TEXT_C)
        ax_a.set_ylim(0, 106)
        ax_a.set_xlim(left=1)
        ax_a.legend(fontsize=9, labelcolor=TEXT_C, facecolor=PANEL_C,
                    edgecolor=BORDER_C, loc="lower right", ncol=2)
        plt.tight_layout(pad=1.5)
        out_a = os.path.join(out_dir, "feature_importance_cumulative_base.png")
        fig_a.savefig(out_a, dpi=DPI, bbox_inches="tight", facecolor=BG_C)
        _log("[ML] feature_importance_cumulative_base.png saved.")
    plt.close(fig_a)

    # ── Figure (b) — stacking meta-learner ───────────────────────────────────
    if not stacking_bundle:
        return

    meta_model = stacking_bundle.get("meta_model")
    meta_fcols = stacking_bundle.get("meta_fcols")
    meta_cols  = stacking_bundle.get("meta_cols", [])
    meta_name  = stacking_bundle.get("meta_name", "Meta-learner")

    if meta_model is None or not meta_fcols:
        return

    try:
        if hasattr(meta_model, "feature_importances_"):
            raw_m = np.array(meta_model.feature_importances_, dtype=float)
        elif hasattr(meta_model, "coef_"):
            raw_m = np.abs(np.asarray(meta_model.coef_, dtype=float).ravel())
        else:
            _log("[ML] Stacking meta-model has no feature importances — skipping figure (b).")
            return

        n_m      = min(len(raw_m), len(meta_fcols))
        raw_m    = raw_m[:n_m]
        names_m  = list(meta_fcols)[:n_m]
        total_m  = raw_m.sum()
        if total_m == 0:
            return
        norm_m   = raw_m / total_m * 100.0
        order_m  = np.argsort(norm_m)[::-1]
        cum_m    = np.cumsum(norm_m[order_m])
        sorted_m = [names_m[i] for i in order_m]

        fig_b, ax_b = _light_fig(1, 1, figsize=(10, 6), dpi=DPI)
        _light_ax(ax_b)
        ax_b.grid(True, alpha=0.35, color=BORDER_C, zorder=0)

        x_m = np.arange(1, len(cum_m) + 1)
        ax_b.plot(x_m, cum_m, lw=2.5, color="#0077b6", label="Stack Model", zorder=3)

        for thresh, ls in [(80, "--"), (95, ":")]:
            ax_b.axhline(thresh, color="#6c757d", lw=1.3, ls=ls,
                         alpha=0.65, zorder=2, label=f"{thresh}% threshold")
            idx = int(np.searchsorted(cum_m, thresh))
            if idx < len(cum_m):
                ax_b.annotate(
                    f"n = {idx + 1}",
                    xy=(idx + 1, cum_m[idx]),
                    xytext=(idx + max(3, len(cum_m) * 0.05),
                            cum_m[idx] - 7),
                    fontsize=9, color=TEXT_C, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=LABEL_C, lw=1.1))

        ax_b.set_xlabel("Number of Predictors (ranked by importance)", fontsize=FONT)
        ax_b.set_ylabel("Cumulative Importance (%)", fontsize=FONT)
        ax_b.set_title(
            f"Sufficient Number of Features — Stack Model  ({meta_name} meta-learner)\n"
            f"Input space: {len(feat_names)} original features  +  "
            f"{len(meta_cols)} OOF base-model predictions",
            fontsize=TITLE_FONT, color=TEXT_C)
        ax_b.set_ylim(0, 106)
        ax_b.set_xlim(left=1)
        ax_b.legend(fontsize=9, labelcolor=TEXT_C, facecolor=PANEL_C,
                    edgecolor=BORDER_C, loc="lower right")
        plt.tight_layout(pad=1.5)
        out_b = os.path.join(out_dir, "feature_importance_cumulative_stack.png")
        fig_b.savefig(out_b, dpi=DPI, bbox_inches="tight", facecolor=BG_C)
        _log("[ML] feature_importance_cumulative_stack.png saved.")
        plt.close(fig_b)

    except Exception as ex:
        _log(f"[ML] Warning — stacking cumulative importance: {ex}")
        plt.close("all")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 11c — BIAS–VARIANCE DECOMPOSITION
# ═════════════════════════════════════════════════════════════════════════════

def plot_bias_variance_decomposition(
        cv_results:      Dict[str, Dict],
        stacking_cv_res: Dict,
        out_path:        str,
        log_fn=None):
    """
    Stacked bar chart decomposing each model's CV error into:
      • Std(residuals)  — random/variance component  (blue bar, bottom)
      • |Bias|          — systematic component        (orange bar, stacked)

    A diamond marker shows the actual RMSE:
        RMSE² = Bias² + Var(residuals)   [exact decomposition]
    so RMSE ≥ max(|Bias|, Std).

    All base models + stacking ensemble are shown side-by-side.
    """
    def _log(m):
        if log_fn:
            log_fn(m)

    _BASE_ORDER = ("Ridge", "ElasticNet", "SVR",
                   "RandomForest", "GradientBoosting", "ExtraTrees", "XGBoost")
    all_res: Dict[str, Dict] = {}
    for k in _BASE_ORDER:
        if k in cv_results:
            all_res[k] = cv_results[k]
    if stacking_cv_res:
        all_res["Stack"] = stacking_cv_res

    if not all_res:
        return

    names:     List[str]   = []
    abs_bias:  List[float] = []
    std_res:   List[float] = []
    rmse_vals: List[float] = []
    bias_sign: List[float] = []   # keep sign for annotation (+/−)

    for mname, res in all_res.items():
        y_t = np.asarray(res["y_true"], dtype=float)
        y_p = np.asarray(res["y_pred"], dtype=float)
        ok  = np.isfinite(y_t) & np.isfinite(y_p)
        if ok.sum() < 2:
            continue
        resid = y_p[ok] - y_t[ok]
        bias  = float(np.mean(resid))
        std   = float(np.std(resid, ddof=1))
        rmse  = float(np.sqrt(np.mean(resid ** 2)))
        names.append(mname)
        abs_bias.append(abs(bias))
        std_res.append(std)
        rmse_vals.append(rmse)
        bias_sign.append(bias)

    if not names:
        return

    x = np.arange(len(names))
    W = 0.55

    fig_w = max(9, len(names) * 1.5)
    fig, ax = _light_fig(1, 1, figsize=(fig_w, 6), dpi=DPI)
    _light_ax(ax)
    ax.grid(True, axis="y", alpha=0.35, color=BORDER_C, zorder=0)

    # Stack: Std on bottom, |Bias| on top
    bar_std  = ax.bar(x, std_res,  W,
                      label="Std of residuals  (random / variance error)",
                      color="#4a90d9", alpha=0.90, zorder=3, edgecolor="white", linewidth=0.5)
    bar_bias = ax.bar(x, abs_bias, W, bottom=std_res,
                      label="|Bias|  (systematic error)",
                      color="#e07b39", alpha=0.90, zorder=3, edgecolor="white", linewidth=0.5)

    # RMSE diamond marker
    ax.scatter(x, rmse_vals, marker="D", s=80, color="#198754",
               zorder=5, label="RMSE  (= √(Bias² + Variance))",
               linewidths=1.0, edgecolors="#0d3518")

    # Annotate each model: RMSE and signed bias
    y_max = max(sv + bv for sv, bv in zip(std_res, abs_bias))
    for xi, rv, bs, ab, sd in zip(x, rmse_vals, bias_sign, abs_bias, std_res):
        # RMSE label above marker
        ax.text(xi, rv + y_max * 0.025, f"RMSE\n{rv:.1f} d",
                ha="center", va="bottom", fontsize=8, color="#155724",
                fontweight="bold", zorder=6, linespacing=1.3)
        # Bias sign inside the orange bar
        if ab > y_max * 0.04:
            label_bias = f"{'+' if bs >= 0 else '−'}{abs(bs):.1f} d"
            ax.text(xi, sd + ab / 2, label_bias,
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold", zorder=7)
        # Std label inside the blue bar
        if sd > y_max * 0.04:
            ax.text(xi, sd / 2, f"±{sd:.1f} d",
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold", zorder=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, rotation=15, ha="right")
    ax.set_ylabel("Error  (days)", fontsize=FONT)
    ax.set_ylim(0, y_max * 1.40)
    ax.set_title(
        "Bias–Variance Decomposition of Leave-One-Trial-Out CV Error\n"
        "Blue = random (variance) component  ·  Orange = systematic (bias) component  ·  ◆ = RMSE",
        fontsize=TITLE_FONT, color=TEXT_C)
    ax.legend(fontsize=9.5, labelcolor=TEXT_C, facecolor=PANEL_C,
              edgecolor=BORDER_C, loc="upper right")

    plt.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG_C)
    plt.close(fig)
    _log("[ML] bias_variance_decomposition.png saved.")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 12 — PER-TRIAL OUTPUT HELPER
# ═════════════════════════════════════════════════════════════════════════════

def save_trial_ml_outputs(
        tr_name:         str,
        tr_root:         str,
        df_j:            pd.DataFrame,
        y:               np.ndarray,
        groups:          np.ndarray,
        X:               pd.DataFrame,
        cv_results:      Dict[str, Dict],
        stacking_cv_res: Dict,
        models:          Dict[str, Any],
        cont_features:   List[str],
        fp_results:      Dict[str, Dict],
        fp_best_name:    str,
        ridge_formulas:  Dict[str, str],
        stacking_bundle: Dict,
        overall_best:    str,
        best_base_name:  str,
        global_out_dir:  str,
        sowing_date      = None,
        log_fn           = None) -> None:
    """
    Save per-trial ML outputs to ``{tr_root}/ML_Analysis/``.

    Mirrors the global output set but filtered / specific to *tr_name*:

    +--------------------------+----------------------------------------+
    | File                     | Content                                |
    +==========================+========================================+
    | predictions.xlsx         | All-model train + CV OOF predictions   |
    | feature_matrix.xlsx      | Feature values for this trial's plots  |
    | cv_report.png            | Scatter + fold bar for this trial      |
    | field_vs_predicted.png   | Genotype-coloured scatter              |
    | feature_importance.png   | Feature importance (trial RMSE tags)   |
    | flight_planning.png      | Global best-model flight plan (copy)   |
    | genotype_formulas.png    | Ridge formula for trial's genotypes    |
    | stacking_cv_report.png   | Stacking scatter for this trial        |
    | ml_summary.txt           | Per-trial performance text summary     |
    +--------------------------+----------------------------------------+
    """
    def _log(m):
        if log_fn:
            log_fn(m)

    tr_ml_dir = os.path.join(tr_root, "ML_Analysis")
    os.makedirs(tr_ml_dir, exist_ok=True)

    # ── Locate this trial's rows in the combined dataset ─────────────────────
    grp_arr    = np.asarray(groups)
    trial_mask = (grp_arr == tr_name)
    if trial_mask.sum() == 0:          # case/whitespace fallback
        trial_mask = np.array([
            str(g).strip().lower() == str(tr_name).strip().lower()
            for g in grp_arr], dtype=bool)
    if trial_mask.sum() == 0:
        _log(f"[ML]   ⚠  {tr_name}: no matching rows in dataset — "
             f"skipping per-trial outputs.")
        return

    df_t = df_j[trial_mask].reset_index(drop=True)
    y_t  = y[trial_mask]
    X_t  = X.iloc[np.where(trial_mask)[0]].reset_index(drop=True)
    n_t  = int(trial_mask.sum())
    _log(f"[ML]   {tr_name}: generating outputs for {n_t} plots ...")

    # ── Build trial-filtered cv_results ──────────────────────────────────────
    # For each model pull y_pred_full[trial_mask] (OOF for exactly this fold)
    trial_cv: Dict[str, Dict] = {}
    for mname, res in cv_results.items():
        ypf = res.get("y_pred_full")
        if ypf is None or len(ypf) != len(y):
            continue
        pred_t = ypf[trial_mask]
        ok_t   = np.isfinite(pred_t) & np.isfinite(y_t)
        if ok_t.sum() < 2:
            continue
        yt_ok  = y_t[ok_t];  yp_ok = pred_t[ok_t]
        rmse_t = float(np.sqrt(mean_squared_error(yt_ok, yp_ok)))
        mae_t  = float(mean_absolute_error(yt_ok, yp_ok))
        r2_t   = (float(r2_score(yt_ok, yp_ok)) if len(yt_ok) >= 2
                  else float("nan"))
        bias_t = float(np.mean(yp_ok - yt_ok))
        trial_cv[mname] = {
            "cv_name":      f"LOTO  (held-out: {tr_name})",
            "y_true":       yt_ok,
            "y_pred":       yp_ok,
            "y_pred_full":  pred_t,          # trial-length (n_t,) array
            "rmse":         rmse_t,
            "mae":          mae_t,
            "r2":           r2_t,
            "bias":         bias_t,
            "fold_results": [{"fold": tr_name, "n": int(ok_t.sum()),
                               "rmse": rmse_t, "mae": mae_t}],
        }

    # Stacking trial predictions
    trial_stacking: Dict = {}
    if stacking_cv_res:
        s_ypf = stacking_cv_res.get("y_pred_full")
        if s_ypf is not None and len(s_ypf) == len(y):
            s_pred_t = s_ypf[trial_mask]
            s_ok     = np.isfinite(s_pred_t) & np.isfinite(y_t)
            if s_ok.sum() >= 2:
                syt  = y_t[s_ok];  syp = s_pred_t[s_ok]
                trial_stacking = {
                    "cv_name":      f"LOTO-Stack  (held-out: {tr_name})",
                    "y_true":       syt,
                    "y_pred":       syp,
                    "y_pred_full":  s_pred_t,
                    "rmse":         float(np.sqrt(mean_squared_error(syt, syp))),
                    "mae":          float(mean_absolute_error(syt, syp)),
                    "r2":           (float(r2_score(syt, syp))
                                    if len(syt) >= 2 else float("nan")),
                    "bias":         float(np.mean(syp - syt)),
                    "base_names":   stacking_cv_res.get("base_names", []),
                    "meta_name":    stacking_cv_res.get("meta_name", "XGBoost"),
                    "fold_results": [{"fold": tr_name, "n": int(s_ok.sum()),
                                      "rmse": float(np.sqrt(mean_squared_error(syt, syp))),
                                      "mae":  float(mean_absolute_error(syt, syp))}],
                }

    # ── 1. Predictions Excel ──────────────────────────────────────────────────
    try:
        id_cols  = [c for c in ("PlotID", "TrialName", "Name", "MTR_field")
                    if c in df_t.columns]
        pred_df  = df_t[id_cols].copy()
        if "TrialName" not in pred_df.columns:
            pred_df.insert(1, "TrialName", tr_name)

        # Training (in-sample) predictions for this trial's plots
        for mname, pipe in models.items():
            try:
                pred_df[f"Train_{mname}"] = np.round(pipe.predict(X_t), 1)
            except Exception:
                pass

        # OOF cross-validated predictions (from held-out fold)
        for mname, trc in trial_cv.items():
            col              = np.full(n_t, np.nan)
            ok_m             = np.isfinite(trc["y_pred_full"])
            col[ok_m]        = np.round(trc["y_pred_full"][ok_m], 1)
            pred_df[f"CV_{mname}"] = col

        # Stacking OOF
        if trial_stacking:
            s_col            = np.full(n_t, np.nan)
            s_ok_m           = np.isfinite(trial_stacking["y_pred_full"])
            s_col[s_ok_m]    = np.round(trial_stacking["y_pred_full"][s_ok_m], 1)
            pred_df["CV_Stack"] = s_col

        # Maturity Date column from the overall-best model's training predictions
        if sowing_date is not None:
            _date_col = (f"Train_{overall_best}"
                         if overall_best in models and f"Train_{overall_best}" in pred_df.columns
                         else f"Train_{best_base_name}"
                         if f"Train_{best_base_name}" in pred_df.columns
                         else None)
            if _date_col:
                pred_df["Predicted_MTR_Date"] = pred_df[_date_col].apply(
                    lambda d: ((sowing_date + timedelta(days=int(d))).strftime("%Y-%m-%d")
                               if np.isfinite(d) else "N/A"))

        pred_df.to_excel(os.path.join(tr_ml_dir, "predictions.xlsx"),
                         index=False, engine="xlsxwriter")
        _log(f"[ML]   ✔ {tr_name}: predictions.xlsx  ({len(pred_df)} plots, "
             f"{len(pred_df.columns)} columns)")
    except Exception as ex:
        _log(f"[ML]   ✘ {tr_name}: predictions.xlsx — {ex}")

    # ── 2. Feature matrix Excel ───────────────────────────────────────────────
    try:
        fm_t = X_t.copy()
        fm_t.insert(0, "PlotID",    df_t["PlotID"].values)
        fm_t.insert(1, "TrialName", tr_name)
        fm_t["MTR_field"] = y_t
        fm_t.to_excel(os.path.join(tr_ml_dir, "feature_matrix.xlsx"),
                      index=False, engine="xlsxwriter")
        _log(f"[ML]   ✔ {tr_name}: feature_matrix.xlsx")
    except Exception as ex:
        _log(f"[ML]   ✘ {tr_name}: feature_matrix.xlsx — {ex}")

    # ── Copy global plots that are dataset-level (no per-trial variant needed) ─
    for _src_name, _dst_name in [
        ("flight_planning.png",    "flight_planning.png"),
        ("trial_comparison.png",   "trial_comparison.png"),
    ]:
        try:
            _src = os.path.join(global_out_dir, _src_name)
            if os.path.exists(_src):
                shutil.copy2(_src, os.path.join(tr_ml_dir, _dst_name))
                _log(f"[ML]   ✔ {tr_name}: {_dst_name} (copied global)")
        except Exception as ex:
            _log(f"[ML]   ✘ {tr_name}: {_dst_name} — {ex}")

    if not trial_cv:
        _log(f"[ML]   ⚠  {tr_name}: no valid OOF predictions — "
             f"plot generation skipped.")
        return

    # ── 3. CV report (scatter + fold bar for this trial's held-out fold) ──────
    try:
        plot_cv_report(trial_cv, os.path.join(tr_ml_dir, "cv_report.png"))
        _log(f"[ML]   ✔ {tr_name}: cv_report.png")
    except Exception as ex:
        _log(f"[ML]   ✘ {tr_name}: cv_report.png — {ex}")

    # ── 4. Field vs predicted (genotype-coloured) ─────────────────────────────
    try:
        # Common ok mask across all models → guarantees df/y alignment
        combined_ok = np.ones(n_t, dtype=bool)
        for trc in trial_cv.values():
            combined_ok &= np.isfinite(trc["y_pred_full"])
        combined_ok &= np.isfinite(y_t)

        df_t_ok = df_t[combined_ok].reset_index(drop=True)
        trial_cv_fvp: Dict[str, Dict] = {}
        for mname, trc in trial_cv.items():
            pred_aln = trc["y_pred_full"][combined_ok]
            y_aln    = y_t[combined_ok]
            ok2      = np.isfinite(pred_aln)
            if ok2.sum() < 2:
                continue
            trial_cv_fvp[mname] = {
                **trc,
                "y_true": y_aln[ok2],
                "y_pred": pred_aln[ok2],
            }

        if trial_cv_fvp:
            plot_field_vs_predicted(
                trial_cv_fvp, df_t_ok,
                os.path.join(tr_ml_dir, "field_vs_predicted.png"))
            _log(f"[ML]   ✔ {tr_name}: field_vs_predicted.png")
    except Exception as ex:
        _log(f"[ML]   ✘ {tr_name}: field_vs_predicted.png — {ex}")

    # ── 5. Feature importance (re-generated with trial-specific RMSE tags) ────
    try:
        plot_feature_importance(
            models, cont_features, trial_cv,
            os.path.join(tr_ml_dir, "feature_importance.png"))
        _log(f"[ML]   ✔ {tr_name}: feature_importance.png")
    except Exception as ex:
        # Fallback: copy global version
        _src = os.path.join(global_out_dir, "feature_importance.png")
        if os.path.exists(_src):
            try:
                shutil.copy2(_src, os.path.join(tr_ml_dir, "feature_importance.png"))
                _log(f"[ML]   ✔ {tr_name}: feature_importance.png (global copy, {ex})")
            except Exception:
                pass
        else:
            _log(f"[ML]   ✘ {tr_name}: feature_importance.png — {ex}")

    # ── 6. Genotype formulas (filter to trial's genotypes) ───────────────────
    if ridge_formulas:
        try:
            trial_genos = (set(df_t["Name"].dropna().unique())
                           if "Name" in df_t.columns else set())
            rf_t = {k: v for k, v in ridge_formulas.items()
                    if k in trial_genos or k == "(global)"}
            if rf_t:
                plot_genotype_formulas(
                    rf_t, trial_cv,
                    os.path.join(tr_ml_dir, "genotype_formulas.png"))
                _log(f"[ML]   ✔ {tr_name}: genotype_formulas.png "
                     f"({len(rf_t)} genotype(s))")
        except Exception as ex:
            _log(f"[ML]   ✘ {tr_name}: genotype_formulas.png — {ex}")

    # ── 7. Stacking CV report for this trial ──────────────────────────────────
    if trial_stacking:
        try:
            plot_stacking_report(
                trial_stacking, trial_cv,
                os.path.join(tr_ml_dir, "stacking_cv_report.png"))
            _log(f"[ML]   ✔ {tr_name}: stacking_cv_report.png")
        except Exception as ex:
            _log(f"[ML]   ✘ {tr_name}: stacking_cv_report.png — {ex}")

    # ── 8. ml_summary.txt ────────────────────────────────────────────────────
    try:
        t_lines = [
            "=" * 60,
            f"ML Analysis Summary  —  Trial: {tr_name}",
            "=" * 60,
            f"  Plots              : {n_t}",
            f"  Overall best model : {overall_best}",
            "",
            "  CV Performance  (held-out fold for this trial):",
        ]
        for mname, trc in trial_cv.items():
            t_lines.append(
                f"    {mname:<20s}  RMSE={trc['rmse']:5.1f} d  "
                f"MAE={trc['mae']:5.1f} d  R²={trc['r2']:.3f}  "
                f"bias={trc['bias']:+.1f} d")
        if trial_stacking:
            ts = trial_stacking
            t_lines.append(
                f"    {'Stack Model':<20s}  RMSE={ts['rmse']:5.1f} d  "
                f"MAE={ts['mae']:5.1f} d  R²={ts['r2']:.3f}  "
                f"bias={ts['bias']:+.1f} d")
        with open(os.path.join(tr_ml_dir, "ml_summary.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(t_lines))
        _log(f"[ML]   ✔ {tr_name}: ml_summary.txt")
    except Exception as ex:
        _log(f"[ML]   ✘ {tr_name}: ml_summary.txt — {ex}")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 13 — MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def run_ml_pipeline(field_excel: str,
                    db_path: str,
                    out_dir: str,
                    sowing_date=None,
                    plot_id_col:  str = "PlotID",
                    trial_col:    str = "Experiment Name",
                    genotype_col: str = "Name",
                    mtr_col:      str = "MTR",
                    trial_roots:  Optional[List[Tuple[str, str]]] = None,
                    _prebuilt_df: Optional[pd.DataFrame] = None,
                    log_fn=None) -> str:
    """
    Full ML analysis pipeline.

    Steps
    -----
    1.  Load persistent trial database
    2.  Join with field Excel (get MTR ground-truth + Name genotype)
    3.  Build feature matrix
    4.  Train Ridge + RandomForest [+ XGBoost]
    5.  Leave-One-Trial-Out (or k-fold) cross-validation
    6.  Extract per-genotype Ridge formula
    7.  Simulate flight planning advisor
    8.  Save global plots, model.pkl, predictions.xlsx, feature_matrix.xlsx
    9.  (optional) Save per-trial predictions into each trial's ML_Analysis/ folder

    Parameters
    ----------
    trial_roots : list of (output_root, trial_name) tuples, optional
        When provided, the trained best model is applied to each trial folder
        and predictions_{trial_name}.xlsx is written into
        {output_root}/ML_Analysis/.

    Returns the global output directory path.
    """
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is not installed.\n"
            "Run:  pip install scikit-learn\nThen restart the app.")

    def _log(m):
        if log_fn:
            log_fn(m)

    os.makedirs(out_dir, exist_ok=True)

    _log("\n[ML] ════════════════════════════════════════")
    _log("[ML]  Starting ML Analysis Pipeline")
    _log(f"[ML]  Output → {out_dir}")
    _log("[ML] ════════════════════════════════════════")

    # ── 1 & 2. Load database + join, OR use pre-built DataFrame ──────────────
    if _prebuilt_df is not None:
        df_j = _prebuilt_df.copy()
        _log(f"[ML] Using pre-built DataFrame: {len(df_j)} rows, "
             f"{df_j['TrialName'].nunique()} trial(s): "
             f"{sorted(df_j['TrialName'].astype(str).unique().tolist())}")
        if len(df_j) < 5:
            raise ValueError(
                f"Pre-built DataFrame has only {len(df_j)} rows — not enough to train.")
    else:
        # ── 1. Load database ─────────────────────────────────────────────────
        db_df = load_database(db_path)
        _log(f"[ML] Database: {len(db_df)} rows, "
             f"{db_df['TrialName'].nunique()} trial(s): "
             f"{sorted(db_df['TrialName'].astype(str).unique().tolist())}")

        # ── 2. Join with field data ──────────────────────────────────────────
        df_j = join_with_field_data(
            db_df, field_excel,
            plot_id_col=plot_id_col,
            trial_col=trial_col,
            genotype_col=genotype_col,
            mtr_col=mtr_col,
            log_fn=_log)
        if len(df_j) < 5:
            raise ValueError(
                f"Only {len(df_j)} matched rows after joining with field data.\n"
                "Check that PlotID values and Trial/Experiment Name match "
                "between pipeline output and field Excel.")

    # ── 3. Feature matrix ──────────────────────────────────────────────────────
    X, y, groups, cont_features, genotypes = build_feature_matrix(df_j, log_fn=_log)

    # ── 3b. Outlier removal (IQR on MTR_field target) ─────────────────────────
    _before = len(y)
    q1, q3  = np.nanpercentile(y, 25), np.nanpercentile(y, 75)
    iqr     = q3 - q1
    _lo, _hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    _keep   = (y >= _lo) & (y <= _hi)
    if _keep.sum() < _before:
        _removed = _before - _keep.sum()
        _log(f"[ML] Outlier removal (IQR): removed {_removed} rows "
             f"(MTR_field outside [{_lo:.1f}, {_hi:.1f}]); "
             f"{_keep.sum()} rows remain.")
        X      = X[_keep].reset_index(drop=True)
        y      = y[_keep]
        groups = np.array(groups)[_keep]
        df_j   = df_j[_keep].reset_index(drop=True)
    else:
        _log(f"[ML] Outlier removal: no outliers found (IQR range [{_lo:.1f}, {_hi:.1f}]).")

    # Save merged_trials_locations.xlsx (shows exactly what data feeds the models)
    try:
        _merged_path = os.path.join(out_dir, "merged_trials_locations.xlsx")
        _export = df_j.copy()
        if _export["TrialName"].astype(str).str.contains("/", regex=False).any():
            _tn = _export["TrialName"].astype(str).str.split("/", n=1)
            _export.insert(0, "Dataset",   _tn.str[0].values)
            _export.insert(1, "TrialOnly", _tn.str[1].values)
        _export.to_excel(_merged_path, index=False, engine="xlsxwriter")
        _log(f"[ML] merged_trials_locations.xlsx saved -> {_merged_path}")
        _log(f"[ML]   ({len(_export)} rows, {_export['TrialName'].nunique()} trial(s))")
    except Exception as ex:
        _log(f"[ML] Warning — merged_trials_locations.xlsx: {ex}")

    # Save feature matrix to Excel
    try:
        fm = X.copy()
        fm.insert(0, "PlotID",    df_j["PlotID"].values)
        fm.insert(1, "TrialName", groups)
        fm["MTR_field"] = y
        fm.to_excel(os.path.join(out_dir, "feature_matrix.xlsx"),
                    index=False, engine="xlsxwriter")
        _log("[ML] feature_matrix.xlsx saved.")
    except Exception as ex:
        _log(f"[ML] Warning — feature_matrix.xlsx: {ex}")

    # Correlation heatmap
    try:
        plot_feature_correlation(
            X, y, cont_features,
            os.path.join(out_dir, "feature_correlation.png"))
        _log("[ML] feature_correlation.png saved.")
    except Exception as ex:
        _log(f"[ML] Warning — feature_correlation.png: {ex}")

    # ── 4. Train ───────────────────────────────────────────────────────────────
    models = train_models(X, y, cont_features, log_fn=_log)

    # ── 5. Cross-validate ─────────────────────────────────────────────────────
    cv_results = cross_validate_loto(X, y, groups, models, cont_features, log_fn=_log)

    # ── 6. Ridge formula ───────────────────────────────────────────────────────
    ridge_formulas: Dict[str, str] = {}
    if "Ridge" in models:
        ridge_formulas = get_ridge_formula(
            models["Ridge"], cont_features, genotypes, log_fn=_log)

    # Best model by CV RMSE
    best_name = min(cv_results, key=lambda k: cv_results[k]["rmse"])
    _log(f"[ML] Best base model: {best_name}  "
         f"(RMSE={cv_results[best_name]['rmse']:.2f} d)")

    # ── 7. Flight planning (base models only first; stack added in 7.5) ───────
    fp_results: Dict[str, Dict] = {}
    try:
        fp_results = simulate_flight_plans(
            df_j, y, models, cont_features, log_fn=_log)
    except Exception as ex:
        _log(f"[ML] Warning — flight planning: {ex}")

    # ── 7.5. Multi-Modal Stacking Ensemble ───────────────────────────────────
    stacking_bundle: Dict = {}
    stacking_cv_res: Dict = {}
    mm_dir = os.path.join(out_dir, "multi_modal")

    try:
        os.makedirs(mm_dir, exist_ok=True)
        _log("\n[ML] ─── Multi-Modal Stacking Ensemble ─────────────────────────")

        stacking_cv_res = cross_validate_stacking(
            X, y, groups, models, cont_features, log_fn=_log)

        if stacking_cv_res:
            stacking_bundle = train_stacking_ensemble(
                X, y, groups, models, cont_features, log_fn=_log)

            # ── Stack flight simulation (adds "Stack Model" to fp_results) ──
            try:
                fp_results = simulate_flight_plans(
                    df_j, y, models, cont_features,
                    stacking_bundle=stacking_bundle, log_fn=_log)
            except Exception as _ex:
                _log(f"[ML] Warning — stack flight simulation: {_ex}")

            # ── Stacking CV report ───────────────────────────────────────
            try:
                plot_stacking_report(
                    stacking_cv_res, cv_results,
                    os.path.join(mm_dir, "stacking_cv_report.png"))
                _log("[ML] multi_modal/stacking_cv_report.png saved.")
            except Exception as _ex:
                _log(f"[ML] Warning — stacking_cv_report.png: {_ex}")

            # ── Model comparison bar chart ───────────────────────────────
            try:
                plot_model_comparison(
                    cv_results, stacking_cv_res,
                    os.path.join(mm_dir, "model_comparison.png"))
                _log("[ML] multi_modal/model_comparison.png saved.")
            except Exception as _ex:
                _log(f"[ML] Warning — model_comparison.png: {_ex}")

            # ── Stacking predictions Excel ───────────────────────────────
            try:
                stack_pred_df = df_j[["PlotID", "TrialName", "Name",
                                       "MTR_field"]].copy()

                # Base-model full-data predictions (in-sample)
                for _bn, _pipe in stacking_bundle["base_models"].items():
                    stack_pred_df[f"Base_{_bn}"] = np.round(
                        _pipe.predict(X), 1)

                # CV predictions for each base model (OOF)
                for _mn, _res in cv_results.items():
                    _fp = _res.get("y_pred_full", _res["y_pred"])
                    if len(_fp) == len(stack_pred_df):
                        stack_pred_df[f"CV_{_mn}"] = np.round(_fp, 1)

                # Stacking OOF predictions (y_pred_full = full-length array)
                _stack_full = stacking_cv_res.get("y_pred_full")
                if _stack_full is not None and len(_stack_full) == len(stack_pred_df):
                    stack_pred_df["CV_Stacking"] = np.round(_stack_full, 1)

                # Full-data stacking (train, not OOF)
                _base_preds_np = np.column_stack([
                    stacking_bundle["base_models"][_bn].predict(X)
                    for _bn in stacking_bundle["base_names"]
                ])
                _oof_full_df   = pd.DataFrame(
                    _base_preds_np, columns=stacking_bundle["meta_cols"])
                _meta_df_full  = pd.concat(
                    [X.reset_index(drop=True), _oof_full_df], axis=1)
                _meta_fcols    = stacking_bundle.get(
                    "meta_fcols", list(_meta_df_full.columns))
                _meta_df_full  = _meta_df_full.reindex(
                    columns=_meta_fcols, fill_value=0)
                stack_pred_df["Train_Stacking"] = np.round(
                    stacking_bundle["meta_model"].predict(_meta_df_full), 1)

                stack_pred_df.to_excel(
                    os.path.join(mm_dir, "stacking_predictions.xlsx"),
                    index=False, engine="xlsxwriter")
                _log("[ML] multi_modal/stacking_predictions.xlsx saved.")
            except Exception as _ex:
                _log(f"[ML] Warning — stacking_predictions.xlsx: {_ex}")

            # ── Save stack_model.pkl ─────────────────────────────────────
            try:
                stack_pkl_path = os.path.join(mm_dir, "stack_model.pkl")
                stack_pkl_bundle = {
                    "stacking_bundle":       stacking_bundle,
                    "cont_features":         cont_features,
                    "base_model_names":      stacking_bundle.get("base_names", []),
                    "meta_model_name":       stacking_bundle.get("meta_name", ""),
                    "meta_fcols":            stacking_bundle.get("meta_fcols", []),
                    "trained_on":            datetime.now().isoformat(),
                    "stacking_cv_summary":   {kk: vv for kk, vv in stacking_cv_res.items()
                                              if kk not in ("y_true", "y_pred",
                                                            "y_pred_full", "fold_results")},
                }
                with open(stack_pkl_path, "wb") as _fh:
                    pickle.dump(stack_pkl_bundle, _fh)
                _log(f"[ML] multi_modal/stack_model.pkl saved → {stack_pkl_path}")
            except Exception as _ex:
                _log(f"[ML] Warning — stack_model.pkl: {_ex}")

            # ── Log stacking summary ─────────────────────────────────────
            _log("\n[ML] Stacking performance summary:")
            _BASE_LOG_ORDER = ("Ridge", "ElasticNet", "SVR",
                               "RandomForest", "GradientBoosting", "ExtraTrees", "XGBoost")
            for _k in _BASE_LOG_ORDER:
                if _k in cv_results:
                    _log(f"[ML]   {_k:<20s} RMSE={cv_results[_k]['rmse']:.2f} d  "
                         f"R²={cv_results[_k]['r2']:.3f}")
            _log(f"[ML]   {'Stack Model':<20s} RMSE={stacking_cv_res['rmse']:.2f} d  "
                 f"R²={stacking_cv_res['r2']:.3f}")
        else:
            _log("[ML] Stacking skipped (insufficient base models).")

    except Exception as _stk_ex:
        _log(f"[ML] Warning — stacking ensemble: {_stk_ex}")

    # ── Determine overall best model (base models + stacking) ────────────────
    # Compare all CV RMSE values; stacking_cv_res uses key "Stack Model"
    overall_best = best_name   # default = best base model
    if stacking_cv_res:
        stack_rmse = stacking_cv_res["rmse"]
        if stack_rmse < cv_results[best_name]["rmse"]:
            overall_best = "Stack Model"
            _log(f"[ML] Overall best model: Stack Model  (RMSE={stack_rmse:.2f} d)")
        else:
            _log(f"[ML] Overall best model: {best_name}  "
                 f"(RMSE={cv_results[best_name]['rmse']:.2f} d)  "
                 f"[Stack RMSE={stack_rmse:.2f} d — no improvement]")
    else:
        _log(f"[ML] Overall best model: {overall_best}  "
             f"(RMSE={cv_results[best_name]['rmse']:.2f} d)")

    # ── 8. Save plots ─────────────────────────────────────────────────────────
    _log("[ML] Saving diagnostic plots ...")

    for fname, fn, args in [
        ("cv_report.png",          plot_cv_report,          (cv_results,)),
        ("feature_importance.png", plot_feature_importance, (models, cont_features, cv_results)),
        ("field_vs_predicted.png", plot_field_vs_predicted, (cv_results, df_j)),
    ]:
        try:
            fn(*args, os.path.join(out_dir, fname))
            _log(f"[ML] {fname} saved.")
        except Exception as ex:
            _log(f"[ML] Warning — {fname}: {ex}")

    if ridge_formulas:
        try:
            plot_genotype_formulas(
                ridge_formulas, cv_results,
                os.path.join(out_dir, "genotype_formulas.png"))
            _log("[ML] genotype_formulas.png saved.")
        except Exception as ex:
            _log(f"[ML] Warning — genotype_formulas.png: {ex}")

    # ── Cumulative feature importance (base + stack) ──────────────────────────
    try:
        plot_cumulative_feature_importance(
            models, X, cont_features, out_dir,
            stacking_bundle=stacking_bundle if stacking_bundle else None,
            log_fn=_log)
    except Exception as ex:
        _log(f"[ML] Warning — cumulative feature importance: {ex}")

    # ── Bias–variance decomposition ────────────────────────────────────────────
    try:
        plot_bias_variance_decomposition(
            cv_results, stacking_cv_res,
            os.path.join(out_dir, "bias_variance_decomposition.png"),
            log_fn=_log)
    except Exception as ex:
        _log(f"[ML] Warning — bias_variance_decomposition.png: {ex}")

    # ── Trial comparison box-plot ──────────────────────────────────────────────
    try:
        plot_trial_comparison(
            df_j, y, groups, cv_results, stacking_cv_res, overall_best,
            os.path.join(out_dir, "trial_comparison.png"))
        _log("[ML] trial_comparison.png saved.")
    except Exception as ex:
        _log(f"[ML] Warning — trial_comparison.png: {ex}")

    if fp_results:
        try:
            # Show the best model (Stack Model if it won, else best base model)
            fp_best_name = (overall_best if overall_best in fp_results
                            else best_name if best_name in fp_results
                            else next(iter(fp_results)))
            plot_flight_planning(
                fp_results, fp_best_name,
                os.path.join(out_dir, "flight_planning.png"))
            _log(f"[ML] flight_planning.png saved  (showing: {fp_best_name}).")
        except Exception as ex:
            _log(f"[ML] Warning — flight_planning.png: {ex}")

    # ── Save model bundle ──────────────────────────────────────────────────────
    model_path = os.path.join(out_dir, "model.pkl")
    bundle = {
        "models":            models,
        "cont_features":     cont_features,
        "best_model_name":   overall_best,      # overall winner (base or stack)
        "best_base_model":   best_name,
        "ridge_formulas":    ridge_formulas,
        "genotypes":         genotypes,
        "trained_on":        datetime.now().isoformat(),
        "n_samples":         int(len(y)),
        "n_trials":          int(len(np.unique(groups))),
        "stacking_bundle":   stacking_bundle,   # {} if stacking was skipped
        "cv_summary": {k: {kk: vv for kk, vv in v.items()
                           if kk not in ("y_true", "y_pred",
                                         "y_pred_full", "fold_results")}
                       for k, v in cv_results.items()},
        "stacking_cv_summary": {kk: vv for kk, vv in stacking_cv_res.items()
                                 if kk not in ("y_true", "y_pred",
                                               "y_pred_full", "fold_results")}
                                if stacking_cv_res else {},
    }
    with open(model_path, "wb") as fh:
        pickle.dump(bundle, fh)
    _log(f"[ML] model.pkl saved → {model_path}")

    # ── Save per-model predictions (training predictions) ─────────────────────
    try:
        # Base columns — always present
        _base_cols = ["PlotID", "TrialName", "Name", "MTR_field"]

        # If TrialName contains "/" it came from a multi-dataset run; extract
        # the dataset label into a separate column for easy filtering.
        _tn_series = df_j["TrialName"].astype(str)
        if _tn_series.str.contains("/", regex=False).any():
            _dataset_col = _tn_series.str.split("/", n=1).str[0]
            _trial_col   = _tn_series.str.split("/", n=1).str[1]
            pred_df = df_j[_base_cols].copy()
            pred_df.insert(1, "Dataset",  _dataset_col.values)
            pred_df.insert(2, "TrialOnly", _trial_col.values)
        else:
            pred_df = df_j[_base_cols].copy()

        # Training (in-sample) predictions from each fitted model
        for mname, pipe in models.items():
            try:
                pred_df[f"Train_{mname}"] = np.round(pipe.predict(X), 1)
            except Exception:
                pass

        # CV predictions — use y_pred_full (full-length NaN array)
        for mname, res in cv_results.items():
            _fp = res.get("y_pred_full")
            if _fp is not None and len(_fp) == len(pred_df):
                pred_df[f"CV_{mname}"] = np.round(_fp, 1)

        # Stacking CV predictions
        _stack_full = stacking_cv_res.get("y_pred_full") if stacking_cv_res else None
        if _stack_full is not None and len(_stack_full) == len(pred_df):
            pred_df["CV_Stack"] = np.round(_stack_full, 1)

        pred_df.to_excel(os.path.join(out_dir, "predictions.xlsx"),
                         index=False, engine="xlsxwriter")
        _log("[ML] predictions.xlsx saved.")
    except Exception as ex:
        _log(f"[ML] Warning — predictions.xlsx: {ex}")

    # ── Text summary ──────────────────────────────────────────────────────────
    lines = [
        "=" * 62,
        "ML Analysis Summary",
        "=" * 62,
        f"  Samples  : {len(y)}  plots × trials",
        f"  Trials   : {', '.join(sorted(set(groups)))}",
        f"  Genotypes: {len(genotypes)}",
        f"  Features : {len(cont_features)} continuous + Name categorical",
        "",
    ]
    for mname, res in cv_results.items():
        lines.append(
            f"  {mname:<20s}  RMSE={res['rmse']:5.1f} d  "
            f"MAE={res['mae']:5.1f} d  R²={res['r2']:.3f}  "
            f"bias={res['bias']:+.1f} d")
    if stacking_cv_res:
        lines.append(f"  {'Stack Model':<20s}  RMSE={stacking_cv_res['rmse']:5.1f} d  "
                     f"MAE={stacking_cv_res['mae']:5.1f} d  "
                     f"R²={stacking_cv_res['r2']:.3f}  "
                     f"bias={stacking_cv_res['bias']:+.1f} d")

    lines += ["", f"  Overall best model: {overall_best}", ""]

    _fp_key = (overall_best if overall_best in fp_results
               else best_name if best_name in fp_results else None)
    if fp_results and _fp_key:
        fp = fp_results[_fp_key]
        lines += [
            f"  Flight Planning Recommendation  ({_fp_key}):",
            f"    Start       : DAP {fp.get('rec_start_dap', '?')}",
            f"    Interval    : every {fp['rec_interval']} days",
            f"    Stop after  : {fp['best_n']} flights",
            f"    Expected RMSE: {fp['best_rmse']:.1f} days",
        ]

    if stacking_cv_res:
        lines += [
            "",
            "  Multi-Modal Stacking Ensemble:",
            f"    Base models : {', '.join(stacking_cv_res.get('base_names', []))}",
            f"    Meta-learner: {stacking_cv_res.get('meta_name', 'XGBoost')}",
            f"    Outputs → {mm_dir}",
        ]

    summary_path = os.path.join(out_dir, "ml_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    for line in lines:
        _log(f"[ML]{line}")

    # ── 9. Per-trial full ML outputs ──────────────────────────────────────────
    if trial_roots:
        _log(f"\n[ML] Saving per-trial ML outputs for {len(trial_roots)} trial(s) ...")
        _fp_best = (fp_best_name if fp_results else
                    overall_best if overall_best in fp_results else
                    best_name)
        for tr_root, tr_name in trial_roots:
            _log(f"\n[ML]  ─── Trial: {tr_name} ───")
            try:
                save_trial_ml_outputs(
                    tr_name        = tr_name,
                    tr_root        = tr_root,
                    df_j           = df_j,
                    y              = y,
                    groups         = groups,
                    X              = X,
                    cv_results     = cv_results,
                    stacking_cv_res= stacking_cv_res,
                    models         = models,
                    cont_features  = cont_features,
                    fp_results     = fp_results,
                    fp_best_name   = _fp_best,
                    ridge_formulas = ridge_formulas,
                    stacking_bundle= stacking_bundle,
                    overall_best   = overall_best,
                    best_base_name = best_name,
                    global_out_dir = out_dir,
                    sowing_date    = sowing_date,
                    log_fn         = _log,
                )
            except Exception as ex:
                _log(f"[ML]   ✘ {tr_name}: unexpected error — {ex}")

    _log(f"\n[ML] ✔  Pipeline complete → {out_dir}")
    return out_dir


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 14 — MULTI-DATASET PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_ml_pipeline_multi(datasets: List[Dict],
                           out_dir: str,
                           sowing_date=None,
                           plot_id_col:  str = "PlotID",
                           trial_col:    str = "Experiment Name",
                           genotype_col: str = "Name",
                           mtr_col:      str = "MTR",
                           log_fn=None) -> str:
    """
    Multi-dataset / multi-location / multi-year ML pipeline.

    Each entry in ``datasets`` is a dict with keys:
        label        - str: display name (e.g. "Iowa_2024")
        scan_root    - str: root folder that was scanned for SUMMARY.xlsx
        trial_roots  - list of (output_root, trial_name)
        field_excel  - str: path to this dataset's field-data Excel
        db_path      - str: path to this dataset's training_data.csv

    Workflow
    --------
    1.  Load & join each dataset individually, then combine.
        TrialName is prefixed with "{label}/" to prevent cross-dataset
        name collisions.
    2.  Run run_ml_pipeline() on the combined DataFrame.
        Trial-level output goes to {tr_root}/ML_Analysis/ as usual.
    3.  Create {out_dir}/ML_Analysis_{safe_label}/ per dataset containing
        filtered predictions.xlsx + ml_summary.txt + copies of global plots.

    Returns the global output directory path (same as run_ml_pipeline).
    """

    def _log(m):
        if log_fn:
            log_fn(m)

    def _safe(label: str) -> str:
        return re.sub(r"[^\w\-]", "_", label)

    if not datasets:
        raise ValueError("run_ml_pipeline_multi: no datasets provided.")

    # ── Single-dataset shortcut — delegate directly ────────────────────────────
    if len(datasets) == 1:
        ds = datasets[0]
        _log("[ML] Single dataset — using standard pipeline.")
        return run_ml_pipeline(
            field_excel  = ds["field_excel"],
            db_path      = ds["db_path"],
            out_dir      = out_dir,
            sowing_date  = sowing_date,
            plot_id_col  = plot_id_col,
            trial_col    = trial_col,
            genotype_col = genotype_col,
            mtr_col      = mtr_col,
            trial_roots  = ds["trial_roots"],
            log_fn       = log_fn)

    # ── Multi-dataset: load, prefix, combine ──────────────────────────────────
    _log(f"\n[ML] ════════════════════════════════════════")
    _log(f"[ML]  Multi-Dataset Pipeline  ({len(datasets)} dataset(s))")
    _log(f"[ML] ════════════════════════════════════════")

    frames: List[pd.DataFrame] = []
    all_trial_roots: List[Tuple[str, str]] = []

    for ds in datasets:
        label  = ds["label"]
        _log(f"\n[ML] ── Loading dataset: {label}")

        # Skip datasets with no DB
        db_path = ds.get("db_path", "")
        if not db_path or not os.path.exists(db_path):
            _log(f"[ML]   ✘ DB not found for '{label}': {db_path}  — skipped")
            continue

        field_xl = ds.get("field_excel", "")
        if not field_xl or not os.path.exists(field_xl):
            _log(f"[ML]   ✘ Field Excel not found for '{label}': {field_xl}  — skipped")
            continue

        try:
            db_df = load_database(db_path)
            _log(f"[ML]   DB: {len(db_df)} rows, "
                 f"{db_df['TrialName'].nunique()} trial(s)")
        except Exception as ex:
            _log(f"[ML]   ✘ DB load failed for '{label}': {ex}  — skipped")
            continue

        try:
            df_j = join_with_field_data(
                db_df, field_xl,
                plot_id_col  = plot_id_col,
                trial_col    = trial_col,
                genotype_col = genotype_col,
                mtr_col      = mtr_col,
                log_fn       = _log)
        except Exception as ex:
            _log(f"[ML]   ✘ Join failed for '{label}': {ex}  — skipped")
            continue

        if len(df_j) < 3:
            _log(f"[ML]   ✘ Too few rows ({len(df_j)}) for '{label}'  — skipped")
            continue

        # Prefix TrialName and add Dataset column for easy identification
        df_j = df_j.copy()
        df_j["TrialName"] = label + "/" + df_j["TrialName"].astype(str)
        frames.append(df_j)
        _log(f"[ML]   ✔ {len(df_j)} rows, "
             f"{df_j['TrialName'].nunique()} trial(s)  → prefixed with '{label}/'")

        # Prefix trial_roots names to match prefixed TrialName
        for root, name in ds["trial_roots"]:
            all_trial_roots.append((root, f"{label}/{name}"))

    if not frames:
        raise ValueError(
            "No datasets could be loaded. "
            "Check DB paths, field Excel files, and that trials have been saved.")

    # ── Dataset loading summary table ─────────────────────────────────────────
    _log(f"\n[ML] ─── Dataset loading summary ({'─'*30})")
    skipped = len(datasets) - len(frames)
    for frame in frames:
        lbl   = frame["TrialName"].str.split("/", n=1).str[0].iloc[0]
        tns   = sorted(frame["TrialName"].unique().tolist())
        _log(f"[ML]   ✔  {lbl:<20s}  {len(frame):>5} rows  "
             f"  {len(tns)} trial(s): {tns}")
    if skipped:
        _log(f"[ML]   ⚠  {skipped} dataset(s) SKIPPED (check log above for details)")
    _log(f"[ML] {'─'*50}")

    # ── Align columns across all frames (fill missing with NaN) ──────────────
    # Different datasets may have different feature columns available.
    # Taking the union ensures pd.concat produces a consistent structure.
    all_cols = list(dict.fromkeys(col for f in frames for col in f.columns))
    frames_aligned = [f.reindex(columns=all_cols) for f in frames]

    df_combined = pd.concat(frames_aligned, ignore_index=True)
    _log(f"\n[ML] Combined total : {len(df_combined)} rows  |  "
         f"{df_combined['TrialName'].nunique()} unique trials  |  "
         f"{len(frames)} of {len(datasets)} dataset(s) loaded")
    _log(f"[ML] Combined columns: {len(all_cols)}")

    # ── Save merged_trials_locations.xlsx ─────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    _merged_path = os.path.join(out_dir, "merged_trials_locations.xlsx")
    try:
        _export = df_combined.copy()
        # Split the "Label/TrialName" into two readable columns at the front
        if _export["TrialName"].astype(str).str.contains("/", regex=False).any():
            _tn = _export["TrialName"].astype(str).str.split("/", n=1)
            _export.insert(0, "Dataset",   _tn.str[0].values)
            _export.insert(1, "TrialOnly", _tn.str[1].values)
        _export.to_excel(_merged_path, index=False, engine="xlsxwriter")
        _log(f"[ML] merged_trials_locations.xlsx saved -> {_merged_path}")
        _log(f"[ML]   ({len(_export)} rows, {_export['TrialName'].nunique()} trials)")
    except Exception as _ex:
        _log(f"[ML] Warning — merged_trials_locations.xlsx: {_ex}")

    # ── Run the full ML pipeline on combined data ──────────────────────────────
    result_dir = run_ml_pipeline(
        field_excel  = "",            # bypassed — using _prebuilt_df
        db_path      = "",            # bypassed — using _prebuilt_df
        out_dir      = out_dir,
        sowing_date  = sowing_date,
        plot_id_col  = plot_id_col,
        trial_col    = trial_col,
        genotype_col = genotype_col,
        mtr_col      = mtr_col,
        trial_roots  = all_trial_roots,
        _prebuilt_df = df_combined,
        log_fn       = log_fn)

    # ── Per-dataset output sub-folders ────────────────────────────────────────
    _log(f"\n[ML] Creating per-dataset output folders ...")

    # Global files to copy into each dataset folder
    _GLOBAL_PLOTS = [
        "cv_report.png", "feature_importance.png", "field_vs_predicted.png",
        "trial_comparison.png", "flight_planning.png",
        "feature_correlation.png", "genotype_formulas.png",
        "ml_summary.txt",
    ]

    # Load global predictions.xlsx once
    global_pred_path = os.path.join(result_dir, "predictions.xlsx")
    global_pred_df: Optional[pd.DataFrame] = None
    try:
        if os.path.exists(global_pred_path):
            global_pred_df = pd.read_excel(global_pred_path, engine="openpyxl")
    except Exception as ex:
        _log(f"[ML] Warning — could not read global predictions.xlsx: {ex}")

    for ds in datasets:
        label  = ds["label"]
        slabel = _safe(label)
        prefix = label + "/"
        ds_out = os.path.join(out_dir, f"ML_Analysis_{slabel}")
        os.makedirs(ds_out, exist_ok=True)

        # Copy global plots / summary
        for fname in _GLOBAL_PLOTS:
            src = os.path.join(result_dir, fname)
            if os.path.exists(src):
                try:
                    shutil.copy2(src, os.path.join(ds_out, fname))
                except Exception:
                    pass

        # Filtered predictions.xlsx  (strip the "{label}/" prefix from TrialName)
        if global_pred_df is not None:
            try:
                mask = global_pred_df["TrialName"].astype(str).str.startswith(prefix)
                ds_pred = global_pred_df[mask].copy()
                ds_pred["TrialName"] = (
                    ds_pred["TrialName"].astype(str).str[len(prefix):]
                )
                ds_pred.to_excel(os.path.join(ds_out, "predictions.xlsx"),
                                 index=False, engine="xlsxwriter")
            except Exception as ex:
                _log(f"[ML]   Warning — filtered predictions for '{label}': {ex}")

        _log(f"[ML]   ✔ {label} → {ds_out}")

    _log(f"\n[ML] ✔  Multi-dataset pipeline complete → {result_dir}")
    return result_dir
