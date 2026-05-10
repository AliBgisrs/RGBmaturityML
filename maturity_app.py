# ======================================================================
# maturity_app.py  –  Desktop GUI for RGB-based crop maturity analysis
# Developed by Aliasghar Bazrafkan  |  bazrafka@msu.edu
# ======================================================================

import os
import re
import sys
import subprocess
import threading
from datetime import datetime
from tkinter import (
    Tk, Frame, Label, Entry, Button, StringVar, IntVar,
    Radiobutton, scrolledtext, filedialog, messagebox, END, DISABLED, NORMAL,
    Canvas, Listbox
)
from tkinter import ttk

# Import analysis module (same directory as this script)
try:
    from analysis import run_pipeline, METHOD_NAMES, METHODS, generate_field_comparison
except ImportError as _e:
    import tkinter as _tk
    _r = _tk.Tk(); _r.withdraw()
    messagebox.showerror("Import Error",
        f"Cannot import analysis.py:\n{_e}\nMake sure analysis.py is in the same folder.")
    sys.exit(1)

# Import ML module (optional — graceful fallback if scikit-learn not installed)
try:
    import ml_analysis as _ml
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

# ────────────────────────────── COLOURS ──────────────────────────────
BG         = "#0d1b2a"   # window background
PANEL_BG   = "#1a2e45"   # input panel
ENTRY_BG   = "#0d1b2a"
ENTRY_FG   = "#e0eaff"
LABEL_FG   = "#a8c7fa"
BTN_RUN    = "#22c55e"   # green
BTN_OPEN   = "#f59e0b"   # amber
BTN_FG     = "#ffffff"
LOG_BG     = "#060d17"
LOG_FG     = "#86efac"   # soft green text
BORDER     = "#2a4a6e"
ACCENT     = "#4a9fd5"
# ─────────────────────────────────────────────────────────────────────

OUTPUT_BASE = os.path.abspath("./Output")


def _fiona_layers(path: str):
    """Return list of layer names inside a .gdb / .shp via fiona."""
    try:
        import fiona
        return fiona.listlayers(path)
    except Exception:
        try:
            import geopandas as gpd
            return [gpd.io.file.fiona.listlayers(path)]
        except Exception:
            return []


class MaturityApp(Tk):
    def __init__(self):
        super().__init__()
        self.title("HMI Precision Agriculture  v3.0 — Desktop Edition")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.geometry("1160x880")
        self.minsize(900, 620)

        self._out_dir: str = ""
        self._running = False
        self._sowing_date = None
        self._field_dap_min = None
        self._field_dap_max = None
        self._ml_model_path: str = ""
        # List of dataset dicts: {label, scan_root, trial_roots, field_excel, db_path}
        self._datasets: list = []

        self._build_ui()

    # ─────────────────────────── UI BUILD ────────────────────────────
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        # ── Left panel — scrollable container ────────────────────────
        left_outer = Frame(self, bg=PANEL_BG, bd=0, highlightthickness=1,
                           highlightbackground=BORDER)
        left_outer.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        left_outer.rowconfigure(0, weight=1)
        left_outer.columnconfigure(0, weight=1)

        # Canvas + vertical scrollbar inside the outer frame
        _canvas = Canvas(left_outer, bg=PANEL_BG, bd=0,
                         highlightthickness=0)
        _vsb = ttk.Scrollbar(left_outer, orient="vertical",
                              command=_canvas.yview)
        _canvas.configure(yscrollcommand=_vsb.set)
        _vsb.grid(row=0, column=1, sticky="ns")
        _canvas.grid(row=0, column=0, sticky="nsew")

        # Inner frame that holds all the actual widgets
        left = Frame(_canvas, bg=PANEL_BG)
        left.columnconfigure(1, weight=1)
        _canvas_window = _canvas.create_window((0, 0), window=left,
                                                anchor="nw")

        # Resize the scroll region whenever inner frame changes size
        def _on_inner_configure(event):
            _canvas.configure(scrollregion=_canvas.bbox("all"))
        left.bind("<Configure>", _on_inner_configure)

        # Stretch inner frame to canvas width
        def _on_canvas_configure(event):
            _canvas.itemconfig(_canvas_window, width=event.width)
        _canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse-wheel scrolling (Windows + Linux)
        def _on_mousewheel(event):
            _canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _on_mousewheel_linux(event):
            _canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
        left_outer.bind_all("<MouseWheel>",     _on_mousewheel)
        left_outer.bind_all("<Button-4>",       _on_mousewheel_linux)
        left_outer.bind_all("<Button-5>",       _on_mousewheel_linux)

        # Title
        title_f = Frame(left, bg=PANEL_BG)
        title_f.grid(row=0, column=0, columnspan=3, sticky="ew", padx=14, pady=(14, 8))
        Label(title_f, text="🌾  HMI Precision Agriculture",
              font=("Segoe UI", 15, "bold"),
              bg=PANEL_BG, fg="#e2e8f0").pack(side="left")
        Label(title_f, text="v3.0",
              font=("Segoe UI", 10), bg=PANEL_BG, fg="#64748b").pack(side="left", padx=(8, 0))

        sep = Frame(left, bg=BORDER, height=1)
        sep.grid(row=1, column=0, columnspan=3, sticky="ew", padx=14, pady=4)

        def _lbl(parent, text, row, col=0, **kw):
            Label(parent, text=text, font=("Segoe UI", 11),
                  bg=PANEL_BG, fg=LABEL_FG, anchor="w", **kw
                  ).grid(row=row, column=col, sticky="w", padx=(14, 4), pady=6)

        def _entry(parent, var, row, col=1):
            e = Entry(parent, textvariable=var, font=("Consolas", 10),
                      bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=ENTRY_FG,
                      relief="flat", bd=0, highlightthickness=1,
                      highlightbackground=BORDER, highlightcolor=ACCENT)
            e.grid(row=row, column=col, sticky="ew", padx=4, pady=6, ipady=5)
            return e

        def _browse_btn(parent, row, cmd):
            b = Button(parent, text="Browse", font=("Segoe UI", 9),
                       bg="#2a4a6e", fg="#e0eaff", activebackground="#3a5a8e",
                       activeforeground="#ffffff", relief="flat", bd=0, cursor="hand2",
                       command=cmd, padx=8, pady=4)
            b.grid(row=row, column=2, padx=(4, 14), pady=6)
            return b

        # Images directory
        self.var_imgs = StringVar()
        _lbl(left, "Images Directory", 2)
        _entry(left, self.var_imgs, 2)
        _browse_btn(left, 2, self._browse_images)

        # Vector format
        _lbl(left, "Vector Format", 3)
        fmt_f = Frame(left, bg=PANEL_BG)
        fmt_f.grid(row=3, column=1, columnspan=2, sticky="w", padx=4, pady=4)
        self.var_fmt = IntVar(value=1)
        for val, txt in ((0, "Shapefile (.shp)"), (1, "File GDB (.gdb)")):
            Radiobutton(fmt_f, text=txt, variable=self.var_fmt, value=val,
                        command=self._on_fmt_change,
                        font=("Segoe UI", 10), bg=PANEL_BG, fg=ENTRY_FG,
                        selectcolor=PANEL_BG, activebackground=PANEL_BG,
                        activeforeground=ENTRY_FG).pack(side="left", padx=(0, 12))

        # Vector path
        self.var_vec = StringVar()
        _lbl(left, "Vector Path", 4)
        _entry(left, self.var_vec, 4)
        _browse_btn(left, 4, self._browse_vector)

        # Layer name
        _lbl(left, "Layer Name (GDB)", 5)
        layer_f = Frame(left, bg=PANEL_BG)
        layer_f.grid(row=5, column=1, sticky="ew", padx=4, pady=6)
        layer_f.columnconfigure(0, weight=1)
        self.var_layer = StringVar()
        self.cmb_layer = ttk.Combobox(layer_f, textvariable=self.var_layer,
                                      font=("Consolas", 10), state="normal")
        self.cmb_layer.grid(row=0, column=0, sticky="ew", ipady=4)
        self._style_combobox()
        detect_btn = Button(left, text="⟳ Detect",
                            font=("Segoe UI", 9), bg="#2a4a6e", fg="#e0eaff",
                            activebackground="#3a5a8e", relief="flat", bd=0,
                            cursor="hand2", command=self._detect_layers, padx=8, pady=4)
        detect_btn.grid(row=5, column=2, padx=(4, 14), pady=6)

        # Sowing date
        self.var_sow = StringVar(value="06_03_2025")
        _lbl(left, "Sowing Date\n(MM_DD_YYYY)", 6)
        _entry(left, self.var_sow, 6)

        # Plot ID field
        self.var_pid_field = StringVar(value="PlotID")
        _lbl(left, "Plot ID Field", 7)
        pid_f = Frame(left, bg=PANEL_BG)
        pid_f.grid(row=7, column=1, sticky="ew", padx=4, pady=6)
        pid_f.columnconfigure(0, weight=1)
        Entry(pid_f, textvariable=self.var_pid_field,
              font=("Consolas", 10), bg=ENTRY_BG, fg=ENTRY_FG,
              insertbackground=ENTRY_FG, relief="flat", bd=0,
              highlightthickness=1, highlightbackground=BORDER,
              highlightcolor=ACCENT).grid(row=0, column=0, sticky="ew", ipady=5)
        Label(pid_f, text="(exact or case-insensitive)",
              font=("Segoe UI", 8), bg=PANEL_BG, fg="#64748b"
              ).grid(row=1, column=0, sticky="w")
        # Preview button — shows first 5 rows of the chosen layer
        Button(left, text="Preview", font=("Segoe UI", 9),
               bg="#2a4a6e", fg="#e0eaff", activebackground="#3a5a8e",
               relief="flat", bd=0, cursor="hand2",
               command=self._preview_layer, padx=8, pady=4
               ).grid(row=7, column=2, padx=(4, 14), pady=6)

        # Output directory
        self.var_out = StringVar(value=OUTPUT_BASE)
        _lbl(left, "Output Directory", 8)
        _entry(left, self.var_out, 8)
        _browse_btn(left, 8, self._browse_output)

        sep2 = Frame(left, bg=BORDER, height=1)
        sep2.grid(row=9, column=0, columnspan=3, sticky="ew", padx=14, pady=4)

        # ── Field maturity DAP range (optional) ─────────────────────
        _lbl(left, "Field Maturity DAP\n(Min  –  Max)", 10)
        field_frame = Frame(left, bg=PANEL_BG)
        field_frame.grid(row=10, column=1, columnspan=2, sticky="ew",
                         padx=4, pady=4)
        field_frame.columnconfigure(1, weight=1)
        field_frame.columnconfigure(3, weight=1)

        self.var_field_min = StringVar(value="")
        self.var_field_max = StringVar(value="")

        Label(field_frame, text="Min:", font=("Segoe UI", 10),
              bg=PANEL_BG, fg=LABEL_FG).grid(row=0, column=0, sticky="w")
        Entry(field_frame, textvariable=self.var_field_min,
              font=("Consolas", 10), bg=ENTRY_BG, fg=ENTRY_FG,
              insertbackground=ENTRY_FG, relief="flat", bd=0,
              highlightthickness=1, highlightbackground=BORDER,
              highlightcolor=ACCENT, width=7
              ).grid(row=0, column=1, sticky="ew", padx=(4, 8), ipady=4)

        Label(field_frame, text="Max:", font=("Segoe UI", 10),
              bg=PANEL_BG, fg=LABEL_FG).grid(row=0, column=2, sticky="w")
        Entry(field_frame, textvariable=self.var_field_max,
              font=("Consolas", 10), bg=ENTRY_BG, fg=ENTRY_FG,
              insertbackground=ENTRY_FG, relief="flat", bd=0,
              highlightthickness=1, highlightbackground=BORDER,
              highlightcolor=ACCENT, width=7
              ).grid(row=0, column=3, sticky="ew", padx=(4, 0), ipady=4)

        # Live label showing computed median
        self.lbl_field_med = Label(
            left, text="(optional — leave blank to skip comparison)",
            font=("Segoe UI", 8, "italic"), bg=PANEL_BG, fg="#64748b")
        self.lbl_field_med.grid(row=11, column=0, columnspan=3,
                                sticky="w", padx=14, pady=(0, 4))

        def _update_field_med(*_):
            try:
                mn = float(self.var_field_min.get())
                mx = float(self.var_field_max.get())
                self.lbl_field_med.config(
                    text=f"Reference median  =  {(mn + mx) / 2:.1f} DAP",
                    fg="#27ae60")
            except ValueError:
                self.lbl_field_med.config(
                    text="(optional — leave blank to skip comparison)",
                    fg="#64748b")

        self.var_field_min.trace_add("write", _update_field_med)
        self.var_field_max.trace_add("write", _update_field_med)

        sep3 = Frame(left, bg=BORDER, height=1)
        sep3.grid(row=12, column=0, columnspan=3, sticky="ew", padx=14, pady=4)

        # RUN button
        self.btn_run = Button(left, text="▶  Run Pipeline",
                              font=("Segoe UI", 13, "bold"),
                              bg=BTN_RUN, fg=BTN_FG,
                              activebackground="#16a34a", activeforeground=BTN_FG,
                              relief="flat", bd=0, cursor="hand2",
                              command=self._run, padx=12, pady=10)
        self.btn_run.grid(row=13, column=0, columnspan=3,
                          sticky="ew", padx=14, pady=(4, 6))

        # Progress
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Green.Horizontal.TProgressbar",
                         troughcolor=LOG_BG, background=BTN_RUN, borderwidth=0)
        self.progress = ttk.Progressbar(left, style="Green.Horizontal.TProgressbar",
                                        orient="horizontal", mode="determinate",
                                        maximum=100)
        self.progress.grid(row=14, column=0, columnspan=3,
                           sticky="ew", padx=14, pady=(0, 4))
        self.lbl_prog = Label(left, text="", font=("Segoe UI", 9),
                              bg=PANEL_BG, fg=LABEL_FG)
        self.lbl_prog.grid(row=15, column=0, columnspan=3, pady=(0, 4))

        # Open output
        self.btn_open = Button(left, text="📂  Open Output Folder",
                               font=("Segoe UI", 11, "bold"),
                               bg=BTN_OPEN, fg=BTN_FG,
                               activebackground="#d97706",
                               relief="flat", bd=0, cursor="hand2",
                               command=self._open_output,
                               padx=12, pady=8, state=DISABLED)
        self.btn_open.grid(row=16, column=0, columnspan=3,
                           sticky="ew", padx=14, pady=(4, 6))

        # Field vs Prediction comparison button (re-run with different range)
        self.btn_compare = Button(
            left,
            text="📊  Re-run Field Comparison",
            font=("Segoe UI", 11, "bold"),
            bg="#6d28d9", fg=BTN_FG,
            activebackground="#5b21b6", activeforeground=BTN_FG,
            relief="flat", bd=0, cursor="hand2",
            command=self._show_field_comparison_dialog,
            padx=12, pady=8, state=DISABLED)
        self.btn_compare.grid(row=17, column=0, columnspan=3,
                              sticky="ew", padx=14, pady=(0, 6))

        # ── ML Analysis section ──────────────────────────────────────────
        sep4 = Frame(left, bg="#27ae60", height=1)
        sep4.grid(row=18, column=0, columnspan=3, sticky="ew", padx=14, pady=(6, 2))

        Label(left, text="🤖  ML Maturity Prediction",
              font=("Segoe UI", 10, "bold"),
              bg=PANEL_BG, fg="#4ade80"
              ).grid(row=19, column=0, columnspan=3, sticky="w", padx=14, pady=(2, 0))

        # DB directory (where ML_Database/ will live)
        self.var_ml_db_dir = StringVar(value=OUTPUT_BASE)
        _lbl(left, "ML Database Dir", 20)
        _entry(left, self.var_ml_db_dir, 20)
        _browse_btn(left, 20, self._browse_ml_db_dir)

        # ── Datasets (Location / Year) list ──────────────────────────────
        Label(left, text="Datasets  (Location / Year):",
              font=("Segoe UI", 9, "bold"), bg=PANEL_BG, fg="#94a3b8"
              ).grid(row=21, column=0, columnspan=3,
                     sticky="w", padx=14, pady=(8, 2))

        lbox_frame = Frame(left, bg=PANEL_BG)
        lbox_frame.grid(row=22, column=0, columnspan=3,
                        sticky="ew", padx=14, pady=(0, 2))
        lbox_frame.columnconfigure(0, weight=1)

        self.trial_listbox = Listbox(
            lbox_frame,
            font=("Consolas", 8),
            bg=ENTRY_BG, fg=ENTRY_FG,
            selectbackground=ACCENT, selectforeground=BTN_FG,
            highlightthickness=1, highlightbackground=BORDER,
            relief="flat", height=5,
            exportselection=False)
        self.trial_listbox.grid(row=0, column=0, sticky="ew")

        _lb_sb = ttk.Scrollbar(lbox_frame, orient="vertical",
                                command=self.trial_listbox.yview)
        _lb_sb.grid(row=0, column=1, sticky="ns")
        self.trial_listbox.configure(yscrollcommand=_lb_sb.set)

        lb_btn_row = Frame(left, bg=PANEL_BG)
        lb_btn_row.grid(row=23, column=0, columnspan=3,
                        sticky="ew", padx=14, pady=(0, 6))

        Button(lb_btn_row, text="➕  Add Location/Year",
               font=("Segoe UI", 9),
               bg="#2a4a6e", fg="#e0eaff",
               activebackground="#3a5a8e", relief="flat", bd=0,
               cursor="hand2", command=self._add_dataset,
               padx=8, pady=4).pack(side="left", padx=(0, 6))

        Button(lb_btn_row, text="✖  Remove Selected",
               font=("Segoe UI", 9),
               bg="#450a0a", fg="#fca5a5",
               activebackground="#7f1d1d", relief="flat", bd=0,
               cursor="hand2", command=self._remove_dataset,
               padx=8, pady=4).pack(side="left")

        # ── Field Excel column names (configurable) ──────────────────────
        Label(left, text="Field Excel Columns",
              font=("Segoe UI", 9, "bold"), bg=PANEL_BG, fg="#94a3b8"
              ).grid(row=24, column=0, columnspan=3,
                     sticky="w", padx=14, pady=(6, 0))

        def _mini_row(parent, row, label_text, var, hint=""):
            """Compact label + short entry on one row."""
            Label(parent, text=label_text, font=("Segoe UI", 9),
                  bg=PANEL_BG, fg=LABEL_FG, anchor="w"
                  ).grid(row=row, column=0, sticky="w", padx=(14, 4), pady=2)
            ef = Frame(parent, bg=PANEL_BG)
            ef.grid(row=row, column=1, columnspan=2, sticky="ew", padx=4, pady=2)
            ef.columnconfigure(0, weight=1)
            Entry(ef, textvariable=var, font=("Consolas", 9),
                  bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=ENTRY_FG,
                  relief="flat", bd=0, highlightthickness=1,
                  highlightbackground=BORDER, highlightcolor=ACCENT
                  ).grid(row=0, column=0, sticky="ew", ipady=3)
            if hint:
                Label(ef, text=hint, font=("Segoe UI", 7),
                      bg=PANEL_BG, fg="#64748b"
                      ).grid(row=1, column=0, sticky="w")

        self.var_ml_pid_col   = StringVar(value="PlotID")
        self.var_ml_trial_col = StringVar(value="Experiment Name")
        self.var_ml_mtr_col   = StringVar(value="MTR")
        self.var_ml_geno_col  = StringVar(value="Name")

        _mini_row(left, 25, "Plot ID col",    self.var_ml_pid_col,
                  "column in field Excel that holds the plot number")
        _mini_row(left, 26, "Trial/Exp col",  self.var_ml_trial_col,
                  "trial or experiment name column")
        _mini_row(left, 27, "MTR col",        self.var_ml_mtr_col,
                  "maturity DAP column (the training target)")
        _mini_row(left, 28, "Genotype col",   self.var_ml_geno_col,
                  "cultivar / variety name column")

        # Save all trials to DB
        self.btn_save_db = Button(
            left,
            text="💾  Save All Trials to ML Database",
            font=("Segoe UI", 10, "bold"),
            bg="#0f766e", fg=BTN_FG,
            activebackground="#0d9488", activeforeground=BTN_FG,
            relief="flat", bd=0, cursor="hand2",
            command=self._save_to_ml_db,
            padx=10, pady=7, state=DISABLED)
        self.btn_save_db.grid(row=29, column=0, columnspan=3,
                              sticky="ew", padx=14, pady=(8, 3))

        # Train & Analyze
        self.btn_ml_train = Button(
            left,
            text="🧠  Load Field Data & Train Models",
            font=("Segoe UI", 10, "bold"),
            bg="#7c3aed", fg=BTN_FG,
            activebackground="#6d28d9", activeforeground=BTN_FG,
            relief="flat", bd=0, cursor="hand2",
            command=self._run_ml_analysis,
            padx=10, pady=7,
            state=DISABLED if not _ML_AVAILABLE else NORMAL)
        self.btn_ml_train.grid(row=30, column=0, columnspan=3,
                               sticky="ew", padx=14, pady=(0, 3))

        # Predict all trials
        self.btn_ml_predict = Button(
            left,
            text="🔮  Predict All Trials",
            font=("Segoe UI", 10, "bold"),
            bg="#b45309", fg=BTN_FG,
            activebackground="#92400e", activeforeground=BTN_FG,
            relief="flat", bd=0, cursor="hand2",
            command=self._predict_trial,
            padx=10, pady=7, state=DISABLED)
        self.btn_ml_predict.grid(row=31, column=0, columnspan=3,
                                 sticky="ew", padx=14, pady=(0, 4))

        if not _ML_AVAILABLE:
            Label(left,
                  text="⚠ scikit-learn not installed — ML disabled\n"
                       "  pip install scikit-learn",
                  font=("Segoe UI", 8), bg=PANEL_BG, fg="#f59e0b",
                  justify="left"
                  ).grid(row=32, column=0, columnspan=3,
                         sticky="w", padx=14, pady=(0, 4))

        # Footer
        Label(left, text="Developed by  Aliasghar Bazrafkan  |  bazrafka@msu.edu",
              font=("Segoe UI", 8), bg=PANEL_BG, fg="#475569"
              ).grid(row=33, column=0, columnspan=3, pady=(0, 8))

        # ── Right panel (log) ─────────────────────────────────────────
        right = Frame(self, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        Label(right, text="Run Log", font=("Segoe UI", 12, "bold"),
              bg=BG, fg=LABEL_FG, anchor="w"
              ).grid(row=0, column=0, sticky="w", pady=(6, 4))

        self.log_box = scrolledtext.ScrolledText(
            right, font=("Consolas", 10), bg=LOG_BG, fg=LOG_FG,
            insertbackground=LOG_FG, relief="flat", bd=0,
            highlightthickness=1, highlightbackground=BORDER,
            state=DISABLED, wrap="word"
        )
        self.log_box.grid(row=1, column=0, sticky="nsew")

        # Method list summary
        Label(right, text=f"Methods included: {len(METHOD_NAMES)}",
              font=("Segoe UI", 9), bg=BG, fg="#64748b", anchor="w"
              ).grid(row=2, column=0, sticky="w", pady=(4, 0))
        mlist = "  •  ".join([METHODS[m]["label"] for m in METHOD_NAMES])
        Label(right, text=mlist, font=("Segoe UI", 8),
              bg=BG, fg="#475569", anchor="w", wraplength=680, justify="left"
              ).grid(row=3, column=0, sticky="w", pady=(0, 6))

    def _style_combobox(self):
        style = ttk.Style()
        style.configure("TCombobox",
                         fieldbackground=ENTRY_BG, background=PANEL_BG,
                         foreground=ENTRY_FG, selectbackground=ACCENT,
                         selectforeground=BTN_FG, borderwidth=1)

    # ─────────────────────── CALLBACKS ───────────────────────────────
    def _on_fmt_change(self):
        is_gdb = bool(self.var_fmt.get())
        self.cmb_layer.config(state="normal" if is_gdb else "disabled")

    def _browse_images(self):
        d = filedialog.askdirectory(title="Select Images Root Directory")
        if d:
            self.var_imgs.set(d)

    def _browse_vector(self):
        if self.var_fmt.get() == 1:
            path = filedialog.askdirectory(title="Select File GDB (.gdb) Directory")
            if path and path.lower().endswith(".gdb"):
                self.var_vec.set(path)
                self._detect_layers()
            elif path:
                self.var_vec.set(path)
                self._detect_layers()
        else:
            path = filedialog.askopenfilename(
                title="Select Shapefile",
                filetypes=[("Shapefile", "*.shp"), ("All Files", "*.*")]
            )
            if path:
                self.var_vec.set(path)

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            self.var_out.set(d)

    def _detect_layers(self):
        vec = self.var_vec.get().strip()
        if not vec:
            messagebox.showwarning("No path", "Enter the vector path first.")
            return
        layers = _fiona_layers(vec)
        if layers:
            self.cmb_layer["values"] = layers
            self.cmb_layer.current(0)
            self._log(f"Layers detected: {', '.join(layers)}")
        else:
            self._log("Could not detect layers — enter layer name manually.")

    # ─────────────────────── PREVIEW LAYER ──────────────────────────
    def _preview_layer(self):
        """Read the vector layer and dump column names + first 5 rows to the log."""
        vec   = self.var_vec.get().strip()
        layer = self.var_layer.get().strip()
        if not vec or not os.path.exists(vec):
            messagebox.showwarning("No vector", "Set the Vector Path first.")
            return
        self._log("--- Layer Preview ---")
        def _do():
            try:
                import geopandas as gpd

                def _try_read(engine_name):
                    kw = {"engine": engine_name} if engine_name else {}
                    if vec.lower().endswith(".gdb") and layer:
                        return gpd.read_file(vec, layer=layer, **kw)
                    return gpd.read_file(vec, **kw)

                # Same order as the pipeline — NO pre-import of pyogrio
                def _try_read_drv(driver=None, engine=None):
                    kw = {}
                    if driver:  kw["driver"] = driver
                    if engine:  kw["engine"] = engine
                    if vec.lower().endswith(".gdb") and layer:
                        return gpd.read_file(vec, layer=layer, **kw)
                    return gpd.read_file(vec, **kw)

                attempts = [
                    ("driver=OpenFileGDB", dict(driver="OpenFileGDB")),
                    ("engine=pyogrio",     dict(engine="pyogrio")),
                    ("driver=FileGDB",     dict(driver="FileGDB")),
                    ("fiona default",      {}),
                ]
                gdf = None
                for lbl, kw in attempts:
                    try:
                        _g = _try_read_drv(**kw)
                        if _g is not None and len(_g) > 0:
                            gdf = _g
                            self._log(f"Engine     : {lbl}  ({len(gdf)} features)")
                            break
                        else:
                            self._log(f"  {lbl}: 0 features — trying next")
                    except Exception as e:
                        self._log(f"  {lbl}: {e}")

                if gdf is None or len(gdf) == 0:
                    self._log("  All engines returned 0 features.")
                    self._log("  Try: pip install pyogrio --force-reinstall")
                    self._log("--- End Preview ---")
                    return

                gdf.columns = [c.strip() for c in gdf.columns]
                dcols = [c for c in gdf.columns if c != "geometry"]
                self._log(f"Columns    : {dcols}")
                preview = gdf[dcols].head(5).to_string(index=False, max_colwidth=18)
                self._log(f"First 5 rows:\n{preview}")
                self._log("--- End Preview ---")
            except Exception as ex:
                self._log(f"Preview error: {ex}")
        threading.Thread(target=_do, daemon=True).start()

    # ─────────────────────── RUN ─────────────────────────────────────
    def _run(self):
        if self._running:
            return

        imgs     = self.var_imgs.get().strip()
        vec      = self.var_vec.get().strip()
        layer    = self.var_layer.get().strip()
        sow      = self.var_sow.get().strip()
        pid_fld  = self.var_pid_field.get().strip() or "PlotID"
        outd     = self.var_out.get().strip() or OUTPUT_BASE

        # Validate
        if not imgs or not os.path.isdir(imgs):
            messagebox.showerror("Error", "Images directory not found.")
            return
        if not vec or not os.path.exists(vec):
            messagebox.showerror("Error", "Vector path not found.")
            return
        if vec.lower().endswith(".gdb") and not layer:
            messagebox.showerror("Error", "Layer name is required for .gdb files.")
            return
        try:
            sowing_date = datetime.strptime(sow, "%m_%d_%Y")
        except ValueError:
            messagebox.showerror("Error", "Sowing date must be MM_DD_YYYY  (e.g. 06_03_2025).")
            return

        # Output folder: outd / layer_or_shapename
        layer_label = layer or os.path.splitext(os.path.basename(vec))[0]
        out_root = os.path.join(outd, layer_label)

        # Parse optional field DAP range
        field_dap_min = None
        field_dap_max = None
        try:
            _fmin = self.var_field_min.get().strip()
            _fmax = self.var_field_max.get().strip()
            if _fmin and _fmax:
                field_dap_min = float(_fmin)
                field_dap_max = float(_fmax)
                if field_dap_min >= field_dap_max:
                    messagebox.showerror(
                        "Error",
                        "Field DAP Minimum must be less than Maximum.")
                    return
        except ValueError:
            messagebox.showerror(
                "Error",
                "Field DAP range must be numbers (e.g. 85 and 100).")
            return

        self._out_dir = out_root
        self._sowing_date = sowing_date
        self._field_dap_min = field_dap_min
        self._field_dap_max = field_dap_max
        self._running = True
        self.btn_run.config(state=DISABLED, bg="#166534", text="Running …")
        self.btn_open.config(state=DISABLED)
        self.progress["value"] = 0
        self._log_clear()
        if field_dap_min is not None:
            self._log(f"Field maturity range: {field_dap_min:.0f}–{field_dap_max:.0f} DAP "
                      f"(median = {(field_dap_min + field_dap_max) / 2:.1f})")
        self._log("Pipeline started …")

        thread = threading.Thread(
            target=self._pipeline_thread,
            args=(imgs, vec, layer, sowing_date, out_root, pid_fld,
                  field_dap_min, field_dap_max),
            daemon=True
        )
        thread.start()

    def _pipeline_thread(self, imgs, vec, layer, sowing_date, out_root, pid_fld,
                         field_dap_min=None, field_dap_max=None):
        try:
            run_pipeline(
                images_dir=imgs,
                vector_path=vec,
                layer_name=layer,
                sowing_date=sowing_date,
                output_root=out_root,
                plot_id_field=pid_fld,
                log_fn=self._log,
                progress_fn=self._set_progress,
                field_dap_min=field_dap_min,
                field_dap_max=field_dap_max,
            )
            self.after(0, self._on_done, True, "")
        except Exception as exc:
            self.after(0, self._on_done, False, str(exc))

    def _on_done(self, success: bool, error_msg: str):
        self._running = False
        self.btn_run.config(state=NORMAL, bg=BTN_RUN, text="▶  Run Pipeline")
        self.progress["value"] = 100 if success else 0

        if success:
            self.lbl_prog.config(text="✔  Done!", fg="#22c55e")
            self.btn_open.config(state=NORMAL)
            self.btn_compare.config(state=NORMAL)
            # Auto-add completed trial to the dataset list
            if _ML_AVAILABLE and self._out_dir and os.path.isdir(self._out_dir):
                norm_out = os.path.normpath(self._out_dir)

                # Already tracked in any dataset?
                already_tracked = any(
                    os.path.normpath(r) == norm_out
                    for ds in self._datasets
                    for r, _ in ds["trial_roots"])

                if not already_tracked:
                    trial_name = (self.var_layer.get().strip() or
                                  os.path.splitext(
                                      os.path.basename(self.var_vec.get()))[0])
                    parent_dir = os.path.dirname(norm_out)

                    # Find dataset whose scan_root is a parent of this output dir
                    match_idx = next(
                        (i for i, ds in enumerate(self._datasets)
                         if norm_out.startswith(
                             os.path.normpath(ds["scan_root"]) + os.sep)
                         or os.path.normpath(ds["scan_root"]) == norm_out),
                        None)

                    if match_idx is not None:
                        # Add trial to the matching dataset
                        ds = self._datasets[match_idx]
                        ds["trial_roots"].append((self._out_dir, trial_name))
                        n = len(ds["trial_roots"])
                        field_base = (os.path.basename(ds.get("field_excel", ""))
                                      or "(add field data)")
                        self.trial_listbox.delete(match_idx)
                        self.trial_listbox.insert(
                            match_idx,
                            f"📂 {ds['label']}  [{n} trial(s)]  ←  {field_base}")
                        self._log(
                            f"[ML] Trial auto-added to '{ds['label']}': {trial_name}")
                    else:
                        # No matching dataset — create a new auto-dataset
                        label = os.path.basename(parent_dir) or "AutoDataset"
                        existing_labels = {ds["label"] for ds in self._datasets}
                        base_label, i = label, 1
                        while label in existing_labels:
                            label = f"{base_label}_{i}"; i += 1
                        db_path = self._ml_db_path_for(label)
                        new_ds = {
                            "label":       label,
                            "scan_root":   parent_dir,
                            "trial_roots": [(self._out_dir, trial_name)],
                            "field_excel": "",
                            "db_path":     db_path,
                        }
                        self._datasets.append(new_ds)
                        self.trial_listbox.insert(
                            END,
                            f"📂 {label}  [1 trial]  ←  (add field data)")
                        self._log(
                            f"[ML] Auto-created dataset '{label}': {trial_name}")
                        self._log(
                            "[ML]   Use ➕ Add Location/Year to set field data.")

                if self._datasets:
                    self.btn_save_db.config(state=NORMAL)
            self._log("\n✔  Pipeline complete!")
        else:
            self.lbl_prog.config(text="✘  Error", fg="#ef4444")
            self._log(f"\n✘  ERROR: {error_msg}")
            messagebox.showerror("Pipeline Error", error_msg)

    # ─────────────────────── PROGRESS / LOG ──────────────────────────
    def _set_progress(self, pct: float):
        self.after(0, self._update_bar, pct)

    def _update_bar(self, pct: float):
        pct = float(pct)
        self.progress["value"] = pct
        self.lbl_prog.config(text=f"{pct:.0f} %")

    def _log(self, msg: str):
        self.after(0, self._append_log, msg)

    def _append_log(self, msg: str):
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%H:%M:%S")
        self.log_box.config(state=NORMAL)
        self.log_box.insert(END, f"[{ts}] {msg}\n")
        self.log_box.see(END)
        self.log_box.config(state=DISABLED)

    def _log_clear(self):
        self.log_box.config(state=NORMAL)
        self.log_box.delete("1.0", END)
        self.log_box.config(state=DISABLED)

    # ─────────────── RE-RUN FIELD COMPARISON (uses main UI range) ────
    def _show_field_comparison_dialog(self):
        """Re-run the Field vs Prediction comparison using the current
        Field DAP range values already entered in the main UI."""
        try:
            dap_min = float(self.var_field_min.get().strip())
            dap_max = float(self.var_field_max.get().strip())
        except ValueError:
            messagebox.showerror(
                "Error",
                "Please enter valid numbers in the\n"
                "'Field Maturity DAP  Min / Max' fields\n"
                "before using this button.")
            return
        if dap_min >= dap_max:
            messagebox.showerror("Error",
                "Field DAP Minimum must be less than Maximum.")
            return

        summary_xlsx = os.path.join(self._out_dir, "SUMMARY.xlsx")
        if not os.path.exists(summary_xlsx):
            messagebox.showerror(
                "Error",
                f"SUMMARY.xlsx not found in:\n{self._out_dir}\n\n"
                "Run the pipeline first.")
            return

        med = (dap_min + dap_max) / 2.0
        self._log(f"Re-running field comparison  "
                  f"DAP range: {dap_min:.0f}–{dap_max:.0f}  "
                  f"(median = {med:.1f}) ...")

        def _do():
            try:
                out = generate_field_comparison(
                    summary_xlsx, dap_min, dap_max,
                    self._out_dir, self._sowing_date)
                self.after(0, lambda: self._log(
                    f"  ✔ Field comparison saved → {out}"))
                self.after(0, lambda: messagebox.showinfo(
                    "Done",
                    f"Field vs Prediction plots saved in:\n{out}"))
            except Exception as ex:
                self.after(0, lambda: self._log(
                    f"  ✘ Field comparison error: {ex}"))
                self.after(0, lambda: messagebox.showerror("Error", str(ex)))

        threading.Thread(target=_do, daemon=True).start()

    # ─────────────────────── ML CALLBACKS ────────────────────────────
    def _browse_ml_db_dir(self):
        d = filedialog.askdirectory(title="Select ML Database Directory")
        if d:
            self.var_ml_db_dir.set(d)

    def _ml_db_path(self) -> str:
        """Return the legacy single-dataset training_data.csv path (backward compat)."""
        return os.path.join(self.var_ml_db_dir.get().strip() or OUTPUT_BASE,
                            "ML_Database", "training_data.csv")

    def _ml_db_path_for(self, label: str) -> str:
        """Return per-dataset DB path: ML_Database/{safe_label}_training_data.csv."""
        safe = re.sub(r"[^\w\-]", "_", label)
        return os.path.join(self.var_ml_db_dir.get().strip() or OUTPUT_BASE,
                            "ML_Database", f"{safe}_training_data.csv")

    def _ml_out_dir(self) -> str:
        """Return the dated global ML_Analysis_{date}/ folder next to ML_Database/."""
        db_dir     = os.path.dirname(self._ml_db_path())   # …/ML_Database
        parent_dir = os.path.dirname(db_dir)               # user-chosen DB root
        today      = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(parent_dir, f"ML_Analysis_{today}")

    def _add_dataset(self):
        """Select a root folder, scan for SUMMARY.xlsx trials, then pick field Excel.

        Creates one dataset entry representing one location/year.
        Multiple datasets can be added for multi-year / multi-location training.
        """
        root = filedialog.askdirectory(
            title="Select Root Folder (Location / Year) — "
                  "sub-folders with SUMMARY.xlsx will be scanned automatically")
        if not root:
            return

        # Walk the entire tree and collect every directory that holds SUMMARY.xlsx
        found: list = []
        for dirpath, _dirnames, filenames in os.walk(root):
            if "SUMMARY.xlsx" in filenames:
                found.append(dirpath)

        if not found:
            messagebox.showwarning(
                "No Trials Found",
                f"No SUMMARY.xlsx files found inside:\n{root}\n\n"
                "Make sure the pipeline has finished running for at "
                "least one trial in this folder.")
            return

        # Auto-label from the selected folder name
        label = os.path.basename(root.rstrip("/\\")) or "Dataset"

        # Avoid duplicate labels
        existing_labels = {ds["label"] for ds in self._datasets}
        base_label, idx = label, 1
        while label in existing_labels:
            label = f"{base_label}_{idx}"; idx += 1

        # Select field Excel for this dataset
        field_xlsx = filedialog.askopenfilename(
            title=f'Select Field Data Excel for "{label}"',
            filetypes=[("Excel files", "*.xlsx *.xls *.xlsm"),
                       ("All files", "*.*")])
        if not field_xlsx:
            return   # user cancelled — abort adding this dataset

        # Build trial_roots list (sorted for determinism)
        trial_roots = [
            (d, os.path.basename(d.rstrip("/\\")))
            for d in sorted(found)
        ]

        db_path = self._ml_db_path_for(label)

        dataset = {
            "label":       label,
            "scan_root":   root,
            "trial_roots": trial_roots,
            "field_excel": field_xlsx,
            "db_path":     db_path,
        }
        self._datasets.append(dataset)

        n = len(trial_roots)
        field_base = os.path.basename(field_xlsx)
        self.trial_listbox.insert(
            END, f"📂 {label}  [{n} trial(s)]  ←  {field_base}")

        self._log(f"[ML] Dataset added: {label}")
        self._log(f"[ML]   Root       : {root}")
        self._log(f"[ML]   Trials     : {n}")
        self._log(f"[ML]   Field Excel: {field_xlsx}")
        self._log(f"[ML]   DB path    : {db_path}")
        for _, tname in trial_roots:
            self._log(f"[ML]     • {tname}")

        if _ML_AVAILABLE:
            self.btn_save_db.config(state=NORMAL)

        messagebox.showinfo(
            "Dataset Added",
            f"✔  {label}\n"
            f"   {n} trial(s) found\n"
            f"   Field data: {field_base}")

    def _remove_dataset(self):
        """Remove the selected dataset(s) from the list."""
        sel = list(self.trial_listbox.curselection())
        if not sel:
            messagebox.showinfo("Nothing selected",
                                "Click a dataset row first.")
            return
        for i in reversed(sel):
            self.trial_listbox.delete(i)
            del self._datasets[i]
        if not self._datasets:
            self.btn_save_db.config(state=DISABLED)

    def _save_to_ml_db(self):
        """Append all trial folders across all datasets to their per-dataset DBs."""
        if not _ML_AVAILABLE:
            messagebox.showerror("ML Unavailable",
                                 "scikit-learn is required.\n"
                                 "Run:  pip install scikit-learn")
            return
        if not self._datasets:
            messagebox.showerror("No Datasets",
                                 "No datasets in the list.\n"
                                 "Use ➕ Add Location/Year to add data first.")
            return

        total = sum(len(ds["trial_roots"]) for ds in self._datasets)
        self._log(f"[ML] Saving {total} trial(s) across "
                  f"{len(self._datasets)} dataset(s) to ML database(s) ...")

        datasets_snapshot = list(self._datasets)

        def _do():
            try:
                last_db = None
                for ds in datasets_snapshot:
                    label   = ds["label"]
                    db_path = ds["db_path"]
                    last_db = db_path
                    self.after(0, lambda l=label, p=db_path:
                               self._log(f"[ML] Dataset '{l}' → {p}"))
                    for out_root, trial_name in ds["trial_roots"]:
                        _ml.save_trial_to_database(
                            out_root, trial_name, db_path, log_fn=self._log)

                # Show summary for the first available DB
                if last_db and os.path.exists(last_db):
                    summary = _ml.database_summary(last_db)
                    self.after(0, lambda: self._log(f"\n{summary}"))

                self.after(0, lambda: self.btn_ml_train.config(state=NORMAL))
                self.after(0, lambda: messagebox.showinfo(
                    "Saved",
                    f"{total} trial(s) across "
                    f"{len(datasets_snapshot)} dataset(s) saved to ML database(s)."))
            except Exception as ex:
                self.after(0, lambda: self._log(f"[ML] ✘ Save error: {ex}"))
                self.after(0, lambda: messagebox.showerror("Error", str(ex)))

        threading.Thread(target=_do, daemon=True).start()

    def _run_ml_analysis(self):
        """Train ML models on all datasets combined (supports multi-location/year)."""
        if not _ML_AVAILABLE:
            messagebox.showerror("ML Unavailable",
                                 "scikit-learn is required.\n"
                                 "Run:  pip install scikit-learn")
            return
        if not self._datasets:
            messagebox.showerror("No Datasets",
                                 "No datasets in the list.\n"
                                 "Use ➕ Add Location/Year to add at least one dataset.")
            return

        # Check and prompt for any missing field Excels (main-thread file dialogs)
        for ds in self._datasets:
            if not ds.get("field_excel") or not os.path.exists(ds["field_excel"]):
                field_xlsx = filedialog.askopenfilename(
                    title=f'Select Field Data Excel for "{ds["label"]}"',
                    filetypes=[("Excel files", "*.xlsx *.xls *.xlsm"),
                               ("All files", "*.*")])
                if field_xlsx:
                    ds["field_excel"] = field_xlsx
                    # Refresh listbox entry to show the file
                    try:
                        idx = self._datasets.index(ds)
                        n   = len(ds["trial_roots"])
                        fb  = os.path.basename(field_xlsx)
                        self.trial_listbox.delete(idx)
                        self.trial_listbox.insert(
                            idx, f"📂 {ds['label']}  [{n} trial(s)]  ←  {fb}")
                    except (ValueError, Exception):
                        pass

        # Verify at least one dataset has a DB
        have_db = [ds for ds in self._datasets if os.path.exists(ds.get("db_path", ""))]
        if not have_db:
            messagebox.showerror(
                "No Database",
                "No ML database found for any dataset.\n\n"
                "Save at least one trial first using\n"
                "💾 Save All Trials to ML Database.")
            return

        missing_dbs = [ds["label"] for ds in self._datasets
                       if not os.path.exists(ds.get("db_path", ""))]
        if missing_dbs:
            self._log(f"[ML] Warning: DB not found for: {', '.join(missing_dbs)}")

        out_dir = self._ml_out_dir()
        self._log(f"\n[ML] Training on {len(self._datasets)} dataset(s) ...")
        self._log(f"[ML] Global output → {out_dir}")
        for ds in self._datasets:
            self._log(f"[ML]   {ds['label']}: "
                      f"{len(ds['trial_roots'])} trial(s)  "
                      f"DB={ds['db_path']}")

        self.btn_ml_train.config(state=DISABLED, text="Training …")
        datasets_snapshot = [dict(d) for d in self._datasets]   # shallow copy

        def _do():
            try:
                sow       = self._sowing_date
                pid_col   = self.var_ml_pid_col.get().strip()   or "PlotID"
                trial_col = self.var_ml_trial_col.get().strip() or "Experiment Name"
                mtr_col   = self.var_ml_mtr_col.get().strip()   or "MTR"
                geno_col  = self.var_ml_geno_col.get().strip()  or "Name"

                if len(datasets_snapshot) == 1:
                    ds = datasets_snapshot[0]
                    result_dir = _ml.run_ml_pipeline(
                        field_excel  = ds["field_excel"],
                        db_path      = ds["db_path"],
                        out_dir      = out_dir,
                        sowing_date  = sow,
                        plot_id_col  = pid_col,
                        trial_col    = trial_col,
                        genotype_col = geno_col,
                        mtr_col      = mtr_col,
                        trial_roots  = ds["trial_roots"],
                        log_fn       = self._log)
                else:
                    result_dir = _ml.run_ml_pipeline_multi(
                        datasets     = datasets_snapshot,
                        out_dir      = out_dir,
                        sowing_date  = sow,
                        plot_id_col  = pid_col,
                        trial_col    = trial_col,
                        genotype_col = geno_col,
                        mtr_col      = mtr_col,
                        log_fn       = self._log)

                model_pkl = os.path.join(result_dir, "model.pkl")
                self._ml_model_path = model_pkl if os.path.exists(model_pkl) else ""

                # Check predictions.xlsx to see how many datasets actually loaded
                _pred_path = os.path.join(result_dir, "predictions.xlsx")
                _n_loaded  = 0
                _loaded_ds = []
                try:
                    import pandas as _pd
                    _p = _pd.read_excel(_pred_path, engine="openpyxl")
                    if "Dataset" in _p.columns:
                        _loaded_ds = sorted(_p["Dataset"].unique().tolist())
                        _n_loaded  = len(_loaded_ds)
                    else:
                        _loaded_ds = sorted(_p["TrialName"].unique().tolist())
                        _n_loaded  = len(datasets_snapshot)  # single-ds path
                except Exception:
                    _n_loaded = len(datasets_snapshot)

                _skipped = len(datasets_snapshot) - _n_loaded

                def _done(n_loaded=_n_loaded, loaded_ds=_loaded_ds,
                          skipped=_skipped, n_ds=len(datasets_snapshot)):
                    self.btn_ml_train.config(state=NORMAL,
                                             text="🧠  Load Field Data & Train Models")
                    if self._ml_model_path:
                        self.btn_ml_predict.config(state=NORMAL)

                    msg = (f"ML analysis complete!\n\n"
                           f"Datasets loaded : {n_loaded} / {n_ds}\n")
                    if loaded_ds and n_ds > 1:
                        msg += "  " + "\n  ".join(loaded_ds) + "\n"
                    msg += f"\nGlobal output:\n  {result_dir}\n"
                    msg += ("\nPer-trial outputs saved inside each\n"
                            "trial folder → ML_Analysis/ subfolder.")

                    if skipped > 0:
                        msg += (f"\n\n⚠  {skipped} dataset(s) were SKIPPED.\n"
                                f"Check the Run Log for PlotID / field-Excel\n"
                                f"mismatch details.")
                        messagebox.showwarning("ML Complete (with warnings)", msg)
                    else:
                        messagebox.showinfo("ML Complete", msg)
                self.after(0, _done)
            except Exception as ex:
                def _err(e=ex):          # capture before Python deletes ex
                    self.btn_ml_train.config(state=NORMAL,
                                             text="🧠  Load Field Data & Train Models")
                    self._log(f"[ML] ✘ Training error: {e}")
                    messagebox.showerror("ML Error", str(e))
                self.after(0, _err)

        threading.Thread(target=_do, daemon=True).start()

    def _predict_trial(self):
        """Apply the saved model to every trial across all datasets."""
        if not _ML_AVAILABLE:
            messagebox.showerror("ML Unavailable",
                                 "scikit-learn is required.\n"
                                 "Run:  pip install scikit-learn")
            return
        if not self._datasets:
            messagebox.showerror("No Datasets",
                                 "No datasets in the list.\n"
                                 "Use ➕ Add Location/Year first.")
            return
        if not self._ml_model_path or not os.path.exists(self._ml_model_path):
            self._ml_model_path = filedialog.askopenfilename(
                title="Select model.pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
            if not self._ml_model_path:
                return

        # Collect every trial across all datasets
        all_trial_roots = [
            (root, name)
            for ds in self._datasets
            for root, name in ds["trial_roots"]
        ]
        self._log(f"[ML] Predicting {len(all_trial_roots)} trial(s) "
                  f"across {len(self._datasets)} dataset(s) ...")
        self.btn_ml_predict.config(state=DISABLED, text="Predicting …")

        def _do():
            saved, errors = [], []
            for out_root, trial_name in all_trial_roots:
                try:
                    pred_df = _ml.predict_new_trial(
                        output_root  = out_root,
                        model_path   = self._ml_model_path,
                        trial_name   = trial_name,
                        sowing_date  = self._sowing_date,
                        log_fn       = self._log)
                    ml_dir    = os.path.join(out_root, "ML_Analysis")
                    os.makedirs(ml_dir, exist_ok=True)
                    pred_path = os.path.join(
                        ml_dir, f"predictions_{trial_name}.xlsx")
                    pred_df.to_excel(pred_path, index=False, engine="xlsxwriter")
                    self.after(0, lambda p=pred_path, t=trial_name:
                               self._log(f"[ML] ✔ {t} → {p}"))
                    saved.append(trial_name)
                except Exception as ex:
                    errors.append(f"{trial_name}: {ex}")
                    self.after(0, lambda t=trial_name, e=str(ex):
                               self._log(f"[ML] ✘ {t}: {e}"))

            def _done():
                self.btn_ml_predict.config(state=NORMAL,
                                           text="🔮  Predict All Trials")
                msg = f"Predictions complete!\n\n✔ {len(saved)} trial(s) saved."
                if errors:
                    msg += f"\n\n✘ {len(errors)} error(s):\n" + "\n".join(errors)
                messagebox.showinfo("Prediction Complete", msg)
            self.after(0, _done)

        threading.Thread(target=_do, daemon=True).start()

    # ─────────────────────── OPEN OUTPUT ─────────────────────────────
    def _open_output(self):
        d = self._out_dir
        if not d or not os.path.isdir(d):
            messagebox.showwarning("Not found", f"Output folder not found:\n{d}")
            return
        if sys.platform == "win32":
            os.startfile(d)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", d])
        else:
            subprocess.Popen(["xdg-open", d])


# ─────────────────────────── ENTRY POINT ─────────────────────────────
if __name__ == "__main__":
    app = MaturityApp()
    app.mainloop()
