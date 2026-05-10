# ======================================================================
# analysis.py  -  RGB-based crop maturity analysis  (all methods)
# ======================================================================
"""
Methods implemented
-------------------
Chromatic coords  : GCC, RCC
Ratio / linear    : NGRDI, VARI, GLI, ExGR, IKAW, NDYI, R_over_G, TGI, WI
Quadratic         : MGRVI, RGBVI
HSV-based         : HMI_MASKED, HMI_MAIN, MPI, desicc_frac, green_cover
CIE Lab           : Lab_a, Lab_b, Lab_Chroma, Lab_HueAngle
Histogram         : hist_ratio  (Bhattacharyya distance ratio)
"""

import os
import re
import math
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import rasterio
from rasterio.mask import mask as rasterio_mask
import geopandas as gpd
from shapely.geometry import mapping, box as shapely_box
from shapely.ops import unary_union

import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import PchipInterpolator
from scipy.stats import gaussian_kde

# ----------------------- CONSTANTS ------------------------
DPI        = 350
FONT       = 13
TITLE_FONT = 14

EXG_MASK_THR = -0.05   # on 0-1 scale; lenient to keep yellowing canopy
MIN_VEG_PIX  = 300

# HMI hue windows (degrees, 0-360)
HMI_GREEN_MIN, HMI_GREEN_MAX = 60.0, 150.0
HMI_YELL_MIN,  HMI_YELL_MAX  = 25.0,  60.0

# ----------------------- METHODS REGISTRY -----------------
# dir   : direction of change at maturity ("increase" | "decrease")
# thr   : absolute threshold (None -> use relative 80 % transition only)
# order : sort prefix for output folders
# label : short axis label for plots
# desc  : long description
METHODS: "OrderedDict[str, dict]" = OrderedDict([
    ("GCC",         {"dir":"decrease","thr":0.36, "order":"01","label":"GCC",         "desc":"Green Chromatic Coordinate"}),
    ("RCC",         {"dir":"increase","thr":0.40, "order":"02","label":"RCC",         "desc":"Red Chromatic Coordinate"}),
    ("NGRDI",       {"dir":"decrease","thr":0.00, "order":"03","label":"NGRDI",       "desc":"Normalized Green-Red Difference Index"}),
    ("VARI",        {"dir":"decrease","thr":0.00, "order":"04","label":"VARI",        "desc":"Visible Atmospherically Resistant Index"}),
    ("GLI",         {"dir":"decrease","thr":0.00, "order":"05","label":"GLI",         "desc":"Green Leaf Index"}),
    ("MGRVI",       {"dir":"decrease","thr":0.00, "order":"06","label":"MGRVI",       "desc":"Modified Green-Red Vegetation Index"}),
    ("RGBVI",       {"dir":"decrease","thr":0.15, "order":"07","label":"RGBVI",       "desc":"RGB Vegetation Index (Gruner 2019)"}),
    ("ExGR",        {"dir":"decrease","thr":0.00, "order":"08","label":"ExGR",        "desc":"Excess Green minus Excess Red"}),
    ("IKAW",        {"dir":"increase","thr":0.05, "order":"09","label":"IKAW",        "desc":"Kawashima & Nakatani Index"}),
    ("NDYI",        {"dir":"increase","thr":0.15, "order":"10","label":"NDYI",        "desc":"Normalized Difference Yellowness Index"}),
    ("R_over_G",    {"dir":"increase","thr":1.10, "order":"11","label":"R/G",         "desc":"Red-to-Green Ratio"}),
    ("TGI",         {"dir":"decrease","thr":None, "order":"12","label":"TGI",         "desc":"Triangular Greenness Index"}),
    ("WI",          {"dir":"increase","thr":None, "order":"13","label":"WI",          "desc":"Woebbecke Index"}),
    ("HMI_MASKED",  {"dir":"increase","thr":0.80, "order":"14","label":"HMI (masked)","desc":"Hue Maturity Index - ExG Masked"}),
    ("HMI_MAIN",    {"dir":"increase","thr":0.80, "order":"15","label":"HMI (all px)","desc":"Hue Maturity Index - All Pixels"}),
    ("MPI",         {"dir":"increase","thr":0.85, "order":"16","label":"MPI",         "desc":"Maturity Progression Index (HSV classes)"}),
    ("desicc_frac", {"dir":"increase","thr":0.50, "order":"17","label":"Desicc. Frac.","desc":"Desiccation Fraction (low-S, high-V pixels)"}),
    ("green_cover", {"dir":"decrease","thr":0.20, "order":"18","label":"Green Cover", "desc":"Green Cover Fraction"}),
    ("Lab_a",       {"dir":"increase","thr":5.0,  "order":"19","label":"Lab a*",      "desc":"CIE Lab a*  (green -> red axis)"}),
    ("Lab_b",       {"dir":"increase","thr":28.0, "order":"20","label":"Lab b*",      "desc":"CIE Lab b*  (blue -> yellow axis)"}),
    ("Lab_Chroma",  {"dir":"increase","thr":25.0, "order":"21","label":"Lab Chroma",  "desc":"CIE Lab Chroma  sqrt(a*2 + b*2)"}),
    ("Lab_HueAngle",{"dir":"decrease","thr":90.0, "order":"22","label":"Lab hdeg",      "desc":"CIE Lab Hue Angle  atan2(b*,a*)  [deg]"}),
    ("hist_ratio",  {"dir":"increase","thr":1.20, "order":"23","label":"Hist. Ratio", "desc":"Bhattacharyya Distance Ratio (ref=date-1 vs date-N)"}),
])
METHOD_NAMES: List[str] = list(METHODS.keys())

# ----------------------- METHOD INTERPRETATION DATA -------------------
# Each entry: formula, range, biology, detect, reading, limits
_METHOD_INFO: Dict[str, Dict[str, str]] = {
    "GCC": {
        "formula": "GCC = G / (R + G + B)",
        "range":   "0 to 1  (green crops: 0.34 – 0.44)",
        "biology": (
            "Measures how much of the total brightness comes from the green channel. "
            "Chlorophyll strongly absorbs red and blue light, so healthy green leaves "
            "reflect disproportionately in green. As chlorophyll degrades during grain "
            "filling and senescence, GCC falls as yellow/brown pigments take over."),
        "detect": (
            "DECREASES at maturity. A GCC below the threshold (0.36) indicates "
            "the canopy has lost enough chlorophyll to be considered senescent. "
            "Uses relative-80% fallback when the threshold is never crossed."),
        "reading": (
            "Time-series: Orange MA-3 line drops from peak (~0.40) toward "
            "senescent values (~0.33). "
            "Ridgeline: Distribution peak shifts LEFT over time. "
            "Chromatic scatter: Points migrate from top-left (green) toward center."),
        "limits": (
            "Sensitive to solar angle, clouds, and shadows. "
            "Soil background inflates GCC before full canopy closure. "
            "Best with consistent nadir imagery under similar illumination."),
    },
    "RCC": {
        "formula": "RCC = R / (R + G + B)",
        "range":   "0 to 1  (green crops: 0.28 – 0.38)",
        "biology": (
            "Red Chromatic Coordinate captures the relative contribution of the red "
            "channel. As chlorophyll breaks down, carotenoids (yellow-orange pigments) "
            "and bare straw become dominant, increasing red reflectance relative to "
            "the total. RCC is the complementary indicator to GCC."),
        "detect": (
            "INCREASES at maturity. When RCC rises above 0.40 the canopy red "
            "reflectance is dominant, signaling late-stage senescence."),
        "reading": (
            "Time-series: Orange MA-3 line rises toward peak as season progresses. "
            "Ridgeline: Distribution peak shifts RIGHT (higher RCC) at maturity. "
            "Chromatic scatter: Points migrate toward high-r / low-g region."),
        "limits": (
            "Mirror of GCC limitations. May show early increases due to "
            "bare soil or dry plant tissue before physiological maturity."),
    },
    "NGRDI": {
        "formula": "NGRDI = (G - R) / (G + R)",
        "range":   "-1 to 1  (healthy canopy: 0.05 – 0.20)",
        "biology": (
            "Normalized Green-Red Difference Index directly contrasts green vs red "
            "reflectance. Positive values indicate green dominance (live tissue); "
            "negative values indicate red dominance (senesced or bare soil). "
            "Crosses zero at equal green and red reflectance."),
        "detect": (
            "DECREASES at maturity. Crossing below 0.0 means red reflectance "
            "now exceeds green — a robust signal of advanced senescence."),
        "reading": (
            "Time-series: Trace drops from positive values toward zero/negative. "
            "Ridgeline: Distribution shifts from positive to near-zero range. "
            "The zero-crossing date is a strong maturity indicator."),
        "limits": (
            "Numerically saturates at very high or very low greenness. "
            "Less sensitive than quadratic indices (MGRVI) to subtle changes."),
    },
    "VARI": {
        "formula": "VARI = (G - R) / (G + R - B)",
        "range":   "typically -0.2 to 0.4",
        "biology": (
            "Visible Atmospherically Resistant Index extends NGRDI by including "
            "blue to partially compensate for atmospheric and illumination effects. "
            "Used for estimating green vegetation fraction in field conditions."),
        "detect": (
            "DECREASES at maturity (threshold 0.0). As green cover disappears, "
            "VARI becomes negative. More stable than NGRDI under variable lighting."),
        "reading": (
            "Very similar to NGRDI but more robust to haze/shadows. "
            "Ridgeline: narrowing and left-shift at maturity. "
            "Watch for denominator instability when G + R ≈ B."),
        "limits": (
            "Denominator can approach zero, causing numerical instability. "
            "Works best when blue channel is reliably calibrated."),
    },
    "GLI": {
        "formula": "GLI = (2G - R - B) / (2G + R + B)",
        "range":   "-1 to 1  (green crops: 0.05 – 0.25)",
        "biology": (
            "Green Leaf Index uses all three channels. Double-weighting of green "
            "vs the sum of red and blue gives a balanced greenness score that is "
            "less affected by soil than single-channel indices."),
        "detect": (
            "DECREASES at maturity (threshold 0.0). Zero crossing indicates the "
            "green component no longer dominates the red+blue sum."),
        "reading": (
            "Behaves similarly to NGRDI with added robustness from the blue channel. "
            "Ridgeline shifts left and narrows at maturity."),
        "limits": ("May lag slightly behind GCC/NGRDI for detecting early senescence."),
    },
    "MGRVI": {
        "formula": "MGRVI = (G^2 - R^2) / (G^2 + R^2)",
        "range":   "-1 to 1",
        "biology": (
            "Modified Green-Red Vegetation Index uses squared values, amplifying "
            "the contrast between green and red. More sensitive than NGRDI to subtle "
            "color changes during early senescence stages."),
        "detect": (
            "DECREASES at maturity (threshold 0.0). Squared formulation makes "
            "this index more sensitive to early-season greenness changes."),
        "reading": (
            "Similar to NGRDI but with steeper response curve. "
            "Ridgeline shows wider distribution due to amplification of extremes."),
        "limits": ("Amplified sensitivity means more noise from soil/shadow pixels."),
    },
    "RGBVI": {
        "formula": "RGBVI = (G^2 - B*R) / (G^2 + B*R)",
        "range":   "-1 to 1  (Gruner et al. 2019)",
        "biology": (
            "RGB Vegetation Index (Gruner 2019) uses the ratio of green squared "
            "to the product of blue and red, exploiting the full trichromatic "
            "relationship. Designed specifically for UAV RGB imagery."),
        "detect": (
            "DECREASES at maturity (threshold 0.15). Effective for high-resolution "
            "UAV data where all three channels are well-calibrated."),
        "reading": (
            "Ridgeline shift at maturity is particularly clear for cereal crops. "
            "Values above 0.15 indicate active green canopy."),
        "limits": ("Less tested than ratio-based indices. Sensitive to blue channel calibration."),
    },
    "ExGR": {
        "formula": "ExGR = 3G - 2.4R - B  (Meyer & Neto 2008)",
        "range":   "variable, ~-50 to +100 for 0-255 scale",
        "biology": (
            "Excess Green minus Excess Red combines the benefit of ExG (emphasizing "
            "green vegetation) and ExR (emphasizing reddish soil/senescent tissue). "
            "Positive = green vegetation dominant; negative = red/bare dominant."),
        "detect": (
            "DECREASES at maturity (threshold 0.0). The ExG trend plot in the HMI "
            "folders shows the PCHIP-fitted trajectory with a slope line from peak "
            "to final measurement."),
        "reading": (
            "Time-series: Clear rise-then-fall pattern. Peak ExGR corresponds to "
            "maximum green biomass. Rate of decline (slope) indicates senescence speed. "
            "Ridgeline: Shifts from large positive to near-zero/negative at maturity."),
        "limits": (
            "Scale-dependent (different for 0-1 vs 0-255 data). Auto-scaled "
            "internally in this application."),
    },
    "IKAW": {
        "formula": "IKAW = (R - B) / (R + B)  (Kawashima & Nakatani 1998)",
        "range":   "-1 to 1",
        "biology": (
            "Proposed by Kawashima & Nakatani for detecting crop stress and maturity. "
            "Compares red to blue; as plants dry out and turn yellow/straw-colored, "
            "red increases and blue decreases, raising IKAW."),
        "detect": (
            "INCREASES at maturity (threshold 0.05). Reliable for desiccation "
            "but may trigger early if soil has high red content."),
        "reading": (
            "Ridgeline: Distribution shifts toward positive values (0.1 – 0.3) "
            "at physiological maturity. Clear rightward progression."),
        "limits": ("Susceptible to soil background and residue color. Less common than GCC-based indices."),
    },
    "NDYI": {
        "formula": "NDYI = ((R+G) - 2B) / ((R+G) + 2B)",
        "range":   "-1 to 1",
        "biology": (
            "Normalized Difference Yellowness Index quantifies yellow-orange "
            "coloration by contrasting the warm (R+G) channels against blue. "
            "Yellow light is R+G in equal proportions; blue reflects cooler/green tones."),
        "detect": (
            "INCREASES at maturity (threshold 0.15). Rising NDYI indicates "
            "the canopy is turning yellow — a direct measure of ripening."),
        "reading": (
            "Ridgeline: Clear rightward shift as ripening occurs. "
            "One of the most biologically intuitive indices for cereal maturity. "
            "Correlates well with grain yellowing."),
        "limits": ("Requires reliable blue channel. Noise-sensitive in short crops with soil exposure."),
    },
    "R_over_G": {
        "formula": "R/G = R / G",
        "range":   "0.6 to 1.5  for crop canopies",
        "biology": (
            "Simple ratio of red to green reflectance. When R/G > 1 the canopy "
            "appears reddish/yellow rather than green. Easy to interpret and highly "
            "correlated with traditional greenness indices."),
        "detect": (
            "INCREASES at maturity (threshold 1.10). Crossing above 1.0 means "
            "red reflectance now exceeds green — a clear senescence signal."),
        "reading": (
            "Ridgeline: Distribution crosses through 1.0 at maturity. "
            "The crossing date is a reliable and interpretable maturity estimate."),
        "limits": ("Division by G can be noisy for very dark/shaded pixels."),
    },
    "TGI": {
        "formula": "TGI = G - 0.39R - 0.61B  (Hunt et al. 2011)",
        "range":   "variable; positive for healthy vegetation",
        "biology": (
            "Triangular Greenness Index exploits the 'triangle' formed by the "
            "chlorophyll absorption features at red and blue wavelengths. "
            "Correlates with chlorophyll content and leaf area index."),
        "detect": (
            "DECREASES at maturity (no fixed threshold — uses relative 80% transition). "
            "Reliable for estimating relative chlorophyll loss."),
        "reading": (
            "Ridgeline: Distribution narrows and shifts to lower values at maturity. "
            "Because there is no fixed threshold, the relative-80% method detects "
            "when TGI has dropped 80% of its seasonal range."),
        "limits": ("Coefficients (0.39, 0.61) derived for broadband sensors; may need tuning for UAV cameras."),
    },
    "WI": {
        "formula": "WI = (G - B) / (R - G)  (Woebbecke et al. 1995)",
        "range":   "unstable near R=G; clipped to [-10, 10]",
        "biology": (
            "Woebbecke Index was developed for segmenting green plants from soil. "
            "Positive when G > B and R > G (typical senescing canopy). "
            "Can be noisy but captures the overall green-to-reddish transition."),
        "detect": (
            "INCREASES at maturity (no fixed threshold — uses relative 80% transition). "
            "Values rise as the denominator (R-G) grows and G-B diminishes."),
        "reading": (
            "Wide ridgeline distribution due to numerical instability. "
            "Focus on the MEDIAN trajectory in the time-series rather than "
            "individual ridgeline shapes."),
        "limits": (
            "Numerically unstable when R ≈ G (denominator near zero). "
            "Values are clipped to [-10, 10] in this application. "
            "Use alongside more stable indices for confirmation."),
    },
    "HMI_MASKED": {
        "formula": "HMI = yellow_px / (green_px + yellow_px)  [ExG-masked]",
        "range":   "0 to 1",
        "biology": (
            "Hue Maturity Index quantifies the fraction of vegetation pixels "
            "that appear yellow (hue 25–60 deg) versus green (hue 60–150 deg). "
            "MASKED version first removes soil/background using an ExG mask "
            "(ExG >= -0.05), isolating only plant tissue."),
        "detect": (
            "INCREASES at maturity (threshold 0.80). When 80% of vegetation "
            "pixels are in the yellow hue zone the crop is considered mature. "
            "HMI_MASKED is generally more accurate than HMI_MAIN for sparse canopies."),
        "reading": (
            "Hue histograms: Orange zone (25-60 deg) fills as crop matures; green "
            "peak (60-120 deg) disappears. 3D stacked: progression from right-peak "
            "(green) to left-peak (yellow) across DAP axis. "
            "HMI trajectory: S-shaped rise from 0 toward 1 at maturity."),
        "limits": (
            "ExG mask may exclude yellowing pixels if they fall below the ExG threshold. "
            "Hue is sensitive to white balance and camera response curves."),
    },
    "HMI_MAIN": {
        "formula": "HMI = yellow_px / (green_px + yellow_px)  [all pixels]",
        "range":   "0 to 1",
        "biology": (
            "Same as HMI_MASKED but computed on ALL pixels in the plot boundary "
            "without any vegetation mask. Includes soil and background pixels. "
            "Useful when the mask is too aggressive or canopy cover is low."),
        "detect": (
            "INCREASES at maturity (threshold 0.80). May be less precise than "
            "HMI_MASKED for open canopies where soil is visible."),
        "reading": (
            "Interpret alongside HMI_MASKED. Large divergence between the two "
            "indicates significant soil background influence. "
            "Hue histograms show the same green-to-yellow progression."),
        "limits": (
            "Soil, residues, and shadows all influence the hue distribution. "
            "For dense canopies (>80% cover) results are similar to HMI_MASKED."),
    },
    "MPI": {
        "formula": "MPI = (f_yellow + f_brown) / (f_green + f_yellow + f_brown)",
        "range":   "0 to 1",
        "biology": (
            "Maturity Progression Index aggregates the non-green HSV classes "
            "(yellow: H=25-60, S>0.2, V>0.3; brown/senescent: H=10-35, S<0.6, V=0.25-0.95) "
            "relative to all detected vegetation classes. Rises continuously from "
            "green stage to full senescence."),
        "detect": (
            "INCREASES at maturity (threshold 0.85). A broad, multi-class index "
            "that is robust to the exact boundary between yellow and brown."),
        "reading": (
            "Cover stack plot: watch the yellow and brown bars grow while green shrinks. "
            "The stacked bar at each date directly shows the class balance. "
            "MPI is the ratio (yellow+brown)/(all) shown in the time-series."),
        "limits": (
            "HSV class boundaries are empirical and crop/camera specific. "
            "May need threshold adjustment for non-cereal crops."),
    },
    "desicc_frac": {
        "formula": "desicc_frac = fraction(S <= 0.25 AND V >= 0.65)",
        "range":   "0 to 1",
        "biology": (
            "Desiccation Fraction detects pixels with low color saturation (grey/white "
            "appearance) and high brightness — characteristic of dried, desiccated "
            "straw and stems. Rises sharply in the final stages of physiological "
            "maturity when green and yellow pigments are fully degraded."),
        "detect": (
            "INCREASES at maturity (threshold 0.50). When >50% of pixels are "
            "desiccated the crop is at or past black-layer formation."),
        "reading": (
            "Cover stack: desiccated (light beige) bar grows in final dates. "
            "Combined with green_cover (decreasing), provides a complementary "
            "view of canopy state. Sharp rise often indicates harvest readiness."),
        "limits": (
            "Bright soil and reflected sky can mimic desiccated pixels "
            "(low S, high V). Use with a vegetation mask when possible."),
    },
    "green_cover": {
        "formula": "green_cover = fraction(H=60-150 deg, S>=0.20, V>=0.15)",
        "range":   "0 to 1",
        "biology": (
            "Green Cover Fraction counts pixels in the green hue zone of HSV space "
            "with sufficient saturation (not grey) and brightness (not black). "
            "Directly measures the proportion of the canopy that is still actively green."),
        "detect": (
            "DECREASES at maturity (threshold 0.20). When less than 20% of pixels "
            "are green, the canopy has undergone substantial senescence."),
        "reading": (
            "Cover stack: green bar shrinks over time; other classes expand. "
            "Green cover is the most directly interpretable of the cover fractions. "
            "Should mirror the GCC time-series trend."),
        "limits": (
            "Hue-based classification is sensitive to camera white balance. "
            "Bare green soil or weeds can inflate the estimate before canopy closure."),
    },
    "Lab_a": {
        "formula": "a* in CIE L*a*b*  (sRGB -> XYZ -> Lab, D65)",
        "range":   "-50 to +50  (negative=green, positive=red/brown)",
        "biology": (
            "The a* axis in CIE Lab color space runs from green (negative) to "
            "red/magenta (positive). As chlorophyll degrades and carotenoids or "
            "browning pigments emerge, a* shifts from negative toward positive. "
            "Perceptually uniform: equal numeric changes = equal visual differences."),
        "detect": (
            "INCREASES at maturity (threshold +5). Crossing zero from negative "
            "to positive is a biologically meaningful boundary (neutral color). "
            "Values above +5 indicate clear reddish/brown canopy coloration."),
        "reading": (
            "Lab scatter: Points migrate from the left (negative a*) toward "
            "the right as maturity progresses. Combined trajectory (a*, b*) "
            "captures both reddening and yellowing simultaneously."),
        "limits": (
            "Requires accurate color calibration. CIE Lab is device-independent "
            "in theory but UAV cameras require proper white balance settings."),
    },
    "Lab_b": {
        "formula": "b* in CIE L*a*b*  (sRGB -> XYZ -> Lab, D65)",
        "range":   "0 to +60  (positive=yellow, negative=blue)",
        "biology": (
            "The b* axis runs from blue (negative) to yellow (positive). "
            "Cereal ripening is characterized by a strong increase in b* as "
            "the canopy transitions from green (low b*, moderate a*) to "
            "yellow (high b*, near-zero a*) and finally straw (very high b*)."),
        "detect": (
            "INCREASES at maturity (threshold +28). High b* values (>30) "
            "represent a clearly yellow canopy. One of the strongest single "
            "indicators of physiological maturity in cereals."),
        "reading": (
            "Lab scatter: Points migrate upward (higher b*) over time. "
            "The b* trajectory often shows a monotonic increase from "
            "green stage onward — look for when it plateaus."),
        "limits": ("Similar to Lab_a — requires consistent white balance for absolute comparisons."),
    },
    "Lab_Chroma": {
        "formula": "Chroma = sqrt(a*^2 + b*^2)",
        "range":   "0 to ~70",
        "biology": (
            "CIE Lab Chroma (also called color saturation or colorfulness) "
            "measures the distance from the neutral grey axis in Lab space. "
            "Green vegetation and yellow grain both have high Chroma. "
            "Desiccated, grey straw has low Chroma. Chroma rises during "
            "ripening (greens to vivid yellows) then falls as straw greys."),
        "detect": (
            "INCREASES at maturity (threshold 25.0). The rise is driven by "
            "yellow pigments in the ripening grain. A subsequent fall after "
            "harvest would indicate full desiccation."),
        "reading": (
            "Lab scatter: Distance of each point from the center (0,0) "
            "represents Chroma. Points far from center = high colorfulness. "
            "Ridgeline: Distribution shifts right during ripening."),
        "limits": ("May not decrease at full desiccation if imaging ends before that stage."),
    },
    "Lab_HueAngle": {
        "formula": "h_ab = atan2(b*, a*) * 180/pi  [0-360 deg]",
        "range":   "0-360 deg  (green ~110-130, yellow ~90, straw ~60-80)",
        "biology": (
            "CIE Lab Hue Angle describes the color direction: ~120 deg = green, "
            "~90 deg = yellow, ~60 deg = orange-straw. As cereals ripen, the hue "
            "angle decreases from ~120 (pure green) toward 90 (yellow) and "
            "eventually ~60-70 (straw/brown). This is the most direct Lab indicator "
            "of the green-to-yellow transition."),
        "detect": (
            "DECREASES at maturity (threshold 90 deg). Dropping below 90 deg "
            "means the canopy has crossed from predominantly green to predominantly "
            "yellow in CIE Lab space — a biologically robust threshold."),
        "reading": (
            "Ridgeline: Distribution peak moves LEFT (lower degrees) over time. "
            "Lab scatter: Same point cloud but colored by time shows clockwise "
            "rotation in the a*-b* plane as hue angle decreases."),
        "limits": ("Undefined near the achromatic axis (very low Chroma). Use Chroma alongside for validation."),
    },
    "hist_ratio": {
        "formula": "hist_ratio = D(H_t, H_first) / D(H_t, H_last)  [Bhattacharyya]",
        "range":   "0 to large values; crosses 1.0 at midpoint",
        "biology": (
            "Bhattacharyya Distance Ratio measures how similar the current date's "
            "hue-saturation histogram is to the FIRST date (green) versus the LAST "
            "date (senescent). A value > 1.0 means the current state is closer to "
            "the final senescent appearance than to the initial green appearance."),
        "detect": (
            "INCREASES at maturity (threshold 1.20). Crossing 1.20 indicates the "
            "canopy has moved clearly past the midpoint toward the final senescent "
            "state. Non-parametric: does not assume any fixed color threshold."),
        "reading": (
            "H-S histogram grid: Earlier dates cluster in the green-hue region "
            "(H ~60-120, S high). Later dates shift toward yellow/brown (H ~20-60). "
            "The Bhattacharyya distance quantifies how different each date is from "
            "the reference states."),
        "limits": (
            "Requires at least 2 valid dates. Sensitive to the quality of the first "
            "and last dates (should be clearly green and clearly senescent). "
            "May not work well if the earliest date is already partially mature."),
    },
}


# ======================= FILE / DATE UTILITIES ========================
_DATE_PATTERNS = [
    r"(\d{2})[_\-](\d{2})[_\-](\d{4})",   # MM_DD_YYYY  or  MM-DD-YYYY
    r"(\d{8})",                             # MMDDYYYY
]

def parse_date_from_path(p: str) -> Optional[datetime]:
    for pat in _DATE_PATTERNS:
        m = re.search(pat, p)
        if not m:
            continue
        g = m.groups()
        if len(g) == 3:
            try:
                return datetime(int(g[2]), int(g[0]), int(g[1]))
            except ValueError:
                continue
        s = g[0]
        try:
            return datetime(int(s[4:8]), int(s[0:2]), int(s[2:4]))
        except ValueError:
            continue
    return None


def _is_rgb_tif(path: str) -> bool:
    try:
        with rasterio.open(path) as ds:
            return ds.count >= 3
    except Exception:
        return False


def collect_dated_images(images_root: str) -> List[Dict]:
    """
    Walk images_root recursively. For each TIF file found, try to parse
    a date from its path. Group by date, prefer multiband (>=3 bands) over
    separate R/G/B files. Returns list of {"date": datetime, "spec": ...}
    sorted by date.
    """
    by_date: Dict[datetime, List[str]] = {}
    for root, _, files in os.walk(images_root):
        for f in files:
            if not f.lower().endswith((".tif", ".tiff")):
                continue
            full = os.path.join(root, f)
            dt = parse_date_from_path(full) or parse_date_from_path(root)
            if dt is not None:
                by_date.setdefault(dt, []).append(full)

    result = []
    for dt in sorted(by_date):
        paths = by_date[dt]
        spec = None
        # 1) prefer multiband stack
        for p in paths:
            if _is_rgb_tif(p):
                spec = {"mode": "stack", "path": p}
                break
        # 2) fall back to separate R/G/B files
        if spec is None:
            band: Dict[str, str] = {}
            for p in paths:
                n = os.path.basename(p).lower()
                if "red" in n and "R" not in band:
                    band["R"] = p
                elif ("green" in n or "grn" in n) and "G" not in band:
                    band["G"] = p
                elif "blue" in n and "B" not in band:
                    band["B"] = p
            if all(k in band for k in ("R", "G", "B")):
                spec = {"mode": "separate", **band}
        if spec is not None:
            result.append({"date": dt, "spec": spec})
    return result


def _spec_main_path(spec: Dict) -> str:
    return spec.get("path") or spec.get("R")


def reproj_geom(geom, src_crs, dst_crs):
    if src_crs == dst_crs:
        return geom
    gs = gpd.GeoSeries([geom], crs=src_crs).to_crs(dst_crs)
    return gs.iloc[0]


def read_rgb_clipped(spec: Dict, geom) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip raster to geom (already in raster CRS).
    Returns (R, G, B) as float32 arrays with NaN for masked/nodata pixels.
    Raises on error so the pipeline can log the specific message.
    Uses filled=False to support integer rasters (uint8/uint16 UAV images)."""
    shapes = [mapping(geom)]

    def _ma_to_float(band_ma):
        """Masked-array band (2-D) -> float32; masked pixels -> NaN."""
        data = band_ma.data.astype("float32")
        data[np.ma.getmaskarray(band_ma)] = np.nan
        return data

    if spec["mode"] == "stack":
        with rasterio.open(spec["path"]) as ds:
            arr, _ = rasterio_mask(ds, shapes, crop=True, filled=False)
        return _ma_to_float(arr[0]), _ma_to_float(arr[1]), _ma_to_float(arr[2])
    else:  # separate single-band files
        def _r(path):
            with rasterio.open(path) as ds:
                a, _ = rasterio_mask(ds, shapes, crop=True, filled=False)
            return _ma_to_float(a[0])
        return _r(spec["R"]), _r(spec["G"]), _r(spec["B"])


# ===================== NORMALISATION & LAB ============================
def normalize_rgb(R: np.ndarray, G: np.ndarray, B: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays in [0,1]. Auto-detects 0-255 scale (maxv > 2)."""
    maxv = max(float(np.nanmax(R)), float(np.nanmax(G)), float(np.nanmax(B)))
    if not np.isfinite(maxv) or maxv == 0:
        return R, G, B
    if maxv > 2.0:
        R, G, B = R / 255.0, G / 255.0, B / 255.0
    return (np.clip(R, 0, 1).astype("float32"),
            np.clip(G, 0, 1).astype("float32"),
            np.clip(B, 0, 1).astype("float32"))


def rgb01_to_lab(R: np.ndarray, G: np.ndarray, B: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sRGB [0,1] -> CIE L*a*b*  (D65 illuminant). Returns L, a, b arrays."""
    def lin(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    Rl, Gl, Bl = lin(R), lin(G), lin(B)
    X = 0.4124564 * Rl + 0.3575761 * Gl + 0.1804375 * Bl
    Y = 0.2126729 * Rl + 0.7151522 * Gl + 0.0721750 * Bl
    Z = 0.0193339 * Rl + 0.1191920 * Gl + 0.9503041 * Bl
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    def f(t):
        return np.where(t > 0.008856, np.cbrt(np.maximum(t, 0.0)),
                        7.787 * t + 16.0 / 116.0)
    fx, fy, fz = f(X / Xn), f(Y / Yn), f(Z / Zn)
    L     = (116.0 * fy - 16.0).astype("float32")
    a_star = (500.0 * (fx - fy)).astype("float32")
    b_star = (200.0 * (fy - fz)).astype("float32")
    return L, a_star, b_star


# ======================= INDEX COMPUTATION ============================
def compute_all_indices(R: np.ndarray, G: np.ndarray, B: np.ndarray
                        ) -> Optional[Dict[str, float]]:
    """
    Compute all scalar maturity indices for one plot x one date.
    R, G, B may be any scale (auto-normalised to 0-1 internally).
    Returns dict of float values, or None if < 20 finite pixels.
    """
    finite = np.isfinite(R) & np.isfinite(G) & np.isfinite(B)
    if finite.sum() < 20:
        return None

    R0, G0, B0 = normalize_rgb(R, G, B)
    r, g, b = R0[finite], G0[finite], B0[finite]
    eps = 1e-6

    # -- Chromatic coordinates --
    denom_c = r + g + b + eps
    gcc = float(np.nanmedian(g / denom_c))
    rcc = float(np.nanmedian(r / denom_c))

    # -- Simple ratio / linear indices --
    ngrdi  = float(np.nanmedian((g - r) / (g + r + eps)))
    vari   = float(np.nanmedian((g - r) / (g + r - b + eps)))
    gli    = float(np.nanmedian((2*g - r - b) / (2*g + r + b + eps)))
    exgr   = float(np.nanmedian(3*g - 2.4*r - b))          # Meyer & Neto 2008
    ikaw   = float(np.nanmedian((r - b) / (r + b + eps)))   # Kawashima 1998
    ndyi   = float(np.nanmedian(((r+g) - 2*b) / ((r+g) + 2*b + eps)))
    r_og   = float(np.nanmedian(np.where(g > eps, r / g, np.nan)))
    tgi    = float(np.nanmedian(g - 0.39*r - 0.61*b))       # Hunt et al. 2011
    wi_val = float(np.nanmedian((g - b) / (r - g + eps)))   # Woebbecke 1995

    # -- Quadratic indices --
    mgrvi  = float(np.nanmedian((g**2 - r**2) / (g**2 + r**2 + eps)))
    rgbvi  = float(np.nanmedian((g**2 - b*r)  / (g**2 + b*r  + eps)))

    # -- HSV (manual, full pixel set) --
    rgb_s  = np.stack([r, g, b], axis=-1)
    maxc   = np.max(rgb_s, axis=-1)
    minc   = np.min(rgb_s, axis=-1)
    dm     = maxc - minc
    V      = maxc
    S      = np.where(maxc > eps, dm / maxc, 0.0)

    H = np.zeros(len(r), dtype="float32")
    mr = (maxc == r) & (dm > eps)
    mg = (maxc == g) & (dm > eps)
    mb = (maxc == b) & (dm > eps)
    H[mr] = ((g[mr] - b[mr]) / dm[mr]) % 6.0
    H[mg] = (b[mg] - r[mg]) / dm[mg] + 2.0
    H[mb] = (r[mb] - g[mb]) / dm[mb] + 4.0
    Hdeg = H * 60.0   # 0-360

    green_cover = float(((Hdeg >= 60) & (Hdeg <= 150) & (S >= 0.20) & (V >= 0.15)).mean())
    desicc_frac = float(((S <= 0.25) & (V >= 0.65)).mean())

    def _frac(Hmin, Hmax, Smin=0, Smax=1, Vmin=0, Vmax=1):
        return float(((Hdeg >= Hmin) & (Hdeg <= Hmax) & (S >= Smin) & (S <= Smax)
                      & (V >= Vmin) & (V <= Vmax)).mean())
    f_g = _frac(60, 150, 0.20, 1.0, 0.15, 1.0)
    f_y = _frac(25,  60, 0.20, 1.0, 0.30, 1.0)
    f_b = _frac(10,  35, 0.00, 0.60, 0.25, 0.95)
    mpi = (f_y + f_b) / max(f_g + f_y + f_b, eps)

    # -- HMI via OpenCV (0-179 -> *2 -> 0-358 deg) --
    rgb8 = np.clip(np.stack([r, g, b], axis=-1) * 255, 0, 255).astype("uint8")
    hsv_cv = cv2.cvtColor(rgb8.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
    h_cv   = hsv_cv[:, 0, 0].astype("float32") * 2.0

    def _hmi(h_arr, min_px=MIN_VEG_PIX):
        if len(h_arr) < min_px:
            return np.nan
        gpx = float(np.count_nonzero((h_arr >= HMI_GREEN_MIN) & (h_arr < HMI_GREEN_MAX)))
        ypx = float(np.count_nonzero((h_arr >= HMI_YELL_MIN)  & (h_arr < HMI_YELL_MAX)))
        veg = gpx + ypx
        return float(ypx / veg) if veg >= 1 else np.nan

    hmi_main   = _hmi(h_cv)
    exg_full   = 2*G0[finite] - R0[finite] - B0[finite]
    veg_mask   = exg_full >= EXG_MASK_THR
    hmi_masked = _hmi(h_cv[veg_mask]) if veg_mask.sum() >= MIN_VEG_PIX else np.nan

    # -- CIE Lab --
    _, a_arr, bst_arr = rgb01_to_lab(R0, G0, B0)
    a_f   = a_arr[finite]
    bst_f = bst_arr[finite]
    lab_a      = float(np.nanmedian(a_f))
    lab_b      = float(np.nanmedian(bst_f))
    lab_chroma = float(np.nanmedian(np.sqrt(a_f**2 + bst_f**2)))
    lab_hue    = float(np.nanmedian(np.degrees(np.arctan2(bst_f, a_f)) % 360.0))

    return {
        "GCC":         gcc,
        "RCC":         rcc,
        "NGRDI":       ngrdi,
        "VARI":        vari,
        "GLI":         gli,
        "MGRVI":       mgrvi,
        "RGBVI":       rgbvi,
        "ExGR":        exgr,
        "IKAW":        ikaw,
        "NDYI":        ndyi,
        "R_over_G":    r_og,
        "TGI":         tgi,
        "WI":          wi_val,
        "HMI_MASKED":  float(hmi_masked) if np.isfinite(hmi_masked) else np.nan,
        "HMI_MAIN":    float(hmi_main)   if np.isfinite(hmi_main)   else np.nan,
        "MPI":         float(mpi),
        "desicc_frac": desicc_frac,
        "green_cover": green_cover,
        "Lab_a":       lab_a,
        "Lab_b":       lab_b,
        "Lab_Chroma":  lab_chroma,
        "Lab_HueAngle":lab_hue,
    }


# =================== HISTOGRAM (for hist_ratio) =======================
def hs_histogram(r: np.ndarray, g: np.ndarray, b: np.ndarray,
                 h_bins: int = 36, s_bins: int = 10) -> np.ndarray:
    """Normalised 2-D Hue-Saturation histogram."""
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    dm   = maxc - minc
    S_   = np.where(maxc > 1e-6, dm / maxc, 0.0)
    H_   = np.zeros(len(r), dtype="float32")
    mr, mg, mb = (maxc == r) & (dm > 1e-6), (maxc == g) & (dm > 1e-6), (maxc == b) & (dm > 1e-6)
    H_[mr] = ((g[mr] - b[mr]) / dm[mr]) % 6.0
    H_[mg] = (b[mg] - r[mg]) / dm[mg] + 2.0
    H_[mb] = (r[mb] - g[mb]) / dm[mb] + 4.0
    Hdeg_  = H_ * 60.0
    Hi = np.clip((Hdeg_ / 360.0 * h_bins).astype(int), 0, h_bins - 1)
    Si = np.clip((S_ * s_bins).astype(int), 0, s_bins - 1)
    hist = np.zeros((h_bins, s_bins), float)
    np.add.at(hist, (Hi, Si), 1.0)
    return hist / (hist.sum() + 1e-12)


def bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    h1 = np.asarray(h1, float).ravel(); h1 /= h1.sum() + 1e-12
    h2 = np.asarray(h2, float).ravel(); h2 /= h2.sum() + 1e-12
    return float(-np.log(np.sum(np.sqrt(h1 * h2)) + 1e-12))


# ===================== MATURITY DETECTION =============================
def smooth3(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    if len(y) < 3:
        return y
    return uniform_filter1d(y, size=3, mode="nearest")


def _first_cross(x: np.ndarray, y: np.ndarray,
                 direction: str, thr: float) -> Optional[float]:
    for i in range(1, len(y)):
        y0, y1 = y[i - 1], y[i]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if direction == "decrease" and y0 > thr >= y1:
            frac = (thr - y0) / (y1 - y0)
            return float(x[i - 1] + frac * (x[i] - x[i - 1]))
        if direction == "increase" and y0 < thr <= y1:
            frac = (thr - y0) / (y1 - y0)
            return float(x[i - 1] + frac * (x[i] - x[i - 1]))
    return None


def _relative80(x: np.ndarray, y: np.ndarray, direction: str) -> Optional[float]:
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return None
    ymin, ymax = np.nanmin(y[valid]), np.nanmax(y[valid])
    if np.isclose(ymin, ymax):
        return None
    thr = (ymax - 0.80 * (ymax - ymin)) if direction == "decrease" \
          else (ymin + 0.80 * (ymax - ymin))
    return _first_cross(x, y, direction, thr)


def estimate_maturity(dap: np.ndarray, values: np.ndarray,
                      method: str) -> Optional[float]:
    cfg = METHODS.get(method)
    if cfg is None:
        return None
    y_sm = smooth3(np.asarray(values, float))
    direction = cfg["dir"]
    thr = cfg.get("thr")
    dap_mat = None
    if thr is not None:
        dap_mat = _first_cross(np.asarray(dap, float), y_sm, direction, thr)
    if dap_mat is None:
        dap_mat = _relative80(np.asarray(dap, float), y_sm, direction)
    return dap_mat


# =========================== PLOTTING =================================
def _style():
    plt.rcParams.update({
        "font.size":        FONT,
        "axes.titlesize":   TITLE_FONT,
        "axes.labelsize":   FONT,
        "xtick.labelsize":  FONT - 1,
        "ytick.labelsize":  FONT - 1,
        "legend.fontsize":  FONT - 1,
        "axes.spines.top":  False,
        "axes.spines.right":False,
    })


def plot_method_ts(pid: str, method: str,
                   dap: np.ndarray, values: np.ndarray,
                   maturity_dap: Optional[float],
                   out_path: str,
                   sowing_date: Optional[datetime] = None,
                   field_dap_min: Optional[float] = None,
                   field_dap_max: Optional[float] = None):
    """Single-method trajectory: raw dots (light blue) + MA-3 (orange) +
    threshold (dotted) + maturity vertical line with date label.
    Optional green band marks the user-supplied field maturity DAP range."""
    _style()
    cfg   = METHODS[method]
    y     = np.asarray(values, float)
    y_sm  = smooth3(y)
    x     = np.asarray(dap, float)
    valid = np.isfinite(y)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Field maturity range band — drawn first so it sits behind everything
    if field_dap_min is not None and field_dap_max is not None:
        ax.axvspan(field_dap_min, field_dap_max,
                   alpha=0.13, color="#27ae60", zorder=1,
                   label=f"Field range  {field_dap_min:.0f}–{field_dap_max:.0f} DAP")
        ax.axvline((field_dap_min + field_dap_max) / 2.0,
                   color="#27ae60", lw=1.6, ls="-.", zorder=2,
                   label=f"Field median  {(field_dap_min + field_dap_max) / 2:.1f} DAP")

    # Raw values  -- light blue dotted + scatter
    if valid.any():
        ax.plot(x[valid], y[valid], "o--", color="#aec6e8", ms=6, lw=1.2,
                alpha=0.80, zorder=3, label=f"{cfg['label']} (raw)")

    # MA-3 smoothed -- solid orange
    ax.plot(x, y_sm, "o-", color="#e67e22", ms=7, lw=2.2,
            zorder=4, label=f"{cfg['label']} (MA-3)")

    # Threshold line -- horizontal dotted blue
    if cfg["thr"] is not None:
        ax.axhline(cfg["thr"], linestyle=":", color="#3498db", lw=1.8,
                   label=f"Threshold = {cfg['thr']}")

    # Maturity vertical line -- dashed red with date label
    if maturity_dap is not None and np.isfinite(maturity_dap):
        ax.axvline(maturity_dap, linestyle="--", color="#e74c3c", lw=2.0,
                   label=f"Maturity ~ {maturity_dap:.0f} DAP")
        ax.scatter([maturity_dap], [np.interp(maturity_dap, x, y_sm)],
                   s=90, color="#e74c3c", zorder=6)
        if sowing_date is not None:
            mat_date = (sowing_date +
                        timedelta(days=int(round(maturity_dap)))).strftime("%Y-%m-%d")
            ymin, ymax = ax.get_ylim()
            ax.text(maturity_dap + 0.3, ymax * 0.97, mat_date,
                    rotation=90, va="top", ha="left",
                    fontsize=FONT - 3, color="#e74c3c", zorder=7)

    ax.set_xlabel("Days After Planting (DAP)")
    ax.set_ylabel(cfg["label"])
    ax.set_title(f"{cfg['desc']}\nPlot {pid}", fontsize=TITLE_FONT)
    ax.legend(framealpha=0.9, fontsize=FONT - 2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(pid: str, dap: np.ndarray,
                    ts_dict: Dict[str, list],
                    mat_dict: Dict[str, Optional[float]],
                    out_path: str,
                    field_dap_min: Optional[float] = None,
                    field_dap_max: Optional[float] = None):
    """All-methods comparison panel for one plot.
    Green band marks the user-supplied field maturity DAP range."""
    _style()
    methods = [m for m in METHOD_NAMES if m in ts_dict]
    n    = len(methods)
    cols = 4
    rows = math.ceil(n / cols)
    cmap = plt.cm.tab20(np.linspace(0, 1, n))

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 4.8, rows * 3.4),
                              squeeze=False)
    axes_flat = axes.ravel()

    for i, method in enumerate(methods):
        ax  = axes_flat[i]
        cfg = METHODS[method]
        y   = np.asarray(ts_dict[method], float)
        y_sm = smooth3(y)
        x   = np.asarray(dap, float)
        valid = np.isfinite(y)

        # Field range band (behind everything)
        if field_dap_min is not None and field_dap_max is not None:
            ax.axvspan(field_dap_min, field_dap_max,
                       alpha=0.13, color="#27ae60", zorder=0)
            ax.axvline((field_dap_min + field_dap_max) / 2.0,
                       color="#27ae60", lw=1.2, ls="-.", zorder=1)

        ax.plot(x[valid], y[valid], "o", color=cmap[i], ms=5, alpha=0.8)
        ax.plot(x, y_sm, "-", color=cmap[i], lw=1.8)

        if cfg["thr"] is not None:
            ax.axhline(cfg["thr"], linestyle="--", color="gray", lw=1.2, alpha=0.7)

        mdap = mat_dict.get(method)
        if mdap is not None and np.isfinite(mdap):
            ax.axvline(mdap, linestyle=":", color="#e74c3c", lw=1.8)
            ax.set_title(f"{cfg['label']}  [{mdap:.0f} DAP]", fontsize=FONT)
        else:
            ax.set_title(cfg["label"], fontsize=FONT)

        ax.set_xlabel("DAP", fontsize=FONT - 2)
        ax.set_ylabel(cfg["label"], fontsize=FONT - 2)
        ax.tick_params(labelsize=FONT - 3)
        ax.grid(True, alpha=0.20)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Add shared legend for field range in unused axis or suptitle note
    field_note = ""
    if field_dap_min is not None and field_dap_max is not None:
        field_note = (f"  |  ▌ Green band = field maturity range "
                      f"{field_dap_min:.0f}–{field_dap_max:.0f} DAP "
                      f"(median {(field_dap_min + field_dap_max) / 2:.1f})")

    fig.suptitle(f"All-Methods Maturity Comparison -- Plot {pid}{field_note}",
                 fontsize=TITLE_FONT + 1, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_summary_heatmap(pids: List[str],
                         mat_matrix: np.ndarray,
                         methods: List[str],
                         out_path: str):
    """Plots x Methods heatmap of maturity DAP."""
    _style()
    labels = [METHODS[m]["label"] for m in methods]
    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.0),
                                    max(6,  len(pids)   * 0.45)))
    im = ax.imshow(mat_matrix, aspect="auto", cmap="RdYlGn_r",
                   interpolation="nearest")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT - 1)
    ax.set_yticks(range(len(pids)))
    n_pid = len(pids)
    fs = max(5, FONT - n_pid // 25)
    ax.set_yticklabels(pids, fontsize=fs)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Maturity DAP", fontsize=FONT)
    ax.set_title("Maturity DAP -- All Plots x All Methods", fontsize=TITLE_FONT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_method_boxplot(method: str,
                        pids: List[str],
                        mat_daps: List[Optional[float]],
                        out_path: str):
    """Distribution of maturity DAP across plots for one method."""
    _style()
    cfg  = METHODS[method]
    vals = [v for v in mat_daps if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(vals, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#4477cc", alpha=0.6),
               medianprops=dict(color="black", linewidth=2))
    ax.set_ylabel("Maturity DAP")
    ax.set_title(f"{cfg['desc']}\nMaturity distribution (n={len(vals)} plots)")
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =================== DIAGNOSTIC PLOT HELPERS =========================

# Methods for which a pixel-distribution ridgeline is meaningful
_RIDGE_METHODS = [
    "GCC", "RCC", "NGRDI", "VARI", "GLI", "MGRVI", "RGBVI",
    "ExGR", "IKAW", "NDYI", "R_over_G", "TGI", "WI",
    "Lab_a", "Lab_b", "Lab_Chroma", "Lab_HueAngle",
]


def _compute_pixel_index(method: str,
                         r: np.ndarray, g: np.ndarray, b: np.ndarray,
                         a_star: Optional[np.ndarray] = None,
                         b_star: Optional[np.ndarray] = None) -> np.ndarray:
    """Return per-pixel index values (flat float32 array). r,g,b in [0,1]."""
    eps = 1e-6
    dc  = r + g + b + eps
    if method == "GCC":        return g / dc
    if method == "RCC":        return r / dc
    if method == "NGRDI":      return (g - r) / (g + r + eps)
    if method == "VARI":       return (g - r) / (g + r - b + eps)
    if method == "GLI":        return (2*g - r - b) / (2*g + r + b + eps)
    if method == "MGRVI":      return (g**2 - r**2) / (g**2 + r**2 + eps)
    if method == "RGBVI":      return (g**2 - b*r)  / (g**2 + b*r  + eps)
    if method == "ExGR":       return 3*g - 2.4*r - b
    if method == "IKAW":       return (r - b) / (r + b + eps)
    if method == "NDYI":       return ((r+g) - 2*b) / ((r+g) + 2*b + eps)
    if method == "R_over_G":   return np.where(g > eps, r / g, np.nan)
    if method == "TGI":        return g - 0.39*r - 0.61*b
    if method == "WI":
        denom = r - g + eps
        raw   = (g - b) / denom
        # WI can blow up when r~g; clip to sane range
        return np.clip(raw, -10, 10).astype("float32")
    if method == "Lab_a"       and a_star is not None: return a_star
    if method == "Lab_b"       and b_star is not None: return b_star
    if method == "Lab_Chroma"  and a_star is not None and b_star is not None:
        return np.sqrt(a_star**2 + b_star**2)
    if method == "Lab_HueAngle" and a_star is not None and b_star is not None:
        return (np.degrees(np.arctan2(b_star, a_star)) % 360.0).astype("float32")
    return np.full(len(r), np.nan, dtype="float32")


def _pixel_hist(values: np.ndarray, n_bins: int = 50,
                clip_pct: float = 1.0):
    """Build a normalised histogram from per-pixel values.
    Returns (centers, density) or (zeros, zeros) if too few values."""
    v = values[np.isfinite(values)]
    if len(v) < 20:
        c = np.linspace(0, 1, n_bins)
        return c, np.zeros(n_bins)
    lo = float(np.percentile(v, clip_pct))
    hi = float(np.percentile(v, 100 - clip_pct))
    if lo >= hi:
        hi = lo + 1e-6
    counts, bins = np.histogram(v, bins=n_bins, range=(lo, hi), density=True)
    centers = (bins[:-1] + bins[1:]) / 2.0
    return centers.astype("float32"), counts.astype("float32")


def plot_index_ridgeline(pid: str, method: str,
                         date_strs: list, daps: list,
                         centers_list: list, counts_list: list,
                         maturity_dap, out_path: str):
    """Stacked per-date ridgeline of pixel-level index distributions.
    Same visual approach as the hue histograms: one row per date,
    progressive color dark-purple -> yellow, threshold as vertical line."""
    n = len(date_strs)
    if n == 0:
        return
    _style()
    cfg    = METHODS[method]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, n))

    fig, axes = plt.subplots(n, 1,
                              figsize=(5.5, max(n * 1.15, 6)),
                              sharex=False,
                              gridspec_kw={"hspace": 0.08})
    if n == 1:
        axes = [axes]

    # Compute global x-range for a shared x-axis feel
    all_c = [c for c in centers_list if len(c) > 0]
    if all_c:
        x_min = float(min(c.min() for c in all_c))
        x_max = float(max(c.max() for c in all_c))
    else:
        x_min, x_max = 0.0, 1.0

    for i, (ax, ds, dp, centers, counts, color) in enumerate(
            zip(axes, date_strs, daps, centers_list, counts_list, colors)):

        if counts.sum() > 0:
            # Smooth with KDE-like gaussian filter for visual polish
            from scipy.ndimage import gaussian_filter1d
            smooth_counts = gaussian_filter1d(counts, sigma=1.5)
            ax.plot(centers, smooth_counts, color=color, lw=1.5, zorder=3)
            ax.fill_between(centers, smooth_counts,
                            alpha=0.15, color=color, zorder=2)
            # Median marker
            median_x = float(centers[np.argmax(counts)])
            ax.axvline(median_x, color=color, lw=0.8, alpha=0.6,
                       linestyle="-", zorder=4)

        # Threshold line (vertical dashed)
        if cfg["thr"] is not None:
            ax.axvline(cfg["thr"], color="#3498db", lw=1.2,
                       linestyle="--", alpha=0.8, zorder=5)

        # Highlight near-maturity row
        if (maturity_dap is not None and np.isfinite(maturity_dap)
                and abs(dp - maturity_dap) <= 6):
            ax.set_facecolor("#fff8f0")

        ax.text(0.98, 0.78, f"{int(dp)} DAP",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=color, fontweight="bold")
        ax.set_ylabel("Density", fontsize=7, labelpad=2)
        ax.set_xlim(x_min, x_max)
        ax.tick_params(labelsize=7, pad=1)
        ax.grid(True, alpha=0.15, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel(cfg["label"], fontsize=FONT)
    # Add threshold label to last axis if applicable
    if cfg["thr"] is not None:
        axes[-1].text(cfg["thr"], 0, f" thr={cfg['thr']}",
                      va="bottom", ha="left", fontsize=7,
                      color="#3498db", transform=axes[-1].get_xaxis_transform())

    fig.suptitle(f"{cfg['desc']}\nPixel Distribution -- Plot {pid}",
                 fontsize=TITLE_FONT, fontweight="bold")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_chromatic_scatter(pid: str, date_strs: list, daps: list,
                            r_chrom_list: list, g_chrom_list: list,
                            maturity_dap, out_path: str):
    """R-G chromatic coordinate scatter: r_c vs g_c per date.
    Progressive colors; medians connected by line showing maturity trajectory."""
    n = len(date_strs)
    if n == 0:
        return
    _style()
    colors = plt.cm.RdYlGn_r(np.linspace(0.05, 0.95, n))
    fig, ax = plt.subplots(figsize=(6.5, 6))

    med_r, med_g = [], []
    for i, (ds, dp, rc, gc, color) in enumerate(
            zip(date_strs, daps, r_chrom_list, g_chrom_list, colors)):
        if len(rc) == 0:
            continue
        # Sub-sample for plot clarity
        step = max(1, len(rc) // 600)
        ax.scatter(rc[::step], gc[::step],
                   s=3, color=color, alpha=0.20, zorder=2)
        mr, mg = float(np.nanmedian(rc)), float(np.nanmedian(gc))
        ax.scatter(mr, mg, s=80, color=color, edgecolors="black",
                   lw=0.7, zorder=5, label=f"DAP {int(dp)}")
        med_r.append(mr)
        med_g.append(mg)

    # Connect medians with arrow line
    if len(med_r) >= 2:
        for k in range(len(med_r) - 1):
            ax.annotate("", xy=(med_r[k+1], med_g[k+1]),
                        xytext=(med_r[k], med_g[k]),
                        arrowprops=dict(arrowstyle="->",
                                        color="gray", lw=1.2, alpha=0.7))

    # Chromatic triangle boundary guides
    ax.plot([0, 1/3], [1/3, 1/3], "--", color="gray", lw=0.6, alpha=0.4)
    ax.plot([1/3, 1/3], [0, 1/3], "--", color="gray", lw=0.6, alpha=0.4)
    ax.axvline(1/3, color="gray", lw=0.5, alpha=0.3)
    ax.axhline(1/3, color="gray", lw=0.5, alpha=0.3)
    ax.text(1/3 + 0.005, 1/3 + 0.005, "neutral\n(1/3, 1/3)",
            fontsize=7, color="gray", alpha=0.7)

    if maturity_dap is not None and np.isfinite(maturity_dap):
        ax.set_title(
            f"Chromatic Scatter -- Plot {pid}\n"
            f"Maturity ~{maturity_dap:.0f} DAP  (green->red = increasing DAP)",
            fontsize=FONT)
    else:
        ax.set_title(f"Chromatic Scatter -- Plot {pid}", fontsize=FONT)

    ax.set_xlabel("r  [R / (R+G+B)]")
    ax.set_ylabel("g  [G / (R+G+B)]")
    ax.set_xlim(0.15, 0.55)
    ax.set_ylim(0.25, 0.55)
    ax.legend(loc="lower right", fontsize=7, markerscale=2.5,
              framealpha=0.85, ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_method_interpretation(method: str, method_dir: str) -> None:
    """Generate and save interpretation.png for one method.
    Called once per method (not per plot)."""
    cfg  = METHODS[method]
    info = _METHOD_INFO.get(method, {})

    W = 14      # figure width inches
    # ---- colour palette ----
    C_BG      = "#f5f7fa"
    C_HEADER  = "#1a3a5c"
    C_ACCENT  = "#2980b9"
    C_BOX1    = "#eaf4fb"   # formula box
    C_BOX2    = "#eafaf1"   # detect box
    C_BOX3    = "#fdfefe"   # general sections
    C_TEXT    = "#2c3e50"
    C_MUTED   = "#555555"

    fig = plt.figure(figsize=(W, 11))
    fig.patch.set_facecolor(C_BG)

    def _rect(x, y, w, h, color, alpha=1.0, radius=0.01):
        ax = fig.add_axes([x, y, w, h])
        ax.set_facecolor(color)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.patch.set_alpha(alpha)
        return ax

    def _txt(ax, s, x=0.5, y=0.5, fs=11, bold=False, color=C_TEXT,
             ha="center", va="center", wrap_w=None):
        """Place wrapped text in an axes."""
        if wrap_w:
            s = "\n".join(textwrap.fill(p, wrap_w) for p in s.split("\n"))
        weight = "bold" if bold else "normal"
        ax.text(x, y, s, transform=ax.transAxes,
                fontsize=fs, fontweight=weight, color=color,
                ha=ha, va=va, multialignment="left",
                linespacing=1.55)

    # ── Header bar ──────────────────────────────────────────────────────
    ax_hdr = _rect(0, 0.89, 1.0, 0.11, C_HEADER)
    _txt(ax_hdr, cfg["desc"],           fs=17, bold=True,  color="white", y=0.65)
    _txt(ax_hdr, f"Method ID:  {method}  |  Folder prefix: {cfg['order']}_{method}",
         fs=10, color="#b0c8e8", y=0.22)

    # ── Row 1: Formula  +  Range  +  Direction/Threshold ────────────────
    ax_f = _rect(0.01, 0.77, 0.48, 0.11, C_BOX1)
    _txt(ax_f, "FORMULA", fs=8, bold=True, color=C_ACCENT, y=0.90, ha="center")
    _txt(ax_f, info.get("formula", "—"), fs=12, bold=True, color=C_HEADER,
         y=0.48, ha="center")
    _txt(ax_f, f"Range: {info.get('range', '—')}", fs=9, color=C_MUTED,
         y=0.12, ha="center")

    ax_d = _rect(0.51, 0.77, 0.48, 0.11, C_BOX2)
    _txt(ax_d, "MATURITY DIRECTION", fs=8, bold=True, color="#27ae60", y=0.90, ha="center")
    thr_txt = (f"Threshold:  {cfg['thr']}" if cfg["thr"] is not None
               else "Threshold:  None  (relative 80 % transition used)")
    dir_str = f"Direction:  {cfg['dir'].upper()}"
    _txt(ax_d, f"{dir_str}\n{thr_txt}", fs=11, bold=True, color=C_HEADER,
         y=0.48, ha="center")

    # ── Section helper ───────────────────────────────────────────────────
    SECTIONS = [
        ("WHAT DOES IT MEASURE?",    info.get("biology",  "—"), C_BOX3,  0.01, 0.54, 0.97, 0.21),
        ("HOW IS MATURITY DETECTED?", info.get("detect",  "—"), C_BOX3,  0.01, 0.37, 0.97, 0.15),
        ("HOW TO READ THE PLOTS",    info.get("reading",  "—"), C_BOX3,  0.01, 0.20, 0.97, 0.15),
        ("KNOWN LIMITATIONS",        info.get("limits",   "—"), "#fef9f0",0.01, 0.05, 0.97, 0.13),
    ]
    for title, body, bg, x, y, w, h in SECTIONS:
        ax_s = _rect(x, y, w, h, bg)
        # title strip at top
        ax_s.text(0.01, 0.92, title, transform=ax_s.transAxes,
                  fontsize=9, fontweight="bold", color=C_ACCENT,
                  va="top", ha="left")
        wrapped = "\n".join(textwrap.fill(line, 120) for line in body.split("\n"))
        ax_s.text(0.015, 0.72, wrapped, transform=ax_s.transAxes,
                  fontsize=10, color=C_TEXT, va="top", ha="left",
                  linespacing=1.5)

    # ── Footer ───────────────────────────────────────────────────────────
    ax_ft = _rect(0, 0.0, 1.0, 0.04, "#dce8f5")
    _txt(ax_ft, "Generated by MaturityAnalyzer  |  RGB-based crop maturity detection",
         fs=8, color="#555", y=0.5)

    out_path = os.path.join(method_dir, "interpretation.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=C_BG)
    plt.close(fig)


def _make_thumb(R: np.ndarray, G: np.ndarray, B: np.ndarray,
                max_dim: int = 220) -> np.ndarray:
    """Return a uint8 (H, W, 3) RGB thumbnail from float R/G/B (any scale)."""
    Rn, Gn, Bn = normalize_rgb(R, G, B)
    rgb = np.stack([Rn, Gn, Bn], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb = np.clip(rgb * 255, 0, 255).astype("uint8")
    h, w = rgb.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return rgb


def plot_rgb_strip(pid: str, date_strs: list, daps: list,
                   thumbnails: list, out_path: str):
    """Vertical stack of RGB crop thumbnails — one row per flight date."""
    n = len(thumbnails)
    if n == 0:
        return
    _style()

    # Scale all thumbnails to a common WIDTH (landscape crops are wide)
    TARGET_W = 300
    scaled = []
    for t in thumbnails:
        h, w = t.shape[:2]
        if w != TARGET_W:
            nh = max(1, int(h * TARGET_W / w))
            t = cv2.resize(t, (TARGET_W, nh), interpolation=cv2.INTER_AREA)
        scaled.append(t)

    # Each row: thumbnail on left, text label on right
    row_h_in = 1.3          # inches per row
    fig_w    = 7.0
    fig, axes = plt.subplots(n, 1,
                              figsize=(fig_w, n * row_h_in),
                              gridspec_kw={"hspace": 0.10})
    if n == 1:
        axes = [axes]

    for ax, img, ds, dp in zip(axes, scaled, date_strs, daps):
        ax.imshow(img, aspect="auto")
        ax.set_ylabel(f"DAP {int(dp)}\n{ds}",
                      fontsize=8, rotation=0, labelpad=4,
                      ha="right", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)

    fig.suptitle(f"RGB Thumbnails  |  Plot {pid}",
                 fontsize=FONT + 1, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_hue_histograms(pid: str, date_strs: list, daps: list,
                        hue_deg_list: list, maturity_dap,
                        out_path: str):
    """Stacked per-date KDE hue histograms (0-120 deg), one row per date.
    Progressive color (dark purple -> yellow), orange zone shading."""
    n = len(date_strs)
    if n == 0:
        return
    _style()
    # Color progression: dark purple -> bright yellow (viridis_r / plasma)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, n))

    ZONE_MIN, ZONE_MAX = 30, 60   # orange-shaded "ripening" hue zone
    X_MIN, X_MAX = 0, 120         # hue range of interest for cereals
    x_hue = np.linspace(X_MIN, X_MAX, 300)

    fig, axes = plt.subplots(n, 1,
                              figsize=(5, max(n * 1.15, 6)),
                              sharex=True,
                              gridspec_kw={"hspace": 0.08})
    if n == 1:
        axes = [axes]

    for i, (ax, ds, dp, hdeg, color) in enumerate(
            zip(axes, date_strs, daps, hue_deg_list, colors)):

        # Filter hue to display range
        h_filt = hdeg[(hdeg >= X_MIN) & (hdeg <= X_MAX)]

        if len(h_filt) >= 20:
            try:
                kde = gaussian_kde(h_filt, bw_method=0.12)
                y_kde = kde(x_hue)
            except Exception:
                counts, bins = np.histogram(h_filt, bins=40,
                                            range=(X_MIN, X_MAX), density=True)
                y_kde = np.interp(x_hue,
                                  (bins[:-1] + bins[1:]) / 2, counts)
        else:
            y_kde = np.zeros(len(x_hue))

        # Orange zone (ripening / yellow hue)
        ax.axvspan(ZONE_MIN, ZONE_MAX, alpha=0.18, color="#f0a500",
                   zorder=0)

        ax.plot(x_hue, y_kde, color=color, lw=1.5, zorder=3)
        ax.fill_between(x_hue, y_kde, alpha=0.12, color=color, zorder=2)

        # Mark near-maturity row with subtle red background
        if (maturity_dap is not None and np.isfinite(maturity_dap)
                and abs(dp - maturity_dap) <= 6):
            ax.set_facecolor("#fff0f0")

        # Label: "X DAP" in top-right
        ax.text(0.98, 0.78, f"{int(dp)} DAP",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=color, fontweight="bold")

        ax.set_ylabel("Rel.Freq.", fontsize=7, labelpad=2)
        ax.tick_params(labelsize=7, pad=1)
        ax.set_xlim(X_MIN, X_MAX)
        ax.grid(True, alpha=0.15, axis="x")
        # Remove most spines for a clean stacked look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Hue (°)", fontsize=FONT)
    fig.suptitle(f"Hue Histograms - PlotID {pid}",
                 fontsize=TITLE_FONT, fontweight="bold")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_hue_hs_density(pid: str, date_strs: list, daps: list,
                         hue_deg_list: list, sat_list: list,
                         out_path: str):
    """2-D Hue vs Saturation density plots across dates (HMI diagnostic)."""
    n = len(date_strs)
    if n == 0:
        return
    _style()
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 4.5, rows * 3.4),
                              squeeze=False)
    axes_flat = axes.ravel()

    h_edges = np.linspace(0, 360, 37)
    s_edges = np.linspace(0, 1, 11)
    vmax = 1e-12
    hists = []
    for hdeg, s in zip(hue_deg_list, sat_list):
        H2d, _, _ = np.histogram2d(hdeg, s, bins=[h_edges, s_edges])
        hists.append(H2d.T)
        vmax = max(vmax, H2d.max())

    for i, (ax, ds, dp, H2d) in enumerate(zip(axes_flat, date_strs, daps, hists)):
        im = ax.contourf(h_edges[:-1], s_edges[:-1], H2d / vmax,
                         levels=15, cmap="hot_r")
        ax.axvline(60,  color="#2ecc71", lw=1.2, alpha=0.7)
        ax.axvline(150, color="#2ecc71", lw=1.2, alpha=0.7, linestyle="--")
        ax.axvline(25,  color="#f1c40f", lw=1.2, alpha=0.7)
        ax.set_title(f"DAP {dp}  ({ds})", fontsize=FONT - 1)
        ax.set_xlabel("Hue (deg)")
        ax.set_ylabel("Saturation")
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 1)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Hue-Saturation Density -- Plot {pid}",
                 fontsize=TITLE_FONT, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_hue_3d(pid: str, date_strs: list, daps: list,
                hue_deg_list: list, out_path: str):
    """3D stacked hue histogram -- each date is a KDE curve at its DAP position."""
    n = len(date_strs)
    if n < 2:
        return
    try:
        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        return

    _style()
    X_MIN, X_MAX = 0, 120
    x_hue = np.linspace(X_MIN, X_MAX, 120)

    # Color: first half grey, second half purple-plasma (matching reference)
    n_grey = max(1, n // 2)
    n_purp = n - n_grey
    c_grey = [plt.cm.Greys(v)   for v in np.linspace(0.35, 0.60, n_grey)]
    c_purp = [plt.cm.plasma(v)  for v in np.linspace(0.10, 0.65, max(1, n_purp))]
    colors = c_grey + c_purp

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    for i, (dp, hdeg, color) in enumerate(zip(daps, hue_deg_list, colors)):
        h_filt = hdeg[(hdeg >= X_MIN) & (hdeg <= X_MAX)]
        if len(h_filt) < 20:
            y_kde = np.zeros(len(x_hue))
        else:
            try:
                y_kde = gaussian_kde(h_filt, bw_method=0.12)(x_hue)
            except Exception:
                y_kde = np.zeros(len(x_hue))

        # Build polygon vertices: baseline (z=0) -> curve -> baseline
        verts_x = np.concatenate([[x_hue[0]], x_hue, [x_hue[-1]]])
        verts_y = np.full_like(verts_x, float(dp))
        verts_z = np.concatenate([[0.0], y_kde, [0.0]])

        verts_3d = [list(zip(verts_x, verts_y, verts_z))]
        poly = Poly3DCollection(verts_3d, alpha=0.35,
                                facecolor=color, edgecolor="none")
        ax.add_collection3d(poly)
        ax.plot(x_hue, np.full(len(x_hue), float(dp)), y_kde,
                color=color, lw=1.4, alpha=0.9)

    ax.set_xlabel("Hue (°)",     fontsize=FONT - 1, labelpad=6)
    ax.set_ylabel("DAP",          fontsize=FONT - 1, labelpad=6)
    ax.set_zlabel("Rel.Freq.",    fontsize=FONT - 1, labelpad=6)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(min(daps), max(daps))
    ax.view_init(elev=28, azim=-55)
    ax.set_title(f"Hue 3D Stacked - PlotID {pid}",
                 fontsize=TITLE_FONT, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_exg_trend(pid: str, daps: list, exg_values: list,
                   maturity_dap, out_path: str,
                   sowing_date=None,
                   field_dap_min: Optional[float] = None,
                   field_dap_max: Optional[float] = None):
    """ExG PCHIP spline with peak-to-end decline slope.
    Returns the slope value (float) or None if insufficient data."""
    _style()
    x = np.asarray(daps, float)
    y = np.asarray(exg_values, float)
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return None

    xv, yv = x[valid], y[valid]
    xi = np.linspace(xv.min(), xv.max(), 400)
    yi = PchipInterpolator(xv, yv)(xi)

    # Find peak in smoothed curve
    peak_i = int(np.argmax(yi))
    peak_x, peak_y = float(xi[peak_i]), float(yi[peak_i])
    last_x, last_y = float(xv[-1]), float(yv[-1])
    slope = (last_y - peak_y) / (last_x - peak_x) if last_x != peak_x else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))

    # Field range band drawn first
    if field_dap_min is not None and field_dap_max is not None:
        ax.axvspan(field_dap_min, field_dap_max,
                   alpha=0.13, color="#27ae60", zorder=0,
                   label=f"Field range  {field_dap_min:.0f}–{field_dap_max:.0f} DAP")
        ax.axvline((field_dap_min + field_dap_max) / 2.0,
                   color="#27ae60", lw=1.5, ls="-.", zorder=1)

    ax.plot(xi, yi, "-", color="#3498db", lw=2.2, label="PCHIP Spline")
    ax.plot(xv, yv, "o", color="black", ms=7, zorder=5, label="Observed")

    # Highlight first and last observed points in orange
    ax.plot(xv[0],  yv[0],  "o", color="#e67e22", ms=11, zorder=6)
    ax.plot(xv[-1], yv[-1], "o", color="#e67e22", ms=11, zorder=6)

    # Decline slope line from peak to last point
    if last_x > peak_x:
        sl_x = np.array([peak_x, last_x])
        sl_y = peak_y + slope * (sl_x - peak_x)
        ax.plot(sl_x, sl_y, "-", color="#e74c3c", lw=2.0,
                label=f"Slope = {slope:.3f}")

    # Maturity line
    if maturity_dap is not None and np.isfinite(maturity_dap):
        lbl = f"Maturity ~{maturity_dap:.0f} DAP"
        if sowing_date:
            lbl += f"  ({(sowing_date + timedelta(days=int(round(maturity_dap)))).strftime('%Y-%m-%d')})"
        ax.axvline(maturity_dap, linestyle="--", color="#8e44ad", lw=1.8, label=lbl)

    ax.set_xlabel("Days After Planting (DAP)", fontsize=FONT)
    ax.set_ylabel("ExG Value", fontsize=FONT)
    ax.set_title(f"ExG Subset Regression - Plot {pid}", fontsize=TITLE_FONT)
    ax.legend(framealpha=0.9, fontsize=FONT - 2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return float(slope)


def plot_lab_scatter(pid: str, date_strs: list, daps: list,
                     a_list: list, b_list: list,
                     maturity_dap, out_path: str):
    """CIE Lab a* vs b* scatter per date, colored by time (Lab diagnostics)."""
    n = len(date_strs)
    if n == 0:
        return
    _style()
    cmap = plt.cm.RdYlGn_r(np.linspace(0.05, 0.95, max(n, 2)))
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (ds, dp, a_f, b_f) in enumerate(zip(date_strs, daps, a_list, b_list)):
        if len(a_f) == 0:
            continue
        ax.scatter(a_f[::max(1, len(a_f) // 400)],
                   b_f[::max(1, len(b_f) // 400)],
                   s=2, color=cmap[i], alpha=0.25)
        ax.scatter(float(np.nanmedian(a_f)), float(np.nanmedian(b_f)),
                   s=90, color=cmap[i], edgecolors="black", lw=0.8,
                   zorder=6, label=f"DAP {dp}")

    ax.axhline(0, color="gray", lw=0.6, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.6, alpha=0.5)
    ax.text(-20, 32, "Green\n(low a*)", fontsize=FONT - 3,
            color="#27ae60", alpha=0.8)
    ax.text(8,  32, "Yellow\n(high b*)", fontsize=FONT - 3,
            color="#e67e22", alpha=0.8)
    ax.text(8, -18, "Red/Brown\n(high a*)", fontsize=FONT - 3,
            color="#c0392b", alpha=0.8)

    tstr = f"  [Maturity ~{maturity_dap:.0f} DAP]" \
           if (maturity_dap is not None and np.isfinite(maturity_dap)) else ""
    ax.set_title(f"CIE Lab a*-b* Scatter -- Plot {pid}{tstr}", fontsize=FONT)
    ax.set_xlabel("a*  (green < 0 < red)")
    ax.set_ylabel("b*  (blue < 0 < yellow)")
    ax.legend(loc="best", fontsize=FONT - 3, markerscale=3, framealpha=0.85)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cover_stack(pid: str, date_strs: list, daps: list,
                     covers_list: list, maturity_dap, out_path: str):
    """Stacked bar of HSV class fractions per flight date (MPI / cover methods)."""
    n = len(date_strs)
    if n == 0:
        return
    _style()
    greens   = [c["green"]  for c in covers_list]
    yellows  = [c["yellow"] for c in covers_list]
    browns   = [c["brown"]  for c in covers_list]
    desiccs  = [c["desicc"] for c in covers_list]

    x = np.arange(n)
    w = 0.58
    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), 5))
    ax.bar(x, greens,                               w, label="Green (60-150 H)",       color="#2ecc71")
    ax.bar(x, yellows,  w, bottom=greens,           label="Yellow/Ripening (25-60 H)", color="#f1c40f")
    bot2 = [g + y for g, y in zip(greens, yellows)]
    ax.bar(x, browns,   w, bottom=bot2,             label="Brown/Senescent",           color="#a0522d")
    bot3 = [b2 + br for b2, br in zip(bot2, browns)]
    ax.bar(x, desiccs,  w, bottom=bot3,             label="Desiccated (low-S/high-V)", color="#d4c5a9")

    if maturity_dap is not None and np.isfinite(maturity_dap):
        # interpolate x position for maturity DAP
        xpos = np.interp(maturity_dap, daps, x)
        ax.axvline(xpos, color="#e74c3c", lw=2.2, linestyle="--",
                   label=f"Maturity ~{maturity_dap:.0f} DAP")

    ax.set_xticks(x)
    ax.set_xticklabels([f"DAP {d}\n{ds}" for d, ds in zip(daps, date_strs)],
                        fontsize=FONT - 2)
    ax.set_ylabel("Fraction of pixels")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"HSV Class Fractions -- Plot {pid}", fontsize=TITLE_FONT)
    ax.legend(loc="upper right", fontsize=FONT - 2)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_hs_hist_grid(pid: str, date_strs: list, daps: list,
                      hs_hists: list, out_path: str):
    """Grid of 2-D H-S histograms per date (hist_ratio diagnostic)."""
    n = len(hs_hists)
    if n == 0:
        return
    _style()
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 4.2, rows * 3.4),
                              squeeze=False)
    axes_flat = axes.ravel()
    vmax = max(h.max() for h in hs_hists) + 1e-12
    ims = []
    for i, (ax, ds, dp, hist) in enumerate(zip(axes_flat, date_strs, daps, hs_hists)):
        im = ax.imshow(hist.T, origin="lower", aspect="auto",
                       extent=[0, 360, 0, 1], vmin=0, vmax=vmax,
                       cmap="hot_r", interpolation="bilinear")
        ims.append(im)
        ax.axvline(60,  color="#2ecc71", lw=1, alpha=0.8)
        ax.axvline(150, color="#2ecc71", lw=1, alpha=0.8, linestyle="--")
        ax.axvline(25,  color="#f1c40f", lw=1, alpha=0.8)
        ax.set_title(f"DAP {dp}  ({ds})", fontsize=FONT - 1)
        ax.set_xlabel("Hue (deg)")
        ax.set_ylabel("Saturation")

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if ims:
        fig.colorbar(ims[-1], ax=axes_flat[:n].tolist(),
                     shrink=0.55, label="Norm. density")
    fig.suptitle(f"H-S Histograms -- Plot {pid}  (hist_ratio)",
                 fontsize=TITLE_FONT, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_plot_consensus(pid: str, mat_dict: Dict[str, Optional[float]],
                        sowing_date: datetime, out_path: str,
                        field_dap_min: Optional[float] = None,
                        field_dap_max: Optional[float] = None):
    """Per-plot: two-panel figure.
    LEFT  – histogram  X=DAP, Y=number of methods detecting maturity at that bin.
    RIGHT – horizontal bar showing every method's estimated DAP (or 'ND').
    Green band marks the user-supplied field maturity DAP range.
    Always saves a file even when 0 methods detected."""
    _style()
    BW = 3   # bin width in days

    detected = {m: float(v) for m, v in mat_dict.items()
                if v is not None and np.isfinite(float(v)
                                                 if v is not None else float("nan"))}

    def _dap2date(d):
        return (sowing_date + timedelta(days=int(round(d)))).strftime("%Y-%m-%d") \
               if sowing_date else str(int(d))

    # ── figure layout: left = histogram, right = per-method bar ────────
    fig, (ax_h, ax_b) = plt.subplots(
        1, 2, figsize=(15, max(6, len(METHOD_NAMES) * 0.32 + 2.5)),
        gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.38})

    # ── LEFT: DAP histogram ─────────────────────────────────────────────
    if len(detected) >= 1:
        daps_arr = np.array(list(detected.values()))
        lo = int(np.floor(daps_arr.min() / BW) * BW)
        hi = int(np.ceil(daps_arr.max()  / BW) * BW) + BW
        bins = np.arange(lo, hi + BW, BW)
        counts, edges = np.histogram(daps_arr, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        peak_idx = int(np.argmax(counts))
        peak_dap = float(centers[peak_idx])

        max_c = max(counts.max(), 1)
        bar_col = [plt.cm.Blues(0.30 + 0.60 * c / max_c) for c in counts]
        bar_col[peak_idx] = "#e74c3c"

        ax_h.bar(centers, counts, width=BW * 0.86,
                 color=bar_col, edgecolor="white", lw=0.6)

        # KDE overlay
        if len(daps_arr) >= 4:
            try:
                xi    = np.linspace(daps_arr.min(), daps_arr.max(), 300)
                kde_y = gaussian_kde(daps_arr, bw_method="silverman")(xi) \
                        * len(daps_arr) * BW
                ax_h.plot(xi, kde_y, "-", color="#e67e22", lw=2.0, label="KDE")
            except Exception:
                pass

        # Field range band on histogram
        if field_dap_min is not None and field_dap_max is not None:
            ax_h.axvspan(field_dap_min, field_dap_max,
                         alpha=0.18, color="#27ae60", zorder=0,
                         label=f"Field {field_dap_min:.0f}–{field_dap_max:.0f} DAP")

        ax_h.axvline(peak_dap, color="#e74c3c", lw=2.0, ls="--",
                     label=f"Peak  DAP {peak_dap:.0f}  ({_dap2date(peak_dap)})")
        if len(daps_arr) >= 2:
            ax_h.axvline(np.mean(daps_arr),   color="#8e44ad", lw=1.6, ls=":",
                         label=f"Mean  DAP {np.mean(daps_arr):.1f}")
            ax_h.axvline(np.median(daps_arr), color="#27ae60", lw=1.6, ls="-.",
                         label=f"Median DAP {np.median(daps_arr):.1f}")

        ax_h.set_xlabel("Days After Planting (DAP)", fontsize=FONT)
        ax_h.set_ylabel("Number of methods", fontsize=FONT)
        ax_h.set_title(
            f"Method Agreement Histogram\n"
            f"{len(detected)} / {len(mat_dict)} methods detected  "
            f"|  bin = {BW} DAP",
            fontsize=FONT)
        ax_h.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax_h.grid(True, alpha=0.25, axis="y")
        ax_h.legend(fontsize=FONT - 3, loc="upper left")
    else:
        ax_h.text(0.5, 0.5, "No methods detected\nmaturity for this plot",
                  ha="center", va="center", fontsize=FONT + 1,
                  color="#999", transform=ax_h.transAxes)
        ax_h.set_title("Method Agreement Histogram", fontsize=FONT)
        ax_h.axis("off")

    # ── RIGHT: horizontal bar, one row per method ───────────────────────
    all_methods = list(METHOD_NAMES)
    y_pos = np.arange(len(all_methods))
    bar_vals, bar_cols_r, tick_labels = [], [], []

    dap_vals = np.array([detected[m] for m in all_methods if m in detected])
    global_min = float(dap_vals.min()) if len(dap_vals) else 0
    global_max = float(dap_vals.max()) if len(dap_vals) else 1
    span       = max(global_max - global_min, 1)

    for m in all_methods:
        if m in detected:
            d = detected[m]
            bar_vals.append(d)
            norm = (d - global_min) / span
            bar_cols_r.append(plt.cm.RdYlGn_r(0.15 + 0.70 * norm))
            tick_labels.append(f"{m}  →  DAP {d:.0f}  ({_dap2date(d)})")
        else:
            bar_vals.append(0)
            bar_cols_r.append("#dddddd")
            tick_labels.append(f"{m}  →  not detected")

    ax_b.barh(y_pos, bar_vals, color=bar_cols_r,
              edgecolor="white", lw=0.4, height=0.75)

    if len(detected) >= 1:
        ax_b.axvline(float(np.mean(list(detected.values()))),
                     color="#8e44ad", lw=1.5, ls=":", label="Mean")
        ax_b.axvline(float(np.median(list(detected.values()))),
                     color="#27ae60", lw=1.5, ls="-.", label="Median")

    # Field range on bar chart
    if field_dap_min is not None and field_dap_max is not None:
        ax_b.axvspan(field_dap_min, field_dap_max,
                     alpha=0.14, color="#27ae60", zorder=0,
                     label=f"Field {field_dap_min:.0f}–{field_dap_max:.0f} DAP")
        ax_b.axvline((field_dap_min + field_dap_max) / 2.0,
                     color="#27ae60", lw=1.5, ls="-.", zorder=1)

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(tick_labels, fontsize=max(6, FONT - 4))
    ax_b.set_xlabel("Estimated maturity  (DAP)", fontsize=FONT)
    ax_b.set_title("Per-method maturity estimates\n(grey bars = not detected)",
                   fontsize=FONT)
    ax_b.grid(True, alpha=0.20, axis="x")
    if len(detected) >= 1 or (field_dap_min is not None):
        ax_b.legend(fontsize=FONT - 3)

    fig.suptitle(f"Maturity Method Consensus  |  Plot {pid}",
                 fontsize=TITLE_FONT + 1, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_trial_histogram(trial_name: str, all_daps: list,
                         n_plots: int, sowing_date: datetime,
                         out_path: str,
                         method_name_daps: Optional[Dict] = None):
    """Trial-level figure: three panels.
    TOP   – main histogram  X=DAP, Y=total method×plot detection frequency.
    MIDDLE – KDE curves per method (stacked, small multiples approach).
    BOTTOM – horizontal bar: per-method mean DAP ± SD across all plots.
    Always saves even when very few detections exist."""
    _style()
    BW = 3

    def _dap2date(d):
        return (sowing_date + timedelta(days=int(round(d)))).strftime("%Y-%m-%d") \
               if sowing_date else str(int(d))

    daps_arr = np.array([v for v in all_daps if np.isfinite(v)], dtype=float)

    # ── Per-method stats ────────────────────────────────────────────────
    m_rows = []   # (name, mean, std, n, all_vals)
    if method_name_daps:
        for mname in METHOD_NAMES:
            vals = np.array([v for v in method_name_daps.get(mname, [])
                             if np.isfinite(v)])
            if len(vals) > 0:
                m_rows.append((mname, float(np.mean(vals)),
                               float(np.std(vals)), len(vals), vals))

    n_methods_with_data = len(m_rows)
    has_method_panel    = n_methods_with_data >= 1

    fig_h = 6.0 + (0.38 * n_methods_with_data if has_method_panel else 0)
    n_rows_fig = 2 if has_method_panel else 1
    height_ratios = [2, max(1, 0.30 * n_methods_with_data)] \
                    if has_method_panel else [1]

    fig, axes = plt.subplots(
        n_rows_fig, 1,
        figsize=(14, fig_h),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.45})
    if n_rows_fig == 1:
        axes = [axes]
    ax_main = axes[0]

    # ── TOP: main histogram ─────────────────────────────────────────────
    if len(daps_arr) >= 1:
        lo = int(np.floor(daps_arr.min() / BW) * BW)
        hi = int(np.ceil(daps_arr.max()  / BW) * BW) + BW
        bins = np.arange(lo, hi + BW, BW)
        counts, edges = np.histogram(daps_arr, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        peak_idx = int(np.argmax(counts))
        peak_dap = float(centers[peak_idx])

        max_c = max(counts.max(), 1)
        bar_col = [plt.cm.YlOrRd(0.25 + 0.65 * c / max_c) for c in counts]
        bar_col[peak_idx] = "#c0392b"

        ax_main.bar(centers, counts, width=BW * 0.88,
                    color=bar_col, edgecolor="white", lw=0.5, alpha=0.90,
                    label="Frequency (method × plot)")

        # KDE
        if len(daps_arr) >= 5:
            try:
                xi    = np.linspace(daps_arr.min(), daps_arr.max(), 400)
                kde_y = gaussian_kde(daps_arr, bw_method="silverman")(xi) \
                        * len(daps_arr) * BW
                ax_main.plot(xi, kde_y, "-", color="#2c3e50", lw=2.5,
                             label="KDE density", zorder=5)
            except Exception:
                pass

        mean_d   = float(np.mean(daps_arr))
        median_d = float(np.median(daps_arr))
        ax_main.axvline(peak_dap, color="#c0392b", lw=2.2, ls="--",
                        label=f"Peak    DAP {peak_dap:.0f}  ({_dap2date(peak_dap)})", zorder=6)
        ax_main.axvline(mean_d,   color="#8e44ad", lw=1.8, ls=":",
                        label=f"Mean    DAP {mean_d:.1f}  ({_dap2date(mean_d)})",   zorder=6)
        ax_main.axvline(median_d, color="#27ae60", lw=1.8, ls="-.",
                        label=f"Median  DAP {median_d:.1f}  ({_dap2date(median_d)})", zorder=6)

        # Dual x-axis (top = calendar dates)
        ax_top = ax_main.twiny()
        raw_ticks = ax_main.get_xticks()
        ticks = [t for t in raw_ticks
                 if daps_arr.min() - BW*2 <= t <= daps_arr.max() + BW*2]
        ax_top.set_xlim(ax_main.get_xlim())
        ax_top.set_xticks(ticks)
        ax_top.set_xticklabels(
            [_dap2date(t) for t in ticks],
            fontsize=FONT - 3, rotation=30, ha="left")
        ax_top.set_xlabel("Calendar date", fontsize=FONT - 2, labelpad=2)

        ax_main.set_xlabel("Days After Planting (DAP)", fontsize=FONT)
        ax_main.set_ylabel("Frequency  (method × plot)", fontsize=FONT)
        ax_main.set_title(
            f"Maturity DAP Distribution  |  Trial: {trial_name}\n"
            f"{n_plots} plots  ×  {len(METHOD_NAMES)} methods  "
            f"=  {len(daps_arr)} total detections   |  bin = {BW} DAP",
            fontsize=TITLE_FONT)
        ax_main.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax_main.grid(True, alpha=0.25, axis="y")
        ax_main.legend(fontsize=FONT - 2, loc="upper left")
    else:
        ax_main.text(0.5, 0.5,
                     f"Trial '{trial_name}' — no maturity detections",
                     ha="center", va="center", fontsize=FONT + 1, color="#999",
                     transform=ax_main.transAxes)
        ax_main.axis("off")

    # ── BOTTOM: per-method mean DAP horizontal bar ──────────────────────
    if has_method_panel:
        ax_m = axes[1]
        m_rows_sorted = sorted(m_rows, key=lambda r: r[1])  # sort by mean DAP
        names  = [f"{r[0]}  (n={r[3]})" for r in m_rows_sorted]
        means  = [r[1] for r in m_rows_sorted]
        stds   = [r[2] for r in m_rows_sorted]
        y_pos  = np.arange(len(names))

        mn_arr = np.array(means)
        norm   = (mn_arr - mn_arr.min()) / (mn_arr.max() - mn_arr.min() + 1e-6)
        colors = [plt.cm.YlOrRd(0.20 + 0.72 * v) for v in norm]

        ax_m.barh(y_pos, means, xerr=stds,
                  color=colors, edgecolor="white", lw=0.4,
                  error_kw=dict(ecolor="#555", lw=1.0, capsize=3),
                  height=0.72)
        ax_m.set_yticks(y_pos)
        ax_m.set_yticklabels(names, fontsize=max(6, FONT - 4))

        if len(daps_arr) >= 1:
            ax_m.axvline(float(np.mean(daps_arr)),
                         color="#8e44ad", lw=1.5, ls=":", label="Overall mean")
            ax_m.axvline(float(np.median(daps_arr)),
                         color="#27ae60", lw=1.5, ls="-.", label="Overall median")
            if len(daps_arr) >= 1:
                ax_m.axvline(peak_dap, color="#c0392b", lw=1.5, ls="--",
                             label="Peak bin")

        ax_m.set_xlabel("Mean maturity DAP  (error bars = ±1 SD across plots)",
                        fontsize=FONT - 1)
        ax_m.set_title(
            "Per-method mean maturity DAP  "
            "(sorted earliest → latest, only methods with ≥1 detection shown)",
            fontsize=FONT - 1)
        ax_m.grid(True, alpha=0.20, axis="x")
        ax_m.legend(fontsize=FONT - 3, loc="lower right")

    fig.suptitle(f"Trial  {trial_name}  —  Maturity DAP Frequency",
                 fontsize=TITLE_FONT + 2, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ========================= ExG SLOPE SUMMARY ==========================

def plot_exg_slope_summary(pid_slopes: Dict[str, float],
                            trial_name: str,
                            out_path: str,
                            sowing_date=None,
                            field_dap_min: Optional[float] = None,
                            field_dap_max: Optional[float] = None):
    """Trial-level summary of ExG decline slopes.
    Two-panel figure:
      TOP  – bar chart of slope per plot (sorted steepest decline first).
      BOTTOM – scatter: peak DAP vs slope with regression line.
    """
    if not pid_slopes:
        return
    _style()

    pids   = list(pid_slopes.keys())
    slopes = np.array([pid_slopes[p] for p in pids], dtype=float)

    # Sort by slope (most negative = fastest decline = first)
    order  = np.argsort(slopes)
    pids_s = [pids[i] for i in order]
    slopes_s = slopes[order]

    # Colors: red for steepest decline, yellow for moderate, green for slow/positive
    norm_c = (slopes_s - slopes_s.min()) / (slopes_s.max() - slopes_s.min() + 1e-8)
    colors = [plt.cm.RdYlGn(v) for v in norm_c]

    fig, (ax_bar, ax_sc) = plt.subplots(
        2, 1, figsize=(max(10, len(pids) * 0.45 + 3), 10),
        gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.45})

    # ── TOP: bar chart ────────────────────────────────────────────────
    y_pos = np.arange(len(pids_s))
    ax_bar.barh(y_pos, slopes_s, color=colors, edgecolor="white", lw=0.5)
    ax_bar.axvline(0, color="black", lw=1.0, alpha=0.5)
    ax_bar.axvline(float(np.mean(slopes)), color="#8e44ad",
                   lw=1.8, ls=":", label=f"Mean slope = {np.mean(slopes):.3f}")
    ax_bar.axvline(float(np.median(slopes)), color="#2980b9",
                   lw=1.8, ls="-.", label=f"Median slope = {np.median(slopes):.3f}")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([f"Plot {p}" for p in pids_s],
                            fontsize=max(6, FONT - 3))
    ax_bar.set_xlabel("ExG Decline Slope  (ExG units / DAP)", fontsize=FONT)
    ax_bar.set_title(
        f"ExG Decline Slope per Plot  |  Trial: {trial_name}\n"
        f"More negative = faster senescence  "
        f"(n={len(pids)} plots,  mean={np.mean(slopes):.3f},  SD={np.std(slopes):.3f})",
        fontsize=TITLE_FONT)
    ax_bar.grid(True, alpha=0.22, axis="x")
    ax_bar.legend(fontsize=FONT - 2)

    # ── BOTTOM: slope distribution as strip / histogram ───────────────
    bins   = min(max(5, len(pids) // 3), 20)
    n_c, e_c = np.histogram(slopes, bins=bins)
    cents  = (e_c[:-1] + e_c[1:]) / 2.0
    mx_c   = max(n_c.max(), 1)
    b_cols = [plt.cm.RdYlGn((c - slopes.min()) / (slopes.max() - slopes.min() + 1e-8))
              for c in cents]
    ax_sc.bar(cents, n_c, width=(e_c[1] - e_c[0]) * 0.88,
              color=b_cols, edgecolor="white", lw=0.5, alpha=0.90)
    ax_sc.axvline(float(np.mean(slopes)), color="#8e44ad",
                  lw=1.8, ls=":", label="Mean")
    ax_sc.axvline(float(np.median(slopes)), color="#2980b9",
                  lw=1.8, ls="-.", label="Median")

    # KDE overlay on bottom panel
    if len(slopes) >= 4:
        try:
            xi_k   = np.linspace(slopes.min(), slopes.max(), 300)
            bw_k   = slopes.std() * (len(slopes) ** -0.2)
            kde_y  = gaussian_kde(slopes, bw_method="silverman")(xi_k) \
                     * len(slopes) * (e_c[1] - e_c[0])
            ax_sc.plot(xi_k, kde_y, "-", color="#2c3e50", lw=2.0, label="KDE")
        except Exception:
            pass

    ax_sc.set_xlabel("ExG Decline Slope  (ExG units / DAP)", fontsize=FONT)
    ax_sc.set_ylabel("Number of plots", fontsize=FONT)
    ax_sc.set_title("Slope Distribution across all plots", fontsize=FONT)
    ax_sc.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_sc.grid(True, alpha=0.22, axis="y")
    ax_sc.legend(fontsize=FONT - 2)

    fig.suptitle(f"ExG Regression Slope Summary  —  Trial: {trial_name}",
                 fontsize=TITLE_FONT + 2, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ========================= FIELD vs PREDICTION COMPARISON =============

def _iqr_trim_arr(arr: np.ndarray) -> np.ndarray:
    """Remove values outside 1.5×IQR fence from Q1/Q3."""
    if len(arr) < 4:
        return arr
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]


def plot_field_vs_prediction(pid: str,
                              field_mean: float, field_sd: float,
                              pred_daps: list,
                              out_path: str,
                              sowing_date=None):
    """Cross-plot: Field Data reference (user-supplied range) vs
    Prediction (IQR-trimmed distribution of all methods).
    Style matches the reference image: vertical error-bar crosses,
    orange dot for the prediction centre, double-headed Δ arrow."""

    pred_arr  = np.array([v for v in pred_daps if np.isfinite(v)], dtype=float)
    pred_trim = _iqr_trim_arr(pred_arr)

    if len(pred_trim) > 0:
        pred_mean = float(np.mean(pred_trim))
        pred_sd   = float(np.std(pred_trim, ddof=0))
    else:
        pred_mean = np.nan
        pred_sd   = 0.0

    delta = abs(field_mean - pred_mean) if np.isfinite(pred_mean) else np.nan

    def _d2cal(d):
        return (sowing_date + timedelta(days=int(round(d)))).strftime("%b %d")  \
               if sowing_date and np.isfinite(d) else ""

    # ── figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    CAP = 22     # capsize in points (wide horizontal bar)
    ELW = 2.5    # error-bar line width
    MW  = 10     # marker size

    # Field data — blue cross + blue filled dot
    ax.errorbar(0, field_mean, yerr=field_sd,
                fmt="o", color="#4472C4", ecolor="#4472C4",
                capsize=CAP, capthick=2.2, elinewidth=ELW,
                markersize=MW, markerfacecolor="#4472C4",
                markeredgecolor="#4472C4", zorder=5)

    # Prediction — blue cross + orange filled dot
    if np.isfinite(pred_mean):
        ax.errorbar(1, pred_mean, yerr=pred_sd,
                    fmt="o", color="#4472C4", ecolor="#4472C4",
                    capsize=CAP, capthick=2.2, elinewidth=ELW,
                    markersize=MW, markerfacecolor="#e67e22",
                    markeredgecolor="#e67e22", zorder=5)

    # Double-headed Δ arrow between the two means
    if np.isfinite(delta) and delta > 0.01:
        y_lo  = min(field_mean, pred_mean)
        y_hi  = max(field_mean, pred_mean)
        mid_x = 0.5
        ax.annotate("",
                    xy=(mid_x, y_lo), xytext=(mid_x, y_hi),
                    arrowprops=dict(arrowstyle="<->", color="black",
                                   lw=2.5, mutation_scale=20))
        ax.text(mid_x + 0.07, (y_lo + y_hi) / 2,
                f"\u0394 = {delta:.2f} DAP",
                fontsize=FONT, va="center", ha="left", fontweight="bold")

    # Calendar-date subtitle below tick labels
    if sowing_date:
        cal_field = _d2cal(field_mean)
        cal_pred  = _d2cal(pred_mean) if np.isfinite(pred_mean) else "N/A"
        ax.set_xticklabels([
            f"Field Data\n({cal_field})",
            f"Prediction\n({cal_pred})"
        ], fontsize=FONT + 1)
    else:
        ax.set_xticklabels(["Field Data", "Prediction"], fontsize=FONT + 1)

    ax.set_xticks([0, 1])
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylabel("Days After Planting (DAP)", fontsize=FONT)

    n_used = len(pred_trim)
    n_total = len(pred_arr)
    ax.set_title(
        f"{pid}  —  Mean \u00b1 SD  (IQR-trimmed)\n"
        f"Prediction: {n_used}/{n_total} methods  |  "
        f"Field range: {field_mean - field_sd:.1f} – {field_mean + field_sd:.1f} DAP",
        fontsize=TITLE_FONT, fontweight="bold")

    ax.grid(True, axis="y", alpha=0.35, ls="--", color="#aaaaaa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def generate_field_comparison(summary_xlsx: str,
                               dap_min: float, dap_max: float,
                               out_dir: str,
                               sowing_date=None) -> str:
    """Read SUMMARY.xlsx and generate Field vs Prediction comparison plots.

    Parameters
    ----------
    summary_xlsx : path to the SUMMARY.xlsx produced by run_pipeline
    dap_min / dap_max : user-specified field maturity DAP range
    out_dir : root output directory (same as pipeline output_root)
    sowing_date : datetime used in the pipeline run (for calendar labels)

    Returns
    -------
    Path to the Field_Comparison/ folder.
    """
    df = pd.read_excel(summary_xlsx)

    field_mean = (dap_min + dap_max) / 2.0
    field_sd   = (dap_max - dap_min) / 2.0

    comp_dir = os.path.join(out_dir, "Field_Comparison")
    os.makedirs(comp_dir, exist_ok=True)

    all_pred_daps: list = []   # pool of every method prediction across all plots

    for _, row in df.iterrows():
        pid = str(row.get("PlotID", "unknown"))
        pred_daps = []
        for m in METHOD_NAMES:
            col = f"{m}_DAP"
            if col in row.index and pd.notna(row[col]):
                try:
                    pred_daps.append(float(row[col]))
                except (TypeError, ValueError):
                    pass

        all_pred_daps.extend(pred_daps)

        plot_field_vs_prediction(
            pid, field_mean, field_sd, pred_daps,
            os.path.join(comp_dir, f"FieldComp_Plot{pid}.png"),
            sowing_date=sowing_date)

    # Overall summary across all plots
    plot_field_vs_prediction(
        "All Plots", field_mean, field_sd, all_pred_daps,
        os.path.join(comp_dir, "FieldComp_AllPlots_Summary.png"),
        sowing_date=sowing_date)

    return comp_dir


# ========================= MAIN PIPELINE ==============================
def run_pipeline(
    images_dir: str,
    vector_path: str,
    layer_name: str,
    sowing_date: datetime,
    output_root: str,
    plot_id_field: str = "PlotID",
    log_fn=None,
    progress_fn=None,
    field_dap_min: Optional[float] = None,
    field_dap_max: Optional[float] = None,
) -> str:
    """
    Full analysis pipeline.
    log_fn(str)  : called with progress messages.
    progress_fn(float 0-100) : called with percentage.
    field_dap_min / field_dap_max : optional user-supplied field maturity DAP range.
      When provided: green band is added to all trajectory/comparison plots,
      Field_Comparison/ folder is auto-generated at end, ExG_Slope/ folder created.
    Returns path to SUMMARY.xlsx.
    """
    def log(msg):
        if log_fn:
            log_fn(msg)

    def prog(p):
        if progress_fn:
            progress_fn(min(100.0, float(p)))

    os.makedirs(output_root, exist_ok=True)

    # -- Locate images --------------------------------------------------
    log("Scanning image directory ...")
    shots = collect_dated_images(images_dir)
    if len(shots) < 2:
        raise RuntimeError(f"Only {len(shots)} dated image set(s) found -- need >= 2.")
    log(f"  Found {len(shots)} dated image sets.")

    # -- Load vector layer ---------------------------------------------
    # Strategy: pyogrio first (handles GDB integer fields correctly),
    # then fall back to fiona. Mirrors the old working app exactly.
    log("Loading vector layer ...")

    gdf: Optional[gpd.GeoDataFrame] = None

    # -- Attempt 1: pyogrio.read_dataframe() directly ------------------
    # This is the most reliable path for ESRI GDB files.
    # Bypasses gpd.read_file() entirely -- works even when geopandas
    # cannot locate pyogrio through its internal plugin loader.
    try:
        import pyogrio as _pyogrio
        _kw = {"layer": layer_name} if (vector_path.lower().endswith(".gdb") and layer_name) else {}
        _g = _pyogrio.read_dataframe(vector_path, **_kw)
        if _g is not None and len(_g) > 0:
            gdf = _g
            log(f"  Loaded with pyogrio.read_dataframe  ({len(gdf)} features)")
    except ImportError:
        log("  pyogrio not available -- trying fiona/gdal ...")
    except Exception as _pe:
        log(f"  pyogrio.read_dataframe: {_pe}")

    # -- Attempts 2-5: gpd.read_file() with various driver/engine options
    if gdf is None or len(gdf) == 0:
        def _read_gpd(driver: Optional[str] = None, engine: Optional[str] = None) -> gpd.GeoDataFrame:
            kw: Dict[str, str] = {}
            if driver:  kw["driver"] = driver
            if engine:  kw["engine"] = engine
            if vector_path.lower().endswith(".gdb"):
                return gpd.read_file(vector_path, layer=layer_name, **kw)
            return gpd.read_file(vector_path, **kw)

        fallbacks = [
            ("gpd driver=OpenFileGDB",  dict(driver="OpenFileGDB")),
            ("gpd engine=pyogrio",      dict(engine="pyogrio")),
            ("gpd driver=FileGDB",      dict(driver="FileGDB")),
            ("gpd fiona default",       {}),
        ]
        for label, kw in fallbacks:
            try:
                _g = _read_gpd(**kw)
                if _g is not None and len(_g) > 0:
                    gdf = _g
                    log(f"  Loaded with {label}  ({len(gdf)} features)")
                    break
                else:
                    log(f"  {label}: 0 features -- trying next")
            except Exception as _err:
                log(f"  {label}: {_err}")

    if gdf is None or len(gdf) == 0:
        raise RuntimeError(
            "Could not load the vector layer.\n\n"
            "All reading methods returned 0 features for this GDB.\n\n"
            "SOLUTIONS (try in order):\n"
            "  1. Run in terminal:  pip install pyogrio --force-reinstall\n"
            "     Then restart the app.\n\n"
            "  2. In QGIS or ArcGIS: export this layer as a Shapefile (.shp),\n"
            "     then select 'Shapefile' format in the app instead of GDB.\n\n"
            "  3. Make sure the Layer Name field matches exactly (case-sensitive)."
        )

    # -- Clean column names --------------------------------------------
    gdf.columns = [c.strip() for c in gdf.columns]
    data_cols = [c for c in gdf.columns if c != "geometry"]
    log(f"  Columns : {data_cols}")

    # Show first 3 rows so the user can verify values immediately
    try:
        log(f"  First 3 rows:\n{gdf[data_cols].head(3).to_string(index=False, max_colwidth=20)}")
    except Exception:
        pass

    # -- Resolve PlotID column (case-insensitive) ----------------------
    plot_id_col: Optional[str] = None
    if plot_id_field in gdf.columns:
        plot_id_col = plot_id_field
    else:
        for col in gdf.columns:
            if col.lower() == plot_id_field.lower():
                plot_id_col = col
                log(f"  Matched '{plot_id_field}' to '{col}' (case-insensitive)")
                break
    if plot_id_col is None:
        raise ValueError(
            f"Plot ID field '{plot_id_field}' not found.\n"
            f"Available columns: {data_cols}"
        )

    if gdf.crs is None:
        raise ValueError("Vector layer has no CRS defined.")
    gdf_crs = gdf.crs

    # -- Log diagnostics of the PlotID column -------------------------
    raw_series = gdf[plot_id_col]
    log(f"  '{plot_id_col}' dtype   : {raw_series.dtype}")
    log(f"  Non-null rows           : {int(raw_series.notna().sum())} / {len(raw_series)}")
    log(f"  Sample values           : {raw_series.head(8).tolist()}")

    # -- Convert to clean string IDs -----------------------------------
    def _to_pid(v) -> Optional[str]:
        s = str(v).strip()
        return None if s.lower() in ("nan", "none", "<na>", "nat", "") else s

    pids = sorted(
        {v for v in (_to_pid(x) for x in raw_series) if v is not None},
        key=lambda x: (len(x), x)
    )

    if len(pids) == 0:
        raise ValueError(
            f"0 unique plot IDs found in column '{plot_id_col}'.\n"
            f"Raw values seen: {raw_series.head(10).tolist()}\n\n"
            f"The GDB was loaded ({len(gdf)} features) but '{plot_id_col}' "
            f"contains only null/empty values.\n"
            f"Try the 'Preview' button to inspect your layer data."
        )

    log(f"  {len(pids)} unique PlotIDs to process.")
    gdf["_pid_str"] = [_to_pid(v) for v in raw_series]

    # -- Detect trial / grouping column --------------------------------
    _trial_col: Optional[str] = None
    for _tc in ["Trial", "trial", "TRIAL", "Experiment", "experiment",
                "Block", "block", "Site", "site", "Year", "year"]:
        if _tc in gdf.columns:
            _trial_col = _tc
            log(f"  Trial grouping column: '{_trial_col}'")
            break
    if _trial_col is None:
        log("  No trial column found -- all plots grouped as 'All_Plots'.")

    # pid_str -> trial label
    _pid_trial: Dict[str, str] = {}
    for _, _row in gdf.iterrows():
        _ps = _to_pid(_row[plot_id_col])
        if _ps:
            _tv = str(_row[_trial_col]).strip() if _trial_col else "All_Plots"
            _pid_trial[_ps] = _tv

    # -- Build output folder tree ---------------------------------------
    ts_dir        = os.path.join(output_root, "TimeSeries");        os.makedirs(ts_dir,        exist_ok=True)
    comp_dir      = os.path.join(output_root, "Comparison");        os.makedirs(comp_dir,      exist_ok=True)
    consensus_dir = os.path.join(output_root, "Method_Consensus");  os.makedirs(consensus_dir, exist_ok=True)
    debug_dir     = os.path.join(output_root, "00_RGB_Thumbnails"); os.makedirs(debug_dir,     exist_ok=True)
    exg_slope_dir = os.path.join(output_root, "ExG_Slope");         os.makedirs(exg_slope_dir, exist_ok=True)

    log("Creating method interpretation cards ...")
    method_dirs: Dict[str, str] = {}
    for m, cfg in METHODS.items():
        d = os.path.join(output_root, f"{cfg['order']}_{m}")
        os.makedirs(d, exist_ok=True)
        method_dirs[m] = d
        # Save one interpretation card per method folder (done once, not per plot)
        try:
            save_method_interpretation(m, d)
        except Exception as _ie:
            log(f"  Interpretation card error ({m}): {_ie}")
    log(f"  {len(method_dirs)} interpretation cards saved.")

    daps = [(s["date"] - sowing_date).days for s in shots]
    dap_arr = np.array(daps, float)

    # -- Accumulators ----------------------------------------------------
    summary_rows: List[Dict] = []
    method_mat_dap: Dict[str, List] = {m: [] for m in METHOD_NAMES}

    # Trial-level accumulator: {trial: {method: [dap_values per plot]}}
    trial_acc: Dict[str, Dict[str, List[float]]] = {}

    # ExG slope per plot: {pid: slope_value}
    exg_slope_data: Dict[str, float] = {}

    total_work = len(pids) * len(shots)
    done_work  = 0

    for pid_idx, pid in enumerate(pids):
        log(f"\n-> PlotID {pid}  ({pid_idx + 1}/{len(pids)})")
        geoms = list(gdf.loc[gdf["_pid_str"] == pid, "geometry"])
        if not geoms:
            log("  No geometries -- skipping.")
            for m in METHOD_NAMES:
                method_mat_dap[m].append(np.nan)
            continue

        ts: Dict[str, List] = {m: [] for m in METHOD_NAMES}
        hs_hists: List[np.ndarray] = []
        valid_shot_indices: List[int] = []

        # -- Diagnostic accumulators (method-specific plots) -----------
        _d_dates:   List[str]           = []
        _d_daps:    List[float]         = []
        _d_thumbs:  List[np.ndarray]    = []
        _d_hdeg:    List[np.ndarray]    = []   # hue angles
        _d_sat:     List[np.ndarray]    = []   # saturation
        _d_a_star:  List[np.ndarray]    = []   # CIE Lab a*
        _d_b_star:  List[np.ndarray]    = []   # CIE Lab b*
        _d_covers:  List[Dict]          = []   # class fractions
        # Per-method pixel histograms for ridgeline plots
        _d_ridge_centers: Dict[str, List] = {m: [] for m in _RIDGE_METHODS}
        _d_ridge_counts:  Dict[str, List] = {m: [] for m in _RIDGE_METHODS}
        # Chromatic coords for GCC/RCC scatter
        _d_r_chrom: List[np.ndarray]    = []
        _d_g_chrom: List[np.ndarray]    = []

        # -- Per-date extraction ----------------------------------------
        for shot_idx, shot in enumerate(shots):
            result = None
            try:
                src_path = _spec_main_path(shot["spec"])
                with rasterio.open(src_path) as src:
                    raster_crs    = src.crs
                    raster_bounds = shapely_box(*src.bounds)

                geoms_rpr  = [reproj_geom(g, gdf_crs, raster_crs) for g in geoms]
                geom_union = unary_union(geoms_rpr)

                # skip if no spatial overlap
                if not geom_union.intersects(raster_bounds):
                    raise ValueError("Plot does not overlap this image.")

                R, G, B = read_rgb_clipped(shot["spec"], geom_union)

                result = compute_all_indices(R, G, B)
                if result is not None:
                    fin        = np.isfinite(R) & np.isfinite(G) & np.isfinite(B)
                    Rn, Gn, Bn = normalize_rgb(R, G, B)
                    hs_hists.append(hs_histogram(Rn[fin], Gn[fin], Bn[fin]))
                    valid_shot_indices.append(shot_idx)

                    # -- collect diagnostic data -------------------------
                    Rf, Gf, Bf = Rn[fin], Gn[fin], Bn[fin]
                    _d_dates.append(shot["date"].strftime("%Y-%m-%d"))
                    _d_daps.append(float(daps[shot_idx]))

                    # RGB thumbnail
                    _d_thumbs.append(_make_thumb(R, G, B))

                    # HSV hue & saturation (subsampled to 3000 px max)
                    _rgb_s = np.stack([Rf, Gf, Bf], axis=-1)
                    _maxc  = np.max(_rgb_s, axis=-1)
                    _minc  = np.min(_rgb_s, axis=-1)
                    _dm    = _maxc - _minc
                    _S     = np.where(_maxc > 1e-6, _dm / _maxc, 0.0)
                    _H     = np.zeros(len(Rf), "float32")
                    _eps   = 1e-6
                    _mr = (_maxc == Rf) & (_dm > _eps)
                    _mg = (_maxc == Gf) & (_dm > _eps)
                    _mb = (_maxc == Bf) & (_dm > _eps)
                    _H[_mr] = ((Gf[_mr] - Bf[_mr]) / _dm[_mr]) % 6.0
                    _H[_mg] = (Bf[_mg] - Rf[_mg]) / _dm[_mg] + 2.0
                    _H[_mb] = (Rf[_mb] - Gf[_mb]) / _dm[_mb] + 4.0
                    _Hdeg  = _H * 60.0
                    _Vf    = _maxc
                    MAX_PX = 3000
                    if len(_Hdeg) > MAX_PX:
                        _idx = np.random.choice(len(_Hdeg), MAX_PX, replace=False)
                        _Hdeg = _Hdeg[_idx]
                        _S    = _S[_idx]
                        _Vf   = _Vf[_idx]
                    _d_hdeg.append(_Hdeg)
                    _d_sat.append(_S)

                    # CIE Lab a*, b* (subsampled)
                    _, _a_f, _b_f = rgb01_to_lab(Rn, Gn, Bn)
                    _a_f, _b_f = _a_f[fin], _b_f[fin]
                    if len(_a_f) > MAX_PX:
                        _idx2 = np.random.choice(len(_a_f), MAX_PX, replace=False)
                        _a_f, _b_f = _a_f[_idx2], _b_f[_idx2]
                    _d_a_star.append(_a_f)
                    _d_b_star.append(_b_f)

                    # Per-method pixel histograms for ridgeline plots
                    for _rm in _RIDGE_METHODS:
                        try:
                            if _rm in ("Lab_a", "Lab_b"):
                                _pv = _a_f if _rm == "Lab_a" else _b_f
                            elif _rm == "Lab_Chroma":
                                _pv = np.sqrt(_a_f**2 + _b_f**2)
                            elif _rm == "Lab_HueAngle":
                                _pv = (np.degrees(np.arctan2(_b_f, _a_f)) % 360.0).astype("float32")
                            else:
                                _pv = _compute_pixel_index(_rm, Rf, Gf, Bf)
                            _c, _cnt = _pixel_hist(_pv)
                            _d_ridge_centers[_rm].append(_c)
                            _d_ridge_counts[_rm].append(_cnt)
                        except Exception:
                            _d_ridge_centers[_rm].append(np.zeros(50))
                            _d_ridge_counts[_rm].append(np.zeros(50))

                    # Chromatic coords for GCC/RCC scatter (subsampled)
                    _denom_chr = Rf + Gf + Bf + 1e-6
                    _rc = (Rf / _denom_chr).astype("float32")
                    _gc = (Gf / _denom_chr).astype("float32")
                    if len(_rc) > MAX_PX:
                        _idx3 = np.random.choice(len(_rc), MAX_PX, replace=False)
                        _rc, _gc = _rc[_idx3], _gc[_idx3]
                    _d_r_chrom.append(_rc)
                    _d_g_chrom.append(_gc)

                    # HSV class cover fractions (use full valid pixels, not subsampled)
                    _Hf_full = _H * 60.0   # _H is still full-size here
                    _Sf_full = np.where(_maxc > 1e-6, _dm / _maxc, 0.0)
                    _Vf_full = _maxc
                    _d_covers.append({
                        "green":  float(np.mean((_Hf_full >= 60)  & (_Hf_full <= 150) & (_Sf_full >= 0.20) & (_Vf_full >= 0.15))),
                        "yellow": float(np.mean((_Hf_full >= 25)  & (_Hf_full <=  60) & (_Sf_full >= 0.20) & (_Vf_full >= 0.30))),
                        "brown":  float(np.mean((_Hf_full >= 10)  & (_Hf_full <=  35) & (_Sf_full <= 0.60) & (_Vf_full >= 0.25) & (_Vf_full <= 0.95))),
                        "desicc": float(np.mean((_Sf_full <= 0.25) & (_Vf_full >= 0.65))),
                    })

            except Exception as e:
                log(f"  Warning [{shot['date'].date()}]: {e}")

            for m in METHOD_NAMES:
                val = result[m] if (result and m in result) else np.nan
                ts[m].append(val)

            done_work += 1
            prog(100.0 * done_work / total_work)

        # -- Compute hist_ratio ----------------------------------------
        hr_full = [np.nan] * len(daps)
        if len(hs_hists) >= 2:
            h_first = hs_hists[0].ravel()
            h_last  = hs_hists[-1].ravel()
            for local_i, global_i in enumerate(valid_shot_indices):
                d_to_first = bhattacharyya(hs_hists[local_i].ravel(), h_first)
                d_to_last  = bhattacharyya(hs_hists[local_i].ravel(), h_last)
                hr_full[global_i] = d_to_first / (d_to_last + 1e-6)
        ts["hist_ratio"] = hr_full

        # -- Save time-series CSV --------------------------------------
        df_ts = pd.DataFrame({
            "DAP":  daps,
            "Date": [s["date"].strftime("%Y-%m-%d") for s in shots],
            **{m: ts[m] for m in METHOD_NAMES}
        })
        df_ts.to_csv(os.path.join(ts_dir, f"ts_{pid}.csv"), index=False)

        # -- Estimate maturity per method ------------------------------
        mat_row = {"PlotID": pid}
        plot_mat: Dict[str, Optional[float]] = {}

        for m in METHOD_NAMES:
            y_arr = np.array(ts[m], float)
            mdap  = estimate_maturity(dap_arr, y_arr, m)
            plot_mat[m] = mdap
            method_mat_dap[m].append(mdap if mdap is not None else np.nan)

            if mdap is not None and np.isfinite(mdap):
                mat_row[f"{m}_DAP"]  = round(mdap, 1)
                mat_row[f"{m}_Date"] = (sowing_date + timedelta(days=int(round(mdap)))).strftime("%Y-%m-%d")
                log(f"  {m}: {mdap:.0f} DAP")
            else:
                mat_row[f"{m}_DAP"]  = pd.NA
                mat_row[f"{m}_Date"] = pd.NA
                log(f"  {m}: not detected")

        summary_rows.append(mat_row)

        # -- Per-plot consensus histogram --------------------------------
        try:
            plot_plot_consensus(
                pid, plot_mat, sowing_date,
                os.path.join(consensus_dir, f"consensus_Plot{pid}.png"),
                field_dap_min=field_dap_min,
                field_dap_max=field_dap_max)
        except Exception as _ce:
            log(f"  Consensus chart error: {_ce}")

        # -- Feed trial accumulator -------------------------------------
        _trial_lbl = _pid_trial.get(pid, "All_Plots")
        if _trial_lbl not in trial_acc:
            trial_acc[_trial_lbl] = {m: [] for m in METHOD_NAMES}
        for _m, _dv in plot_mat.items():
            if _dv is not None and np.isfinite(_dv):
                trial_acc[_trial_lbl][_m].append(float(_dv))

        # -- Method-specific diagnostic plots (now plot_mat is ready) -----
        if _d_dates:
            # 1. RGB thumbnails strip (shared debug folder)
            try:
                plot_rgb_strip(pid, _d_dates, _d_daps, _d_thumbs,
                               os.path.join(debug_dir,
                                            f"rgb_strip_Plot{pid}.png"))
            except Exception as _e:
                log(f"  rgb_strip error: {_e}")

            # 2. HMI methods: hue histograms + 3D + H-S density + ExG trend
            for _hm in ("HMI_MASKED", "HMI_MAIN"):
                try:
                    plot_hue_histograms(
                        pid, _d_dates, _d_daps, _d_hdeg,
                        plot_mat.get(_hm),
                        os.path.join(method_dirs[_hm],
                                     f"hue_hist_Plot{pid}.png"))
                    plot_hue_3d(
                        pid, _d_dates, _d_daps, _d_hdeg,
                        os.path.join(method_dirs[_hm],
                                     f"hue_3D_Plot{pid}.png"))
                    plot_hue_hs_density(
                        pid, _d_dates, _d_daps, _d_hdeg, _d_sat,
                        os.path.join(method_dirs[_hm],
                                     f"hue_density_Plot{pid}.png"))
                    plot_exg_trend(
                        pid, list(dap_arr), list(ts["ExGR"]),
                        plot_mat.get(_hm),
                        os.path.join(method_dirs[_hm],
                                     f"ExG_trend_Plot{pid}.png"),
                        sowing_date=sowing_date,
                        field_dap_min=field_dap_min,
                        field_dap_max=field_dap_max)
                except Exception as _e:
                    log(f"  HMI diag error ({_hm}): {_e}")

            # ExG_Slope folder: dedicated per-plot slope figure (run once, not per HMI)
            try:
                _slope_val = plot_exg_trend(
                    pid, list(dap_arr), list(ts["ExGR"]),
                    plot_mat.get("HMI_MASKED"),
                    os.path.join(exg_slope_dir, f"ExG_slope_Plot{pid}.png"),
                    sowing_date=sowing_date,
                    field_dap_min=field_dap_min,
                    field_dap_max=field_dap_max)
                if _slope_val is not None and np.isfinite(_slope_val):
                    exg_slope_data[pid] = _slope_val
                    log(f"  ExG slope: {_slope_val:.4f}")
            except Exception as _e:
                log(f"  ExG slope error: {_e}")

            # 3. Lab methods: a* vs b* scatter + ridgeline
            for _lm in ("Lab_a", "Lab_b", "Lab_Chroma", "Lab_HueAngle"):
                try:
                    plot_lab_scatter(
                        pid, _d_dates, _d_daps, _d_a_star, _d_b_star,
                        plot_mat.get(_lm),
                        os.path.join(method_dirs[_lm],
                                     f"lab_scatter_Plot{pid}.png"))
                except Exception as _e:
                    log(f"  Lab scatter error ({_lm}): {_e}")
                try:
                    plot_index_ridgeline(
                        pid, _lm, _d_dates, _d_daps,
                        _d_ridge_centers[_lm], _d_ridge_counts[_lm],
                        plot_mat.get(_lm),
                        os.path.join(method_dirs[_lm],
                                     f"dist_ridgeline_Plot{pid}.png"))
                except Exception as _e:
                    log(f"  Lab ridgeline error ({_lm}): {_e}")

            # 4. Cover methods: stacked bar
            for _cm in ("desicc_frac", "green_cover", "MPI"):
                try:
                    plot_cover_stack(
                        pid, _d_dates, _d_daps, _d_covers,
                        plot_mat.get(_cm),
                        os.path.join(method_dirs[_cm],
                                     f"cover_stack_Plot{pid}.png"))
                except Exception as _e:
                    log(f"  Cover stack error ({_cm}): {_e}")

            # 5. Ridgeline distribution for ratio/chromatic methods
            for _rm in _RIDGE_METHODS:
                if _rm in ("Lab_a", "Lab_b", "Lab_Chroma", "Lab_HueAngle"):
                    continue   # already handled above with scatter
                if _rm not in method_dirs:
                    continue
                try:
                    plot_index_ridgeline(
                        pid, _rm, _d_dates, _d_daps,
                        _d_ridge_centers[_rm], _d_ridge_counts[_rm],
                        plot_mat.get(_rm),
                        os.path.join(method_dirs[_rm],
                                     f"dist_ridgeline_Plot{pid}.png"))
                except Exception as _e:
                    log(f"  Ridgeline error ({_rm}): {_e}")

            # 6. Chromatic scatter for GCC and RCC
            for _cm2 in ("GCC", "RCC"):
                if _cm2 not in method_dirs:
                    continue
                try:
                    plot_chromatic_scatter(
                        pid, _d_dates, _d_daps,
                        _d_r_chrom, _d_g_chrom,
                        plot_mat.get(_cm2),
                        os.path.join(method_dirs[_cm2],
                                     f"chromatic_scatter_Plot{pid}.png"))
                except Exception as _e:
                    log(f"  Chromatic scatter error ({_cm2}): {_e}")

            # 7. hist_ratio: H-S histogram grid
            if hs_hists:
                try:
                    _hs_dates = [shots[i]["date"].strftime("%Y-%m-%d")
                                 for i in valid_shot_indices]
                    _hs_daps  = [float(daps[i]) for i in valid_shot_indices]
                    plot_hs_hist_grid(
                        pid, _hs_dates, _hs_daps, hs_hists,
                        os.path.join(method_dirs["hist_ratio"],
                                     f"hs_grid_Plot{pid}.png"))
                except Exception as _e:
                    log(f"  HS hist grid error: {_e}")

        # -- Per-method trajectory figures -----------------------------
        for m in METHOD_NAMES:
            out_fig = os.path.join(method_dirs[m], f"{m}_Plot{pid}.png")
            try:
                plot_method_ts(pid, m, dap_arr, np.array(ts[m], float),
                               plot_mat[m], out_fig,
                               sowing_date=sowing_date,
                               field_dap_min=field_dap_min,
                               field_dap_max=field_dap_max)
            except Exception as e:
                log(f"  Figure error ({m}): {e}")

        # -- Comparison figure -----------------------------------------
        try:
            plot_comparison(pid, dap_arr,
                            {m: ts[m] for m in METHOD_NAMES},
                            plot_mat,
                            os.path.join(comp_dir, f"Comparison_Plot{pid}.png"),
                            field_dap_min=field_dap_min,
                            field_dap_max=field_dap_max)
        except Exception as e:
            log(f"  Comparison figure error: {e}")

    # -- Per-method boxplot + stats CSV ---------------------------------
    log("\nSaving per-method statistics ...")
    for m in METHOD_NAMES:
        vals = [v for v in method_mat_dap[m] if np.isfinite(v)]
        # boxplot
        try:
            plot_method_boxplot(m, pids, method_mat_dap[m],
                                os.path.join(method_dirs[m], f"{m}_distribution.png"))
        except Exception:
            pass
        # stats CSV
        if vals:
            stats = {
                "method": m, "n_plots": len(pids),
                "n_detected": len(vals),
                "mean_DAP": round(float(np.mean(vals)), 1),
                "std_DAP":  round(float(np.std(vals)), 1),
                "min_DAP":  round(float(np.min(vals)), 1),
                "max_DAP":  round(float(np.max(vals)), 1),
            }
            pd.DataFrame([stats]).to_csv(
                os.path.join(method_dirs[m], f"stats_{m}.csv"), index=False)

    # -- Summary heatmap ------------------------------------------------
    log("Generating summary heatmap ...")
    hm_methods = [m for m in METHOD_NAMES
                  if any(np.isfinite(v) for v in method_mat_dap[m])]
    if pids and hm_methods:
        mat_matrix = np.array(
            [[float(method_mat_dap[m][i])
              if (method_mat_dap[m][i] is not None and np.isfinite(method_mat_dap[m][i]))
              else np.nan
              for m in hm_methods]
             for i in range(len(pids))],
            dtype=float
        )
        try:
            plot_summary_heatmap(
                pids, mat_matrix, hm_methods,
                os.path.join(output_root, "Summary_Heatmap.png")
            )
        except Exception as e:
            log(f"Heatmap error: {e}")

    # -- Trial maturity histograms ---------------------------------------
    log("Generating trial maturity histograms ...")
    trials_dir = os.path.join(output_root, "Trial_Histograms")
    os.makedirs(trials_dir, exist_ok=True)

    # Always generate a histogram for every trial group (even when 0 detections –
    # plot_trial_histogram renders a "no detections" notice in that case).
    for _tname, _m_daps in trial_acc.items():
        # Flatten all method×plot DAPs for this trial
        _all_trial_daps = [v for vals in _m_daps.values() for v in vals]
        _n_plots_trial  = len({pid for pid in pids
                                if _pid_trial.get(pid, "All_Plots") == _tname})
        # Log how many detections we have (informational only — never skip)
        log(f"  Trial '{_tname}': {len(_all_trial_daps)} detection(s) from "
            f"{_n_plots_trial} plot(s).")
        try:
            plot_trial_histogram(
                trial_name      = _tname,
                all_daps        = _all_trial_daps,
                n_plots         = _n_plots_trial,
                sowing_date     = sowing_date,
                out_path        = os.path.join(trials_dir,
                                               f"Trial_{_tname}_histogram.png"),
                method_name_daps= _m_daps)
            log(f"    → histogram saved.")
        except Exception as _te:
            log(f"  Trial histogram error ({_tname}): {_te}")

    # -- ExG slope trial summary ----------------------------------------
    log("Generating ExG slope summary ...")
    if exg_slope_data:
        # One summary per trial (group by trial label)
        _trial_slope_acc: Dict[str, Dict[str, float]] = {}
        for _spid, _sval in exg_slope_data.items():
            _strial = _pid_trial.get(_spid, "All_Plots")
            _trial_slope_acc.setdefault(_strial, {})[_spid] = _sval
        for _stname, _sdata in _trial_slope_acc.items():
            try:
                plot_exg_slope_summary(
                    pid_slopes=_sdata,
                    trial_name=_stname,
                    out_path=os.path.join(
                        exg_slope_dir, f"ExG_slope_trial_{_stname}.png"),
                    sowing_date=sowing_date,
                    field_dap_min=field_dap_min,
                    field_dap_max=field_dap_max)
                log(f"  ExG slope summary saved: trial {_stname} "
                    f"({len(_sdata)} plots)")
            except Exception as _se:
                log(f"  ExG slope summary error ({_stname}): {_se}")
    else:
        log("  No ExG slope data collected (ExGR values all NaN?).")

    # -- Auto field comparison (if range was supplied) ------------------
    if field_dap_min is not None and field_dap_max is not None:
        log(f"Generating Field vs Prediction comparison "
            f"(range {field_dap_min:.0f}–{field_dap_max:.0f} DAP) ...")
        try:
            _sumxlsx = os.path.join(output_root, "SUMMARY.xlsx")
            # Write a temporary SUMMARY first so generate_field_comparison can read it
            _df_tmp = pd.DataFrame(summary_rows)
            if not _df_tmp.empty:
                # Collect per-plot predictions directly from summary_rows
                _field_mean = (field_dap_min + field_dap_max) / 2.0
                _field_sd   = (field_dap_max  - field_dap_min) / 2.0
                _fc_dir = os.path.join(output_root, "Field_Comparison")
                os.makedirs(_fc_dir, exist_ok=True)
                _all_pred: list = []
                for _sr in summary_rows:
                    _pid_fc = str(_sr.get("PlotID", ""))
                    _pred_fc = [float(_sr[f"{_m}_DAP"])
                                for _m in METHOD_NAMES
                                if f"{_m}_DAP" in _sr
                                and _sr[f"{_m}_DAP"] is not None
                                and pd.notna(_sr[f"{_m}_DAP"])]
                    _all_pred.extend(_pred_fc)
                    plot_field_vs_prediction(
                        _pid_fc, _field_mean, _field_sd, _pred_fc,
                        os.path.join(_fc_dir,
                                     f"FieldComp_Plot{_pid_fc}.png"),
                        sowing_date=sowing_date)
                # Overall summary
                plot_field_vs_prediction(
                    "All Plots", _field_mean, _field_sd, _all_pred,
                    os.path.join(_fc_dir,
                                 "FieldComp_AllPlots_Summary.png"),
                    sowing_date=sowing_date)
                log(f"  Field comparison saved → {_fc_dir}")
        except Exception as _fce:
            log(f"  Field comparison error: {_fce}")

    # -- Write SUMMARY.xlsx ---------------------------------------------
    log("Writing SUMMARY.xlsx ...")
    df_sum = pd.DataFrame(summary_rows)
    if df_sum.empty:
        log("  WARNING: No plots were processed -- SUMMARY.xlsx will be empty.")
    elif "PlotID" in df_sum.columns:
        df_sum = df_sum.sort_values("PlotID")
    ordered_cols = ["PlotID"]
    for m in METHOD_NAMES:
        ordered_cols += [f"{m}_DAP", f"{m}_Date"]
    existing = [c for c in ordered_cols if c in df_sum.columns]
    df_sum = df_sum[existing]

    summary_xlsx = os.path.join(output_root, "SUMMARY.xlsx")
    with pd.ExcelWriter(summary_xlsx, engine="xlsxwriter") as writer:
        df_sum.to_excel(writer, index=False, sheet_name="Maturity_Summary")
        ws  = writer.sheets["Maturity_Summary"]
        fmt_hdr  = writer.book.add_format({"bold": True, "bg_color": "#1f4e79",
                                           "font_color": "#ffffff", "border": 1})
        fmt_cell = writer.book.add_format({"border": 1})
        for col_idx, col_name in enumerate(df_sum.columns):
            width = max(len(str(col_name)),
                        df_sum[col_name].astype(str).str.len().max() + 2)
            ws.set_column(col_idx, col_idx, min(int(width), 22), fmt_cell)
            ws.write(0, col_idx, col_name, fmt_hdr)

        # Second sheet: method descriptions
        desc_data = [{"Method": m, "Label": cfg["label"],
                      "Description": cfg["desc"],
                      "Maturity direction": cfg["dir"],
                      "Threshold": cfg["thr"] if cfg["thr"] is not None else "relative 80%"}
                     for m, cfg in METHODS.items()]
        pd.DataFrame(desc_data).to_excel(writer, index=False, sheet_name="Method_Descriptions")

    log(f"\nSUMMARY.xlsx  ->  {summary_xlsx}")
    prog(100)
    return summary_xlsx
