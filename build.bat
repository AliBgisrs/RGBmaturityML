@echo off
REM ============================================================
REM  Build MaturityAnalyzer.exe  (Windows)
REM  Run this from the project folder after:
REM    pip install -r requirements.txt
REM ============================================================

echo Installing / upgrading PyInstaller and pyogrio...
pip install --upgrade pyinstaller
pip install pyogrio

echo.
echo Building executable...

pyinstaller ^
  --onefile ^
  --windowed ^
  --name "MaturityAnalyzer" ^
  --add-data "analysis.py;." ^
  --hidden-import pyogrio ^
  --hidden-import pyogrio._io ^
  --hidden-import rasterio ^
  --hidden-import rasterio._shim ^
  --hidden-import rasterio.control ^
  --hidden-import rasterio.crs ^
  --hidden-import rasterio.drivers ^
  --hidden-import rasterio._warp ^
  --hidden-import fiona ^
  --hidden-import fiona._shim ^
  --hidden-import fiona.ogrext ^
  --hidden-import geopandas ^
  --hidden-import shapely ^
  --hidden-import shapely.geometry ^
  --hidden-import shapely.ops ^
  --hidden-import cv2 ^
  --hidden-import pandas ^
  --hidden-import numpy ^
  --hidden-import matplotlib ^
  --hidden-import scipy ^
  --hidden-import xlsxwriter ^
  --collect-all pyogrio ^
  --collect-all rasterio ^
  --collect-all fiona ^
  --collect-all pyproj ^
  maturity_app.py

echo.
echo ============================================================
echo  Done!  Find your EXE in:  dist\MaturityAnalyzer.exe
echo ============================================================
pause
