@echo off
chcp 65001 >nul

:: === Force all PyTorch / CUDA caches to F drive (inline, do NOT rely on setx) ===
set CUDA_CACHE_PATH=F:\cuda_pytorch_cache\cuda_compute
set TORCH_EXTENSIONS_DIR=F:\cuda_pytorch_cache\torch_extensions
set TORCH_HOME=F:\cuda_pytorch_cache\torch_hub
set TMP=F:\cuda_pytorch_cache\tmp
set TEMP=F:\cuda_pytorch_cache\tmp
if not exist "F:\cuda_pytorch_cache\tmp" mkdir "F:\cuda_pytorch_cache\tmp"

echo ============================================================
echo PointMLP-Elite v6 - Breed-grouped 5-Fold CV Training
echo Resume run: T + ALL (A and LY already completed 2026-04-17)
echo Started: %date% %time%
echo CUDA_CACHE_PATH   = %CUDA_CACHE_PATH%
echo TORCH_EXTENSIONS_DIR = %TORCH_EXTENSIONS_DIR%
echo TMP/TEMP          = %TMP%
echo ============================================================

set PYTHON=D:\Anaconda\envs\pytorch\python.exe
cd /d "%~dp0"

REM === Skipped: A and LY already finished in prior run ===
REM echo [1/4] Training Breed A (456 pigs)...
REM echo Start: %date% %time%
REM "%PYTHON%" train_v6_breed_kfold.py --breed A
REM echo Breed A finished: %date% %time%
REM echo.
REM
REM echo [2/4] Training Breed LY (233 pigs)...
REM echo Start: %date% %time%
REM "%PYTHON%" train_v6_breed_kfold.py --breed LY
REM echo Breed LY finished: %date% %time%
REM echo.

echo.
echo [1/2] Training Breed T (65 pigs)...
echo Start: %date% %time%
"%PYTHON%" train_v6_breed_kfold.py --breed T
echo Breed T finished: %date% %time%
echo.

echo [2/2] Training ALL breeds (753 pigs)...
echo Start: %date% %time%
"%PYTHON%" train_v6_breed_kfold.py --breed ALL
echo Breed ALL finished: %date% %time%
echo.

echo ============================================================
echo All training completed: %date% %time%
echo ============================================================
pause
