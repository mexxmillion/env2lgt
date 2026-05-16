@echo off
REM env2lgt launcher. Launches the GUI via the `env2lgt` conda env.
REM
REM Requires `conda` on PATH. If your DA-2 env / repo / model caches are not
REM on the default E: paths, set the ENV2LGT_DA2_* and HF_HOME / TORCH_HOME
REM vars below (or once, system-wide, with `setx`). HF_TOKEN should come from
REM your user environment — never hardcode it here.

setlocal
if not defined ENV2LGT_DA2_ENV  set "ENV2LGT_DA2_ENV=E:\conda\envs\env2lgt-da2"
if not defined ENV2LGT_DA2_REPO set "ENV2LGT_DA2_REPO=E:\models\DA-2"
if not defined HF_HOME          set "HF_HOME=E:\models\huggingface"
if not defined TORCH_HOME       set "TORCH_HOME=E:\models\torch"

where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: `conda` not found on PATH. Activate conda or open an Anaconda Prompt.
    pause
    exit /b 1
)

conda run --no-capture-output -n env2lgt python -m env2lgt.app %*
endlocal
