@echo off
REM env2lgt launcher. Edit the paths below to match your system, then double-click.

setlocal
set "ENV2LGT_ENV=E:\conda\envs\env2lgt"
set "ENV2LGT_DA2_ENV=E:\conda\envs\env2lgt-da2"
set "ENV2LGT_DA2_REPO=E:\models\DA-2"
set "HF_HOME=E:\models\huggingface"
set "TORCH_HOME=E:\models\torch"
REM HF_TOKEN should come from your user environment (setx) — never hardcode here.

if not exist "%ENV2LGT_ENV%\python.exe" (
    echo ERROR: env2lgt conda env not found at %ENV2LGT_ENV%
    pause
    exit /b 1
)

"%ENV2LGT_ENV%\python.exe" -m env2lgt.app %*
endlocal
