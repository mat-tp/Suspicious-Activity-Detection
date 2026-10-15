@echo off
REM ============================================================================
REM build.bat - set up the environment and run ALL training and testing (Windows).
REM
REM Same stages as build.sh:
REM   env         create the virtual environment (default: sdr-env), activate it,
REM               and check / install every required library
REM   record      write Python version, pip freeze, git commit, config checksums
REM   dataset     check that the AD-SVD dataset is where the configs expect it
REM   tests       run the unit tests (synthetic data; no dataset needed)
REM   configs     audit that the 8 factorial cells differ only in detector,
REM               preprocessing and features
REM   duplicates  find copies of one recording and write data\groups.csv
REM   factorial   train/evaluate the 8 cells of the 2x2x2 design
REM   detectors   ViBe, PBAS, SuBSENSE (needs pybgs) against MOG2
REM   features    feature-family ablations and the minimum-track-length filter
REM   groups      remaining experiment groups (B, E, F, G, H, J, neural C and D)
REM   threshold   fixed vs adaptive MOG2 over a range of variance thresholds
REM   ablation    one-at-a-time ablation of every selected parameter
REM   maskquality validate the mask-quality score against ground-truth boxes
REM   verify      check every required result exists and no classifier failed
REM   analysis    factorial effects, comparisons, feature-family importance
REM   figures     copy the paper's figures into docs\images
REM
REM Usage (double-click, or from a command prompt):
REM   build.bat                      every stage, full dataset (many hours)
REM   build.bat --smoke              every stage on a tiny sample, to test the setup
REM   build.bat env tests            only these stages
REM   build.bat --workers 8 --resume
REM
REM Options:
REM   --smoke             tiny sample; results go to experiments_smoke\
REM   --sample-size N     videos per class in --smoke mode (default 5)
REM   --workers N         video-processing worker processes (default 1)
REM   --resume            skip sweep/ablation settings already finished
REM   --env-name NAME     virtual environment folder (default sdr-env)
REM   --python PATH       interpreter used to create the environment (3.10+)
REM   --recreate-env      delete and rebuild the environment
REM   --skip-optional     do not install torch / fiftyone / pybgs / seaborn
REM   --fail-fast         stop at the first failed step
REM
REM A failing step is recorded and the batch continues; every failure is listed at
REM the end and the exit code is 1. Use --fail-fast to stop at the first one.
REM ============================================================================
setlocal EnableExtensions EnableDelayedExpansion

REM ---- settings --------------------------------------------------------------
set "ENV_NAME=sdr-env"
set "PYTHON_BIN="
set "WORKERS=1"
set "SAMPLE_PER_CLASS=5"
set "SMOKE=0"
set "RESUME=0"
set "FAIL_FAST=0"
set "RECREATE_ENV=0"
set "SKIP_OPTIONAL=0"
set "SELECTED_STAGES="
set "FAILURES=0"
set "SUMMARY_COUNT=0"

set "ALL_STAGES=env record dataset tests configs duplicates factorial detectors features groups threshold ablation maskquality verify analysis figures"

set "FACTORIAL_CONFIGS=group_a_classical ablation_1_detector_only ablation_2_preprocessing_only ablation_3_features_only ablation_4_full_pipeline_fixed_mog2 ablation_5_detector_preprocessing ablation_6_detector_features group_i_adaptive_mog2"
set "DETECTOR_CONFIGS=detector_vibe detector_vibe_fullfeat detector_pbas detector_pbas_fullfeat detector_subsense detector_subsense_fullfeat"
set "FEATURE_CONFIGS=ablation_0_baseline_no_meta ablation_7_no_image_quality_meta ablation_8_motion_only group_i_minlen30_filtered"
set "OTHER_GROUP_CONFIGS=group_b_distortion_aware group_e_gt_upper_bound group_f_tracking_ablation group_g_feature_comparison group_h_cascaded_fallback group_h_optical_flow_detector group_j_tfcr_tracker group_c_neural group_d_hybrid"

REM pip-name:import-name. REQUIRED libraries are needed for the study's results;
REM OPTIONAL ones only switch parts off (seaborn plots, neural groups, FiftyOne, SuBSENSE).
set "REQUIRED_PACKAGES=numpy:numpy pandas:pandas opencv-python:cv2 PyYAML:yaml scipy:scipy scikit-learn:sklearn openpyxl:openpyxl joblib:joblib imbalanced-learn:imblearn lightgbm:lightgbm xgboost:xgboost PyWavelets:pywt matplotlib:matplotlib pytest:pytest"
set "OPTIONAL_PACKAGES=seaborn:seaborn torch:torch fiftyone:fiftyone pybgs:pybgs"

REM ---- go to the project root (the folder of this file) -----------------------
cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"

REM ---- parse options ---------------------------------------------------------
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--smoke"         ( set "SMOKE=1" & shift & goto parse_args )
if /i "%~1"=="--sample-size"   ( set "SAMPLE_PER_CLASS=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--workers"       ( set "WORKERS=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--env-name"      ( set "ENV_NAME=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--python"        ( set "PYTHON_BIN=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--resume"        ( set "RESUME=1" & shift & goto parse_args )
if /i "%~1"=="--fail-fast"     ( set "FAIL_FAST=1" & shift & goto parse_args )
if /i "%~1"=="--recreate-env"  ( set "RECREATE_ENV=1" & shift & goto parse_args )
if /i "%~1"=="--skip-optional" ( set "SKIP_OPTIONAL=1" & shift & goto parse_args )
if /i "%~1"=="--help"          goto show_help
if /i "%~1"=="-h"              goto show_help
set "IS_STAGE=0"
for %%S in (%ALL_STAGES%) do if /i "%%S"=="%~1" set "IS_STAGE=1"
if "!IS_STAGE!"=="0" (
    echo ERROR: unknown option or stage: %~1
    echo Stages: %ALL_STAGES%
    exit /b 2
)
set "SELECTED_STAGES=!SELECTED_STAGES! %~1"
shift
goto parse_args

:show_help
for /f "usebackq tokens=* delims=" %%L in ("%~f0") do (
    set "LINE=%%L"
    if "!LINE:~0,4!"=="REM " echo !LINE:~4!
    if "!LINE:~0,3!"=="REM" if "!LINE!"=="REM" echo.
    if "!LINE:~0,7!"=="setloca" goto help_done
)
:help_done
exit /b 0

:args_done
REM Smoke runs write to their own folder so they cannot be mistaken for real results.
if "%SMOKE%"=="1" set "SDR_EXPERIMENTS_DIR=experiments_smoke"
if defined SDR_EXPERIMENTS_DIR ( set "RESULTS_DIR=%SDR_EXPERIMENTS_DIR%" ) else ( set "RESULTS_DIR=experiments" )

for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%T"
set "LOG_DIR=%RESULTS_DIR%\build_logs\%STAMP%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Arguments shared by every training run.
set "RUN_ARGS=--target both --workers %WORKERS%"
set "SAMPLE_ARGS="
if "%SMOKE%"=="1" (
    set "RUN_ARGS=%RUN_ARGS% --sample --n %SAMPLE_PER_CLASS%"
    set "SAMPLE_ARGS=--sample --n %SAMPLE_PER_CLASS%"
)
set "RESUME_ARGS="
if "%RESUME%"=="1" set "RESUME_ARGS=--resume"
set "GROUP_ARGS="

echo.
echo ==^> Project root: %PROJECT_ROOT%
echo     environment: %ENV_NAME%   workers: %WORKERS%   smoke: %SMOKE%   results: %RESULTS_DIR%

REM ============================================================================
REM Stage: env - create + activate the virtual environment, check libraries
REM ============================================================================
call :wanted env
if "!WANTED!"=="1" call :stage_env
if errorlevel 1 ( call :summarize & exit /b 1 )

REM Every later stage needs the environment; activate it quietly if env was skipped.
if exist "%PROJECT_ROOT%\%ENV_NAME%\Scripts\activate.bat" (
    call "%PROJECT_ROOT%\%ENV_NAME%\Scripts\activate.bat"
) else (
    echo WARNING: environment %ENV_NAME% does not exist; using the Python on PATH. Run "build.bat env" first.
)

REM ============================================================================
REM Stage: record - the environment behind this run
REM ============================================================================
call :wanted record
if "!WANTED!"=="1" call :stage_record

REM ============================================================================
REM Stage: dataset - the data must be where the configs expect it
REM ============================================================================
set "DATASET_OK=1"
call :wanted dataset
if "!WANTED!"=="1" (
    call :run_step "dataset check" python source\scripts\check_dataset.py
    if errorlevel 1 set "DATASET_OK=0"
)

REM ============================================================================
REM Stage: tests - unit tests on synthetic data
REM ============================================================================
call :wanted tests
if "!WANTED!"=="1" call :run_step "unit tests" python -m pytest source\tests -q

REM ============================================================================
REM Stage: configs - the factorial cells must differ only in the intended factors
REM ============================================================================
call :wanted configs
if "!WANTED!"=="1" call :run_step "audit factorial configs" python source\scripts\audit_configs.py --out "%RESULTS_DIR%\config_audit"

REM ============================================================================
REM Stage: duplicates - group copies of one recording; later runs use the groups
REM ============================================================================
call :wanted duplicates
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" (
        call :skip_step "duplicate-recording audit" "dataset check failed"
    ) else if "%SMOKE%"=="1" (
        call :skip_step "duplicate-recording audit" "smoke mode samples random videos the group table would not cover"
    ) else (
        call :run_step "duplicate-recording audit" python source\scripts\audit_duplicates.py --out data\groups.csv
    )
)
if "%SMOKE%"=="0" (
    if exist data\groups.csv (
        set "GROUP_ARGS=--set training.group_map_csv=data/groups.csv"
        echo     cross-validation will keep each recording in one fold ^(data\groups.csv^)
    ) else (
        echo WARNING: data\groups.csv not found: cross-validation will NOT be group-aware. Run "build.bat duplicates" first.
    )
)

REM ============================================================================
REM Stage: factorial - the 8 cells of the 2x2x2 design
REM ============================================================================
call :wanted factorial
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "factorial cells" "dataset check failed" ) else (
        for %%C in (%FACTORIAL_CONFIGS%) do call :train_config %%C
    )
)

REM ============================================================================
REM Stage: detectors - alternative background subtractors
REM ============================================================================
call :wanted detectors
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "alternative detectors" "dataset check failed" ) else (
        for %%C in (%DETECTOR_CONFIGS%) do (
            set "CFG=%%C"
            set "SKIP_IT=0"
            if not "!CFG:subsense=!"=="!CFG!" (
                python -c "import pybgs" >nul 2>&1
                if errorlevel 1 set "SKIP_IT=1"
            )
            if "!SKIP_IT!"=="1" ( call :skip_step "train: %%C" "pybgs is not installed" ) else ( call :train_config %%C )
        )
    )
)

REM ============================================================================
REM Stage: features - feature-family ablations, minimum-track-length filter
REM ============================================================================
call :wanted features
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "feature ablations" "dataset check failed" ) else (
        for %%C in (%FEATURE_CONFIGS%) do call :train_config %%C
    )
)

REM ============================================================================
REM Stage: groups - remaining experiment groups (neural ones need PyTorch)
REM ============================================================================
call :wanted groups
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "other experiment groups" "dataset check failed" ) else (
        for %%C in (%OTHER_GROUP_CONFIGS%) do (
            set "CFG=%%C"
            set "SKIP_IT=0"
            if "!CFG:~0,7!"=="group_c" set "NEEDS_TORCH=1"
            if "!CFG:~0,7!"=="group_d" set "NEEDS_TORCH=1"
            if defined NEEDS_TORCH (
                python -c "import torch" >nul 2>&1
                if errorlevel 1 set "SKIP_IT=1"
            )
            set "NEEDS_TORCH="
            if "!SKIP_IT!"=="1" ( call :skip_step "train: %%C" "PyTorch is not installed" ) else ( call :train_config %%C )
        )
    )
)

REM ============================================================================
REM Stage: threshold - is the adaptive gain just a lower fixed threshold?
REM ============================================================================
call :wanted threshold
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "MOG2 threshold sweep" "dataset check failed" ) else (
        for %%T in (activity distortion) do (
            call :run_step "threshold sweep (%%T)" python source\scripts\mog2_threshold_sweep.py --target %%T --workers %WORKERS% %RESUME_ARGS% %SAMPLE_ARGS%
        )
    )
)

REM ============================================================================
REM Stage: ablation - every selected parameter, one at a time
REM ============================================================================
call :wanted ablation
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "parameter ablation" "dataset check failed" ) else (
        call :run_step "parameter ablation" python source\scripts\parameter_ablation.py --workers %WORKERS% %RESUME_ARGS% %SAMPLE_ARGS%
    )
)

REM ============================================================================
REM Stage: maskquality - does Q_mask track real mask quality?
REM ============================================================================
call :wanted maskquality
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "mask-quality validation" "dataset check failed" ) else (
        set "MASK_N=10"
        if "%SMOKE%"=="1" set "MASK_N=2"
        call :run_step "mask-quality validation" python source\scripts\validate_mask_quality.py --n !MASK_N! --out "%RESULTS_DIR%\mask_quality_validation"
    )
)

REM ============================================================================
REM Stage: verify - a finished script is not a finished study; inspect the outputs
REM ============================================================================
call :wanted verify
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "verify outputs" "dataset check failed" ) else (
        call :run_step "verify outputs" python source\scripts\verify_outputs.py --configs %FACTORIAL_CONFIGS% %FEATURE_CONFIGS% --experiments-dir "%RESULTS_DIR%"
    )
)

REM ============================================================================
REM Stage: analysis - effects, comparisons, importance
REM ============================================================================
call :wanted analysis
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "analysis" "dataset check failed" ) else (
        for %%T in (activity distortion) do (
            for %%U in (all common) do (
                call :run_step "factorial analysis (%%T, %%U videos)" python source\scripts\factorial_analysis.py --target %%T --universe %%U --experiments-dir "%RESULTS_DIR%"
            )
            call :run_step "compare: detector only (%%T)" python source\scripts\compare_experiments.py group_a_classical ablation_1_detector_only --target %%T --universe both --experiments-dir "%RESULTS_DIR%"
            call :run_step "compare: adaptive vs fixed, full pipeline (%%T)" python source\scripts\compare_experiments.py ablation_4_full_pipeline_fixed_mog2 group_i_adaptive_mog2 --target %%T --universe both --experiments-dir "%RESULTS_DIR%"
            call :run_step "feature-family importance (%%T)" python source\scripts\feature_family_importance.py --latest group_i_adaptive_mog2 --target %%T --experiments-dir "%RESULTS_DIR%"
        )
    )
)

REM ============================================================================
REM Stage: figures - copy what the paper uses into docs\images
REM ============================================================================
call :wanted figures
if "!WANTED!"=="1" (
    if "%SMOKE%"=="1" (
        call :skip_step "collect paper figures" "smoke results must not overwrite the paper's figures"
    ) else (
        call :run_step "collect paper figures" python source\scripts\collect_paper_figures.py --strict
    )
)

call :summarize
if %FAILURES% GTR 0 exit /b 1
exit /b 0

REM ============================================================================
REM Subroutines
REM ============================================================================

:wanted
REM Sets WANTED=1 when the stage was named on the command line, or none was named.
set "WANTED=0"
if "!SELECTED_STAGES!"=="" ( set "WANTED=1" & exit /b 0 )
for %%S in (!SELECTED_STAGES!) do if /i "%%S"=="%~1" set "WANTED=1"
exit /b 0

:record_step
REM %1 = name, %2 = status
set /a SUMMARY_COUNT+=1
set "STEP_NAME_!SUMMARY_COUNT!=%~1"
set "STEP_STATUS_!SUMMARY_COUNT!=%~2"
if "%~2"=="FAILED" set /a FAILURES+=1
exit /b 0

:run_step
REM %1 = label, remaining arguments = the command. Output goes to the console and a log.
set "LABEL=%~1"
set "SAFE=%LABEL: =_%"
set "SAFE=%SAFE::=_%"
set "SAFE=%SAFE:(=_%"
set "SAFE=%SAFE:)=_%"
set "SAFE=%SAFE:,=_%"
echo.
echo ==^> %LABEL%
shift
set "CMD="
:collect_cmd
if "%~1"=="" goto run_cmd
set "CMD=!CMD! %1"
shift
goto collect_cmd
:run_cmd
echo     $ !CMD!
powershell -NoProfile -Command "& { $ErrorActionPreference='Continue'; cmd /c '!CMD! 2^>^&1' | Tee-Object -FilePath '%LOG_DIR%\%SAFE%.log'; exit $LASTEXITCODE }"
set "STATUS=!ERRORLEVEL!"
if "!STATUS!"=="0" (
    call :record_step "%LABEL%" ok
) else (
    echo WARNING: step failed ^(exit !STATUS!^): %LABEL%   ^(log: %LOG_DIR%\%SAFE%.log^)
    call :record_step "%LABEL%" FAILED
    if "%FAIL_FAST%"=="1" ( call :summarize & exit /b 1 )
)
exit /b 0

:skip_step
echo.
echo ==^> %~1
echo     skipped: %~2
call :record_step "%~1" skipped
exit /b 0

:train_config
REM %1 = config name. One experiment, both targets.
call :run_step "train: %~1" python source\run.py --config %~1 %RUN_ARGS% !GROUP_ARGS!
exit /b 0

:stage_env
echo.
echo ==^> Stage: env
set "ENV_PATH=%PROJECT_ROOT%\%ENV_NAME%"
if "%RECREATE_ENV%"=="1" if exist "%ENV_PATH%" (
    echo     removing the old environment: %ENV_PATH%
    rmdir /s /q "%ENV_PATH%"
)
if not exist "%ENV_PATH%\Scripts\activate.bat" (
    call :find_python
    if not defined BASE_PYTHON (
        echo ERROR: Python 3.10 or newer was not found. Install it, or pass --python C:\path\to\python.exe
        exit /b 1
    )
    echo     creating %ENV_NAME% with !BASE_PYTHON!
    !BASE_PYTHON! -m venv "%ENV_PATH%"
    if errorlevel 1 ( echo ERROR: could not create the virtual environment & exit /b 1 )
) else (
    echo     reusing the existing environment: %ENV_PATH%
)
call "%ENV_PATH%\Scripts\activate.bat"
if errorlevel 1 ( echo ERROR: could not activate %ENV_PATH% & exit /b 1 )
for /f "delims=" %%V in ('python --version 2^>^&1') do echo     activated: %%V

python -m pip install --quiet --upgrade pip setuptools wheel

echo     checking required libraries
set "ENV_OK=1"
for %%P in (%REQUIRED_PACKAGES%) do call :check_package required %%P
if "!ENV_OK!"=="1" ( call :record_step "env: required libraries" ok ) else ( call :record_step "env: required libraries" FAILED )

if "%SKIP_OPTIONAL%"=="1" (
    echo     skipping optional libraries ^(--skip-optional^)
) else (
    echo     checking optional libraries
    for %%P in (%OPTIONAL_PACKAGES%) do call :check_package optional %%P
)
exit /b 0

:find_python
REM Picks a Python >= 3.10 to create the environment with.
set "BASE_PYTHON="
for %%P in ("%PYTHON_BIN%" py python3 python) do (
    if not defined BASE_PYTHON if not "%%~P"=="" (
        %%~P -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
        if not errorlevel 1 set "BASE_PYTHON=%%~P"
    )
)
exit /b 0

:check_package
REM %1 = required|optional, %2 = pip-name:import-name
set "KIND=%~1"
for /f "tokens=1,2 delims=:" %%A in ("%~2") do ( set "PIP_NAME=%%A" & set "IMPORT_NAME=%%B" )
python -c "import !IMPORT_NAME!" >nul 2>&1
if not errorlevel 1 (
    echo     found    !PIP_NAME!
    exit /b 0
)
echo     missing  !PIP_NAME!  - installing ...
python -m pip install --quiet !PIP_NAME!
python -c "import !IMPORT_NAME!" >nul 2>&1
if errorlevel 1 (
    if "!KIND!"=="required" (
        echo WARNING: could not install required package: !PIP_NAME!
        set "ENV_OK=0"
    ) else (
        echo WARNING: optional package not installed: !PIP_NAME! ^(the parts that need it are switched off^)
    )
)
exit /b 0

:stage_record
echo.
echo ==^> Stage: record environment
set "ENV_FILE=%LOG_DIR%\environment.txt"
> "%ENV_FILE%" echo date:       %DATE% %TIME%
>> "%ENV_FILE%" echo host:       %COMPUTERNAME% %OS%
>> "%ENV_FILE%" python --version
>> "%ENV_FILE%" echo smoke mode: %SMOKE%   workers: %WORKERS%
>> "%ENV_FILE%" echo git commit:
>> "%ENV_FILE%" git rev-parse HEAD 2^>^&1
>> "%ENV_FILE%" echo git uncommitted changes:
>> "%ENV_FILE%" git status --porcelain 2^>^&1
>> "%ENV_FILE%" echo.
>> "%ENV_FILE%" echo --- pip freeze ---
>> "%ENV_FILE%" python -m pip freeze
>> "%ENV_FILE%" echo.
>> "%ENV_FILE%" echo --- sha256 of configs ---
for %%F in (source\configs\*.yaml) do >> "%ENV_FILE%" certutil -hashfile "%%F" SHA256 ^| findstr /v ":"
if exist data\groups.csv (
    >> "%ENV_FILE%" echo --- sha256 of data\groups.csv ---
    >> "%ENV_FILE%" certutil -hashfile data\groups.csv SHA256
)
copy /y requirements.txt "%LOG_DIR%\requirements.txt" >nul 2>&1
echo     written: %ENV_FILE%
call :record_step "record environment" ok
exit /b 0

:summarize
echo.
echo ==^> Summary
> "%LOG_DIR%\build_summary.txt" echo Summary
for /l %%I in (1,1,%SUMMARY_COUNT%) do (
    echo   !STEP_STATUS_%%I!   !STEP_NAME_%%I!
    >> "%LOG_DIR%\build_summary.txt" echo   !STEP_STATUS_%%I!   !STEP_NAME_%%I!
)
echo.
if %FAILURES% GTR 0 (
    echo %FAILURES% step^(s^) FAILED. Logs: %LOG_DIR%
) else (
    echo All steps finished without a failure. Logs: %LOG_DIR%
)
exit /b 0
