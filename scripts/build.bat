@echo off
rem Sets up the environment and runs all the training and testing, stage by stage (Windows).
rem It does the same as build.sh. Run scripts\build.bat --help for the options.
setlocal EnableExtensions EnableDelayedExpansion

rem This is the project root (the folder above scripts\), so the script works from any folder.
cd /d "%~dp0.."
set "PROJECT_ROOT=%CD%"

rem These are the settings; each can be changed with an option.
set "ENV_NAME=sdr-env"
set "PYTHON_BIN="
set "WORKERS=1"
set "CV_JOBS=1"
set "SAMPLE_PER_CLASS=5"
set "SMOKE=0"
set "RESUME=0"
set "FAIL_FAST=0"
set "RECREATE_ENV=0"
set "SKIP_OPTIONAL=0"
set "SELECTED_STAGES="
set "FAILURES=0"
set "SUMMARY_COUNT=0"

rem This is the list of stages in the order they run.
set "ALL_STAGES=env record dataset tests configs duplicates factorial detectors features groups threshold ablation maskquality verify analysis figures"

rem These are the config names (the files in source\configs without .yaml) that each stage trains.
set "FACTORIAL_CONFIGS=group_a_classical ablation_1_detector_only ablation_2_preprocessing_only ablation_3_features_only ablation_4_full_pipeline_fixed_mog2 ablation_5_detector_preprocessing ablation_6_detector_features group_i_adaptive_mog2"
set "DETECTOR_CONFIGS=detector_vibe detector_vibe_fullfeat detector_pbas detector_pbas_fullfeat detector_subsense detector_subsense_fullfeat"
set "FEATURE_CONFIGS=ablation_0_baseline_no_meta ablation_7_no_image_quality_meta ablation_8_motion_only group_i_minlen30_filtered"
set "OTHER_GROUP_CONFIGS=group_b_distortion_aware group_e_gt_upper_bound group_f_tracking_ablation group_g_feature_comparison group_h_cascaded_fallback group_h_optical_flow_detector group_j_tfcr_tracker group_c_neural group_d_hybrid"

rem These are the Python packages as pip-name:import-name. Without a required one the study
rem cannot run; a missing optional one only switches a part off.
set "REQUIRED_PACKAGES=numpy:numpy pandas:pandas opencv-python:cv2 PyYAML:yaml scipy:scipy scikit-learn:sklearn openpyxl:openpyxl joblib:joblib imbalanced-learn:imblearn lightgbm:lightgbm xgboost:xgboost PyWavelets:pywt matplotlib:matplotlib pytest:pytest"
set "OPTIONAL_PACKAGES=seaborn:seaborn torch:torch fiftyone:fiftyone pybgs:pybgs"

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--smoke"         ( set "SMOKE=1" & shift & goto parse_args )
if /i "%~1"=="--sample-size"   ( set "SAMPLE_PER_CLASS=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--workers"       ( set "WORKERS=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--cv-jobs"       ( set "CV_JOBS=%~2" & shift & shift & goto parse_args )
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
for /f "usebackq tokens=1* delims= " %%A in (`findstr /b /c:"::help" "%~f0"`) do echo(%%B
exit /b 0

::help Usage: scripts\build.bat [options] [stage ...]
::help
::help Stages (all of them run when none is named):
::help   env          create the virtual environment and install the libraries
::help   record       write the Python version, pip freeze, git commit and config checksums
::help   dataset      check that the AD-SVD dataset is where the configs expect it
::help   tests        run the unit tests (synthetic data, no dataset needed)
::help   configs      check that the 8 factorial cells differ only in the intended factors
::help   duplicates   find copies of one recording and write data\groups.csv
::help   factorial    train the 8 cells of the 2x2x2 design
::help   detectors    ViBe, PBAS and SuBSENSE (needs pybgs) against MOG2
::help   features     feature-family ablations and the minimum-track-length filter
::help   groups       the other experiment groups (B, E, F, G, H, J and the neural C, D)
::help   threshold    fixed against adaptive MOG2 over a range of variance thresholds
::help   ablation     one-at-a-time ablation of the selected parameters
::help   maskquality  check the mask-quality score against the ground-truth boxes
::help   verify       check that every required result exists and no classifier failed
::help   analysis     factorial effects, comparisons and feature-family importance
::help   figures      copy the figures of the paper into docs\manuscript\images
::help
::help Options:
::help   --smoke           tiny sample (5 videos per class); results go to experiments_smoke\
::help   --sample-size N   videos per class in smoke mode
::help   --workers N       processes that read the videos, or "auto" for all cores but one (default 1)
::help   --cv-jobs N       cross-validation folds trained at the same time, -1 for all cores (default 1)
::help   --resume          skip the sweep and ablation settings that are already finished
::help   --env-name NAME   virtual environment folder (default sdr-env)
::help   --python PATH     interpreter that creates the environment (3.10 or newer)
::help   --recreate-env    delete and rebuild the environment
::help   --skip-optional   do not install torch, fiftyone, pybgs and seaborn
::help   --fail-fast       stop at the first failed step
::help   -h, --help        show this text
::help
::help A failed step is recorded and the run goes on, because one bad config should not stop a
::help long run. The failures are listed at the end and the exit code is then 1.
::help
::help Examples:
::help   scripts\build.bat --smoke
::help   scripts\build.bat env tests
::help   scripts\build.bat --workers auto --resume

:args_done
rem "auto" becomes all the cores but one, so the machine stays usable.
if /i "%WORKERS%"=="auto" (
    set /a WORKERS=NUMBER_OF_PROCESSORS-1
    if !WORKERS! LSS 1 set "WORKERS=1"
)

rem Smoke runs write to their own folders, so they cannot be mistaken for real results.
for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%T"
if "%SMOKE%"=="1" (
    set "SDR_EXPERIMENTS_DIR=experiments_smoke"
    set "STAMP=smoke_!STAMP!"
)
if defined SDR_EXPERIMENTS_DIR ( set "RESULTS_DIR=%SDR_EXPERIMENTS_DIR%" ) else ( set "RESULTS_DIR=experiments" )
if defined SDR_LOGS_DIR ( set "LOGS_ROOT=%SDR_LOGS_DIR%" ) else ( set "LOGS_ROOT=docs\logs" )
set "LOG_DIR=%LOGS_ROOT%\%STAMP%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

rem These are the arguments that every training run gets.
set "RUN_ARGS=--target both --workers %WORKERS% --set training.n_jobs=%CV_JOBS%"
set "SAMPLE_ARGS="
set "RESUME_ARGS="
set "GROUP_ARGS="
if "%SMOKE%"=="1" (
    set "RUN_ARGS=%RUN_ARGS% --sample --n %SAMPLE_PER_CLASS%"
    set "SAMPLE_ARGS=--sample --n %SAMPLE_PER_CLASS%"
)
if "%RESUME%"=="1" set "RESUME_ARGS=--resume"

echo.
echo ==^> Project root: %PROJECT_ROOT%
echo     environment: %ENV_NAME%   workers: %WORKERS%   cv jobs: %CV_JOBS%   smoke: %SMOKE%   results: %RESULTS_DIR%

call :wanted env
if "!WANTED!"=="1" call :stage_env
if errorlevel 1 ( call :summarize & exit /b 1 )

rem Every stage after env needs the environment, so this activates it when env was skipped.
if exist "%PROJECT_ROOT%\%ENV_NAME%\Scripts\activate.bat" (
    call "%PROJECT_ROOT%\%ENV_NAME%\Scripts\activate.bat"
) else (
    echo WARNING: environment %ENV_NAME% does not exist, so the Python on PATH is used ^(run scripts\build.bat env first^)
)

call :wanted record
if "!WANTED!"=="1" call :stage_record

rem This stays 1 unless the dataset check fails; the stages that need videos are then skipped.
set "DATASET_OK=1"
call :wanted dataset
if "!WANTED!"=="1" (
    call :run_step "dataset check" python scripts\check_dataset.py
    if errorlevel 1 set "DATASET_OK=0"
)

call :wanted tests
if "!WANTED!"=="1" call :run_step "unit tests" python -m pytest source\tests -q

call :wanted configs
if "!WANTED!"=="1" call :run_step "audit factorial configs" python scripts\audit_configs.py --out "%RESULTS_DIR%\config_audit"

call :wanted duplicates
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" (
        call :skip_step "duplicate-recording audit" "the dataset check failed"
    ) else if "%SMOKE%"=="1" (
        call :skip_step "duplicate-recording audit" "a smoke run samples random videos, and the group table must cover all of them"
    ) else (
        call :run_step "duplicate-recording audit" python scripts\audit_duplicates.py --out data\groups.csv
    )
)
rem This is the group table that keeps copies of one recording in the same fold.
if "%SMOKE%"=="0" (
    if exist data\groups.csv (
        set "GROUP_ARGS=--set training.group_map_csv=data/groups.csv"
        echo     cross-validation keeps each recording in one fold ^(data\groups.csv^)
    ) else (
        echo WARNING: data\groups.csv not found, so cross-validation is NOT group-aware; run scripts\build.bat duplicates first
    )
)

call :wanted factorial
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "factorial cells" "the dataset check failed" ) else (
        for %%C in (%FACTORIAL_CONFIGS%) do call :train_config %%C
    )
)

call :wanted detectors
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "alternative detectors" "the dataset check failed" ) else (
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

call :wanted features
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "feature ablations" "the dataset check failed" ) else (
        for %%C in (%FEATURE_CONFIGS%) do call :train_config %%C
    )
)

call :wanted groups
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "other experiment groups" "the dataset check failed" ) else (
        for %%C in (%OTHER_GROUP_CONFIGS%) do (
            set "CFG=%%C"
            set "SKIP_IT=0"
            set "NEEDS_TORCH="
            if "!CFG:~0,7!"=="group_c" set "NEEDS_TORCH=1"
            if "!CFG:~0,7!"=="group_d" set "NEEDS_TORCH=1"
            if defined NEEDS_TORCH (
                python -c "import torch" >nul 2>&1
                if errorlevel 1 set "SKIP_IT=1"
            )
            if "!SKIP_IT!"=="1" ( call :skip_step "train: %%C" "PyTorch is not installed" ) else ( call :train_config %%C )
        )
    )
)

call :wanted threshold
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "MOG2 threshold sweep" "the dataset check failed" ) else (
        for %%T in (activity distortion) do (
            call :run_step "threshold sweep (%%T)" python scripts\mog2_threshold_sweep.py --target %%T --workers %WORKERS% %RESUME_ARGS% %SAMPLE_ARGS%
        )
    )
)

call :wanted ablation
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "parameter ablation" "the dataset check failed" ) else (
        call :run_step "parameter ablation" python scripts\parameter_ablation.py --workers %WORKERS% %RESUME_ARGS% %SAMPLE_ARGS%
    )
)

call :wanted maskquality
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "mask-quality validation" "the dataset check failed" ) else (
        set "MASK_N=10"
        if "%SMOKE%"=="1" set "MASK_N=2"
        call :run_step "mask-quality validation" python scripts\validate_mask_quality.py --n !MASK_N! --out "%RESULTS_DIR%\mask_quality_validation"
    )
)

call :wanted verify
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "verify outputs" "the dataset check failed" ) else (
        call :run_step "verify outputs" python scripts\verify_outputs.py --configs %FACTORIAL_CONFIGS% %FEATURE_CONFIGS% --experiments-dir "%RESULTS_DIR%"
    )
)

call :wanted analysis
if "!WANTED!"=="1" (
    if "!DATASET_OK!"=="0" ( call :skip_step "analysis" "the dataset check failed" ) else (
        for %%T in (activity distortion) do (
            for %%U in (all common) do (
                call :run_step "factorial analysis (%%T, %%U videos)" python scripts\factorial_analysis.py --target %%T --universe %%U --experiments-dir "%RESULTS_DIR%"
            )
            rem Does adaptive MOG2 alone help, and does it help with everything else switched on?
            call :run_step "compare: detector only (%%T)" python scripts\compare_experiments.py group_a_classical ablation_1_detector_only --target %%T --universe both --experiments-dir "%RESULTS_DIR%"
            call :run_step "compare: adaptive against fixed, full pipeline (%%T)" python scripts\compare_experiments.py ablation_4_full_pipeline_fixed_mog2 group_i_adaptive_mog2 --target %%T --universe both --experiments-dir "%RESULTS_DIR%"
            rem Which feature families does the classifier rely on?
            call :run_step "feature-family importance (%%T)" python scripts\feature_family_importance.py --latest group_i_adaptive_mog2 --target %%T --experiments-dir "%RESULTS_DIR%"
        )
    )
)

call :wanted figures
if "!WANTED!"=="1" (
    if "%SMOKE%"=="1" (
        call :skip_step "collect paper figures" "smoke results must not overwrite the figures of the paper"
    ) else (
        call :run_step "collect paper figures" python scripts\collect_paper_figures.py --strict
    )
)

call :summarize
if %FAILURES% GTR 0 exit /b 1
exit /b 0

:wanted
rem Sets WANTED=1 when the stage was named on the command line, or when none was named.
set "WANTED=0"
if "!SELECTED_STAGES!"=="" ( set "WANTED=1" & exit /b 0 )
for %%S in (!SELECTED_STAGES!) do if /i "%%S"=="%~1" set "WANTED=1"
exit /b 0

:record_step
rem %1 is the name and %2 the status.
set /a SUMMARY_COUNT+=1
set "STEP_NAME_!SUMMARY_COUNT!=%~1"
set "STEP_STATUS_!SUMMARY_COUNT!=%~2"
if "%~2"=="FAILED" set /a FAILURES+=1
exit /b 0

:run_step
rem %1 is the label and the other arguments are the command. The output goes to the console and to a log.
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
    echo WARNING: step failed ^(exit !STATUS!^): %LABEL% ^(log: %LOG_DIR%\%SAFE%.log^)
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
rem %1 is the config name. This trains one experiment for both targets.
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
        echo ERROR: Python 3.10 or newer was not found; install it or pass --python C:\path\to\python.exe
        exit /b 1
    )
    echo     creating %ENV_NAME% with !BASE_PYTHON!
    !BASE_PYTHON! -m venv "%ENV_PATH%"
    if errorlevel 1 ( echo ERROR: could not create the virtual environment & exit /b 1 )
) else (
    echo     reusing the environment: %ENV_PATH%
)
call "%ENV_PATH%\Scripts\activate.bat"
if errorlevel 1 ( echo ERROR: could not activate %ENV_PATH% & exit /b 1 )
for /f "delims=" %%V in ('python --version 2^>^&1') do echo     activated: %%V

python -m pip install --quiet --upgrade pip setuptools wheel

echo     checking the required libraries
set "ENV_OK=1"
for %%P in (%REQUIRED_PACKAGES%) do call :check_package required %%P
if "!ENV_OK!"=="1" ( call :record_step "env: required libraries" ok ) else ( call :record_step "env: required libraries" FAILED )

if "%SKIP_OPTIONAL%"=="1" (
    echo     skipping the optional libraries ^(--skip-optional^)
) else (
    echo     checking the optional libraries
    for %%P in (%OPTIONAL_PACKAGES%) do call :check_package optional %%P
)
exit /b 0

:find_python
rem Sets BASE_PYTHON to the first interpreter that is Python 3.10 or newer.
set "BASE_PYTHON="
for %%P in ("%PYTHON_BIN%" py python3 python) do (
    if not defined BASE_PYTHON if not "%%~P"=="" (
        %%~P -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
        if not errorlevel 1 set "BASE_PYTHON=%%~P"
    )
)
exit /b 0

:check_package
rem %1 is required or optional, %2 is pip-name:import-name.
set "KIND=%~1"
for /f "tokens=1,2 delims=:" %%A in ("%~2") do ( set "PIP_NAME=%%A" & set "IMPORT_NAME=%%B" )
python -c "import !IMPORT_NAME!" >nul 2>&1
if not errorlevel 1 (
    echo     found    !PIP_NAME!
    exit /b 0
)
echo     missing  !PIP_NAME!, installing
python -m pip install --quiet !PIP_NAME!
python -c "import !IMPORT_NAME!" >nul 2>&1
if errorlevel 1 (
    if "!KIND!"=="required" (
        echo WARNING: could not install the required package !PIP_NAME!
        set "ENV_OK=0"
    ) else (
        echo WARNING: optional package !PIP_NAME! not installed ^(the parts that need it are switched off^)
    )
)
exit /b 0

:stage_record
echo.
echo ==^> Stage: record the environment
set "ENV_FILE=%LOG_DIR%\environment.txt"
> "%ENV_FILE%" echo date:       %DATE% %TIME%
>> "%ENV_FILE%" echo host:       %COMPUTERNAME% %OS%
>> "%ENV_FILE%" python --version
>> "%ENV_FILE%" echo smoke mode: %SMOKE%   workers: %WORKERS%   cv jobs: %CV_JOBS%
>> "%ENV_FILE%" echo git commit:
>> "%ENV_FILE%" git rev-parse HEAD 2^>^&1
>> "%ENV_FILE%" echo git uncommitted changes:
>> "%ENV_FILE%" git status --porcelain 2^>^&1
>> "%ENV_FILE%" echo.
>> "%ENV_FILE%" echo --- pip freeze ---
>> "%ENV_FILE%" python -m pip freeze
>> "%ENV_FILE%" echo.
>> "%ENV_FILE%" echo --- sha256 of the configs ---
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
