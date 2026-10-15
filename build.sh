#!/usr/bin/env bash
# =============================================================================
# build.sh - set up the environment and run ALL training and testing, in one go.
#
# What it does, in order (each is a "stage" you can run on its own):
#
#   env         create the Python virtual environment (default: sdr-env), activate it,
#               and check / install every required library
#   record      write the Python version, pip freeze, git commit and config checksums
#               next to the results, so the exact setup behind a number is on record
#   dataset     check that the AD-SVD dataset is where the configs expect it
#   tests       run the unit tests (synthetic data; no dataset needed)
#   configs     audit that the 8 factorial cells differ only in detector,
#               preprocessing and features
#   duplicates  find videos that are copies of one recording and write data/groups.csv
#               (all later cross-validation then keeps each recording in one fold)
#   factorial   train/evaluate the 8 cells of the 2x2x2 design
#   detectors   ViBe, PBAS, SuBSENSE (needs pybgs) against MOG2
#   features    feature-family ablations and the minimum-track-length filter
#   groups      the remaining experiment groups (B, E, F, G, H, J, and the neural C, D)
#   threshold   fixed vs adaptive MOG2 over a range of variance thresholds
#   ablation    one-at-a-time ablation of every selected parameter
#   maskquality validate the mask-quality score against ground-truth boxes
#   verify      check every required result exists and no classifier failed
#   analysis    factorial effects, pairwise comparisons, feature-family importance
#   figures     copy the figures the paper uses into docs/images
#
# Usage (from anywhere; the script moves to the project root):
#
#   ./build.sh                       # every stage, full dataset (takes many hours)
#   ./build.sh --smoke               # every stage on a tiny sample, to check the setup
#   ./build.sh env tests             # only these stages
#   ./build.sh --workers 8 --resume  # 8 video-processing processes; skip finished work
#   ./build.sh --help
#
# A failing step is recorded and the batch continues (a long run should not stop for one
# bad config), but the script prints every failure at the end and exits with status 1.
# Use --fail-fast to stop at the first failure instead.
#
# Options:
#   --smoke             tiny sample (5 videos per class); results go to experiments_smoke/
#   --sample-size N     videos per class in --smoke mode
#   --workers N         video-processing worker processes (default 1)
#   --resume            skip sweep/ablation settings already finished
#   --env-name NAME     virtual environment folder (default sdr-env)
#   --python PATH       interpreter used to create the environment (3.10 or newer)
#   --recreate-env      delete and rebuild the environment
#   --skip-optional     do not install torch / fiftyone / pybgs / seaborn
#   --fail-fast         stop at the first failed step
# =============================================================================

set -u -o pipefail

# ---- settings (override with options or environment variables) --------------------------
ENV_NAME="${SDR_ENV_NAME:-sdr-env}"      # name of the virtual environment folder
PYTHON_BIN="${SDR_PYTHON:-}"             # interpreter used to CREATE the env (auto-detected)
WORKERS="${SDR_WORKERS:-1}"              # video-processing worker processes
SAMPLE_PER_CLASS=5                       # videos per class in --smoke mode
SMOKE=0                                  # 1 = tiny sample, results kept apart
RESUME=0                                 # 1 = skip sweep settings that are already done
FAIL_FAST=0                              # 1 = stop at the first failed step
RECREATE_ENV=0                           # 1 = delete and rebuild the environment
SKIP_OPTIONAL=0                          # 1 = do not install torch / fiftyone / pybgs
SELECTED_STAGES=()                       # stages named on the command line

ALL_STAGES="env record dataset tests configs duplicates factorial detectors features groups threshold ablation maskquality verify analysis figures"

# Experiment configs (names of files in source/configs, without .yaml)
FACTORIAL_CONFIGS="group_a_classical ablation_1_detector_only ablation_2_preprocessing_only ablation_3_features_only ablation_4_full_pipeline_fixed_mog2 ablation_5_detector_preprocessing ablation_6_detector_features group_i_adaptive_mog2"
DETECTOR_CONFIGS="detector_vibe detector_vibe_fullfeat detector_pbas detector_pbas_fullfeat detector_subsense detector_subsense_fullfeat"
FEATURE_CONFIGS="ablation_0_baseline_no_meta ablation_7_no_image_quality_meta ablation_8_motion_only group_i_minlen30_filtered"
OTHER_GROUP_CONFIGS="group_b_distortion_aware group_e_gt_upper_bound group_f_tracking_ablation group_g_feature_comparison group_h_cascaded_fallback group_h_optical_flow_detector group_j_tfcr_tracker group_c_neural group_d_hybrid"

# Python packages: "pip-name:import-name". REQUIRED ones are needed for the study's results.
REQUIRED_PACKAGES="numpy:numpy pandas:pandas opencv-python:cv2 PyYAML:yaml scipy:scipy scikit-learn:sklearn openpyxl:openpyxl joblib:joblib imbalanced-learn:imblearn lightgbm:lightgbm xgboost:xgboost PyWavelets:pywt matplotlib:matplotlib pytest:pytest"
# OPTIONAL ones only switch parts off when missing (seaborn plots, neural groups C/D,
# FiftyOne visualisation, the SuBSENSE detector).
OPTIONAL_PACKAGES="seaborn:seaborn torch:torch fiftyone:fiftyone pybgs:pybgs"

# ---- locate the project root and set up logging ----------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT" || exit 1

STEP_NAMES=()      # one entry per step that ran
STEP_STATUS=()     # "ok" / "FAILED" / "skipped"
FAILURES=0

# ---- small helpers -------------------------------------------------------------------
say()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33mWARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

usage() { sed -n '3,50p' "$0" | sed 's/^# \{0,1\}//'; }

stage_wanted() {
    # True when the stage was named on the command line, or no stage was named.
    local stage="$1"
    if [ "${#SELECTED_STAGES[@]}" -eq 0 ]; then return 0; fi
    local s
    for s in "${SELECTED_STAGES[@]}"; do [ "$s" = "$stage" ] && return 0; done
    return 1
}

record_step() {  # record_step <name> <status>
    STEP_NAMES+=("$1"); STEP_STATUS+=("$2")
    if [ "$2" = "FAILED" ]; then FAILURES=$((FAILURES + 1)); fi
}

run_step() {
    # run_step <label> <command...> : run, log to the build log, record success/failure.
    local label="$1"; shift
    say "$label"
    note "\$ $*"
    local logfile="$LOG_DIR/$(echo "$label" | tr -c 'A-Za-z0-9._-' '_').log"
    "$@" 2>&1 | tee "$logfile"
    local status="${PIPESTATUS[0]}"
    if [ "$status" -eq 0 ]; then
        record_step "$label" "ok"
    else
        warn "step failed (exit $status): $label   (log: $logfile)"
        record_step "$label" "FAILED"
        if [ "$FAIL_FAST" -eq 1 ]; then summarize; exit 1; fi
    fi
    return 0
}

skip_step() { say "$1"; note "skipped: $2"; record_step "$1" "skipped"; }

summarize() {
    say "Summary"
    local i
    for i in "${!STEP_NAMES[@]}"; do
        printf '  %-8s %s\n' "${STEP_STATUS[$i]}" "${STEP_NAMES[$i]}"
    done | tee "$LOG_DIR/build_summary.txt"
    if [ "$FAILURES" -gt 0 ]; then
        printf '\n\033[1;31m%d step(s) FAILED.\033[0m Logs: %s\n' "$FAILURES" "$LOG_DIR"
    else
        printf '\n\033[1;32mAll steps finished without a failure.\033[0m Logs: %s\n' "$LOG_DIR"
    fi
}

# ---- command-line options ------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --smoke)         SMOKE=1 ;;
        --sample-size)   SAMPLE_PER_CLASS="$2"; shift ;;
        --workers)       WORKERS="$2"; shift ;;
        --env-name)      ENV_NAME="$2"; shift ;;
        --python)        PYTHON_BIN="$2"; shift ;;
        --resume)        RESUME=1 ;;
        --fail-fast)     FAIL_FAST=1 ;;
        --recreate-env)  RECREATE_ENV=1 ;;
        --skip-optional) SKIP_OPTIONAL=1 ;;
        -h|--help)       usage; exit 0 ;;
        -*)              die "unknown option: $1 (try --help)" ;;
        *)
            case " $ALL_STAGES " in
                *" $1 "*) SELECTED_STAGES+=("$1") ;;
                *) die "unknown stage: $1. Stages: $ALL_STAGES" ;;
            esac ;;
    esac
    shift
done

# Smoke runs write to their own folder so they can never be mistaken for real results.
if [ "$SMOKE" -eq 1 ]; then
    export SDR_EXPERIMENTS_DIR="experiments_smoke"
fi
RESULTS_DIR="${SDR_EXPERIMENTS_DIR:-experiments}"
LOG_DIR="$RESULTS_DIR/build_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Arguments shared by every training run.
RUN_ARGS=(--target both --workers "$WORKERS")
if [ "$SMOKE" -eq 1 ]; then RUN_ARGS+=(--sample --n "$SAMPLE_PER_CLASS"); fi
RESUME_ARGS=()
if [ "$RESUME" -eq 1 ]; then RESUME_ARGS+=(--resume); fi
SAMPLE_ARGS=()
if [ "$SMOKE" -eq 1 ]; then SAMPLE_ARGS+=(--sample --n "$SAMPLE_PER_CLASS"); fi

say "Project root: $PROJECT_ROOT"
note "environment: $ENV_NAME   workers: $WORKERS   smoke: $SMOKE   results: $RESULTS_DIR"

# =============================================================================
# Stage: env - create + activate the virtual environment, check libraries
# =============================================================================
find_python() {
    # Pick a Python >= 3.10 to create the environment with.
    local candidate
    for candidate in "$PYTHON_BIN" python3.12 python3.11 python3.10 python3 python; do
        [ -n "$candidate" ] || continue
        if command -v "$candidate" >/dev/null 2>&1 &&
           "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            echo "$candidate"; return 0
        fi
    done
    return 1
}

package_installed() {  # package_installed <import-name>
    python -c "import $1" >/dev/null 2>&1
}

check_and_install() {
    # check_and_install <required|optional> "<pip:import> ..."
    local kind="$1" entry pip_name import_name missing=()
    for entry in $2; do
        pip_name="${entry%%:*}"; import_name="${entry##*:}"
        if package_installed "$import_name"; then
            note "found    $pip_name"
        else
            note "missing  $pip_name"; missing+=("$pip_name")
        fi
    done
    local pkg
    for pkg in ${missing[@]+"${missing[@]}"}; do
        note "installing $pkg ..."
        if ! python -m pip install --quiet "$pkg"; then
            if [ "$kind" = "required" ]; then
                warn "could not install required package: $pkg"
            else
                warn "optional package not installed: $pkg (the parts that need it are switched off)"
            fi
        fi
    done
    # Re-check required packages; the environment is unusable without them.
    if [ "$kind" = "required" ]; then
        for entry in $2; do
            pip_name="${entry%%:*}"; import_name="${entry##*:}"
            package_installed "$import_name" || { warn "$pip_name is still missing"; return 1; }
        done
    fi
    return 0
}

stage_env() {
    say "Stage: env"
    local env_path="$PROJECT_ROOT/$ENV_NAME"
    if [ "$RECREATE_ENV" -eq 1 ] && [ -d "$env_path" ]; then
        note "removing the old environment: $env_path"
        rm -rf "$env_path"
    fi
    if [ ! -f "$env_path/bin/activate" ]; then
        local base_python
        base_python="$(find_python)" || die "Python 3.10 or newer was not found. Install it, or pass --python /path/to/python."
        note "creating $ENV_NAME with $("$base_python" --version 2>&1)"
        "$base_python" -m venv "$env_path" || die "could not create the virtual environment (on Debian/Ubuntu: apt install python3-venv)"
    else
        note "reusing the existing environment: $env_path"
    fi
    # shellcheck disable=SC1091
    source "$env_path/bin/activate" || die "could not activate $env_path"
    note "activated: $(command -v python)  ($(python --version 2>&1))"

    python -m pip install --quiet --upgrade pip setuptools wheel
    note "checking required libraries"
    if check_and_install required "$REQUIRED_PACKAGES"; then
        record_step "env: required libraries" "ok"
    else
        record_step "env: required libraries" "FAILED"
    fi
    if [ "$SKIP_OPTIONAL" -eq 1 ]; then
        note "skipping optional libraries (--skip-optional)"
    else
        note "checking optional libraries"
        check_and_install optional "$OPTIONAL_PACKAGES"
    fi
}

# Every stage after "env" needs the environment; activate it quietly if env was skipped.
activate_env() {
    local env_path="$PROJECT_ROOT/$ENV_NAME"
    if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" = "$env_path" ]; then return 0; fi
    if [ -f "$env_path/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$env_path/bin/activate"
    else
        warn "environment $ENV_NAME does not exist; using the current Python ($(command -v python3 || command -v python)). Run './build.sh env' first."
        if ! command -v python >/dev/null 2>&1; then
            python() { python3 "$@"; }
        fi
    fi
}

if stage_wanted env; then stage_env; fi
activate_env

# =============================================================================
# Stage: record - the environment behind this run
# =============================================================================
stage_record() {
    say "Stage: record environment"
    {
        echo "date:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "host:       $(uname -a)"
        echo "python:     $(python --version 2>&1)  ($(command -v python))"
        echo "git commit: $(git rev-parse HEAD 2>/dev/null || echo 'not a git repository')"
        echo "git status: $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') uncommitted change(s)"
        echo "smoke mode: $SMOKE   workers: $WORKERS"
        echo
        echo "--- pip freeze ---"
        python -m pip freeze
        echo
        echo "--- sha256 of configs ---"
        if command -v sha256sum >/dev/null 2>&1; then
            (cd source/configs && sha256sum ./*.yaml)
        else
            (cd source/configs && shasum -a 256 ./*.yaml)
        fi
        if [ -f data/groups.csv ]; then
            echo; echo "--- sha256 of data/groups.csv ---"
            if command -v sha256sum >/dev/null 2>&1; then sha256sum data/groups.csv; else shasum -a 256 data/groups.csv; fi
        fi
    } > "$LOG_DIR/environment.txt"
    cp requirements.txt "$LOG_DIR/requirements.txt" 2>/dev/null || true
    note "written: $LOG_DIR/environment.txt"
    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
        warn "the git working tree has uncommitted changes; commit before the final run so the result maps to one commit."
    fi
    record_step "record environment" "ok"
}
if stage_wanted record; then stage_record; fi

# =============================================================================
# Stage: dataset - the data must be where the configs expect it
# =============================================================================
stage_dataset() {
    run_step "dataset check" python source/scripts/check_dataset.py
}
DATASET_OK=1
if stage_wanted dataset; then
    stage_dataset
    if [ "${STEP_STATUS[${#STEP_STATUS[@]}-1]}" = "FAILED" ]; then DATASET_OK=0; fi
fi

# =============================================================================
# Stage: tests - unit tests on synthetic data
# =============================================================================
if stage_wanted tests; then
    run_step "unit tests" python -m pytest source/tests -q
fi

# =============================================================================
# Stage: configs - the factorial cells must differ only in the intended factors
# =============================================================================
if stage_wanted configs; then
    run_step "audit factorial configs" python source/scripts/audit_configs.py --out "$RESULTS_DIR/config_audit"
fi

# Everything below trains on videos, so it needs the dataset.
needs_dataset() {  # needs_dataset <label> : returns 1 (and records a skip) without data
    if [ "$DATASET_OK" -eq 0 ]; then skip_step "$1" "dataset check failed"; return 1; fi
    return 0
}

# =============================================================================
# Stage: duplicates - group copies of one recording; later runs use the groups
# =============================================================================
GROUP_ARGS=()
if stage_wanted duplicates; then
    if needs_dataset "duplicate-recording audit"; then
        if [ "$SMOKE" -eq 1 ]; then
            skip_step "duplicate-recording audit" "smoke mode samples random videos, which the group table must cover completely"
        else
            run_step "duplicate-recording audit" python source/scripts/audit_duplicates.py --out data/groups.csv
        fi
    fi
fi
# Use the group table whenever it exists (and tell the user when it does not).
if [ "$SMOKE" -eq 0 ] && [ -f data/groups.csv ]; then
    GROUP_ARGS=(--set "training.group_map_csv=data/groups.csv")
    note "cross-validation will keep each recording in one fold (data/groups.csv)"
elif [ "$SMOKE" -eq 0 ]; then
    warn "data/groups.csv not found: cross-validation will NOT be group-aware. Run './build.sh duplicates' first."
fi

# train_config <config-name> [extra run.py args...] : one experiment, both targets.
train_config() {
    local config_name="$1"; shift
    run_step "train: $config_name" python source/run.py --config "$config_name" \
        "${RUN_ARGS[@]}" ${GROUP_ARGS[@]+"${GROUP_ARGS[@]}"} "$@"
}

# =============================================================================
# Stage: factorial - the 8 cells of the 2x2x2 design
# =============================================================================
if stage_wanted factorial; then
    if needs_dataset "factorial cells"; then
        for cfg_name in $FACTORIAL_CONFIGS; do train_config "$cfg_name"; done
    fi
fi

# =============================================================================
# Stage: detectors - alternative background subtractors
# =============================================================================
if stage_wanted detectors; then
    if needs_dataset "alternative detectors"; then
        for cfg_name in $DETECTOR_CONFIGS; do
            if [[ "$cfg_name" == *subsense* ]] && ! python -c "import pybgs" >/dev/null 2>&1; then
                skip_step "train: $cfg_name" "pybgs is not installed (SuBSENSE needs it)"
                continue
            fi
            train_config "$cfg_name"
        done
    fi
fi

# =============================================================================
# Stage: features - feature-family ablations, minimum-track-length filter
# =============================================================================
if stage_wanted features; then
    if needs_dataset "feature ablations"; then
        for cfg_name in $FEATURE_CONFIGS; do train_config "$cfg_name"; done
    fi
fi

# =============================================================================
# Stage: groups - remaining experiment groups (neural ones need PyTorch)
# =============================================================================
if stage_wanted groups; then
    if needs_dataset "other experiment groups"; then
        for cfg_name in $OTHER_GROUP_CONFIGS; do
            if [[ "$cfg_name" == group_c_* || "$cfg_name" == group_d_* ]] && ! python -c "import torch" >/dev/null 2>&1; then
                skip_step "train: $cfg_name" "PyTorch is not installed"
                continue
            fi
            train_config "$cfg_name"
        done
    fi
fi

# =============================================================================
# Stage: threshold - is the adaptive gain just a lower fixed threshold?
# =============================================================================
if stage_wanted threshold; then
    if needs_dataset "MOG2 threshold sweep"; then
        for target_name in activity distortion; do
            run_step "threshold sweep ($target_name)" python source/scripts/mog2_threshold_sweep.py \
                --target "$target_name" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} ${SAMPLE_ARGS[@]+"${SAMPLE_ARGS[@]}"}
        done
    fi
fi

# =============================================================================
# Stage: ablation - every selected parameter, one at a time
# =============================================================================
if stage_wanted ablation; then
    if needs_dataset "parameter ablation"; then
        run_step "parameter ablation" python source/scripts/parameter_ablation.py \
            --workers "$WORKERS" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} ${SAMPLE_ARGS[@]+"${SAMPLE_ARGS[@]}"}
    fi
fi

# =============================================================================
# Stage: maskquality - does Q_mask track real mask quality?
# =============================================================================
if stage_wanted maskquality; then
    if needs_dataset "mask-quality validation"; then
        n_per_condition=10
        if [ "$SMOKE" -eq 1 ]; then n_per_condition=2; fi
        run_step "mask-quality validation" python source/scripts/validate_mask_quality.py \
            --n "$n_per_condition" --out "$RESULTS_DIR/mask_quality_validation"
    fi
fi

# =============================================================================
# Stage: verify - a finished script is not a finished study; inspect the outputs
# =============================================================================
if stage_wanted verify; then
    if needs_dataset "verify outputs"; then
        run_step "verify outputs" python source/scripts/verify_outputs.py \
            --configs $FACTORIAL_CONFIGS $FEATURE_CONFIGS --experiments-dir "$RESULTS_DIR"
    fi
fi

# =============================================================================
# Stage: analysis - effects, comparisons, importance
# =============================================================================
if stage_wanted analysis; then
    if needs_dataset "analysis"; then
        for target_name in activity distortion; do
            for universe_name in all common; do
                run_step "factorial analysis ($target_name, $universe_name videos)" \
                    python source/scripts/factorial_analysis.py --target "$target_name" --universe "$universe_name" \
                    --experiments-dir "$RESULTS_DIR"
            done
            # Does adaptive MOG2 alone help? (detector-only vs baseline), and with everything else on.
            run_step "compare: detector only ($target_name)" python source/scripts/compare_experiments.py \
                group_a_classical ablation_1_detector_only --target "$target_name" --universe both \
                --experiments-dir "$RESULTS_DIR"
            run_step "compare: adaptive vs fixed, full pipeline ($target_name)" python source/scripts/compare_experiments.py \
                ablation_4_full_pipeline_fixed_mog2 group_i_adaptive_mog2 --target "$target_name" --universe both \
                --experiments-dir "$RESULTS_DIR"
            # Which feature families does the classifier rely on?
            run_step "feature-family importance ($target_name)" python source/scripts/feature_family_importance.py \
                --latest group_i_adaptive_mog2 --target "$target_name" --experiments-dir "$RESULTS_DIR"
        done
    fi
fi

# =============================================================================
# Stage: figures - copy what the paper uses into docs/images
# =============================================================================
if stage_wanted figures; then
    if [ "$SMOKE" -eq 1 ]; then
        skip_step "collect paper figures" "smoke results must not overwrite the paper's figures"
    else
        run_step "collect paper figures" python source/scripts/collect_paper_figures.py --strict
    fi
fi

summarize
if [ "$FAILURES" -gt 0 ]; then exit 1; fi
exit 0
