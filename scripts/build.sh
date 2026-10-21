#!/usr/bin/env bash
# Sets up the environment and runs all the training and testing, stage by stage.
# Run ./scripts/build.sh --help for the options.

set -u -o pipefail

# This is the project root (the folder above scripts/), so the script works from any folder.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# These are the settings; each can be changed with an option or an environment variable.
ENV_NAME="${SDR_ENV_NAME:-sdr-env}"
PYTHON_BIN="${SDR_PYTHON:-}"
WORKERS="${SDR_WORKERS:-1}"
CV_JOBS="${SDR_CV_JOBS:-1}"
SAMPLE_PER_CLASS=5
SMOKE=0
RESUME=0
FAIL_FAST=0
RECREATE_ENV=0
SKIP_OPTIONAL=0
SELECTED_STAGES=()

# This is the list of stages in the order they run.
ALL_STAGES="env record dataset tests configs duplicates factorial detectors features groups threshold ablation maskquality verify analysis figures"

# These are the config names (the files in source/configs without .yaml) that each stage trains.
FACTORIAL_CONFIGS="group_a_classical ablation_1_detector_only ablation_2_preprocessing_only ablation_3_features_only ablation_4_full_pipeline_fixed_mog2 ablation_5_detector_preprocessing ablation_6_detector_features group_i_adaptive_mog2"
DETECTOR_CONFIGS="detector_vibe detector_vibe_fullfeat detector_pbas detector_pbas_fullfeat detector_subsense detector_subsense_fullfeat"
FEATURE_CONFIGS="ablation_0_baseline_no_meta ablation_7_no_image_quality_meta ablation_8_motion_only group_i_minlen30_filtered"
OTHER_GROUP_CONFIGS="group_b_distortion_aware group_e_gt_upper_bound group_f_tracking_ablation group_g_feature_comparison group_h_cascaded_fallback group_h_optical_flow_detector group_j_tfcr_tracker group_c_neural group_d_hybrid"

# These are the Python packages as pip-name:import-name. Without a required one the study
# cannot run; a missing optional one only switches a part off.
REQUIRED_PACKAGES="numpy:numpy pandas:pandas opencv-python:cv2 PyYAML:yaml scipy:scipy scikit-learn:sklearn openpyxl:openpyxl joblib:joblib imbalanced-learn:imblearn lightgbm:lightgbm xgboost:xgboost PyWavelets:pywt matplotlib:matplotlib pytest:pytest"
OPTIONAL_PACKAGES="seaborn:seaborn torch:torch fiftyone:fiftyone pybgs:pybgs"

# This is where the results of the step list are collected.
STEP_NAMES=()
STEP_STATUS=()
FAILURES=0

say()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33mWARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: scripts/build.sh [options] [stage ...]

Stages (all of them run when none is named):
  env          create the virtual environment and install the libraries
  record       write the Python version, pip freeze, git commit and config checksums
  dataset      check that the AD-SVD dataset is where the configs expect it
  tests        run the unit tests (synthetic data, no dataset needed)
  configs      check that the 8 factorial cells differ only in the intended factors
  duplicates   find copies of one recording and write data/groups.csv
  factorial    train the 8 cells of the 2x2x2 design
  detectors    ViBe, PBAS and SuBSENSE (needs pybgs) against MOG2
  features     feature-family ablations and the minimum-track-length filter
  groups       the other experiment groups (B, E, F, G, H, J and the neural C, D)
  threshold    fixed against adaptive MOG2 over a range of variance thresholds
  ablation     one-at-a-time ablation of the selected parameters
  maskquality  check the mask-quality score against the ground-truth boxes
  verify       check that every required result exists and no classifier failed
  analysis     factorial effects, comparisons and feature-family importance
  figures      copy the figures of the paper into docs/manuscript/images

Options:
  --smoke           tiny sample (5 videos per class); results go to experiments_smoke/
  --sample-size N   videos per class in smoke mode
  --workers N       processes that read the videos, or "auto" for all cores but one (default 1)
  --cv-jobs N       cross-validation folds trained at the same time, -1 for all cores (default 1)
  --resume          skip the sweep and ablation settings that are already finished
  --env-name NAME   virtual environment folder (default sdr-env)
  --python PATH     interpreter that creates the environment (3.10 or newer)
  --recreate-env    delete and rebuild the environment
  --skip-optional   do not install torch, fiftyone, pybgs and seaborn
  --fail-fast       stop at the first failed step
  -h, --help        show this text

A failed step is recorded and the run goes on, because one bad config should not stop a
long run. The failures are listed at the end and the exit status is then 1.

Examples:
  scripts/build.sh --smoke
  scripts/build.sh env tests
  scripts/build.sh --workers auto --resume
EOF
}

stage_wanted() {
    # This succeeds when the stage was named on the command line, or when none was named.
    local stage="$1" s
    if [ "${#SELECTED_STAGES[@]}" -eq 0 ]; then return 0; fi
    for s in "${SELECTED_STAGES[@]}"; do
        if [ "$s" = "$stage" ]; then return 0; fi
    done
    return 1
}

record_step() {
    STEP_NAMES+=("$1")
    STEP_STATUS+=("$2")
    if [ "$2" = "FAILED" ]; then FAILURES=$((FAILURES + 1)); fi
}

run_step() {
    # run_step <label> <command...>: run it, keep its output in the log folder and record the result.
    local label="$1"
    shift
    say "$label"
    note "\$ $*"
    local logfile="$LOG_DIR/$(echo "$label" | tr -c 'A-Za-z0-9._-' '_').log"
    "$@" 2>&1 | tee "$logfile"
    local status="${PIPESTATUS[0]}"
    if [ "$status" -eq 0 ]; then
        record_step "$label" "ok"
    else
        warn "step failed (exit $status): $label (log: $logfile)"
        record_step "$label" "FAILED"
        if [ "$FAIL_FAST" -eq 1 ]; then summarize; exit 1; fi
    fi
    return 0
}

skip_step() {
    say "$1"
    note "skipped: $2"
    record_step "$1" "skipped"
}

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

cpu_count() {
    nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --smoke)         SMOKE=1 ;;
        --sample-size)   SAMPLE_PER_CLASS="$2"; shift ;;
        --workers)       WORKERS="$2"; shift ;;
        --cv-jobs)       CV_JOBS="$2"; shift ;;
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
                *) die "unknown stage: $1 (stages: $ALL_STAGES)" ;;
            esac ;;
    esac
    shift
done

# "auto" becomes all the cores but one, so the machine stays usable.
if [ "$WORKERS" = "auto" ]; then
    WORKERS=$(( $(cpu_count) - 1 ))
    if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
fi

# Smoke runs write to their own folders, so they cannot be mistaken for real results.
LOG_STAMP="$(date +%Y%m%d_%H%M%S)"
if [ "$SMOKE" -eq 1 ]; then
    export SDR_EXPERIMENTS_DIR="experiments_smoke"
    LOG_STAMP="smoke_$LOG_STAMP"
fi
RESULTS_DIR="${SDR_EXPERIMENTS_DIR:-experiments}"
LOG_DIR="${SDR_LOGS_DIR:-docs/logs}/$LOG_STAMP"
mkdir -p "$LOG_DIR"

# These are the arguments that every training run gets.
RUN_ARGS=(--target both --workers "$WORKERS" --set "training.n_jobs=$CV_JOBS")
SAMPLE_ARGS=()
RESUME_ARGS=()
if [ "$SMOKE" -eq 1 ]; then
    RUN_ARGS+=(--sample --n "$SAMPLE_PER_CLASS")
    SAMPLE_ARGS+=(--sample --n "$SAMPLE_PER_CLASS")
fi
if [ "$RESUME" -eq 1 ]; then RESUME_ARGS+=(--resume); fi

say "Project root: $PROJECT_ROOT"
note "environment: $ENV_NAME   workers: $WORKERS   cv jobs: $CV_JOBS   smoke: $SMOKE   results: $RESULTS_DIR"

find_python() {
    # This prints the first interpreter that is Python 3.10 or newer.
    local candidate
    for candidate in "$PYTHON_BIN" python3.12 python3.11 python3.10 python3 python; do
        [ -n "$candidate" ] || continue
        if command -v "$candidate" >/dev/null 2>&1 &&
           "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

package_installed() {
    python -c "import $1" >/dev/null 2>&1
}

check_and_install() {
    # check_and_install <required|optional> "<pip-name:import-name> ..."
    local kind="$1" entry pip_name import_name pkg
    local missing=()
    for entry in $2; do
        pip_name="${entry%%:*}"
        import_name="${entry##*:}"
        if package_installed "$import_name"; then
            note "found    $pip_name"
        else
            note "missing  $pip_name"
            missing+=("$pip_name")
        fi
    done
    for pkg in ${missing[@]+"${missing[@]}"}; do
        note "installing $pkg"
        if ! python -m pip install --quiet "$pkg"; then
            if [ "$kind" = "required" ]; then
                warn "could not install the required package $pkg"
            else
                warn "optional package $pkg not installed (the parts that need it are switched off)"
            fi
        fi
    done
    # The environment is unusable without the required packages, so check them again.
    if [ "$kind" = "required" ]; then
        for entry in $2; do
            pip_name="${entry%%:*}"
            import_name="${entry##*:}"
            if ! package_installed "$import_name"; then
                warn "$pip_name is still missing"
                return 1
            fi
        done
    fi
    return 0
}

stage_env() {
    say "Stage: env"
    local env_path="$PROJECT_ROOT/$ENV_NAME" base_python
    if [ "$RECREATE_ENV" -eq 1 ] && [ -d "$env_path" ]; then
        note "removing the old environment: $env_path"
        rm -rf "$env_path"
    fi
    if [ ! -f "$env_path/bin/activate" ]; then
        base_python="$(find_python)" || die "Python 3.10 or newer was not found; install it or pass --python PATH"
        note "creating $ENV_NAME with $("$base_python" --version 2>&1)"
        "$base_python" -m venv "$env_path" || die "could not create the environment (on Debian or Ubuntu: apt install python3-venv)"
    else
        note "reusing the environment: $env_path"
    fi
    # shellcheck disable=SC1091
    source "$env_path/bin/activate" || die "could not activate $env_path"
    note "activated: $(command -v python) ($(python --version 2>&1))"

    python -m pip install --quiet --upgrade pip setuptools wheel
    note "checking the required libraries"
    if check_and_install required "$REQUIRED_PACKAGES"; then
        record_step "env: required libraries" "ok"
    else
        record_step "env: required libraries" "FAILED"
    fi
    if [ "$SKIP_OPTIONAL" -eq 1 ]; then
        note "skipping the optional libraries (--skip-optional)"
    else
        note "checking the optional libraries"
        check_and_install optional "$OPTIONAL_PACKAGES"
    fi
}

activate_env() {
    # Every stage after env needs the environment, so this activates it when env was skipped.
    local env_path="$PROJECT_ROOT/$ENV_NAME"
    if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" = "$env_path" ]; then return 0; fi
    if [ -f "$env_path/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$env_path/bin/activate"
    else
        warn "environment $ENV_NAME does not exist, so the current Python is used (run scripts/build.sh env first)"
        if ! command -v python >/dev/null 2>&1; then
            python() { python3 "$@"; }
        fi
    fi
}

stage_record() {
    say "Stage: record the environment"
    {
        echo "date:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "host:       $(uname -a)"
        echo "python:     $(python --version 2>&1) ($(command -v python))"
        echo "git commit: $(git rev-parse HEAD 2>/dev/null || echo 'not a git repository')"
        echo "git status: $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') uncommitted change(s)"
        echo "smoke mode: $SMOKE   workers: $WORKERS   cv jobs: $CV_JOBS"
        echo
        echo "--- pip freeze ---"
        python -m pip freeze
        echo
        echo "--- sha256 of the configs ---"
        if command -v sha256sum >/dev/null 2>&1; then
            (cd source/configs && sha256sum ./*.yaml)
        else
            (cd source/configs && shasum -a 256 ./*.yaml)
        fi
        if [ -f data/groups.csv ]; then
            echo
            echo "--- sha256 of data/groups.csv ---"
            if command -v sha256sum >/dev/null 2>&1; then sha256sum data/groups.csv; else shasum -a 256 data/groups.csv; fi
        fi
    } > "$LOG_DIR/environment.txt"
    cp requirements.txt "$LOG_DIR/requirements.txt" 2>/dev/null || true
    note "written: $LOG_DIR/environment.txt"
    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
        warn "the git working tree has uncommitted changes; commit before the final run so the results map to one commit"
    fi
    record_step "record environment" "ok"
}

needs_dataset() {
    # needs_dataset <label>: fails, and records a skipped step, when the dataset check failed.
    if [ "$DATASET_OK" -eq 0 ]; then
        skip_step "$1" "the dataset check failed"
        return 1
    fi
    return 0
}

train_config() {
    # train_config <config-name> [more run.py arguments]: one experiment, both targets.
    local config_name="$1"
    shift
    run_step "train: $config_name" python source/run.py --config "$config_name" \
        "${RUN_ARGS[@]}" ${GROUP_ARGS[@]+"${GROUP_ARGS[@]}"} "$@"
}

if stage_wanted env; then stage_env; fi
activate_env

if stage_wanted record; then stage_record; fi

# This stays 1 unless the dataset check fails; the stages that need videos are then skipped.
DATASET_OK=1
if stage_wanted dataset; then
    run_step "dataset check" python scripts/check_dataset.py
    if [ "${STEP_STATUS[${#STEP_STATUS[@]}-1]}" = "FAILED" ]; then DATASET_OK=0; fi
fi

if stage_wanted tests; then
    run_step "unit tests" python -m pytest source/tests -q
fi

if stage_wanted configs; then
    run_step "audit factorial configs" python scripts/audit_configs.py --out "$RESULTS_DIR/config_audit"
fi

# This is the group table that keeps copies of one recording in the same fold.
GROUP_ARGS=()
if stage_wanted duplicates; then
    if needs_dataset "duplicate-recording audit"; then
        if [ "$SMOKE" -eq 1 ]; then
            skip_step "duplicate-recording audit" "a smoke run samples random videos, and the group table must cover all of them"
        else
            run_step "duplicate-recording audit" python scripts/audit_duplicates.py --out data/groups.csv
        fi
    fi
fi
if [ "$SMOKE" -eq 0 ] && [ -f data/groups.csv ]; then
    GROUP_ARGS=(--set "training.group_map_csv=data/groups.csv")
    note "cross-validation keeps each recording in one fold (data/groups.csv)"
elif [ "$SMOKE" -eq 0 ]; then
    warn "data/groups.csv not found, so cross-validation is NOT group-aware; run scripts/build.sh duplicates first"
fi

if stage_wanted factorial; then
    if needs_dataset "factorial cells"; then
        for cfg_name in $FACTORIAL_CONFIGS; do train_config "$cfg_name"; done
    fi
fi

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

if stage_wanted features; then
    if needs_dataset "feature ablations"; then
        for cfg_name in $FEATURE_CONFIGS; do train_config "$cfg_name"; done
    fi
fi

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

if stage_wanted threshold; then
    if needs_dataset "MOG2 threshold sweep"; then
        for target_name in activity distortion; do
            run_step "threshold sweep ($target_name)" python scripts/mog2_threshold_sweep.py \
                --target "$target_name" --workers "$WORKERS" \
                ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} ${SAMPLE_ARGS[@]+"${SAMPLE_ARGS[@]}"}
        done
    fi
fi

if stage_wanted ablation; then
    if needs_dataset "parameter ablation"; then
        run_step "parameter ablation" python scripts/parameter_ablation.py --workers "$WORKERS" \
            ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} ${SAMPLE_ARGS[@]+"${SAMPLE_ARGS[@]}"}
    fi
fi

if stage_wanted maskquality; then
    if needs_dataset "mask-quality validation"; then
        n_per_condition=10
        if [ "$SMOKE" -eq 1 ]; then n_per_condition=2; fi
        run_step "mask-quality validation" python scripts/validate_mask_quality.py \
            --n "$n_per_condition" --out "$RESULTS_DIR/mask_quality_validation"
    fi
fi

if stage_wanted verify; then
    if needs_dataset "verify outputs"; then
        run_step "verify outputs" python scripts/verify_outputs.py \
            --configs $FACTORIAL_CONFIGS $FEATURE_CONFIGS --experiments-dir "$RESULTS_DIR"
    fi
fi

if stage_wanted analysis; then
    if needs_dataset "analysis"; then
        for target_name in activity distortion; do
            for universe_name in all common; do
                run_step "factorial analysis ($target_name, $universe_name videos)" \
                    python scripts/factorial_analysis.py --target "$target_name" --universe "$universe_name" \
                    --experiments-dir "$RESULTS_DIR"
            done
            # Does adaptive MOG2 alone help, and does it help with everything else switched on?
            run_step "compare: detector only ($target_name)" python scripts/compare_experiments.py \
                group_a_classical ablation_1_detector_only --target "$target_name" --universe both \
                --experiments-dir "$RESULTS_DIR"
            run_step "compare: adaptive against fixed, full pipeline ($target_name)" python scripts/compare_experiments.py \
                ablation_4_full_pipeline_fixed_mog2 group_i_adaptive_mog2 --target "$target_name" --universe both \
                --experiments-dir "$RESULTS_DIR"
            # Which feature families does the classifier rely on?
            run_step "feature-family importance ($target_name)" python scripts/feature_family_importance.py \
                --latest group_i_adaptive_mog2 --target "$target_name" --experiments-dir "$RESULTS_DIR"
        done
    fi
fi

if stage_wanted figures; then
    if [ "$SMOKE" -eq 1 ]; then
        skip_step "collect paper figures" "smoke results must not overwrite the figures of the paper"
    else
        run_step "collect paper figures" python scripts/collect_paper_figures.py --strict
    fi
fi

summarize
if [ "$FAILURES" -gt 0 ]; then exit 1; fi
exit 0
