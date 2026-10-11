#!/usr/bin/env bash
# Runs the experiments the SACAIR reviewers asked for, in priority order.
# Run from the project root:   bash scripts/run_review_experiments.sh [--smoke]
#   --smoke   tiny subset of videos per class, to check everything works first (minutes)
# Each step is resumable on its own; a failed step does not stop the rest.
set -u
EXTRA=""
if [ "${1:-}" = "--smoke" ]; then EXTRA="--sample --n 5"; fi
SWEEP_EXTRA="$EXTRA"
step() { echo; echo "######## $*"; }

step "1. R3 / R1: single-factor ablations + baseline + full pipeline (all 7 classifiers)"
for cfg in group_a_classical ablation_1_detector_only ablation_2_preprocessing_only \
           ablation_3_features_only ablation_4_full_pipeline_fixed_mog2 group_i_adaptive_mog2; do
  python run.py --config "$cfg" --target both $EXTRA || echo "FAILED: $cfg"
done

step "2. R1: other adaptive background subtractors"
for cfg in detector_vibe detector_pbas detector_subsense; do
  python run.py --config "$cfg" --target both $EXTRA || echo "FAILED: $cfg (subsense needs pybgs)"
done

step "3. var_threshold ablation (fixed vs adaptive MOG2)"
for t in activity distortion; do
  python scripts/mog2_threshold_sweep.py --target $t --resume $SWEEP_EXTRA || echo "FAILED: threshold sweep $t"
done

step "4. R1: sensitivity of q, w, tau_r, tau_min"
for t in activity distortion; do
  python scripts/sensitivity_sweep.py --target $t --resume $SWEEP_EXTRA || echo "FAILED: sensitivity sweep $t"
done

echo; echo "Done. Results are under experiments/. Compare runs with scripts/compare_experiments.py."
