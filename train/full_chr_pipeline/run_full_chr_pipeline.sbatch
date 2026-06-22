#!/bin/bash
#SBATCH --job-name=chr_14
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --time=20:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/full_chr_pipeline_%j.out
#SBATCH --error=logs/full_chr_pipeline_%j.err
#SBATCH --account=minjilab99
# Recommended invocation (2 GPUs, fine near-diagonal coverage, 5 Mbp band):
#   CHROM=2 NUM_GPUS=2 BATCH_SIZE=64 \
#     DIAG_STEP_NEAR_BP=80000 DIAG_STEP_FAR_BP=320000 NEAR_BAND_BP=5000000 \
#     sbatch --gres=gpu:2 full_chr_pipeline/run_full_chr_pipeline.sbatch
#
# Single GPU (slower):
#   NUM_GPUS=1 sbatch --gres=gpu:1 full_chr_pipeline/run_full_chr_pipeline.sbatch


# Do not use `set -u` before sourcing bashrc/conda: /etc/bashrc uses
# BASHRCSOURCED before it is set, which aborts under nounset.
WORKDIR="/nfs/turbo/umms-minjilab/lpullela/cell-cycle/train"
PIPE_DIR="${WORKDIR}/full_chr_pipeline"

CHROM="${CHROM:-14}"
#CHECKPOINT="${CHECKPOINT:-${WORKDIR}/checkpoints/oe_log_4phase_epoch40_4-16_asym-residual-loss.pth}"
#CHECKPOINT="${CHECKPOINT:-${WORKDIR}/checkpoints/oe_log_4phase_epoch40_4-30.pth}"
#CHECKPOINT="${CHECKPOINT:-${WORKDIR}/checkpoints/observed_log_5phase_epoch40_5_15.pth}" # chr14
#CHECKPOINT="${CHECKPOINT:-${WORKDIR}/checkpoints/observed_log_5phase_epoch40_5_30_bulk_norm_ssim_and_phase_sim.pth}" # chr14
#CHECKPOINT="${CHECKPOINT:-${WORKDIR}/checkpoints/observed_log_5phase_epoch40_6-6_chip_loop_loss_holdout_2.pth}" # chr14
CHECKPOINT="${CHECKPOINT:-${WORKDIR}/checkpoints/observed_log_5phase_epoch40_6-12_chip_oe_loss_holdout_14.pth}" # chr14

# These default to your repo structure; override as needed.
DATA_DIR="${DATA_DIR:-/nfs/turbo/umms-minjilab/lpullela/cell-cycle/raw_data/zhang_4dn}"
CHIP_DIR="${CHIP_DIR:-$DATA_DIR}"

OUT_BASE="${OUT_BASE:-${WORKDIR}/full_chr_outputs_zhang/tukey/no_window_actual/chr${CHROM}}"
ARRAYS_DIR="${ARRAYS_DIR:-${OUT_BASE}/arrays}"
NEAR_DIR="${NEAR_DIR:-${OUT_BASE}/near_pred}"
FINAL_DIR="${FINAL_DIR:-${OUT_BASE}/final}"
JUICER_DIR="${JUICER_DIR:-${OUT_BASE}/juicer_txt}"

HIC_UNIT="${HIC_UNIT:-observed}"   # oe or observed
NORM="${NORM:-KR}"
NEAR_BAND_BP="${NEAR_BAND_BP:-5000000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
# Step sizes for diagonal tile row positions.
# DIAG_STEP_NEAR_BP applies to tiles within NEAR_FAR_THRESHOLD_BP of the diagonal.
# DIAG_STEP_FAR_BP  applies to tiles beyond NEAR_FAR_THRESHOLD_BP.
# Lower step = finer overlap (e.g. 80000 = 87.5%, 320000 = 50%, 640000 = no overlap).
DIAG_STEP_NEAR_BP="${DIAG_STEP_NEAR_BP:-160000}"
DIAG_STEP_FAR_BP="${DIAG_STEP_FAR_BP:-320000}"
NEAR_FAR_THRESHOLD_BP="${NEAR_FAR_THRESHOLD_BP:-1000000}"
# Number of GPUs to use for the near-diagonal inference step.
# Each GPU gets a contiguous spatial block of regions (e.g. with 2 GPUs:
#   GPU 0 → first half of the chromosome, GPU 1 → second half).
# IMPORTANT: --gres=gpu:NUM_GPUS must match (pass via sbatch --gres=gpu:N or
# edit the #SBATCH --gres line at the top of this file).
NUM_GPUS="${NUM_GPUS:-1}"

# Testing shortcut: set TEST_FRAC to a fraction in (0,1] to only run inference on the
# TRAILING block of the chromosome (e.g. TEST_FRAC=0.1 = last 10%).  The end of the
# chromosome is used because it is less likely to contain unmappable regions.  Everything
# outside that block is written as zeros so the exported .hic only contains the tested
# region.  Leave at 0 (default) to process the whole chromosome.
#   Example: TEST_FRAC=0.1 sbatch full_chr_pipeline/run_full_chr_pipeline.sbatch
TEST_FRAC="${TEST_FRAC:-0}"

# Tile merge weights: Tukey window (default) or uniform if NO_HANNING=1.
TUKEY_MIN_WEIGHT="${TUKEY_MIN_WEIGHT:-0.25}"
TUKEY_TAPER_FRAC="${TUKEY_TAPER_FRAC:-0.08}"
# Set NO_HANNING=1 to disable tile weighting (uniform weights over overlapping tiles).
NO_HANNING="${NO_HANNING:-0}"

# Step controls (set to 1 to skip). This script never deletes outputs.
# Typical Juicer text-only rerun:
#   ONLY_EXPORT=1 sbatch full_chr_pipeline/run_full_chr_pipeline.sbatch
ONLY_EXPORT="${ONLY_EXPORT:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
SKIP_INFER="${SKIP_INFER:-0}"
SKIP_FILL="${SKIP_FILL:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"

# Juicer text export format (see export_juicer_sparse.py)
EXPORT_FORMAT="${EXPORT_FORMAT:-short_with_score}"
# Set EXPORT_ROUND_INT=1 to round scores to integers (often zeros out weak O/E off-diagonal).
EXPORT_ROUND_INT="${EXPORT_ROUND_INT:-0}"

source ~/.bashrc
conda activate test_env

set -euo pipefail

if [[ "$ONLY_EXPORT" == "1" ]]; then
  SKIP_EXTRACT=1
  SKIP_INFER=1
  SKIP_FILL=1
fi

cd "$WORKDIR"
mkdir -p logs "$OUT_BASE" "$ARRAYS_DIR" "$NEAR_DIR" "$FINAL_DIR" "$JUICER_DIR"

echo "========== Full chromosome pipeline =========="
echo "Job: $SLURM_JOB_ID  Node: $HOSTNAME"
echo "Chrom: $CHROM"
echo "Checkpoint: $CHECKPOINT"
echo "DATA_DIR: $DATA_DIR"
echo "OUT_BASE: $OUT_BASE"
echo "HIC_UNIT: $HIC_UNIT   NORM: $NORM   NEAR_BAND_BP: $NEAR_BAND_BP"
echo "BATCH_SIZE: $BATCH_SIZE   NUM_GPUS: $NUM_GPUS"
echo "DIAG_STEP_NEAR_BP: $DIAG_STEP_NEAR_BP   DIAG_STEP_FAR_BP: $DIAG_STEP_FAR_BP   NEAR_FAR_THRESHOLD_BP: $NEAR_FAR_THRESHOLD_BP"
echo "TEST_FRAC: $TEST_FRAC   TILE_WEIGHT: tukey(min=$TUKEY_MIN_WEIGHT,taper=$TUKEY_TAPER_FRAC)   NO_HANNING: $NO_HANNING"
echo "ONLY_EXPORT: $ONLY_EXPORT"
echo "SKIP_EXTRACT / SKIP_INFER / SKIP_FILL / SKIP_EXPORT: $SKIP_EXTRACT / $SKIP_INFER / $SKIP_FILL / $SKIP_EXPORT"
echo "EXPORT_FORMAT: $EXPORT_FORMAT"
echo "EXPORT_ROUND_INT: $EXPORT_ROUND_INT"
echo "=============================================="

if [[ "$SKIP_EXTRACT" != "1" ]]; then
  echo "[1/4] Extract full-chrom arrays -> $ARRAYS_DIR"
  python "${PIPE_DIR}/extract_chr_numpy.py" \
    --chrom "$CHROM" \
    --data_dir "$DATA_DIR" \
    --chip_dir "$CHIP_DIR" \
    --output_dir "$ARRAYS_DIR" \
    --hic_unit "$HIC_UNIT" \
    --norm "$NORM"
else
  echo "[1/4] SKIP extract (reuse arrays in $ARRAYS_DIR)"
fi

if [[ "$SKIP_INFER" != "1" ]]; then
  echo "[2/4] Near-diagonal GPU inference -> $NEAR_DIR  (NUM_GPUS=$NUM_GPUS)"
  infer_extra=(
    --tukey_min_weight "$TUKEY_MIN_WEIGHT"
    --tukey_taper_frac "$TUKEY_TAPER_FRAC"
  )
  if [[ "$NO_HANNING" == "1" ]]; then
    infer_extra+=(--no_hanning)
  fi
  if [[ "$NUM_GPUS" -gt 1 ]]; then
    # Launch one worker per GPU in parallel; each owns every NUM_GPUS-th region.
    pids=()
    for (( k=0; k<NUM_GPUS; k++ )); do
      CUDA_VISIBLE_DEVICES=$k python "${PIPE_DIR}/infer_chr_near_diag_gpu.py" \
        --chrom "$CHROM" \
        --checkpoint "$CHECKPOINT" \
        --arrays_dir "$ARRAYS_DIR" \
        --output_dir "$NEAR_DIR" \
        --near_band_bp "$NEAR_BAND_BP" \
        --batch_size "$BATCH_SIZE" \
        --diag_step_near_bp "$DIAG_STEP_NEAR_BP" \
        --diag_step_far_bp "$DIAG_STEP_FAR_BP" \
        --near_far_threshold_bp "$NEAR_FAR_THRESHOLD_BP" \
        --test_frac "$TEST_FRAC" \
        --shard_id "$k" \
        --num_shards "$NUM_GPUS" \
        "${infer_extra[@]}" &
      pids+=($!)
    done
    # Wait for all workers; abort the job if any fail.
    for pid in "${pids[@]}"; do
      wait "$pid" || { echo "ERROR: inference shard failed (pid $pid)"; exit 1; }
    done
    echo "  All shards complete. Merging..."
    python "${PIPE_DIR}/merge_near_diag_shards.py" \
      --chrom "$CHROM" \
      --near_dir "$NEAR_DIR" \
      --num_shards "$NUM_GPUS" \
      --delete_shards
  else
    python "${PIPE_DIR}/infer_chr_near_diag_gpu.py" \
      --chrom "$CHROM" \
      --checkpoint "$CHECKPOINT" \
      --arrays_dir "$ARRAYS_DIR" \
      --output_dir "$NEAR_DIR" \
      --near_band_bp "$NEAR_BAND_BP" \
      --batch_size "$BATCH_SIZE" \
      --diag_step_near_bp "$DIAG_STEP_NEAR_BP" \
      --diag_step_far_bp "$DIAG_STEP_FAR_BP" \
      --near_far_threshold_bp "$NEAR_FAR_THRESHOLD_BP" \
      --test_frac "$TEST_FRAC" \
      "${infer_extra[@]}"
  fi
else
  echo "[2/4] SKIP near-diagonal inference (reuse $NEAR_DIR)"
fi

if [[ "$SKIP_FILL" != "1" ]]; then
  echo "[3/4] CPU off-diagonal fill + stitch -> $FINAL_DIR"
  python "${PIPE_DIR}/fill_chr_offdiag_cpu.py" \
    --chrom "$CHROM" \
    --arrays_dir "$ARRAYS_DIR" \
    --near_dir "$NEAR_DIR" \
    --output_dir "$FINAL_DIR" \
    --near_band_bp "$NEAR_BAND_BP" \
    --test_frac "$TEST_FRAC"
else
  echo "[3/4] SKIP fill (reuse final matrices in $FINAL_DIR)"
fi

if [[ "$SKIP_EXPORT" != "1" ]]; then
  echo "[4/4] Export Juicer sparse text -> $JUICER_DIR"
  for PHASE in earlyG1 midG1 lateG1 anatelo prometa; do
    extra=()
    if [[ "$EXPORT_ROUND_INT" == "1" ]]; then
      extra+=(--round_int)
    fi
    python "${PIPE_DIR}/export_juicer_sparse.py" \
      --chrom "$CHROM" \
      --matrix_npy "${FINAL_DIR}/chr${CHROM}_${PHASE}_final_raw.npy" \
      --output_txt "${JUICER_DIR}/chr${CHROM}_${PHASE}_uppertri.txt" \
      --format "$EXPORT_FORMAT" \
      "${extra[@]}"
  done
else
  echo "[4/4] SKIP Juicer text export"
fi

echo "Done. Juicer text outputs in: $JUICER_DIR"



# TEST_FRAC=0.10 NO_HANNING=1 sbatch full_chr_pipeline/run_full_chr_pipeline.sbatch