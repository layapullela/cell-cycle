#!/bin/bash
#SBATCH --job-name=chr_full_pipe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/full_chr_pipeline_%j.out
#SBATCH --error=logs/full_chr_pipeline_%j.err
#SBATCH --account=minjilab99


# Do not use `set -u` before sourcing bashrc/conda: /etc/bashrc uses
# BASHRCSOURCED before it is set, which aborts under nounset.
WORKDIR="/nfs/turbo/umms-minjilab/lpullela/cell-cycle/train"
PIPE_DIR="${WORKDIR}/full_chr_pipeline"

CHROM="${CHROM:-2}"
#CHECKPOINT="${CHECKPOINT:-${WORKDIR}/checkpoints/oe_log_4phase_epoch40_4-16_asym-residual-loss.pth}"
CHECKPOINT="${CHECKPOINT:-${WORKDIR}/checkpoints/oe_log_4phase_epoch40_4-30.pth}"

# These default to your repo structure; override as needed.
DATA_DIR="${DATA_DIR:-/nfs/turbo/umms-minjilab/lpullela/cell-cycle/raw_data/zhang_4dn}"
CHIP_DIR="${CHIP_DIR:-$DATA_DIR}"

OUT_BASE="${OUT_BASE:-${WORKDIR}/full_chr_outputs/chr${CHROM}}"
ARRAYS_DIR="${ARRAYS_DIR:-${OUT_BASE}/arrays}"
NEAR_DIR="${NEAR_DIR:-${OUT_BASE}/near_pred}"
FINAL_DIR="${FINAL_DIR:-${OUT_BASE}/final}"
JUICER_DIR="${JUICER_DIR:-${OUT_BASE}/juicer_txt}"

HIC_UNIT="${HIC_UNIT:-oe}"   # oe or observed
NORM="${NORM:-KR}"
NEAR_BAND_BP="${NEAR_BAND_BP:-640000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DIAG_STEP_BP="${DIAG_STEP_BP:-640000}"

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
conda activate cell-cycle

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
echo "BATCH_SIZE: $BATCH_SIZE"
echo "DIAG_STEP_BP: $DIAG_STEP_BP"
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
  echo "[2/4] Near-diagonal GPU inference -> $NEAR_DIR"
  python "${PIPE_DIR}/infer_chr_near_diag_gpu.py" \
    --chrom "$CHROM" \
    --checkpoint "$CHECKPOINT" \
    --arrays_dir "$ARRAYS_DIR" \
    --output_dir "$NEAR_DIR" \
    --near_band_bp "$NEAR_BAND_BP" \
    --batch_size "$BATCH_SIZE" \
    --diag_step_bp "$DIAG_STEP_BP"
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
    --near_band_bp "$NEAR_BAND_BP"
else
  echo "[3/4] SKIP fill (reuse final matrices in $FINAL_DIR)"
fi

if [[ "$SKIP_EXPORT" != "1" ]]; then
  echo "[4/4] Export Juicer sparse text -> $JUICER_DIR"
  for PHASE in earlyG1 midG1 lateG1 anatelo; do
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

