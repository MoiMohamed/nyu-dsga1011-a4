#!/bin/bash
#SBATCH --job-name=t5-scratch
#SBATCH --output=t5-scratch-%j.out
#SBATCH --error=t5-scratch-%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --export=ALL

set -euo pipefail

cd /scratch/$USER/nyu-dsga1011-a4/part2

# Experiment name (reuse on subsequent runs)
export EXPERIMENT_NAME="experiment_extra_credit"

# Check for a resumable checkpoint
CHECKPOINT="checkpoints/scr_experiments/${EXPERIMENT_NAME}/last/training_state.pt"

export RESUME_FLAG=""

if [ -f "$CHECKPOINT" ]; then
    echo "Found checkpoint: $CHECKPOINT"
    echo "Resuming training..."
    export RESUME_FLAG="--resume"
else
    echo "No checkpoint found. Starting fresh."
fi

# Singularity cache
export SINGULARITY_CACHEDIR=/scratch/$USER/tmp/cache
mkdir -p "$SINGULARITY_CACHEDIR"

# Temporary files
export TMPDIR=/tmp/$USER/t5-scratch-$SLURM_JOB_ID
export SINGULARITY_TMPDIR="$TMPDIR"
mkdir -p "$TMPDIR"

singularity exec --nv \
  --overlay /scratch/$USER/overlay-hw4-part2.ext3:rw \
  /scratch/$USER/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
  /bin/bash -c '
    set -e
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate dsga1011-hw4-part2

    cd /scratch/$USER/nyu-dsga1011-a4/part2

    python train_t5.py \
      --experiment_name "$EXPERIMENT_NAME" \
      $RESUME_FLAG \
      --optimizer_type Adafactor \
      --max_n_epochs 100 \
      --patience_epochs 5 \
      --scheduler_type none \
      --batch_size 16 \
      --test_batch_size 16 \
      --weight_decay 0
  '
