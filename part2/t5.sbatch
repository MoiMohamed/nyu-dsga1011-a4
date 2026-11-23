#!/bin/bash
#SBATCH --job-name=t5-finetune
#SBATCH --output=t5-finetune-%j.out
#SBATCH --error=t5-finetune-%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --export=ALL

set -euo pipefail

cd /scratch/$USER/nyu-dsga1011-a4/part2

# Singularity cache
export SINGULARITY_CACHEDIR=/scratch/$USER/tmp/cache
mkdir -p "$SINGULARITY_CACHEDIR"

# Temporary files on local disk
export TMPDIR=/tmp/$USER/t5-$SLURM_JOB_ID
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
      --finetune \
      --experiment_name experiment \
      --max_n_epochs 15 \
      --patience_epochs 4 \
      --learning_rate 3e-4 \
      --scheduler_type none \
      --batch_size 16 \
      --test_batch_size 16
  '
