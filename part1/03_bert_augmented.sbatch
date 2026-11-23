#!/bin/bash
#SBATCH --job-name=bert-augmented
#SBATCH --output=bert-augmented-%j.out
#SBATCH --error=bert-augmented-%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --export=ALL


set -euo pipefail

cd /scratch/$USER/nyu-dsga1011-a4/part1

# Singularity cache (NFS)
export SINGULARITY_CACHEDIR=/scratch/$USER/tmp/cache
mkdir -p "$SINGULARITY_CACHEDIR"

# Temporary files (local XFS)
export TMPDIR=/tmp/$USER/bert-$SLURM_JOB_ID
export SINGULARITY_TMPDIR="$TMPDIR"
mkdir -p "$TMPDIR"

singularity exec --nv \
  --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw \
  /scratch/$USER/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
  /bin/bash -c '
    set -e
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate dsga1011-hw4-part1
    cd /scratch/$USER/nyu-dsga1011-a4/part1
    python -u main.py --train_augmented --eval --eval_transformed
  '
