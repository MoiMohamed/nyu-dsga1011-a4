#!/bin/bash
#SBATCH --job-name=bert-baseline
#SBATCH --output=bert-baseline-%j.out
#SBATCH --error=bert-baseline-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --export=ALL

set -euo pipefail
cd /scratch/$USER/nyu-dsga1011-a4/part1
export SINGULARITY_CACHEDIR=/scratch/$USER/tmp/cache
export SINGULARITY_TMPDIR=/scratch/$USER/tmp/singularity
export TMPDIR=/scratch/$USER/tmp/singularity
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

singularity exec --nv \
  --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw \
  /scratch/$USER/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
  /bin/bash -c '
    set -e
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate dsga1011-hw4-part1
    cd /scratch/$USER/nyu-dsga1011-a4/part1
    python -u main.py --train --eval
  '
