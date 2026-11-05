#!/bin/bash
# QFI sweep through the package channel (lab frame) on one GPU.
#
#   sbatch scripts/sweep_gpu.slurm --tag channel_loss_eta0p95 --eta-ch 0.95
#
# Everything after the script name is passed to scripts/sweep_labframe.py.
# The H200 pool (18 nodes) is the default GPU; --gres=gpu:h100:1 on the sbatch command line
# overrides it. GPU resource names on the cluster: sinfo -p gpu -o "%N %G"
#
#SBATCH --job-name=qfi-sweep
#SBATCH --account=ligo-controls
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sweep-%j.out

set -euo pipefail
mkdir -p logs
cd "$HOME/backaction-support-modules"
module load cuda/12.9
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate backaction-support

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u scripts/sweep_labframe.py --device gpu "$@"
