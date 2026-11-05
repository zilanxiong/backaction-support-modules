#!/bin/bash
# GPU diagnostic: the lab-frame tests (CuPy test included) plus a CPU-vs-GPU comparison.
#   sbatch --qos=debug --time=00:30:00 scripts/gpu_check.slurm
#SBATCH --job-name=qfi-gpucheck
#SBATCH --account=ligo-controls
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gpucheck-%j.out
set -uo pipefail
mkdir -p logs
cd "$HOME/backaction-support-modules"
module load cuda/12.9
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate backaction-support
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import cupy, numpy, qutip, scipy; print('cupy', cupy.__version__, 'numpy', numpy.__version__, 'qutip', qutip.__version__, 'scipy', scipy.__version__)"
echo "=== tests ==="; python -m pytest -q tests/test_labframe.py 2>&1 | tail -15
echo "=== gpu_check ==="; python -u scripts/gpu_check.py
