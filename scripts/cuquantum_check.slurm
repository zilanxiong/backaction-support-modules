#!/bin/bash
# Trial of qutip-cuquantum as the GPU backend for the unchanged quantum_sensing channel.
#   sbatch --qos=debug --time=00:30:00 scripts/cuquantum_check.slurm
#SBATCH --job-name=qfi-cuquantum
#SBATCH --account=ligo-controls
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cuquantum-%j.out
set -uo pipefail
mkdir -p logs
cd "$HOME/backaction-support-modules"
module load cuda/12.9
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate backaction-support
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import qutip, cupy; print('qutip', qutip.__version__, 'cupy', cupy.__version__)"
python -c "import qutip_cuquantum, cuquantum; print('qutip_cuquantum ok; cuquantum', getattr(cuquantum, '__version__', '?'))" || echo "qutip-cuquantum not installed: pip install 'qutip-cuquantum[cuda12]'"
python -u scripts/cuquantum_check.py
