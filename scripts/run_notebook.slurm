#!/bin/bash
# Execute a Jupyter notebook top to bottom on one GPU, no browser needed.
#
#   sbatch scripts/run_notebook.slurm notebooks/backaction_sweeps.ipynb
#
# The executed copy, with every cell's output and figure, is written next to the
# original as <name>.executed.ipynb; the original is left untouched.  Figures the
# notebook saves with fig.savefig land wherever the notebook writes them (paths are
# relative to the notebook's folder).  A failing cell stops the run and the error
# appears both in the log and in the executed copy.
#
#SBATCH --job-name=qfi-notebook
#SBATCH --account=ligo-controls
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/notebook-%j.out

set -euo pipefail
mkdir -p logs
cd "$HOME/backaction-support-modules"
module load cuda/12.9
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate backaction-support

nb="${1:?usage: sbatch scripts/run_notebook.slurm path/to/notebook.ipynb}"
out="${nb%.ipynb}.executed.ipynb"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "executing $nb -> $out"
jupyter nbconvert --to notebook --execute "$nb" --output "$(basename "$out")" \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3
echo "done: $out"
