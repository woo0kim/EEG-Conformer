#!/bin/bash
#SBATCH --job-name=eegconf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --array=0-71%30
#SBATCH --output=logs/%A_%a.out

# Submit from the project root (the dir holding data_proc/, results/, logs/, venv/).
ROOT=${EEGCONF_ROOT:-$SLURM_SUBMIT_DIR}
cd "$ROOT"
PY=${EEGCONF_PY:-./venv/bin/python}
export OPENBLAS_NUM_THREADS=1
i=$SLURM_ARRAY_TASK_ID
SEEDS=(2023 2024 2025)
if [ $i -lt 27 ]; then
    TAG=A; ARGS="--val-frac 0.0";                 j=$i
elif [ $i -lt 54 ]; then
    TAG=B; ARGS="--val-frac 0.2";                 j=$((i-27))
else
    TAG=C; ARGS="--val-frac 0.0 --permute-labels"; j=$((i-54))
fi
SUB=$(( j / 3 + 1 ))
SEED=${SEEDS[$(( j % 3 ))]}
"$PY" code/run_validation.py --subject $SUB --seed $SEED --epochs 2000 $ARGS --tag $TAG
