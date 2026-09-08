#!/bin/bash
#SBATCH --job-name=rq1_dataprep
#SBATCH --output=/home/yazanme/BachelorThesis/BachelorThesis/slurm/preparation/logs/rq1_dataprep_%j.log
#SBATCH --error=/home/yazanme/BachelorThesis/BachelorThesis/slurm/preparation/logs/rq1_dataprep_%j.log
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=CPU

set -eo pipefail

export TZ=Europe/Berlin

echo "========================================="
echo "1. SLURM ALLOKATIONS-INFO"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Ausfuehrender Node: $SLURMD_NODENAME"
echo "Zugewiesene CPUs: $SLURM_CPUS_PER_TASK"
echo "Zugewiesener Speicher: ${SLURM_MEM_PER_NODE:-$SLURM_MEM_PER_CPU} MB"
echo "Zugewiesene GPUs: ${SLURM_JOB_GPUS:-keine (CPU-only Job)}"
echo "Zugewiesene GPUs ueber CUDA: ${CUDA_VISIBLE_DEVICES:-nicht gesetzt}"
echo ""
echo "Zeitzonen-Check -> Node-Uhr (UTC): $(date -u '+%Y-%m-%d %H:%M:%S %Z')"
echo "Zeitzonen-Check -> korrigiert    : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

# =========================================
# 2. UMGEBUNG LADEN
# =========================================
source /home/yazanme/miniforge3/etc/profile.d/conda.sh
conda activate /home/yazanme/miniforge3/envs/venv/

echo "Python: $(which python)"
echo "Version: $(python --version)"
echo ""

# =========================================
# 3. AUSFÜHRUNG RQ1 DATA PREPARATION
# =========================================
REPO_ROOT="/home/yazanme/BachelorThesis/BachelorThesis"
cd "${REPO_ROOT}/src"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTORCH_NVML_BASED_CUDA_CHECK=0

export MPLBACKEND=Agg

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

RQ1_CONFIG="${REPO_ROOT}/src/yaml/preparation/rq1_data_preparation.yaml"
DATA_CONFIG="${REPO_ROOT}/src/implementation/data_preparation/config.yaml"

for cfg in "${RQ1_CONFIG}" "${DATA_CONFIG}"; do
    if [ ! -f "${cfg}" ]; then
        echo "ABBRUCH: Config nicht gefunden: ${cfg}" >&2
        exit 1
    fi
done
echo "Trainings-Config: ${RQ1_CONFIG}"
echo "Daten-Config:     ${DATA_CONFIG}"
echo ""

echo "Startzeit: $(date '+%Y-%m-%d %H:%M:%S %Z')"
START_EPOCH=$(date +%s)
echo ""

set +e
python -m implementation.run_pipeline \
    --config "${RQ1_CONFIG}" \
    --data-config "${DATA_CONFIG}"
EXIT_CODE=$?
set -e

echo ""
END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
echo "Endzeit: $(date '+%Y-%m-%d %H:%M:%S %Z')"
printf 'Laufzeit: %02d:%02d:%02d\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "Exit-Code: ${EXIT_CODE}"


if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "Erzeugte HDF5-Splits:"
    ls -lh "${REPO_ROOT}/data/processed/resisc45/"*.h5
    echo ""
    echo "Erzeugte RQ1-Label-Tensoren (erwartet: 3 Seeds x 5 Raten = 15):"
    find "${REPO_ROOT}/data/processed/resisc45/poisoned/RQ1" -name 'labels_*.pt' | sort
    echo "Anzahl: $(find "${REPO_ROOT}/data/processed/resisc45/poisoned/RQ1" -name 'labels_*.pt' | wc -l)"
fi

exit ${EXIT_CODE}
