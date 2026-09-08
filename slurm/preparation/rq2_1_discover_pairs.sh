#!/bin/bash
#SBATCH --job-name=rq2_1_discover_pairs
#SBATCH --output=/home/yazanme/BachelorThesis/BachelorThesis/slurm/preparation/logs/rq2_1_discover_pairs_%j.log
#SBATCH --error=/home/yazanme/BachelorThesis/BachelorThesis/slurm/preparation/logs/rq2_1_discover_pairs_%j.log
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:L40S:1

set -eo pipefail

export TZ=Europe/Berlin

echo "========================================="
echo "1. SLURM ALLOKATIONS-INFO"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Ausfuehrender Node: $SLURMD_NODENAME"
echo "Zugewiesene CPUs: $SLURM_CPUS_PER_TASK"
echo "Zugewiesener Speicher: ${SLURM_MEM_PER_NODE:-$SLURM_MEM_PER_CPU} MB"
echo "Zugewiesene GPUs: ${SLURM_JOB_GPUS:-nicht gesetzt}"
echo "Zugewiesene GPUs ueber CUDA: ${CUDA_VISIBLE_DEVICES:-nicht gesetzt}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
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
# 3. AUSFÜHRUNG RQ2.1 PAARSUCHE
# =========================================
REPO_ROOT="/home/yazanme/BachelorThesis/BachelorThesis"
cd "${REPO_ROOT}/src"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTORCH_NVML_BASED_CUDA_CHECK=0

export MPLBACKEND=Agg

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

RQ_CONFIG="${REPO_ROOT}/src/yaml/preparation/rq2_1_discover_pairs.yaml"
DATA_CONFIG="${REPO_ROOT}/src/implementation/data_preparation/config.yaml"
PROCESSED="${REPO_ROOT}/data/processed/resisc45"
BASELINE_MODEL="resnet18"
BASELINE_DIR="${REPO_ROOT}/experiments/RQ1/${BASELINE_MODEL}/seed_42/rate_0.0"

for cfg in "${RQ_CONFIG}" "${DATA_CONFIG}"; do
    if [ ! -f "${cfg}" ]; then
        echo "ABBRUCH: Config nicht gefunden: ${cfg}" >&2
        exit 1
    fi
done
if [ ! -f "${PROCESSED}/train.h5" ]; then
    echo "ABBRUCH: ${PROCESSED}/train.h5 fehlt -- erst slurm/preparation/rq1_data_preparation.sbatch laufen lassen." >&2
    exit 1
fi
if ! ls "${BASELINE_DIR}"/best_model_weights_rate_0.00_epoch_*.pt >/dev/null 2>&1; then
    echo "ABBRUCH: kein Baseline-Checkpoint in ${BASELINE_DIR}" >&2
    echo "         Erst slurm/rq1/rq1_train_${BASELINE_MODEL}.sbatch (Seed 42, rate 0.0) laufen lassen." >&2
    exit 1
fi

echo "Baseline-Checkpoint: $(ls "${BASELINE_DIR}"/best_model_weights_rate_0.00_epoch_*.pt | tail -1)"
echo "Ausgabe:             ${REPO_ROOT}/src/implementation/rq2/artifacts/discovered_target_pairs.json"
echo ""

echo "Startzeit: $(date '+%Y-%m-%d %H:%M:%S %Z')"
START_EPOCH=$(date +%s)
echo ""

set +e
python -m implementation.rq2.rq2_1.discover_pairs \
    --config "${RQ_CONFIG}" \
    --data-config "${DATA_CONFIG}"
EXIT_CODE=$?
set -e

echo ""
END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
echo "Endzeit: $(date '+%Y-%m-%d %H:%M:%S %Z')"
printf 'Laufzeit: %02d:%02d:%02d\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "Exit-Code: ${EXIT_CODE}"

echo ""
echo "HINWEIS: poisoning.target_pairs in data_preparation/config.yaml wird NICHT"
echo "         automatisch aktualisiert. Abweichungen von Hand uebertragen und danach"
echo "         rq2_data_preparation.sbatch neu laufen lassen."

exit ${EXIT_CODE}
