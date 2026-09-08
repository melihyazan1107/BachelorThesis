#!/bin/bash
#SBATCH --job-name=rq2_3_review_sheet
#SBATCH --output=/home/yazanme/BachelorThesis/BachelorThesis/slurm/rq2/rq2_3/logs/rq2_3_review_sheet_%j.log
#SBATCH --error=/home/yazanme/BachelorThesis/BachelorThesis/slurm/rq2/rq2_3/logs/rq2_3_review_sheet_%j.log
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
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
# 3. AUSFÜHRUNG RQ2.3 REVIEW-SHEET
# =========================================
REPO_ROOT="/home/yazanme/BachelorThesis/BachelorThesis"
cd "${REPO_ROOT}/src"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTORCH_NVML_BASED_CUDA_CHECK=0

export MPLBACKEND=Agg

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

RQ_CONFIG="${REPO_ROOT}/src/yaml/rq2/rq2_3/rq2_3_review.yaml"
DATA_CONFIG="${REPO_ROOT}/src/implementation/data_preparation/config.yaml"
PROCESSED="${REPO_ROOT}/data/processed/resisc45"

for cfg in "${RQ_CONFIG}" "${DATA_CONFIG}"; do
    if [ ! -f "${cfg}" ]; then
        echo "ABBRUCH: Config nicht gefunden: ${cfg}" >&2
        exit 1
    fi
done
if [ ! -f "${PROCESSED}/train.h5" ]; then
    echo "ABBRUCH: ${PROCESSED}/train.h5 fehlt." >&2
    exit 1
fi
if [ ! -f "${PROCESSED}/poisoned/RQ2/seed_42/labels_0.20.pt" ]; then
    echo "ABBRUCH: ${PROCESSED}/poisoned/RQ2/seed_42/labels_0.20.pt fehlt." >&2
    echo "         Erst slurm/preparation/rq2_data_preparation.sbatch laufen lassen." >&2
    exit 1
fi

echo "Startzeit: $(date '+%Y-%m-%d %H:%M:%S %Z')"
START_EPOCH=$(date +%s)
echo ""

set +e
python -m implementation.rq2.rq2_3.build_review_sheet \
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

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "Review-Sheet:"
    ls -la "${REPO_ROOT}/human_review/"
    echo "Bilder: $(find "${REPO_ROOT}/human_review/images" -name '*.png' | wc -l)"
    echo ""
    echo "NAECHSTER SCHRITT (manuell): review.csv + images/ begutachten und eine"
    echo "annotations.csv mit den Spalten id,flagged in human_review/ ablegen."
    echo "Danach rq2_3_evaluate_detection.sbatch starten."
fi

exit ${EXIT_CODE}
