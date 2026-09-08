#!/bin/bash
#SBATCH --job-name=rq2_4_dataprep
#SBATCH --output=/home/yazanme/BachelorThesis/BachelorThesis/slurm/preparation/logs/rq2_4_dataprep_%j.log
#SBATCH --error=/home/yazanme/BachelorThesis/BachelorThesis/slurm/preparation/logs/rq2_4_dataprep_%j.log
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
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
# 3. AUSFÜHRUNG RQ2.4 LABEL-GENERIERUNG
# =========================================
REPO_ROOT="/home/yazanme/BachelorThesis/BachelorThesis"
cd "${REPO_ROOT}/src"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTORCH_NVML_BASED_CUDA_CHECK=0

export MPLBACKEND=Agg

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

RQ_CONFIG="${REPO_ROOT}/src/yaml/preparation/rq2_4_data_preparation.yaml"
DATA_CONFIG="${REPO_ROOT}/src/implementation/data_preparation/config.yaml"
PROCESSED="${REPO_ROOT}/data/processed/resisc45"
BASELINE_MODEL="resnet18"

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
MISSING=0
for seed in 42 2001 2026; do
    tel="${REPO_ROOT}/experiments/RQ1/${BASELINE_MODEL}/seed_${seed}/rate_0.0/telemetry/sample_telemetry.csv"
    if [ ! -f "${tel}" ]; then
        echo "FEHLT: ${tel}" >&2
        MISSING=$((MISSING + 1))
    fi
done
if [ ${MISSING} -gt 0 ]; then
    echo "ABBRUCH: ${MISSING} von 3 Baseline-Telemetrien fehlen." >&2
    echo "         Erst slurm/rq1/rq1_train_${BASELINE_MODEL}.sbatch komplett durchlaufen lassen." >&2
    exit 1
fi

echo "Forschungsfrage:  RQ2.4 (Uncertainty)"
echo "Baseline:         experiments/RQ1/${BASELINE_MODEL}/seed_*/rate_0.0/telemetry/"
echo "Trainings-Config: ${RQ_CONFIG}"
echo "Preflight ok: train.h5 + 3 Baseline-Telemetrien vorhanden."
echo ""

echo "Startzeit: $(date '+%Y-%m-%d %H:%M:%S %Z')"
START_EPOCH=$(date +%s)
echo ""

set +e
python -m implementation.run_pipeline \
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
    echo "Erzeugte Label-Tensoren (erwartet: 3, Sentinel-Name ohne Rate):"
    find "${PROCESSED}/poisoned/RQ2.4" -name 'labels.pt' | sort
    echo ""
    echo "Empirische Raten:"
    find "${PROCESSED}/poisoned/RQ2.4" -name 'poisoning_summary.json' -exec sh -c 'echo "  $1:"; cat "$1"' _ {} \;
fi

exit ${EXIT_CODE}
