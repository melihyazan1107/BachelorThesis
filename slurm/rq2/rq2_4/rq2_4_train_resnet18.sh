#!/bin/bash
#SBATCH --job-name=rq2_4_train_resnet18
#SBATCH --output=/home/yazanme/BachelorThesis/BachelorThesis/slurm/rq2/rq2_4/logs/rq2_4_train_resnet18_%j.log
#SBATCH --error=/home/yazanme/BachelorThesis/BachelorThesis/slurm/rq2/rq2_4/logs/rq2_4_train_resnet18_%j.log
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:L40S:1

set -eo pipefail

export TZ=Europe/Berlin

MODEL="resnet18"

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
# 3. AUSFÜHRUNG RQ2.4 TRAINING (${MODEL})
# =========================================
REPO_ROOT="/home/yazanme/BachelorThesis/BachelorThesis"
cd "${REPO_ROOT}/src"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTORCH_NVML_BASED_CUDA_CHECK=0

export MPLBACKEND=Agg

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

COMPILE_CACHE="${SLURM_TMPDIR:-/tmp}/torchcompile_${SLURM_JOB_ID}"
mkdir -p "${COMPILE_CACHE}/inductor" "${COMPILE_CACHE}/triton"
export TORCHINDUCTOR_CACHE_DIR="${COMPILE_CACHE}/inductor"
export TRITON_CACHE_DIR="${COMPILE_CACHE}/triton"
trap 'rm -rf "${COMPILE_CACHE}"' EXIT

TRAIN_CONFIG="${REPO_ROOT}/src/yaml/rq2/rq2_4/rq2_4_train_resnet18.yaml"
DATA_CONFIG="${REPO_ROOT}/src/implementation/data_preparation/config.yaml"
PROCESSED="${REPO_ROOT}/data/processed/resisc45"

for cfg in "${TRAIN_CONFIG}" "${DATA_CONFIG}"; do
    if [ ! -f "${cfg}" ]; then
        echo "ABBRUCH: Config nicht gefunden: ${cfg}" >&2
        exit 1
    fi
done
for split in train val test; do
    if [ ! -f "${PROCESSED}/${split}.h5" ]; then
        echo "ABBRUCH: ${PROCESSED}/${split}.h5 fehlt -- erst slurm/preparation/rq1_data_preparation.sbatch laufen lassen." >&2
        exit 1
    fi
done
if [ ! -f "${PROCESSED}/poisoned/RQ2.4/seed_42/labels.pt" ]; then
    echo "ABBRUCH: ${PROCESSED}/poisoned/RQ2.4/seed_42/labels.pt fehlt." >&2
    echo "         Erst slurm/preparation/rq2_4_data_preparation.sbatch laufen lassen." >&2
    exit 1
fi
if [ ! -f "${PROCESSED}/poisoned/RQ2.4/seed_2001/labels.pt" ]; then
    echo "ABBRUCH: ${PROCESSED}/poisoned/RQ2.4/seed_2001/labels.pt fehlt." >&2
    echo "         Erst slurm/preparation/rq2_4_data_preparation.sbatch laufen lassen." >&2
    exit 1
fi
if [ ! -f "${PROCESSED}/poisoned/RQ2.4/seed_2026/labels.pt" ]; then
    echo "ABBRUCH: ${PROCESSED}/poisoned/RQ2.4/seed_2026/labels.pt fehlt." >&2
    echo "         Erst slurm/preparation/rq2_4_data_preparation.sbatch laufen lassen." >&2
    exit 1
fi

echo "Forschungsfrage:  RQ2.4"
echo "Modell:           ${MODEL}"
echo "Trainings-Config: ${TRAIN_CONFIG}"
echo "Ausgabe:          ${REPO_ROOT}/experiments/RQ2.4/resnet18"
echo ""

echo "Startzeit: $(date '+%Y-%m-%d %H:%M:%S %Z')"
START_EPOCH=$(date +%s)
echo ""

set +e
python -m implementation.shared.training.run \
    --config "${TRAIN_CONFIG}" \
    --data-config "${DATA_CONFIG}"
EXIT_CODE=$?
set -e

echo ""
END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
echo "Endzeit: $(date '+%Y-%m-%d %H:%M:%S %Z')"
printf 'Laufzeit: %02d:%02d:%02d\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "Exit-Code: ${EXIT_CODE}"

EXP_DIR="${REPO_ROOT}/experiments/RQ2.4/resnet18"
if [ -d "${EXP_DIR}" ]; then
    echo ""
    echo "Abgeschlossene Laeufe (erwartet: 3 Seeds x 2 Raten = 6):"
    find "${EXP_DIR}" -mindepth 2 -maxdepth 2 -type d -name 'rate_*' | sort
    echo "Anzahl: $(find "${EXP_DIR}" -mindepth 2 -maxdepth 2 -type d -name 'rate_*' | wc -l)"
fi

exit ${EXIT_CODE}
