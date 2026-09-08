#!/bin/bash
#SBATCH --job-name=rq3_mixup_resnet18
#SBATCH --output=/home/yazanme/BachelorThesis/BachelorThesis/slurm/rq3/logs/rq3_mixup_resnet18_%A_%a.log
#SBATCH --error=/home/yazanme/BachelorThesis/BachelorThesis/slurm/rq3/logs/rq3_mixup_resnet18_%A_%a.log
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=GPU
#SBATCH --gres=gpu
#SBATCH --array=0-2

set -eo pipefail

export TZ=Europe/Berlin

MODEL="resnet18"
DEFENSE="mixup"
SEEDS=(42 2001 2026)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"

echo "========================================="
echo "1. SLURM ALLOKATIONS-INFO"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID (Array ${SLURM_ARRAY_JOB_ID}, Task ${SLURM_ARRAY_TASK_ID})"
echo "Job Name: ${SLURM_JOB_NAME}"
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
# 3. AUSFUEHRUNG RQ3 TRAINING (${DEFENSE} / ${MODEL} / Seed ${SEED})
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

TRAIN_CONFIG="${REPO_ROOT}/src/yaml/rq3/rq3_train_${MODEL}.yaml"
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

MISSING=0
for rq in RQ1 RQ2; do
    for r in 0.01 0.02 0.05 0.10 0.20; do
        lbl="${PROCESSED}/poisoned/${rq}/seed_${SEED}/labels_${r}.pt"
        if [ ! -f "${lbl}" ]; then
            echo "FEHLT: ${lbl}" >&2
            MISSING=$((MISSING + 1))
        fi
    done
done
if [ ${MISSING} -gt 0 ]; then
    echo "ABBRUCH: ${MISSING} von 10 Label-Dateien fehlen (rate 0.0 braucht keine)." >&2
    exit 1
fi

echo "Modell:           ${MODEL}"
echo "Verteidigung:     ${DEFENSE}"
echo "Seed:             ${SEED}"
echo "Trainings-Config: ${TRAIN_CONFIG}"
echo "Daten-Config:     ${DATA_CONFIG}"
echo "Ausgabe:          ${REPO_ROOT}/experiments/RQ3/${DEFENSE}/${MODEL}/{clean,RQ1,RQ2}"
echo "Preflight ok: 3 HDF5-Splits + 10 Label-Tensoren (RQ1+RQ2) fuer Seed ${SEED}."
echo "Erwartet: 1 clean + 2 Angriffe x 5 Raten = 11 Laeufe (~4.6 h)."
echo ""

echo "Startzeit: $(date '+%Y-%m-%d %H:%M:%S %Z')"
START_EPOCH=$(date +%s)
echo ""

set +e
python -m implementation.shared.training.run \
    --config "${TRAIN_CONFIG}" \
    --data-config "${DATA_CONFIG}" \
    --defense "${DEFENSE}" \
    --seed "${SEED}"
EXIT_CODE=$?
set -e

echo ""
END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
echo "Endzeit: $(date '+%Y-%m-%d %H:%M:%S %Z')"
printf 'Laufzeit: %02d:%02d:%02d\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "Exit-Code: ${EXIT_CODE}"

ARM_DIR="${REPO_ROOT}/experiments/RQ3/${DEFENSE}/${MODEL}"
if [ -d "${ARM_DIR}" ]; then
    echo ""
    echo "Abgeschlossene Laeufe fuer Seed ${SEED} (erwartet: 11):"
    find "${ARM_DIR}" -type d -name 'rate_*' -path "*seed_${SEED}*" | sort
    echo "Anzahl: $(find "${ARM_DIR}" -type d -name 'rate_*' -path "*seed_${SEED}*" | wc -l)"
    for attack in clean RQ1 RQ2; do
        if [ -f "${ARM_DIR}/${attack}/summary_metrics.csv" ]; then
            echo ""
            echo "${attack}/summary_metrics.csv:"
            cat "${ARM_DIR}/${attack}/summary_metrics.csv"
        fi
    done
fi

exit ${EXIT_CODE}
