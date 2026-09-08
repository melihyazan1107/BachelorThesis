#!/bin/bash
#SBATCH --job-name=rq2_3_evaluate
#SBATCH --output=/home/yazanme/BachelorThesis/BachelorThesis/slurm/rq2/rq2_3/logs/rq2_3_evaluate_%j.log
#SBATCH --error=/home/yazanme/BachelorThesis/BachelorThesis/slurm/rq2/rq2_3/logs/rq2_3_evaluate_%j.log
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
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
# 3. AUSFÜHRUNG RQ2.3 DETEKTIONSAUSWERTUNG
# =========================================
REPO_ROOT="/home/yazanme/BachelorThesis/BachelorThesis"
cd "${REPO_ROOT}/src"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTORCH_NVML_BASED_CUDA_CHECK=0

export MPLBACKEND=Agg

REVIEW="${REPO_ROOT}/human_review"

for f in key.csv annotations.csv; do
    if [ ! -f "${REVIEW}/${f}" ]; then
        echo "ABBRUCH: ${REVIEW}/${f} fehlt." >&2
        if [ "${f}" = "annotations.csv" ]; then
            echo "         annotations.csv (Spalten: id,flagged) wird VON HAND erstellt," >&2
            echo "         nachdem das Review-Sheet begutachtet wurde." >&2
        else
            echo "         Erst rq2_3_build_review_sheet.sbatch laufen lassen." >&2
        fi
        exit 1
    fi
done

echo "Startzeit: $(date '+%Y-%m-%d %H:%M:%S %Z')"
START_EPOCH=$(date +%s)
echo ""

set +e
python -m implementation.rq2.rq2_3.evaluate_detection
EXIT_CODE=$?
set -e

echo ""
END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
echo "Endzeit: $(date '+%Y-%m-%d %H:%M:%S %Z')"
printf 'Laufzeit: %02d:%02d:%02d\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "Exit-Code: ${EXIT_CODE}"

exit ${EXIT_CODE}
