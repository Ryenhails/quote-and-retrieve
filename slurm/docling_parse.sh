#!/bin/bash
#SBATCH --job-name=docling
#SBATCH --partition=<PARTITION>
#SBATCH --gpus=h200:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --array=0-3
#SBATCH --output=${PROJECT_ROOT}/../logs/docling_%A_%a.out
#SBATCH --error=${PROJECT_ROOT}/../logs/docling_%A_%a.err

# E9 cross-parser: Docling layout parse of the 440 pub-719 PDFs (GPU-accelerated layout
# model + easyocr for the scan-like minority). 4 shards in parallel on V100s.
module load mamba; module load <cluster-toolchain>; module load gcc/13.3.0; module load cuda/12.6.2
export CC=$(which gcc); export CXX=$(which g++)
export HF_HOME=${PROJECT_ROOT}/../cache
export EASYOCR_MODULE_PATH=${PROJECT_ROOT}/../cache/easyocr
mkdir -p ${PROJECT_ROOT}/../tmp
export TMPDIR=${PROJECT_ROOT}/../tmp

ENV=${CONDA_ENVS}/docling
source activate "$ENV"

cd ${PROJECT_ROOT}
python src/cross_parser_docling.py --shard ${SLURM_ARRAY_TASK_ID} --num-shards 4
EXIT_CODE=$?
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
exit $EXIT_CODE
