#!/bin/bash
#SBATCH --job-name=docling_match
#SBATCH --partition=<PARTITION>
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --output=${PROJECT_ROOT}/../logs/docling_match_%j.out
#SBATCH --error=${PROJECT_ROOT}/../logs/docling_match_%j.err

# E9 step 2: Docling parser ceiling + NL rematch with Docling blocks (2B embedder on V100).
module load mamba; module load <cluster-toolchain>; module load gcc/13.3.0; module load cuda/12.6.2
export CC=$(which gcc); export CXX=$(which g++)
export HF_HOME=${PROJECT_ROOT}/../cache HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
source activate ${ENV_ANALYSIS}
cd ${PROJECT_ROOT}

python src/cross_parser_match.py
EXIT_CODE=$?
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
exit $EXIT_CODE
