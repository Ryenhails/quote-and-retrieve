#!/bin/bash
#SBATCH --job-name=reb_dump
#SBATCH --partition=<PARTITION>
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --output=${PROJECT_ROOT}/../logs/revision_dump_%j.out
#SBATCH --error=${PROJECT_ROOT}/../logs/revision_dump_%j.err

# Camera-ready analyses: dump q-only / q+answer retrieval boxes and
# per-element failure buckets for base vs GRPO.  2B embedder + cached
# block embeddings, so a V100-32G is enough (same as slurm/controls_embed.sh).
module load mamba; module load <cluster-toolchain>; module load gcc/13.3.0; module load cuda/12.6.2
export CC=$(which gcc); export CXX=$(which g++)
export HF_HOME=${PROJECT_ROOT}/../cache HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
source activate ${ENV_MAIN}
cd ${PROJECT_ROOT}

python src/revision_gpu_dump.py
EXIT_CODE=$?
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
exit $EXIT_CODE
