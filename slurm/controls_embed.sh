#!/bin/bash
#SBATCH --job-name=ctrl_emb
#SBATCH --partition=<PARTITION>
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --output=${PROJECT_ROOT}/../logs/ctrl_emb_%j.out
#SBATCH --error=${PROJECT_ROOT}/../logs/ctrl_emb_%j.err

# E1 retrieval-only baselines + E8 assigned-similarity export (2B embedder, block embeddings
# already cached, so V100-32G suffices and queues faster than H200).
module load mamba; module load <cluster-toolchain>; module load gcc/13.3.0; module load cuda/12.6.2
export CC=$(which gcc); export CXX=$(which g++)
export HF_HOME=${PROJECT_ROOT}/../cache HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
source activate ${ENV_ANALYSIS}
cd ${PROJECT_ROOT}

python src/retrieval_baselines.py
EXIT_CODE=$?
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
exit $EXIT_CODE
