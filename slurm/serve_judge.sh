#!/bin/bash
#SBATCH --job-name=judge
#SBATCH --gpus=h200:1
#SBATCH --partition=<PARTITION>
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=${PROJECT_ROOT}/../logs/judge_%j.out
#SBATCH --error=${PROJECT_ROOT}/../logs/judge_%j.err

# External answer-correctness judge for GRPO (reward_citeground http backend).
# Qwen3-4B-Instruct-2507 text judge via vLLM OpenAI server. Start this FIRST, read
# the printed node, then submit GRPO with:
#   JUDGE_BACKEND=http JUDGE_BASE_URL=http://<node>:8000/v1 sbatch slurm/grpo_citeground.sh
module load mamba; module load <cluster-toolchain>; module load gcc/13.3.0; module load cuda/12.6.2
source activate ${ENV_TRAIN}
export CC=$(which gcc); export CXX=$(which g++)
export TRITON_CACHE_DIR=${PROJECT_ROOT}/../cache/triton_persist
export HF_HUB_OFFLINE=1

JUDGE=$(ls -d ${PROJECT_ROOT}/../cache/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ 2>/dev/null | head -1)
[ -z "$JUDGE" ] && { echo "judge model not found in cache"; exit 1; }
echo "=================================================================="
echo " JUDGE SERVER on node: $(hostname)"
echo " -> submit GRPO with: JUDGE_BACKEND=http JUDGE_BASE_URL=http://$(hostname):8000/v1 \\"
echo "      JUDGE_MODEL=judge sbatch slurm/grpo_citeground.sh"
echo "=================================================================="
python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE" --served-model-name judge \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.3 --max-model-len 4096 \
    --trust-remote-code
