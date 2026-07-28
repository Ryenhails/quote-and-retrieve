#!/bin/bash
#SBATCH --job-name=matrix_grpo
#SBATCH --partition=<PARTITION>
#SBATCH --gpus=h200:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=140GB
#SBATCH --output=${PROJECT_ROOT}/../logs/matrix_grpo_%j.out
#SBATCH --error=${PROJECT_ROOT}/../logs/matrix_grpo_%j.err
#
# Part-2 eval: GRPO last-ckpt vs base, NL-grounding interface, pub 719, IDENTICAL pipeline to
# the Part-1 matrix (same prompt = GRPO training prompt, same full-page adaptive resolution as
# qwen3vl, same match_wholedoc, same block_emb cache). Only difference vs base is the LoRA adapter,
# served through vLLM --enable-lora. bboxR scored locally; judge inputs cached for the later Gemini
# pass. Base-NL is the matrix run outputs/matrix/qwen3vl_nl_wd (do NOT recompute).
# Usage: sbatch --export=ALL,CKPT=outputs/grpo_longdoc/global_step_144 slurm/matrix_grpo.sh
module load mamba; module load <cluster-toolchain>; module load gcc/13.3.0; module load cuda/12.6.2
export CC=$(which gcc); export CXX=$(which g++)
export HF_HOME=${PROJECT_ROOT}/../cache HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT=${PROJECT_ROOT}/../cache/vllm_cache
ROOT=${PROJECT_ROOT}
CACHE=${PROJECT_ROOT}/../cache
cd $ROOT
PORT=$((22000 + SLURM_JOB_ID % 20000))
MODEL=$(ls -d $CACHE/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/ | head -1)
CKPT=${CKPT:-outputs/grpo_longdoc/global_step_144}
ADAPTER=$ROOT/$CKPT/actor/lora_adapter
BUDGET=187000000   # same as qwen3vl matrix cell (1M/page for all pub docs)
TAG=$(basename $CKPT)
echo "MODEL=$MODEL ADAPTER=$ADAPTER TAG=$TAG"
[ -d "$ADAPTER" ] || { echo "adapter not found: $ADAPTER"; exit 1; }

# ---- serve base Qwen3-VL + GRPO LoRA (vLLM), served-model-name=grpo ----
SVLOG=$ROOT/logs/matrix_grpo_serve_${SLURM_JOB_ID}.log
( source activate ${ENV_INFER}
  VLLM_ATTENTION_BACKEND=TORCH_SDPA \
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host 127.0.0.1 --port $PORT \
    --enable-lora --max-lora-rank 64 --lora-modules "grpo=$ADAPTER" \
    --gpu-memory-utilization 0.90 --max-model-len 262144 \
    --limit-mm-per-prompt '{"image":200}' --trust-remote-code ) > "$SVLOG" 2>&1 &
SV=$!
for i in $(seq 1 180); do
  kill -0 $SV 2>/dev/null || { echo "[serve] DIED"; tail -40 "$SVLOG"; exit 1; }
  curl -sf http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q grpo && { echo "[serve] READY ${i}0s"; break; }
  sleep 10
done

# ---- NL inference with the GRPO adapter (served_model=grpo) ----
source activate ${ENV_MAIN}
OUT=$ROOT/outputs/matrix/grpo_${TAG}_nl/predictions.jsonl
python src/infer_matrix.py --url http://127.0.0.1:$PORT --served_model grpo --interface nl \
  --bench data/citevqa/citevqa_singledoc_pub.jsonl --budget $BUDGET --out $OUT
kill $SV 2>/dev/null; sleep 3

# ---- quote -> whole-doc block bbox ----
source activate ${ENV_ANALYSIS}
WD=$ROOT/outputs/matrix/grpo_${TAG}_nl_wd/predictions.jsonl
python src/match_wholedoc.py --pred $OUT --bench data/citevqa/citevqa_singledoc_pub.jsonl \
  --cache data/citevqa/mineru_cache --out $WD

# ---- local bboxR: base-NL (matrix) vs GRPO-NL, paired ----
echo "===== Part-2 bboxR (pub 719): base-NL vs GRPO-NL ($TAG) ====="
python - <<PY
import json, sys, statistics
sys.path.insert(0,'src'); from score_citevqa import score_item, iou_hit
def load(f):
    d={}
    for l in open(f):
        r=json.loads(l)
        if r.get('error') or not r.get('gt_necessary'): continue
        rec=score_item(r)[0]
        if rec is not None: d[r['index']]=rec
    return d
base=load('$ROOT/outputs/matrix/qwen3vl_nl_wd/predictions.jsonl')
grpo=load('$WD')
keys=sorted(set(base)&set(grpo))
b=statistics.mean(base[k] for k in keys); g=statistics.mean(grpo[k] for k in keys)
print(f"  paired n={len(keys)} | base-NL bboxR={100*b:.1f}%  GRPO-NL bboxR={100*g:.1f}%  delta={100*(g-b):+.1f}")
PY
echo "Exit: $?  End: $(date)"
