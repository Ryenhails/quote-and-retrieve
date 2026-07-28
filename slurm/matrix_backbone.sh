#!/bin/bash
#SBATCH --job-name=matrix
#SBATCH --partition=<PARTITION>
#SBATCH --gpus=h200:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=140GB
#SBATCH --output=${PROJECT_ROOT}/../logs/matrix_%x_%j.out
#SBATCH --error=${PROJECT_ROOT}/../logs/matrix_%x_%j.err
#
# P1 matrix, one backbone: serve via vLLM (OpenAI API) -> coord + NL inference on pub 719
# (full-page, adaptive resolution, see docs/protocol_matrix_eval.md) -> match_wholedoc (NL)
# -> local bbox scoring. NO judge (judge inputs cached later for Gemini).
# Usage: sbatch --export=ALL,BACKBONE=qwen3vl [,LIMIT=20] -J matrix_qwen3vl slurm/matrix_backbone.sh
module load mamba; module load <cluster-toolchain>; module load gcc/13.3.0; module load cuda/12.6.2
export CC=$(which gcc); export CXX=$(which g++)
export HF_HOME=${PROJECT_ROOT}/../cache HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT=${PROJECT_ROOT}/../cache/vllm_cache
ROOT=${PROJECT_ROOT}
CACHE=${PROJECT_ROOT}/../cache
cd $ROOT
PORT=$((21000 + SLURM_JOB_ID % 20000))
LIM=""; [ -n "$LIMIT" ] && LIM="--limit $LIMIT"
IFACES=${IFACES:-"coord nl"}

# ---- per-backbone config: model path, total pixel budget, max-model-len, extra vllm flags ----
snap () { ls -d $CACHE/models--$1/snapshots/*/ 2>/dev/null | head -1; }
case "$BACKBONE" in
  qwen3vl)   MODEL=$(snap Qwen--Qwen3-VL-8B-Instruct);      BUDGET=187000000; MAXLEN=262144; XFLAG="" ;;
  qwen35_9b) MODEL=$(snap Qwen--Qwen3.5-9B);                BUDGET=187000000; MAXLEN=262144; XFLAG="" ;;
  qwen35_27b)MODEL=$(snap Qwen--Qwen3.5-27B);               BUDGET=187000000; MAXLEN=262144; XFLAG="" ;;
  qwen3vl_30b)MODEL=$(snap Qwen--Qwen3-VL-30B-A3B-Instruct);BUDGET=187000000; MAXLEN=262144; XFLAG="" ;;   # reproduce CiteVQA (ref single-doc Rec 5.6; 3.5 was Overall)
  gemma4_31b)MODEL=$(snap google--gemma-4-31B-it);            BUDGET=187000000; MAXLEN=262144; XFLAG="" ;;   # cross-family reproduce (ref single-doc Rec 10.9)
  internvl)  MODEL=$(snap OpenGVLab--InternVL3_5-8B);       BUDGET=18750000;  MAXLEN=40960;  XFLAG="" ;;
  gemma3)    MODEL=$(snap google--gemma-3-12b-it);          BUDGET=187000000; MAXLEN=131072; XFLAG="" ;;
  pixtral)   MODEL=$(snap mistral-community--pixtral-12b);  BUDGET=28700000;  MAXLEN=131072; XFLAG="" ;;
  *) echo "unknown BACKBONE=$BACKBONE"; exit 1 ;;
esac
echo "BACKBONE=$BACKBONE MODEL=$MODEL BUDGET=$BUDGET MAXLEN=$MAXLEN PORT=$PORT"
[ -z "$MODEL" ] && { echo "model snapshot not found"; exit 1; }

# ---- serve (${ENV_INFER} has vllm + ninja for GDN qwen3.5) ----
SVLOG=$ROOT/logs/matrix_serve_${BACKBONE}_${SLURM_JOB_ID}.log
( source activate ${ENV_INFER}
  VLLM_ATTENTION_BACKEND=TORCH_SDPA \
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name matrix --host 127.0.0.1 --port $PORT \
    --gpu-memory-utilization 0.90 --max-model-len $MAXLEN \
    --limit-mm-per-prompt '{"image":200}' --trust-remote-code $XFLAG ) > "$SVLOG" 2>&1 &
SV=$!
for i in $(seq 1 180); do
  kill -0 $SV 2>/dev/null || { echo "[serve] DIED"; tail -40 "$SVLOG"; exit 1; }
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { echo "[serve] READY ${i}0s"; break; }
  sleep 10
done

# ---- inference: coord + nl ----
source activate ${ENV_MAIN}
URL=http://127.0.0.1:$PORT
for IFACE in $IFACES; do
  OUT=$ROOT/outputs/matrix/${BACKBONE}_${IFACE}/predictions.jsonl
  python src/infer_matrix.py --url $URL --served_model matrix --interface $IFACE \
    --bench data/citevqa/citevqa_singledoc_pub.jsonl --budget $BUDGET --out $OUT $LIM \
    --max_tokens ${MAXTOK:-1536}
done
kill $SV 2>/dev/null; sleep 3

# ---- NL: quote -> whole-doc block bbox (${ENV_ANALYSIS}); skipped when IFACES has no nl (coord-only rerun) ----
WD=$ROOT/outputs/matrix/${BACKBONE}_nl_wd/predictions.jsonl
source activate ${ENV_ANALYSIS}
if echo "$IFACES" | grep -qw nl; then
  NLG=$ROOT/outputs/matrix/${BACKBONE}_nl/predictions.jsonl
  python src/match_wholedoc.py --pred $NLG --bench data/citevqa/citevqa_singledoc_pub.jsonl \
    --cache data/citevqa/mineru_cache --out $WD
fi

# ---- local bbox scoring (no judge) ----
echo "===== bbox scores ($BACKBONE) ====="
echo "--- coord ---"; python src/score_citevqa.py $ROOT/outputs/matrix/${BACKBONE}_coord/predictions.jsonl 2>&1 | grep -E "recall|precision|page_recall|format|n=" | head
[ -f "$WD" ] && { echo "--- nl (whole-doc) ---"; python src/score_citevqa.py $WD 2>&1 | grep -E "recall|precision|page_recall|format|n=" | head; }
echo "Exit: $?  End: $(date)"
