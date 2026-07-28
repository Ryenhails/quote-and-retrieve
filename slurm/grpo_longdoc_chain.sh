#!/bin/bash
#SBATCH --job-name=grpo_chain
#SBATCH --gpus=h200:3
#SBATCH --partition=<PARTITION>
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=480GB
#SBATCH --output=${PROJECT_ROOT}/../logs/grpo_chain_%j.out
#SBATCH --error=${PROJECT_ROOT}/../logs/grpo_chain_%j.err

# GRPO = C2 reward (soft answer-gate x additive evidence, long-tail) + DAPO online filtering
# (oversample mini_rollout 64 -> drop all-pass/all-fail groups -> fill batch 32) over the
# ~1667-item pool (2252 minus consistently-aced); zero-reward prompts re-enter as policy improves.
# Self-chaining 2h-segment GRPO with auto-resume. Each segment: (re)start judge+reward
# services, resume the actor from the latest checkpoint (verl find_last_checkpoint=True,
# restores LR scheduler + optimizer + step), train until the 2h SLURM limit kills it,
# whereupon the already-queued successor (afterany) resumes from the last complete ckpt.
# Trainer exit 0 == training complete -> write .done + cancel the queued successor.
module load mamba; module load <cluster-toolchain>; module load gcc/13.3.0; module load cuda/12.6.2
export CC=$(which gcc); export CXX=$(which g++)
export HF_HOME=${PROJECT_ROOT}/../cache HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=${PROJECT_ROOT}/../cache/triton_persist
export VLLM_CACHE_ROOT=${PROJECT_ROOT}/../cache/vllm_cache    # persist GDN JIT across segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True                     # fix 107GB reserved-but-unallocated fragmentation (lm_head logits OOM)
ROOT=${PROJECT_ROOT}
OUT=$ROOT/outputs/grpo_longdoc
MODEL=${PROJECT_ROOT}/../cache/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
JUDGE=$(ls -d ${PROJECT_ROOT}/../cache/models--Qwen--Qwen3.5-9B/snapshots/*/ 2>/dev/null | head -1)
${ENV_MAIN}=${ENV_MAIN}/bin/python
MAXSEG=24
# unique localhost ports per segment (avoid 'Address already in use' when a node already
# has something on 8000/8001 from another user/job)
JPORT=$((20000 + SLURM_JOB_ID % 20000)); RPORT=$((JPORT + 1))
mkdir -p $OUT
cd $ROOT

# ---- chain bookkeeping + runaway guard ----
[ -f "$OUT/.done" ]   && { echo "[chain] training complete (.done). exit."; exit 0; }
[ -f "$OUT/.failed" ] && { echo "[chain] chain marked .failed. exit (no successor)."; exit 0; }
SEG=$(( $(cat $OUT/.segment 2>/dev/null || echo 0) + 1 )); echo $SEG > $OUT/.segment
echo "[chain] segment $SEG / $MAXSEG  (job $SLURM_JOB_ID)"
[ $SEG -gt $MAXSEG ] && { echo "[chain] MAXSEG reached, stop."; exit 0; }
# progress guard: if the PREVIOUS segment advanced no checkpoint step, count a strike;
# 2 consecutive stuck segments (hard failure, not a timeout) -> stop the chain.
latest_step() { ls -d $OUT/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1; }
CUR=$(latest_step); CUR=${CUR:-0}; LAST=$(cat $OUT/.laststep 2>/dev/null || echo -1)
if [ "$CUR" -le "$LAST" ]; then
  FC=$(( $(cat $OUT/.failcount 2>/dev/null || echo 0) + 1 )); echo $FC > $OUT/.failcount
  echo "[chain] NO progress since last segment (step=$CUR<=$LAST) strike=$FC"
  [ $FC -ge 2 ] && { echo "[chain] 2 stuck segments -> .failed, stop, NO successor."; touch $OUT/.failed; exit 1; }
else
  echo 0 > $OUT/.failcount
fi
echo $CUR > $OUT/.laststep
# queue the successor NOW (runs after this one ends, even if timeout-killed)
NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID slurm/grpo_longdoc_chain.sh)
echo $NEXT > $OUT/.nextjob; echo "[chain] queued successor $NEXT (afterany)"

# ---- preserve PREVIOUS segment's metrics (val curve!) before verl truncates the log ----
$${ENV_MAIN} $ROOT/src/accumulate_metrics.py $OUT 2>&1 || echo "[chain] accumulate_metrics failed (non-fatal)"

# ---- 1. vLLM vision judge on GPU 2 (${ENV_INFER} env -> ninja on PATH for GDN JIT) ----
JUDGE_LOG=$ROOT/logs/grpo_chain_judge_${SLURM_JOB_ID}.log
( source activate ${ENV_INFER}
  CUDA_VISIBLE_DEVICES=2 VLLM_ATTENTION_BACKEND=TORCH_SDPA \
  python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE" --served-model-name judge --host 127.0.0.1 --port $JPORT \
    --gpu-memory-utilization 0.45 --max-model-len 8192 --trust-remote-code \
    --limit-mm-per-prompt '{"image":6}' ) > "$JUDGE_LOG" 2>&1 &
JUDGE_PID=$!
for i in $(seq 1 120); do
  kill -0 $JUDGE_PID 2>/dev/null || { echo "[judge] DIED"; tail -30 "$JUDGE_LOG"; exit 1; }
  curl -sf http://127.0.0.1:$JPORT/v1/models >/dev/null 2>&1 && { echo "[judge] READY ${i}0s"; break; }
  sleep 10
done

# ---- 2. reward_service on GPU 2 (${ENV_MAIN} env) ----
REWARD_LOG=$ROOT/logs/grpo_chain_reward_${SLURM_JOB_ID}.log
CUDA_VISIBLE_DEVICES=2 JUDGE_BASE_URL=http://127.0.0.1:$JPORT/v1 JUDGE_MODEL=judge \
  REWARD_PORT=$RPORT PYTHONPATH=$ROOT/src \
  $${ENV_MAIN} $ROOT/src/reward_service.py > "$REWARD_LOG" 2>&1 &
REWARD_PID=$!
for i in $(seq 1 60); do
  kill -0 $REWARD_PID 2>/dev/null || { echo "[reward] DIED"; tail -30 "$REWARD_LOG"; kill $JUDGE_PID; exit 1; }
  curl -sf http://127.0.0.1:$RPORT/health >/dev/null 2>&1 && { echo "[reward] READY ${i}0s"; break; }
  sleep 10
done

# ---- 3. actor on GPU 0,1 (easyr1 env); auto-resume via find_last_checkpoint ----
source activate ${ENV_TRAIN}
export REWARD_SERVICE_URL=http://127.0.0.1:$RPORT
cd $ROOT/EasyR1
CUDA_VISIBLE_DEVICES=0,1 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=$ROOT/data/longdocurl/grpo/train.jsonl \
    data.val_files=$ROOT/data/longdocurl/grpo/val.jsonl \
    data.format_prompt=$ROOT/configs/grounding_passthrough.jinja \
    data.max_prompt_length=48000 data.max_response_length=768 \
    data.rollout_batch_size=32 data.val_batch_size=24 \
    data.mini_rollout_batch_size=64 \
    data.min_pixels=65536 data.max_pixels=1000000 \
    algorithm.online_filtering=true algorithm.filter_key=overall \
    algorithm.filter_low=0.01 algorithm.filter_high=0.99 \
    worker.actor.model.model_path=$MODEL \
    worker.actor.model.lora.rank=64 worker.actor.model.lora.exclude_modules='.*visual.*' \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.optim.lr=1e-5 worker.actor.global_batch_size=16 \
    worker.actor.clip_ratio_low=0.2 worker.actor.clip_ratio_high=0.28 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.rollout.n=8 worker.rollout.tensor_parallel_size=2 \
    worker.rollout.temperature=1.0 worker.rollout.limit_images=32 \
    worker.rollout.gpu_memory_utilization=0.55 worker.rollout.max_num_batched_tokens=49152 \
    worker.rollout.val_override_config.temperature=0 worker.rollout.val_override_config.n=1 \
    worker.reward.reward_function=$ROOT/src/reward_longdoc_grpo.py:compute_score \
    trainer.total_epochs=3 trainer.experiment_name=grpo_longdoc \
    trainer.project_name=doc_grounding trainer.logger='["file"]' \
    trainer.n_gpus_per_node=2 trainer.find_last_checkpoint=true \
    trainer.val_freq=10 trainer.val_before_train=false \
    trainer.save_freq=1 trainer.save_limit=4 \
    trainer.save_checkpoint_path=$OUT
EC=$?
echo "[trainer] exit=$EC"
kill $JUDGE_PID $REWARD_PID 2>/dev/null

# ---- 4. completion handling ----
if [ $EC -eq 0 ]; then
  echo "[chain] trainer finished cleanly -> training COMPLETE. writing .done + cancelling successor."
  touch $OUT/.done
  scancel $(cat $OUT/.nextjob 2>/dev/null) 2>/dev/null
else
  echo "[chain] trainer exit=$EC (likely 2h timeout). successor $NEXT will resume from last ckpt."
fi
echo "End: $(date)"
exit $EC
