#!/bin/bash
#SBATCH --job-name=judge_batch
#SBATCH --partition=<PARTITION>
#SBATCH --time=1-12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=${PROJECT_ROOT}/../logs/judge_batch_%j.out
#SBATCH --error=${PROJECT_ROOT}/../logs/judge_batch_%j.err
#
# Submit the Gemini Batch judge (upload JSONL -> create -> poll -> download) and score.
# Pure API task (no GPU); <PARTITION> has network + long walltime for the 24h batch SLA.
# API key read from ~/.config/gemini/api_key (HOME reachable from compute nodes).
module load mamba
source activate ${ENV_MAIN}
cd ${PROJECT_ROOT}

echo "=== submit batch judge in sequential chunks (<=2M enqueued tok each; Tier-1 cap 3M) ==="
python src/submit_judge_chunked.py \
  --jsonl outputs/judge/batch_requests.jsonl \
  --out outputs/judge/batch_results.jsonl --poll_sec 90
RC=$?
echo "submit rc=$RC"
[ ! -s outputs/judge/batch_results.jsonl ] && { echo "no results downloaded"; exit 1; }

echo "=== parse rate ==="
python -c "
import json
R=[json.loads(l) for l in open('outputs/judge/batch_results.jsonl')]
def txt(o):
    r=o.get('response') or {}
    for c in (r.get('candidates') or []):
        for p in ((c.get('content') or {}).get('parts') or []):
            if p.get('text'): return p['text']
    return r.get('text','')
qa=[x for x in R if x['key'].endswith('|qa')]; rel=[x for x in R if x['key'].endswith('|rel')]
qok=sum(1 for x in qa if '<qa_acc>' in txt(x)); rok=sum(1 for x in rel if '<relevance_score>' in txt(x))
err=sum(1 for x in R if x.get('error'))
print(f'  total={len(R)} QA={qok}/{len(qa)} REL={rok}/{len(rel)} api_err={err}')
"
echo "=== SAA / AH / Rel (fixed conditions) ==="
python src/score_judge_batch.py --results outputs/judge/batch_results.jsonl \
  --preds \
    qwen3vl_coord=outputs/matrix/qwen3vl_coord/predictions.jsonl \
    qwen3vl_nl=outputs/matrix/qwen3vl_nl_wd/predictions.jsonl \
    gemma3_coord=outputs/matrix/gemma3_coord/predictions.jsonl \
    gemma3_nl=outputs/matrix/gemma3_nl_wd/predictions.jsonl \
    grpo_nl=outputs/matrix/grpo_global_step_144_lr1e5_nl_wd/predictions.jsonl
echo "End: $(date)"
