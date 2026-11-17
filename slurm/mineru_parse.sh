#!/bin/bash
#SBATCH --job-name=mineru_pub
#SBATCH --partition=<PARTITION>
#SBATCH --gpus=h200:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --output=${PROJECT_ROOT}/../logs/mineru_pub_%j.out
#SBATCH --error=${PROJECT_ROOT}/../logs/mineru_pub_%j.err

# Parse the PUB single-doc bench's PDFs that are NOT yet in mineru_cache (145 of 440,
# the long documents added by removing the 48-page cap). Same recipe as the small-version
# parse: pipeline backend, official offline models, table/formula OFF, split by language.
# Resumable: skips PDFs already extracted. Longer --time because pub docs are long (up to 182pg).
module load mamba; module load <cluster-toolchain>; module load gcc/13.3.0; module load cuda/12.6.2
export CC=$(which gcc); export CXX=$(which g++)
export HF_HOME=${PROJECT_ROOT}/../cache
export MINERU_MODEL_SOURCE=local
export OMP_NUM_THREADS=8
source activate ${ENV_PARSE}
cd ${PROJECT_ROOT}

CACHE=data/citevqa/mineru_cache
RAW=${PROJECT_ROOT}/../cache/mineru_cache_raw
LINK_EN=${PROJECT_ROOT}/../cache/mineru_todo_en
LINK_ZH=${PROJECT_ROOT}/../cache/mineru_todo_zh
mkdir -p "$CACHE" "$RAW" "$LINK_EN" "$LINK_ZH"
rm -f "$LINK_EN"/*.pdf "$LINK_ZH"/*.pdf

# stage NOT-yet-cached unique PDFs from the PUB bench into per-language symlink dirs
python - <<'PY'
import json, os
R=[json.loads(l) for l in open("data/citevqa/citevqa_singledoc_pub.jsonl")]
lang={}
for r in R: lang.setdefault(r["pdf_path"], r.get("language","en"))
cache="data/citevqa/mineru_cache"
dirs={"zh":"${PROJECT_ROOT}/../cache/mineru_todo_zh",
      "en":"${PROJECT_ROOT}/../cache/mineru_todo_en"}
n={"en":0,"zh":0}
for p,lg in sorted(lang.items()):
    stem=os.path.splitext(os.path.basename(p))[0]
    if os.path.exists(os.path.join(cache,f"{stem}.json")): continue
    key="zh" if lg=="zh" else "en"
    dst=os.path.join(dirs[key],f"{stem}.pdf")
    if not os.path.exists(dst):
        try: os.symlink(os.path.abspath(p),dst); n[key]+=1
        except FileExistsError: pass
print(f"[stage] to-parse en={n['en']} zh={n['zh']}")
PY

for pair in "en:$LINK_EN" "ch:$LINK_ZH"; do
  L=${pair%%:*}; D=${pair##*:}
  N=$(ls "$D"/*.pdf 2>/dev/null | wc -l)
  echo "[parse] lang=$L  $N PDFs"
  [ "$N" -gt 0 ] && mineru -p "$D" -o "$RAW" -b pipeline -m auto -t False -f False -l "$L" 2>&1 \
      | grep -viE "pthread_setaffinity|ThreadMain" | tail -25
done

echo "[extract] content_list -> mineru_cache"
python src/mineru_extract_cache.py "$RAW" "$CACHE"
echo "[cache] total: $(ls "$CACHE"/*.json 2>/dev/null | wc -l) json (target 440 for pub)"
echo "End: $(date)"
