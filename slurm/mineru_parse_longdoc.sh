#!/bin/bash
#SBATCH --job-name=mineru_ldoc
#SBATCH --partition=<PARTITION>
#SBATCH --gpus=h200:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --output=${PROJECT_ROOT}/../logs/mineru_ldoc_%j.out
#SBATCH --error=${PROJECT_ROOT}/../logs/mineru_ldoc_%j.err

# MinerU-parse LongDocURL's 396 PDFs (CCpdf, English) -> per-doc element cache for the
# GRPO reward's quote->block matching. table/formula OFF, lang en, MINERU_MODEL_SOURCE=local.
module load mamba; module load <cluster-toolchain>; module load gcc/13.3.0; module load cuda/12.6.2
export CC=$(which gcc); export CXX=$(which g++)
export HF_HOME=${PROJECT_ROOT}/../cache MINERU_MODEL_SOURCE=local OMP_NUM_THREADS=8
source activate ${ENV_PARSE}
cd ${PROJECT_ROOT}

CACHE=data/longdocurl/mineru_cache
RAW=${PROJECT_ROOT}/../cache/ldoc_mineru_raw
LINK=${PROJECT_ROOT}/../cache/ldoc_todo
mkdir -p "$CACHE" "$RAW" "$LINK"; rm -f "$LINK"/*.pdf

# stage NOT-yet-cached PDFs (named by doc_no) as symlinks
python - <<'PY'
import json, os
d2p=json.load(open("data/longdocurl/doc2pdf.json"))
cache="data/longdocurl/mineru_cache"; link="${PROJECT_ROOT}/../cache/ldoc_todo"
n=0
for did,p in d2p.items():
    if os.path.exists(os.path.join(cache,f"{did}.json")): continue
    dst=os.path.join(link,f"{did}.pdf")
    if not os.path.exists(dst):
        try: os.symlink(os.path.abspath(p),dst); n+=1
        except FileExistsError: pass
print(f"[stage] {n} PDFs to parse")
PY

N=$(ls "$LINK"/*.pdf 2>/dev/null | wc -l)
echo "[parse] $N PDFs (lang en)"
[ "$N" -gt 0 ] && mineru -p "$LINK" -o "$RAW" -b pipeline -m auto -t False -f False -l en 2>&1 \
    | grep -viE "pthread_setaffinity|ThreadMain" | tail -30
echo "[extract] -> mineru_cache"
python src/mineru_extract_cache.py "$RAW" "$CACHE"
echo "[cache] total: $(ls "$CACHE"/*.json 2>/dev/null | wc -l) / 396"
echo "End: $(date)"
