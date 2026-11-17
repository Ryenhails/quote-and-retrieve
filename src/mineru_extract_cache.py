#!/usr/bin/env python3
"""Extract per-page layout elements from MinerU output into a compact cache.

Per official MinerU output schema (opendatalab.github.io/MinerU/reference/output_files):
  content_list.json = flat reading-order blocks, each {type, text, text_level,
  page_idx (0-based), bbox [x0,y0,x1,y1] ALREADY normalized to 0-1000}.
So we use content_list.json bbox DIRECTLY (no page_size division — that would
double-normalize). CiteVQA GT elements are themselves MinerU-parsed, so this
aligns our candidate block grid with the GT element grid.

Writes data/citevqa/mineru_cache/<stem>.json == {page_idx(str): [{type,bbox,text}]}.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qr_paths  # noqa: F401,E402  sets PROJECT_ROOT / QR_CACHE / QWEN3VL_EMB_REPO defaults
import json, glob, os, sys
from pathlib import Path

RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars("${QR_CACHE}/mineru_cache_raw")
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/citevqa/mineru_cache"
Path(OUT).mkdir(parents=True, exist_ok=True)


def extract(stem_dir):
    stem = os.path.basename(stem_dir)
    cl = os.path.join(stem_dir, "auto", f"{stem}_content_list.json")
    if not os.path.isfile(cl):
        return stem, None
    items = json.load(open(cl))
    pages = {}
    for it in items:
        pidx = it.get("page_idx")
        bbox = it.get("bbox")            # already 0-1000 per official schema
        txt = it.get("text") or ""
        typ = it.get("type") or "text"
        if bbox is None or pidx is None or len(bbox) != 4:
            continue
        pages.setdefault(str(pidx), []).append(
            {"type": typ, "bbox": [float(v) for v in bbox], "text": txt})
    return stem, pages


def main():
    ok = empty = 0
    for stem_dir in sorted(glob.glob(os.path.join(RAW, "*"))):
        if not os.path.isdir(stem_dir):
            continue
        stem, pages = extract(stem_dir)
        if pages is None:
            print(f"  MISS {stem} (no content_list.json)")
            empty += 1
            continue
        json.dump(pages, open(os.path.join(OUT, f"{stem}.json"), "w"), ensure_ascii=False)
        ok += 1
    print(f"[extract] wrote {ok} caches, {empty} missing -> {OUT}")


if __name__ == "__main__":
    main()
