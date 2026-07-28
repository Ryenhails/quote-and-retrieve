#!/usr/bin/env python3
"""Pre-build the whole-document block embedding cache for the PUB single-doc bench, so
the eval matrix does not pay the embedding cost inline. Replicates exactly the block_emb_for_pdf
logic of match_wholedoc.py: render each page at 150 DPI, crop every MinerU block, embed the block
image with Qwen3-VL-Embedding-2B, and save E/pages/bboxes to cache/block_emb/<stem>.npz.
Idempotent: skips docs already cached. Run AFTER MinerU parsing (needs mineru_cache/<stem>.json)."""
import sys, json, os
sys.path.insert(0, "${PROJECT_ROOT}/../rerank-highlight")
import numpy as np, torch, fitz
from PIL import Image
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

ROOT = "${PROJECT_ROOT}"
EMB_CACHE = "${PROJECT_ROOT}/../cache/block_emb"
MINERU = f"{ROOT}/data/citevqa/mineru_cache"
BENCH = f"{ROOT}/data/citevqa/citevqa_singledoc_pub.jsonl"
os.makedirs(EMB_CACHE, exist_ok=True)

def crop(page_img, bbox):
    W, H = page_img.size
    x1, y1, x2, y2 = bbox[0]/1000*W, bbox[1]/1000*H, bbox[2]/1000*W, bbox[3]/1000*H
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return page_img.crop((x1, y1, x2, y2))

def main():
    mp = "${PROJECT_ROOT}/../cache/models--Qwen--Qwen3-VL-Embedding-2B/snapshots"
    mp = os.path.join(mp, sorted(os.listdir(mp))[0])
    pairs = {}
    for l in open(BENCH):
        r = json.loads(l)
        stem = os.path.splitext(os.path.basename(r["pdf_path"]))[0]
        pairs[stem] = r["pdf_path"]
    todo = [(s, p) for s, p in sorted(pairs.items())
            if not os.path.exists(os.path.join(EMB_CACHE, f"{s}.npz"))]
    print(f"pub docs: {len(pairs)} | block_emb to build: {len(todo)}", flush=True)
    if not todo:
        print("nothing to build"); return

    print(f"loading embedder {mp}", flush=True)
    emb = Qwen3VLEmbedder(model_name_or_path=mp, torch_dtype=torch.float16, attn_implementation="sdpa")

    @torch.no_grad()
    def embed(items):
        out = []
        for k in range(0, len(items), 48):
            out.append(emb.process(items[k:k+48]).float().cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, 2048), dtype=np.float32)

    built = skipped = 0
    for i, (stem, pdf) in enumerate(todo):
        fp = os.path.join(EMB_CACHE, f"{stem}.npz")
        cpath = os.path.join(MINERU, f"{stem}.json")
        if not os.path.exists(cpath):
            skipped += 1
            print(f"  [{i+1}/{len(todo)}] SKIP (no mineru cache) {stem}", flush=True)
            continue
        try:
            cache = json.load(open(cpath))
            doc = fitz.open(pdf)
            imgs, pages, boxes = [], [], []
            for pidx_s, els in cache.items():
                pidx = int(pidx_s)
                if pidx >= len(doc):
                    continue
                pix = doc[pidx].get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
                pim = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                for b in els:
                    c = crop(pim, b["bbox"])
                    if c is None:
                        continue
                    imgs.append({"image": c}); pages.append(pidx); boxes.append(b["bbox"])
            doc.close()
            E = embed(imgs) if imgs else np.zeros((0, 2048), dtype=np.float32)
            pages = np.array(pages, int)
            boxes = np.array(boxes, float) if boxes else np.zeros((0, 4))
            np.savez(fp, E=E, pages=pages, bboxes=boxes)
            built += 1
            if (i+1) % 10 == 0 or i == len(todo)-1:
                print(f"  [{i+1}/{len(todo)}] built {stem} ({E.shape[0]} blocks)", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(todo)}] ERROR {stem}: {str(e)[:100]}", flush=True)
    print(f"\nDONE: built {built}, skipped-no-mineru {skipped}")

if __name__ == "__main__":
    main()
