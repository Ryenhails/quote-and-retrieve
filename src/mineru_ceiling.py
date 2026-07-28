#!/usr/bin/env python3
"""MinerU ceiling bbox recall: the UPPER BOUND of the NL (quote->block) interface. If retrieval were
perfect (every GT necessary evidence matched to its exact MinerU block), what recall is achievable?
= recall of GT necessary bboxes when the candidate set is ALL MinerU blocks of the document. A GT box
above the ceiling is fundamentally unrecoverable by ANY quote-retrieval method (MinerU never parsed a
block that overlaps it). Reported per threshold (page, IoU>=0.1/0.3/0.5/0.7). Bench = pub 719.
Usage: python src/mineru_ceiling.py [--bench ...] [--cache ...]
"""
import argparse, json, os, sys, statistics
sys.path.insert(0, "src")
from score_citevqa import iou_hit

THRESH = [0.1, 0.3, 0.5, 0.7]

def blocks_for(pdf_path, cache_dir):
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    cpath = os.path.join(cache_dir, f"{stem}.json")
    if not os.path.exists(cpath):
        return None   # MinerU parse missing entirely
    cache = json.load(open(cpath))
    blocks = []
    for pidx_s, els in cache.items():
        p = int(pidx_s) + 1  # 0-based -> 1-based to match GT
        for b in els:
            bb = b.get("bbox")
            if bb and len(bb) == 4:
                blocks.append((1, p, bb[0], bb[1], bb[2], bb[3]))
    return blocks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/citevqa/citevqa_singledoc_pub.jsonl")
    ap.add_argument("--cache", default="data/citevqa/mineru_cache")
    args = ap.parse_args()
    bench = [json.loads(l) for l in open(args.bench)]

    page_rec, thr_rec = [], {t: [] for t in THRESH}
    n_no_cache = 0
    n_blocks = []
    for r in bench:
        gt = [(1, b[0], b[1], b[2], b[3], b[4]) for b in r["gt_necessary"]]
        if not gt:
            continue
        blocks = blocks_for(r["pdf_path"], args.cache)
        if blocks is None:
            n_no_cache += 1
            page_rec.append(0.0)
            for t in THRESH:
                thr_rec[t].append(0.0)
            continue
        n_blocks.append(len(blocks))
        bpages = {bl[1] for bl in blocks}
        page_rec.append(sum(1 for g in gt if g[1] in bpages) / len(gt))
        for t in THRESH:
            thr_rec[t].append(sum(1 for g in gt if any(iou_hit(g, bl, t) for bl in blocks)) / len(gt))

    n = len(page_rec)
    print(f"MinerU CEILING bboxR (pub, n={n} items with GT; {n_no_cache} lack a MinerU cache -> counted 0)")
    print(f"  median blocks/doc = {int(statistics.median(n_blocks)) if n_blocks else 0}")
    print(f"  {'page':>6}{'@0.1':>7}{'@0.3':>7}{'@0.5':>7}{'@0.7':>7}")
    print(f"  {100*statistics.mean(page_rec):>6.1f}"
          + "".join(f"{100*statistics.mean(thr_rec[t]):>7.1f}" for t in THRESH))
    print("Interpretation: NL bboxR cannot exceed these. @0.5 ceiling is the fraction of GT evidence that "
          "MinerU parses as a block overlapping GT at IoU>=0.5; the rest is a hard parser limit, not a model failure.")

if __name__ == "__main__":
    main()
