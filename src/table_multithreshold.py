#!/usr/bin/env python3
"""Multi-threshold bbox recall table for the P1/P2 matrix. Reports, per condition, the fraction of
NECESSARY GT evidence recovered at increasing strictness:
    page   = same page, no IoU requirement          (did it find the right page at all?)
    @0.1   = same page & IoU>=0.1 (any real overlap) (roughly right region)
    @0.3   = same page & IoU>=0.3
    @0.5   = same page & IoU>=0.5 (CiteVQA standard)
    @0.7   = same page & IoU>=0.7 (strict, precise box)
The gap between `page` and `@0.5` localizes the failure: high page + low @0.5 = "finds the page but
draws an imprecise box" (the coordinate-interface pathology); NL recovers precision via block matching.
Usage: python src/table_multithreshold.py tag=path [tag=path ...]
"""
import argparse, json, sys, statistics
sys.path.insert(0, "src")
from score_citevqa import iou_hit

THRESH = [0.1, 0.3, 0.5, 0.7]

def page_recall(r):
    gp = {b[0] for b in r["gt_necessary"]}
    pp = {p[1] for p in r.get("pred_tuples", [])}
    return len(gp & pp) / len(gp) if gp else None

def recall_at(r, th):
    gt = [(1, b[0], b[1], b[2], b[3], b[4]) for b in r["gt_necessary"]]
    preds = [tuple(t) for t in r.get("pred_tuples", [])]
    if not gt:
        return None
    return sum(1 for g in gt if any(iou_hit(g, p, th) for p in preds)) / len(gt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help="tag=path pairs")
    args = ap.parse_args()
    hdr = f"{'condition':<24} {'n':>4} {'page':>6} " + " ".join(f"{'@'+str(t):>6}" for t in THRESH) + f" {'fmt%':>6}"
    print(hdr); print("-" * len(hdr))
    for pair in args.preds:
        tag, path = pair.split("=", 1)
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if not r.get("error") and r.get("gt_necessary")]
        pr = [page_recall(r) for r in rows]; pr = [v for v in pr if v is not None]
        cols = []
        for th in THRESH:
            v = [recall_at(r, th) for r in rows]; v = [a for a in v if a is not None]
            cols.append(100 * statistics.mean(v) if v else 0.0)
        fmt = 100 * statistics.mean([r.get("format_ok", 0) for r in rows]) if rows else 0.0
        line = f"{tag:<24} {len(pr):>4} {100*statistics.mean(pr):>6.1f} " + " ".join(f"{c:>6.2f}" for c in cols) + f" {fmt:>6.1f}"
        print(line)

if __name__ == "__main__":
    main()
