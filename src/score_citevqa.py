"""Score CiteVQA local predictions — bbox metrics, replicating eval/run.py EXACTLY.

Metrics (no judge needed): recall, precision, f1, page_recall.
Matching: same doc & page, IoU >= 0.5. GT for recall = necessary evidence;
precision matches against ALL evidence (our protocol decision, applied
identically across arms; see build_citevqa_eval.py docstring).

Note: precision replicates the official loop semantics verbatim
(hit = #GT boxes matched, divided by #pred boxes).

Usage: python src/score_citevqa.py outputs/citevqa_g0_gt/predictions.jsonl
"""
import argparse, json
from collections import defaultdict


def iou_hit(gt, pred, thresh=0.5):
    _, gp, gx1, gy1, gx2, gy2 = gt
    _, pp, px1, py1, px2, py2 = pred
    if gt[0] != pred[0] or gp != pp:
        return False
    ix1, iy1 = max(gx1, px1), max(gy1, py1)
    ix2, iy2 = min(gx2, px2), min(gy2, py2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_g = (gx2 - gx1) * (gy2 - gy1)
    a_p = (px2 - px1) * (py2 - py1)
    return inter / (a_g + a_p - inter + 1e-6) >= thresh


def score_item(r):
    gt_nec = [(1, b[0], b[1], b[2], b[3], b[4]) for b in r["gt_necessary"]]
    gt_all = [(1, b[0], b[1], b[2], b[3], b[4]) for b in r["gt_all"]]
    preds = [tuple(t) for t in r.get("pred_tuples", [])]

    # recall over necessary GT
    hit_r = sum(1 for g in gt_nec if any(iou_hit(g, p) for p in preds))
    recall = hit_r / len(gt_nec) if gt_nec else None
    # precision: official loop semantics (GT hits / #preds), vs ALL evidence
    if preds:
        hit_p = sum(1 for g in gt_all if any(iou_hit(g, p) for p in preds))
        precision = hit_p / len(preds)
    else:
        precision = None
    f1 = None
    if recall is not None and precision is not None and (recall + precision) > 0:
        f1 = 2 * recall * precision / (recall + precision)
    elif recall == 0.0 or precision == 0.0:
        f1 = 0.0
    # page recall
    gt_pages = {b[0] for b in r["gt_necessary"]}
    pred_pages = {p[1] for p in preds}
    page_recall = len(gt_pages & pred_pages) / len(gt_pages) if gt_pages else None
    return recall, precision, f1, page_recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_jsonl", nargs="+")
    args = ap.parse_args()

    for path in args.pred_jsonl:
        recs = [json.loads(l) for l in open(path)]
        agg = defaultdict(list)
        by_lang = defaultdict(lambda: defaultdict(list))
        by_qtype = defaultdict(lambda: defaultdict(list))
        by_len = defaultdict(lambda: defaultdict(list))
        n_err = 0
        for r in recs:
            if r.get("error"):
                n_err += 1
                continue
            rec, prec, f1, pr = score_item(r)
            npg = r.get("n_input_pages", 0)
            lb = "1-10pg" if npg <= 10 else ("11-24pg" if npg <= 24 else "25-48pg")
            for k, v in [("recall", rec), ("precision", prec), ("f1", f1),
                         ("page_recall", pr), ("format", r.get("format_ok"))]:
                if v is not None:
                    agg[k].append(v)
                    by_lang[r["language"]][k].append(v)
                    by_qtype[r.get("qtype", "?")][k].append(v)
                    by_len[lb][k].append(v)
        n = len(recs) - n_err
        print(f"\n=== {path} (n={n}, infra_err={n_err}) ===")
        for k in ["recall", "precision", "f1", "page_recall", "format"]:
            v = agg[k]
            print(f"  {k:12s} {100*sum(v)/len(v):6.1f}  (n={len(v)})" if v else f"  {k:12s}   --")
        for tag, table in [("lang", by_lang), ("qtype", by_qtype), ("doclen", by_len)]:
            for key, d in sorted(table.items()):
                n_k = len(d.get("recall", d.get("page_recall", [])))
                vals = " ".join(f"{k}={100*sum(v)/len(v):.1f}" for k, v in d.items() if v and k != "format")
                print(f"  [{tag}:{key} n={n_k}] {vals}")


if __name__ == "__main__":
    main()
