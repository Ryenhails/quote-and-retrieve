#!/usr/bin/env python3
"""Assemble the detailed CiteVQA main table: per condition, multi-threshold bbox recall (local) +
full Gemini-judge statistics (Ans/Rel/SAA/AH). Also prints an n-accounting block explaining why the
per-condition n differs (total -> valid -> judged, with qa/rel parse failures).
Usage: python src/build_main_table.py --results outputs/judge/batch_results.jsonl \
         --preds tag=path [tag=path ...]
Judge scores are raw 0-5; Ans/Rel shown x20 (0-100) as in the paper. SAA=1(qa>=4 & (rel>=4 or bboxR>=0.6)).
AH (attribution hallucination) = P(SAA=0 | qa>=4). bboxR thresholds: page (same page), IoU>=0.1/0.3/0.5/0.7.
"""
import argparse, json, re, sys, statistics
sys.path.insert(0, "src")
from score_citevqa import iou_hit
from mineru_ceiling import blocks_for, THRESH as CEIL_THRESH

# lenient: gemini sometimes drops the leading '<' (writes ```relevance_score>3</...); the score is valid
QA_RE = re.compile(r'<?qa_acc>\s*(\d+)\s*</qa_acc>')
REL_RE = re.compile(r'<?relevance_score>\s*(\d+)\s*</relevance_score>')
THRESH = [0.1, 0.3, 0.5, 0.7]

def resp_text(o):
    r = o.get("response") or {}
    if r.get("text"):
        return r["text"]
    for c in (r.get("candidates") or []):
        for p in ((c.get("content") or {}).get("parts") or []):
            if p.get("text"):
                return p["text"]
    return ""

def page_recall(r):
    gp = {b[0] for b in r["gt_necessary"]}; pp = {p[1] for p in r.get("pred_tuples", [])}
    return len(gp & pp) / len(gp) if gp else None

def recall_at(r, th):
    gt = [(1, b[0], b[1], b[2], b[3], b[4]) for b in r["gt_necessary"]]
    preds = [tuple(t) for t in r.get("pred_tuples", [])]
    if not gt:
        return None
    return sum(1 for g in gt if any(iou_hit(g, p, th) for p in preds)) / len(gt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/judge/batch_results.jsonl")
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--bench", default="data/citevqa/citevqa_singledoc_pub.jsonl")
    ap.add_argument("--cache", default="data/citevqa/mineru_cache")
    args = ap.parse_args()

    # parse judge: (tag,idx) -> {qa,rel}
    J, qa_fail, rel_fail = {}, {}, {}
    for l in open(args.results):
        l = l.strip()
        if not l:
            continue
        o = json.loads(l); key = o.get("key", ""); parts = key.split("|")
        if len(parts) != 3:
            continue
        tag, idx, kind = parts
        m = (QA_RE if kind == "qa" else REL_RE).search(resp_text(o))
        if m:
            J.setdefault((tag, idx), {})[kind] = int(m.group(1))
        else:
            (qa_fail if kind == "qa" else rel_fail).setdefault(tag, []).append(idx)

    print("=" * 118)
    print(f"{'condition':<16}{'n':>5}{'fmt%':>6} | {'page':>6}{'@.1':>6}{'@.3':>6}{'@.5':>6}{'@.7':>6} | "
          f"{'Ans':>6}{'Rel':>6}{'SAA':>6}{'AH':>6} | {'total':>6}{'valid':>6}{'judg':>6}{'qaF':>4}{'relF':>4}")
    print("-" * 118)
    # MinerU ceiling row: bench-level upper bound for the NL (quote->block) bboxR columns
    bench = [json.loads(l) for l in open(args.bench)]
    cp, ct = [], {t: [] for t in CEIL_THRESH}
    for r in bench:
        gt = [(1, b[0], b[1], b[2], b[3], b[4]) for b in r["gt_necessary"]]
        if not gt:
            continue
        bl = blocks_for(r["pdf_path"], args.cache) or []
        bpg = {x[1] for x in bl}
        cp.append(sum(1 for g in gt if g[1] in bpg) / len(gt))
        for t in CEIL_THRESH:
            ct[t].append(sum(1 for g in gt if any(iou_hit(g, x, t) for x in bl)) / len(gt))
    print(f"{'MinerU-ceiling':<16}{len(cp):>5}{'':>6} | {100*statistics.mean(cp):>6.1f}"
          + "".join(f"{100*statistics.mean(ct[t]):>6.1f}" for t in CEIL_THRESH)
          + f" |    -- {'--':>5}{'--':>6}{'--':>6} | upper bound for NL bboxR")
    print("-" * 118)
    for pair in args.preds:
        tag, path = pair.split("=", 1)
        rows = [json.loads(l) for l in open(path)]
        n_total = len(rows)
        valid = [r for r in rows if not r.get("error") and r.get("gt_necessary")]
        # judged = valid rows with BOTH qa and rel parsed
        items = []
        for r in valid:
            sc = J.get((tag, r["index"]))
            if not sc or "qa" not in sc or "rel" not in sc:
                continue
            rec = recall_at(r, 0.5)
            if rec is None:
                continue
            items.append((r, sc["qa"], sc["rel"], rec))
        n = len(items)
        if not n:
            print(f"{tag:<16}{0:>5}  (no judged items)"); continue
        # local multi-threshold over the JUDGED set (so columns are comparable)
        pageR = 100 * statistics.mean(page_recall(r) for r, _, _, _ in items)
        thr = [100 * statistics.mean(recall_at(r, t) for r, _, _, _ in items) for t in THRESH]
        fmt = 100 * statistics.mean(r.get("format_ok", 0) for r, _, _, _ in items)
        ans = 20 * statistics.mean(q for _, q, _, _ in items)      # x20
        rel = 20 * statistics.mean(rl for _, _, rl, _ in items)
        saa_list = [1.0 if (q >= 4 and (rl >= 4 or rc >= 0.6)) else 0.0 for _, q, rl, rc in items]
        saa = 100 * statistics.mean(saa_list)
        corr = [(s) for (_, q, _, _), s in zip(items, saa_list) if q >= 4]
        ah = 100 * sum(1 for s in corr if s == 0) / len(corr) if corr else 0.0
        print(f"{tag:<16}{n:>5}{fmt:>6.1f} | {pageR:>6.1f}{thr[0]:>6.1f}{thr[1]:>6.1f}{thr[2]:>6.1f}{thr[3]:>6.1f} | "
              f"{ans:>6.1f}{rel:>6.1f}{saa:>6.1f}{ah:>6.1f} | {n_total:>6}{len(valid):>6}{n:>6}"
              f"{len(qa_fail.get(tag,[])):>4}{len(rel_fail.get(tag,[])):>4}")
    print("-" * 118)
    tot_qaf = sum(len(v) for v in qa_fail.values()); tot_relf = sum(len(v) for v in rel_fail.values())
    print(f"parse failures total: qa={tot_qaf} rel={tot_relf}  (these items drop from n; re-judgeable as a cleanup batch)")
    print("Ans/Rel = mean judge score x20 (0-100). SAA=1(qa>=4 & (rel>=4 or bboxR>=0.6)). AH=P(SAA=0|qa>=4).")
    print("total=all 719, valid=has GT & no infra error, judg=valid with BOTH qa+rel parsed (=n).")

if __name__ == "__main__":
    main()
