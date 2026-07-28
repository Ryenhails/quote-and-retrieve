#!/usr/bin/env python3
"""Parse Gemini Batch judge results, combine with local bbox recall, and print the CiteVQA main table.
Scores are raw 0-5 (we do NOT apply the repo's /20 bug). SAA = 1(qa_acc>=4 and (rel>=4 or recall>=0.6)),
matching the paper. AH (hallucination rate) = P(SAA=0 | qa_acc>=4). Rel/Ans shown x20 (0-100) as in the paper.
Usage: python src/score_judge_batch.py --results outputs/judge/batch_results.jsonl \
         --preds qwen3vl_coord=outputs/matrix/qwen3vl_coord/predictions.jsonl qwen3vl_nl=outputs/matrix/qwen3vl_nl_wd/predictions.jsonl ...
"""
import argparse, json, re, sys, statistics
sys.path.insert(0, "src")
from score_citevqa import score_item

QA_RE = re.compile(r'<?qa_acc>\s*(\d+)\s*</qa_acc>')       # lenient: gemini sometimes drops leading '<'
REL_RE = re.compile(r'<?relevance_score>\s*(\d+)\s*</relevance_score>')

def resp_text(obj):
    r = obj.get("response") or {}
    if isinstance(r, dict):
        if r.get("text"):
            return r["text"]
        for c in (r.get("candidates") or []):
            for p in ((c.get("content") or {}).get("parts") or []):
                if p.get("text"):
                    return p["text"]
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/judge/batch_results.jsonl")
    ap.add_argument("--preds", nargs="+", required=True)
    args = ap.parse_args()

    # parse judge outputs: (tag, idx) -> {qa, rel}
    J = {}
    n_parse_err = 0
    for l in open(args.results):
        l = l.strip()
        if not l:
            continue
        o = json.loads(l)
        key = o.get("key", "")
        parts = key.split("|")
        if len(parts) != 3:
            continue
        tag, idx, kind = parts
        txt = resp_text(o)
        m = (QA_RE if kind == "qa" else REL_RE).search(txt)
        if not m:
            n_parse_err += 1
            continue
        J.setdefault((tag, idx), {})[kind] = int(m.group(1))   # raw 0-5

    # local bbox recall per (tag, idx)
    REC = {}
    for pair in args.preds:
        tag, path = pair.split("=", 1)
        for l in open(path):
            r = json.loads(l)
            if r.get("error") or not r.get("gt_necessary"):
                continue
            rec = score_item(r)[0]
            if rec is not None:
                REC[(tag, r["index"])] = rec

    # aggregate per tag
    tags = sorted({t for t, _ in J} | {t for t, _ in REC})
    print(f"\n{'tag':<28} {'n':>4} {'QA>=4%':>7} {'Rel':>6} {'bboxR%':>7} {'SAA%':>6} {'AH%':>6}")
    print("-" * 70)
    for tag in tags:
        items = []
        for (t, idx), sc in J.items():
            if t != tag:
                continue
            qa = sc.get("qa"); rel = sc.get("rel"); rec = REC.get((t, idx))
            if qa is None or rel is None or rec is None:
                continue
            saa = 1.0 if (qa >= 4 and (rel >= 4 or rec >= 0.6)) else 0.0
            items.append((qa, rel, rec, saa))
        if not items:
            continue
        n = len(items)
        qa4 = 100 * sum(1 for q, _, _, _ in items if q >= 4) / n
        relm = 20 * statistics.mean(r for _, r, _, _ in items)      # x20 -> 0-100 as in paper
        bboxr = 100 * statistics.mean(rc for _, _, rc, _ in items)
        saa = 100 * statistics.mean(s for _, _, _, s in items)
        corr = [it for it in items if it[0] >= 4]
        ah = 100 * sum(1 for it in corr if it[3] == 0) / len(corr) if corr else 0.0
        print(f"{tag:<28} {n:>4} {qa4:>7.1f} {relm:>6.1f} {bboxr:>7.1f} {saa:>6.1f} {ah:>6.1f}")
    print(f"\n(parse errors: {n_parse_err}; SAA=1(qa>=4 & (rel>=4 or bboxR>=0.6)); AH=P(SAA=0|qa>=4); raw 0-5, no /20 bug)")

if __name__ == "__main__":
    main()
