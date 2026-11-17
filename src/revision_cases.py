#!/usr/bin/env python3
"""Camera-ready: text-form 'side-by-side' cases where quote retrieval and
question+answer retrieval land on different blocks, for the same model output.

For each direction (quotes hit / q+a missed; q+a hit / quotes missed) pick English questions with one
necessary element, and print: question, gold element type, the model's quotes, the block text each
quote was assigned to (with page + IoU vs gold), and the block text q+a retrieval returned.
Block texts come from data/citevqa/mineru_cache (matched by page + bbox).

Usage: python src/revision_cases.py [--backbone qwen3vl] [--n 4]
Writes outputs/revision/analysis/cases.md
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qr_paths  # noqa: F401,E402  sets PROJECT_ROOT / QR_CACHE / QWEN3VL_EMB_REPO defaults
import argparse, json, os, sys
sys.path.insert(0, "src")
from revision_analysis import (valid_rows, load_preds, gt_nec, preds_of, REB, OUT, CACHE, CITE, parse_obj)
from score_citevqa import iou_hit

TYPE_GROUP = {"text": "text", "list": "text", "ref_text": "text", "code": "text", "title": "text",
              "table": "table", "table_caption": "table caption", "table_footnote": "table note",
              "image": "figure", "image_caption": "figure caption"}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def block_text(pdf, page1, bbox):
    stem = os.path.splitext(os.path.basename(pdf))[0]
    cp = os.path.join(CACHE, f"{stem}.json")
    if not os.path.exists(cp):
        return "?"
    cache = json.load(open(cp))
    best, bt = 0.0, ""
    for b in cache.get(str(int(page1) - 1), []):
        v = iou(tuple(b["bbox"]), tuple(bbox))
        if v > best:
            best, bt = v, (b.get("text") or f"[{b.get('type','?')} block, no text]")
    return bt.replace("\n", " ").strip()


def clip(s, n=220):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + " …"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="qwen3vl")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    os.chdir(os.path.expandvars("${PROJECT_ROOT}"))
    cite = {d["index"]: d for d in json.load(open(CITE))}
    qa = {json.loads(l)["index"]: json.loads(l) for l in open(f"{REB}/qa_boxes_{args.backbone}.jsonl")}
    nl = {r["index"]: r for r in valid_rows(load_preds(f"{args.backbone}_nl_wd"))}
    quotes_hit_qa_miss, qa_hit_quotes_miss = [], []
    for i, r in nl.items():
        if r["language"] != "en" or len(r["gt_necessary"]) != 1 or i not in qa:
            continue
        g = gt_nec(r)[0]
        U = preds_of(r); QA = [tuple(x) for x in qa[i]["qa_tuples"]]
        hu = any(iou_hit(g, p) for p in U); hq = any(iou_hit(g, p) for p in QA)
        if hu and not hq and QA:
            quotes_hit_qa_miss.append(i)
        elif hq and not hu and U:
            qa_hit_quotes_miss.append(i)

    def render(i):
        r = nl[i]; g = gt_nec(r)[0]; gbox = (g[2], g[3], g[4], g[5])
        ev = [e for e in cite[i]["Evidence"] if e["necessity"] == "necessary"]
        etype = TYPE_GROUP.get(ev[0].get("type", "?"), ev[0].get("type", "?")) if ev else "?"
        o = parse_obj(r.get("raw_predict", "")) or {}
        quotes = [str(x.get("quote", "")) for x in (o.get("evidence") or []) if isinstance(x, dict)]
        ans = str(o.get("answer", ""))
        out = [f"**Question** ({i[:12]}, {r['qtype']}, gold element: {etype}, page {g[1]} of {r['n_pdf_pages']}): {clip(r['question'], 300)}",
               f"- Gold answer: {clip(r['standard_answer'], 200)}",
               f"- Model answer: {clip(ans, 200)}",
               "- Quote retrieval (ours):"]
        for q, p in zip(quotes, preds_of(r)):
            v = iou(gbox, tuple(p[2:])) if p[1] == g[1] else 0.0
            out.append(f"  - quote: “{clip(q, 160)}” → page {p[1]} block “{clip(block_text(r['pdf_path'], p[1], p[2:]), 160)}” (IoU with gold {v:.2f})")
        out.append("- Question + answer retrieval (same budget):")
        for p in qa[i]["qa_tuples"]:
            v = iou(gbox, tuple(p[2:])) if p[1] == g[1] else 0.0
            out.append(f"  - page {p[1]} block “{clip(block_text(r['pdf_path'], p[1], p[2:]), 160)}” (IoU with gold {v:.2f})")
        return "\n".join(out)

    lines = [f"# Side-by-side cases: quote retrieval vs question+answer retrieval ({args.backbone}, English, single-element questions)", "",
             f"Counts over the {sum(1 for r in nl.values() if r['language']=='en' and len(r['gt_necessary'])==1)} English single-element questions: "
             f"quotes hit / q+a missed = {len(quotes_hit_qa_miss)}; q+a hit / quotes missed = {len(qa_hit_quotes_miss)}. "
             "Both readings use the same encoder, candidate blocks and per-question budget; only the query differs.", ""]
    lines += ["## A. The quote reaches the gold element, question+answer retrieval does not", ""]
    for i in quotes_hit_qa_miss[:args.n]:
        lines += [render(i), ""]
    lines += ["## B. Question+answer retrieval reaches the gold element, the quote does not", ""]
    for i in qa_hit_quotes_miss[:args.n]:
        lines += [render(i), ""]
    lines += ["Pattern to look for: in A the quote is a specific cell / caption / sentence that pins one block while the answer text is generic and "
              "drifts to a topically similar block; in B the quote is paraphrased or spans two blocks (assignment lands on a neighbour), while the "
              "answer restates the evidence content closely enough for dense retrieval."]
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/cases.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {OUT}/cases.md")


if __name__ == "__main__":
    main()
