#!/usr/bin/env python3
"""Assemble a single Gemini Batch API JSONL for judging all matrix predictions, using the CiteVQA
OFFICIAL judge prompts (CiteVQA/prompts/eval_qa_acc_*.txt, eval_rel_*.txt) VERBATIM. Two requests per
prediction: qa_acc (answer quality, text-only) and rel (evidence relevance, answer text + the crops of
the predicted bboxes). Follows the CiteVQA protocol exactly, EXCEPT we keep the judge score on the raw
0-5 scale (the repo's /20 is a bug: it is inconsistent with compute_saa's >=4 threshold and with the
paper's x20 table; only raw 0-5 satisfies both). SAA/AH are computed later from raw 0-5 + local bbox recall.

JSONL line: {"key": "<tag>|<index>|<qa|rel>", "request": {contents, system_instruction, generation_config}}
Crops are rendered from the page at 150 DPI and cropped by the 0-1000 bbox (crop_bbox), capped per item.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qr_paths  # noqa: F401,E402  sets PROJECT_ROOT / QR_CACHE / QWEN3VL_EMB_REPO defaults
import argparse, json, os, io, base64, re
import fitz
from PIL import Image

ROOT = os.path.expandvars("${PROJECT_ROOT}")
CVP = f"{ROOT}/CiteVQA/prompts"
QA_SYS = open(f"{CVP}/eval_qa_acc_system.txt").read()
QA_USER = open(f"{CVP}/eval_qa_acc_user.txt").read()
REL_SYS = open(f"{CVP}/eval_rel_system.txt").read()
REL_USER = open(f"{CVP}/eval_rel_user.txt").read()
MAX_CROPS = 6   # CiteVQA judge uses limit-mm image:6
# gemini-3.5-flash is a THINKING model: with thinking on, the token budget is eaten by internal
# thoughts and the score tag (emitted last) gets truncated (finishReason=MAX_TOKENS). A scoring task
# does not need extended thinking (the rubric asks for a one-line reason + score), so disable it.
GEN_CFG = {"temperature": 0.0, "max_output_tokens": 1024, "thinking_config": {"thinking_budget": 0}}

def remove_bbox(t):
    return re.sub(r'<bbox[^>]*/?>', '', t or '').strip()

def crop_bbox(img, x1, y1, x2, y2):
    w, h = img.size
    x1, y1, x2, y2 = int(x1*w/1000), int(y1*h/1000), int(x2*w/1000), int(y2*h/1000)
    if x2 <= x1 or y2 <= y1:
        return None
    return img.crop((x1, y1, x2, y2))

def img_b64(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()

def page_image(doc, pidx0, dpi=150):
    if pidx0 < 0 or pidx0 >= len(doc):
        return None
    pix = doc[pidx0].get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

def crops_for(pred_tuples, pdf_path):
    """pred_tuples: [(1, page_1based, x1,y1,x2,y2), ...] -> list of base64 JPEG crops."""
    out = []
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return out
    cache = {}
    for t in pred_tuples[:MAX_CROPS]:
        _, p1, x1, y1, x2, y2 = t
        p0 = int(p1) - 1
        if p0 not in cache:
            cache[p0] = page_image(doc, p0)
        pim = cache[p0]
        if pim is None:
            continue
        c = crop_bbox(pim, x1, y1, x2, y2)
        if c is not None and c.size[0] > 3 and c.size[1] > 3:
            out.append(img_b64(c))
    doc.close()
    return out

def qa_request(question, gt_ans, model_ans):
    user = QA_USER.format(question=question, standard_answer=gt_ans, model_answer_no_bbox=model_ans)
    return {"contents": [{"role": "user", "parts": [{"text": user}]}],
            "system_instruction": {"parts": [{"text": QA_SYS}]},
            "generation_config": dict(GEN_CFG)}

def rel_request(question, gt_ans, model_ans, crops_b64):
    user = REL_USER.format(question=question, standard_answer=gt_ans, answer_with_images=model_ans)
    parts = [{"text": user}] + [{"inline_data": {"mime_type": "image/jpeg", "data": b}} for b in crops_b64]
    return {"contents": [{"role": "user", "parts": parts}],
            "system_instruction": {"parts": [{"text": REL_SYS}]},
            "generation_config": dict(GEN_CFG)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True,
                    help="tag=path pairs, e.g. qwen3vl_coord=outputs/matrix/qwen3vl_coord/predictions.jsonl")
    ap.add_argument("--out", default=f"{ROOT}/outputs/judge/batch_requests.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_done", nargs="*", default=[],
                    help="prior batch_results.jsonl file(s); (tag,idx) whose qa AND rel already succeeded are skipped (no re-pay)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # resume: an item counts as done only if BOTH its qa and rel responses PARSE (lenient tag regex);
    # responses whose score tag is malformed are NOT done, so they get re-enqueued (re-judged).
    QA_OK = re.compile(r'<?qa_acc>\s*\d+\s*</qa_acc>')
    REL_OK = re.compile(r'<?relevance_score>\s*\d+\s*</relevance_score>')
    def resp_text(o):
        rr = o.get("response") or {}
        if rr.get("text"):
            return rr["text"]
        for c in (rr.get("candidates") or []):
            for p in ((c.get("content") or {}).get("parts") or []):
                if p.get("text"):
                    return p["text"]
        return ""
    done_keys = set()
    for rf in args.skip_done:
        if os.path.exists(rf):
            got = {}
            for l in open(rf):
                try:
                    o = json.loads(l); k = o.get("key", "")
                    kind = k.split("|")[-1]
                    ok = (QA_OK if kind == "qa" else REL_OK).search(resp_text(o))
                    if o.get("response") and not o.get("error") and ok:
                        got.setdefault("|".join(k.split("|")[:2]), set()).add(kind)
                except Exception:
                    pass
            for base, kinds in got.items():
                if {"qa", "rel"} <= kinds:
                    done_keys.add(base)   # both qa and rel parsed OK
    if done_keys:
        print(f"[resume] skipping {len(done_keys)} already-judged items from {len(args.skip_done)} prior result file(s)")
    n_qa = n_rel = n_item = 0
    with open(args.out, "w") as f:
        for pair in args.preds:
            tag, path = pair.split("=", 1)
            rows = [json.loads(l) for l in open(path)]
            if args.limit:
                rows = rows[:args.limit]
            for r in rows:
                if r.get("error") or not r.get("gt_necessary"):
                    continue
                idx = r["index"]
                if f"{tag}|{idx}" in done_keys:   # already judged in a prior batch -> skip, no re-pay
                    continue
                q = r.get("question", "")
                gt = r.get("standard_answer", "")
                # model answer without bbox markers (both interfaces)
                ans = r.get("answer_text") or remove_bbox(r.get("raw_predict", ""))
                # qa_acc (text only)
                f.write(json.dumps({"key": f"{tag}|{idx}|qa", "request": qa_request(q, gt, ans)},
                                   ensure_ascii=False) + "\n"); n_qa += 1
                # rel (answer + crops of predicted bboxes)
                crops = crops_for(r.get("pred_tuples", []), r.get("pdf_path", ""))
                f.write(json.dumps({"key": f"{tag}|{idx}|rel", "request": rel_request(q, gt, ans, crops)},
                                   ensure_ascii=False) + "\n"); n_rel += 1
                n_item += 1
                if n_item % 200 == 0:
                    print(f"  {n_item} items ({n_qa} qa + {n_rel} rel)", flush=True)
    sz = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}: {n_item} items -> {n_qa+n_rel} requests ({n_qa} qa + {n_rel} rel), {sz:.0f} MB")
    print("judge scores are raw 0-5; SAA = 1(qa_acc>=4 and (rel>=4 or bboxR>=0.6)) computed later.")

if __name__ == "__main__":
    main()
