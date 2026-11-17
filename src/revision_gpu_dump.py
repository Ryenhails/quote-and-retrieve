#!/usr/bin/env python3
"""Camera-ready GPU dump, one V100.  Two things the CPU analyses need that the frozen
outputs do not contain:

(a) q-only and q+answer retrieval BOXES per question (src/retrieval_baselines.py stored only the
    per-question recall in outputs/controls/e1_perq_*.jsonl).  Same encoder, same cached block
    embeddings, same budget k = number of quotes that backbone emitted on that question.
    -> outputs/revision/qa_boxes_<backbone>.jsonl
       {"index", "k", "q_tuples": [[1,page,x1,y1,x2,y2],...], "qa_tuples": [...], "rec_q", "rec_qa"}
    Used for:  side-by-side cases, exact (not oracle-max) quote/q+a union.

(b) per-necessary-GT-element bucket labels for base Qwen3-VL-8B lang. and 8B+GRPO lang.
    (src/diag_rerank_headroom.py logic, plus parser reachability and containment), joined with
    question type / language / annotated element type.
    -> outputs/revision/buckets_<cond>.jsonl
       {"index","qtype","language","elem_type","bucket","contain"}   bucket in
       {matched@1, recoverable@5, not_retrieved, parser_unreachable}
    Used for:  failure analysis by question type.

Run under sbatch: slurm/revision_gpu_dump.sh
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qr_paths  # noqa: F401,E402  sets PROJECT_ROOT / QR_CACHE / QWEN3VL_EMB_REPO defaults
import json, os, re, sys
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, os.path.expandvars("${QWEN3VL_EMB_REPO}"))
import numpy as np
import torch
from score_citevqa import iou_hit
from mineru_ceiling import blocks_for
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

ROOT = os.path.expandvars("${PROJECT_ROOT}")
BENCH = f"{ROOT}/data/citevqa/citevqa_singledoc_pub.jsonl"
MATRIX = f"{ROOT}/outputs/matrix"
CITE = f"{ROOT}/CiteVQA/data/validation/CiteVQA.json"
MINERU = f"{ROOT}/data/citevqa/mineru_cache"
EMB_CACHE = os.path.expandvars("${QR_CACHE}/block_emb")
OUT_DIR = os.environ.get("QR_REVISION_DIR", f"{ROOT}/outputs/revision")

BACKBONES = [
    ("qwen3vl", "Qwen3-VL-8B"),
    ("qwen3vl_30b", "Qwen3-VL-30B-A3B"),
    ("qwen35_9b", "Qwen3.5-9B"),
    ("qwen35_27b", "Qwen3.5-27B"),
    ("gemma3", "Gemma-3-12B"),
    ("gemma4_31b", "Gemma-4-31B"),
    ("grpo_global_step_144_lr1e5", "8B + GRPO"),
]
BUCKET_CONDS = [("qwen3vl", "base"), ("grpo_global_step_144_lr1e5", "grpo")]
TYPE_GROUP = {"text": "text", "list": "text", "ref_text": "text", "code": "text",
              "title": "text", "header": "text", "footer": "text",
              "table": "table", "table_caption": "table", "table_footnote": "table",
              "image": "figure", "image_caption": "figure", "image_footnote": "figure",
              "equation": "other", "page_footnote": "other"}


def parse_obj(raw):
    t = re.sub(r"^\s*```(?:json)?|```\s*$", "", (raw or "").strip(), flags=re.I).strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return None


def parse_quotes(raw):
    o = parse_obj(raw)
    if not o:
        return []
    return [str(x.get("quote", "")) for x in (o.get("evidence") or []) if isinstance(x, dict) and x.get("quote")]


def parse_answer(raw):
    o = parse_obj(raw)
    return str(o.get("answer", "")) if o else ""


def block_emb(stem):
    fp = os.path.join(EMB_CACHE, f"{stem}.npz")
    if not os.path.exists(fp):
        return None
    d = np.load(fp)
    return d["E"], d["pages"], d["bboxes"]


def rec05(gt, preds):
    return sum(1 for g in gt if any(iou_hit(g, tuple(p)) for p in preds)) / len(gt)


def containment(g, blocks):
    """max over blocks of |g ∩ b| / |g| (same page)."""
    _, gp, gx1, gy1, gx2, gy2 = g
    ag = max(1e-6, (gx2 - gx1) * (gy2 - gy1))
    best = 0.0
    for b in blocks:
        if b[1] != gp:
            continue
        ix1, iy1 = max(gx1, b[2]), max(gy1, b[3])
        ix2, iy2 = min(gx2, b[4]), min(gy2, b[5])
        if ix2 > ix1 and iy2 > iy1:
            best = max(best, (ix2 - ix1) * (iy2 - iy1) / ag)
    return best


def main():
    os.chdir(ROOT)
    os.makedirs(OUT_DIR, exist_ok=True)
    # the HF cache moved from cache/models--* to cache/hub/models--* after the paper runs; try both
    mp = None
    for base in [os.path.expandvars("${QR_CACHE}/hub"), os.path.expandvars("${QR_CACHE}")]:
        cand = os.path.join(base, "models--Qwen--Qwen3-VL-Embedding-2B", "snapshots")
        if os.path.isdir(cand):
            mp = os.path.join(cand, sorted(os.listdir(cand))[0]); break
    if mp is None:
        raise FileNotFoundError("Qwen3-VL-Embedding-2B snapshot not found under cache/hub or cache")
    print(f"loading embedder {mp}", flush=True)
    emb = Qwen3VLEmbedder(model_name_or_path=mp, torch_dtype=torch.float16, attn_implementation="sdpa")

    @torch.no_grad()
    def embed(items):
        out = []
        for k in range(0, len(items), 48):
            out.append(emb.process(items[k:k + 48]).float().cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, 2048), dtype=np.float32)

    bench = [json.loads(l) for l in open(BENCH)]
    by_idx = {r["index"]: r for r in bench}
    stem_of = {r["index"]: os.path.splitext(os.path.basename(r["pdf_path"]))[0] for r in bench}
    q_emb = {r["index"]: v for r, v in zip(bench, embed([{"text": r["question"]} for r in bench]))}

    # ------------------------------------------------ (a) q-only / q+answer boxes
    print("\n=== (a) retrieval boxes: q-only and q+answer, budget-matched ===", flush=True)
    for dir_pfx, name in BACKBONES:
        nl = [json.loads(l) for l in open(f"{MATRIX}/{dir_pfx}_nl/predictions.jsonl")]
        valid = [r for r in nl if not r.get("error") and r.get("gt_necessary")]
        idxs, budgets, texts = [], [], []
        for r in valid:
            quotes = parse_quotes(r.get("raw_predict", ""))
            ans = parse_answer(r.get("raw_predict", "")) or r.get("answer_text", "") or ""
            idxs.append(r["index"])
            budgets.append(max(1, len(quotes)))
            texts.append((by_idx[r["index"]]["question"] + "\n" + ans)[:4000])
        qa_emb = embed([{"text": t} for t in texts])
        rows, agg_q, agg_qa = [], [], []
        for pos, i in enumerate(idxs):
            meta = by_idx[i]
            gt = [(1, b[0], b[1], b[2], b[3], b[4]) for b in meta["gt_necessary"]]
            be = block_emb(stem_of[i])
            if be is None or be[0].shape[0] == 0:
                rows.append({"index": i, "k": budgets[pos], "q_tuples": [], "qa_tuples": [], "rec_q": 0.0, "rec_qa": 0.0})
                agg_q.append(0.0); agg_qa.append(0.0)
                continue
            E, pages, boxes = be
            k = budgets[pos]
            oq = np.argsort(-(q_emb[i] @ E.T))[:k]
            oqa = np.argsort(-(qa_emb[pos] @ E.T))[:k]
            tq = [[1, int(pages[j]) + 1, *[float(v) for v in boxes[j]]] for j in oq]
            tqa = [[1, int(pages[j]) + 1, *[float(v) for v in boxes[j]]] for j in oqa]
            rq, rqa = rec05(gt, tq), rec05(gt, tqa)
            rows.append({"index": i, "k": k, "q_tuples": tq, "qa_tuples": tqa, "rec_q": rq, "rec_qa": rqa})
            agg_q.append(rq); agg_qa.append(rqa)
        fp = f"{OUT_DIR}/qa_boxes_{dir_pfx}.jsonl"
        with open(fp, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"{name:<20} n={len(rows)}  q-only@0.5={100*np.mean(agg_q):.1f}  q+a@0.5={100*np.mean(agg_qa):.1f}  -> {fp}", flush=True)

    # ------------------------------------------------ (b) per-element buckets
    print("\n=== (b) per-element buckets (base vs GRPO, language interface) ===", flush=True)
    cite = {it["index"]: it for it in json.load(open(CITE))}
    for dir_pfx, tag in BUCKET_CONDS:
        wd = [json.loads(l) for l in open(f"{MATRIX}/{dir_pfx}_nl_wd/predictions.jsonl")]
        valid = [r for r in wd if not r.get("error") and r.get("gt_necessary")]
        out_rows, cnt = [], defaultdict(int)
        blk_cache = {}
        for n_done, r in enumerate(valid):
            i = r["index"]; meta = by_idx[i]
            gt = [(1, b[0], b[1], b[2], b[3], b[4]) for b in meta["gt_necessary"]]
            ev = [e for e in cite.get(i, {}).get("Evidence", []) if e.get("necessity") == "necessary"]
            types = [TYPE_GROUP.get(e.get("type", "other"), "other") for e in ev] if len(ev) == len(gt) else ["?"] * len(gt)
            pdf = meta["pdf_path"]
            if pdf not in blk_cache:
                blk_cache[pdf] = blocks_for(pdf, MINERU) or []
            blocks = blk_cache[pdf]
            preds1 = [tuple(t) for t in r.get("pred_tuples", [])]
            quotes = parse_quotes(r.get("raw_predict", ""))
            preds5 = []
            be = block_emb(stem_of[i])
            if quotes and be is not None and be[0].shape[0] > 0:
                E, pages, boxes = be
                S = embed([{"text": q} for q in quotes]) @ E.T
                kk = min(5, E.shape[0])
                for row in S:
                    for j in np.argsort(-row)[:kk]:
                        preds5.append((1, int(pages[int(j)]) + 1, *[float(v) for v in boxes[int(j)]]))
            for g, ty in zip(gt, types):
                reachable = any(iou_hit(g, b) for b in blocks)
                if any(iou_hit(g, p) for p in preds1):
                    bucket = "matched@1"
                elif any(iou_hit(g, p) for p in preds5):
                    bucket = "recoverable@5"
                elif not reachable:
                    bucket = "parser_unreachable"
                else:
                    bucket = "not_retrieved"
                cnt[bucket] += 1
                out_rows.append({"index": i, "qtype": meta["qtype"], "language": meta["language"],
                                 "elem_type": ty, "bucket": bucket,
                                 "contain": round(containment(g, blocks), 3)})
            if (n_done + 1) % 100 == 0:
                print(f"  [{tag} {n_done+1}/{len(valid)}]", flush=True)
        fp = f"{OUT_DIR}/buckets_{tag}.jsonl"
        with open(fp, "w") as f:
            for row in out_rows:
                f.write(json.dumps(row) + "\n")
        tot = len(out_rows)
        print(f"{tag}: N={tot}  " + "  ".join(f"{k}={100*v/tot:.1f}" for k, v in sorted(cnt.items())) + f"  -> {fp}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
