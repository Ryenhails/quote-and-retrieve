#!/usr/bin/env python3
"""E1 retrieval-only baselines + E8 assigned-similarity export (GPU job).

E1: how much recall does the retriever achieve WITHOUT model quotes?
  (i)  question-only: embed the question text, retrieve top-k blocks from the cached
       whole-document image-crop embeddings (same encoder, same candidate set).
       k is budget-matched per backbone (the number of citations that backbone's NL run
       emitted for that question); fixed k in {1,3,5} reported as a model-free curve.
  (ii) question+answer: embed question + the backbone's own answer text (from the NL run).
Scored identically to the main table (page + IoU 0.1/0.3/0.5/0.7 over necessary GT).

E8: for each NL condition, re-embed the quotes and export the cosine similarity of every
assigned (quote, block) pair, plus whether that block hits a GT element at IoU 0.5.
Supports a judge-free selective-prediction curve (abstain below a similarity threshold).

Run under sbatch on one GPU (slurm/controls_embed.sh).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qr_paths  # noqa: F401,E402  sets PROJECT_ROOT / QR_CACHE / QWEN3VL_EMB_REPO defaults
import argparse, json, os, re, sys, statistics
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, os.path.expandvars("${QWEN3VL_EMB_REPO}"))
import numpy as np
import torch
from score_citevqa import iou_hit
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
from scipy.optimize import linear_sum_assignment

ROOT = os.path.expandvars("${PROJECT_ROOT}")
BENCH = f"{ROOT}/data/citevqa/citevqa_singledoc_pub.jsonl"
MATRIX = f"{ROOT}/outputs/matrix"
EMB_CACHE = os.path.expandvars("${QR_CACHE}/block_emb")
OUT_DIR = f"{ROOT}/outputs/controls"
THRESH = [0.1, 0.3, 0.5, 0.7]

BACKBONES = [
    ("qwen3vl", "Qwen3-VL-8B"),
    ("qwen3vl_30b", "Qwen3-VL-30B-A3B"),
    ("qwen35_9b", "Qwen3.5-9B"),
    ("qwen35_27b", "Qwen3.5-27B"),
    ("gemma3", "Gemma-3-12B"),
    ("gemma4_31b", "Gemma-4-31B"),
    ("grpo_global_step_144_lr1e5", "GRPO (8B) NL"),
]


def parse_quotes(raw):
    t = re.sub(r"^\s*```(?:json)?|```\s*$", "", (raw or "").strip(), flags=re.I).strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e <= s:
        return []
    try:
        ev = json.loads(t[s:e + 1]).get("evidence") or []
    except Exception:
        return []
    return [str(x.get("quote", "")) for x in ev if isinstance(x, dict) and x.get("quote")]


def parse_answer(raw):
    t = re.sub(r"^\s*```(?:json)?|```\s*$", "", (raw or "").strip(), flags=re.I).strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e <= s:
        return ""
    try:
        return str(json.loads(t[s:e + 1]).get("answer", ""))
    except Exception:
        return ""


def block_emb(stem):
    fp = os.path.join(EMB_CACHE, f"{stem}.npz")
    if not os.path.exists(fp):
        return None
    d = np.load(fp)
    return d["E"], d["pages"], d["bboxes"]


def rec_multi(gt, preds):
    out = {}
    gp = {g[1] for g in gt}
    pp = {p[1] for p in preds}
    out["page"] = len(gp & pp) / len(gp)
    for t in THRESH:
        out[t] = sum(1 for g in gt if any(iou_hit(g, tuple(p), t) for p in preds)) / len(gt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expandvars("${QR_CACHE}/models--Qwen--Qwen3-VL-Embedding-2B/snapshots"))
    args = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(OUT_DIR, exist_ok=True)
    mp = args.model
    if os.path.isdir(mp) and os.path.basename(mp) == "snapshots":
        mp = os.path.join(mp, sorted(os.listdir(mp))[0])
    print(f"loading embedder {mp}", flush=True)
    emb = Qwen3VLEmbedder(model_name_or_path=mp, torch_dtype=torch.float16, attn_implementation="sdpa")

    @torch.no_grad()
    def embed(items):
        out = []
        for k in range(0, len(items), 48):
            out.append(emb.process(items[k:k + 48]).float().cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, 2048), dtype=np.float32)

    bench = [json.loads(l) for l in open(BENCH)]
    bench_by_idx = {r["index"]: r for r in bench}

    # ---------------- E1(i) question-only, fixed k ----------------
    print("\n=== E1(i) question-only retrieval, fixed k ===", flush=True)
    q_emb = embed([{"text": r["question"]} for r in bench])
    fixed = {k: defaultdict(list) for k in [1, 3, 5]}
    q_rank_cache = {}
    for i, r in enumerate(bench):
        gt = [(1, b[0], b[1], b[2], b[3], b[4]) for b in r["gt_necessary"]]
        stem = os.path.splitext(os.path.basename(r["pdf_path"]))[0]
        be = block_emb(stem)
        if be is None or be[0].shape[0] == 0 or not gt:
            for k in fixed:
                for th in ["page"] + THRESH:
                    fixed[k][th].append(0.0)
            continue
        E, pages, boxes = be
        sims = q_emb[i] @ E.T
        order = np.argsort(-sims)
        q_rank_cache[r["index"]] = (order, pages, boxes, sims)
        for k in fixed:
            preds = [(1, int(pages[j]) + 1, *[float(v) for v in boxes[j]]) for j in order[:k]]
            m = rec_multi(gt, preds)
            for th in ["page"] + THRESH:
                fixed[k][th].append(m[th])
    hdr = f"{'k':>3}{'page':>7}" + "".join(f"{'@'+str(t):>7}" for t in THRESH)
    print(hdr)
    for k in [1, 3, 5]:
        line = f"{k:>3}{100*statistics.mean(fixed[k]['page']):>7.1f}"
        for t in THRESH:
            line += f"{100*statistics.mean(fixed[k][t]):>7.1f}"
        print(line, flush=True)

    # ---------------- E1 budget-matched + (ii) question+answer ----------------
    print("\n=== E1 budget-matched (k = that backbone's citations on that question) ===", flush=True)
    print(f"{'backbone':<26}{'variant':<12}{'page':>7}" + "".join(f"{'@'+str(t):>7}" for t in THRESH), flush=True)
    for dir_pfx, name in BACKBONES:
        nl = [json.loads(l) for l in open(f"{MATRIX}/{dir_pfx}_nl/predictions.jsonl")]
        valid = [r for r in nl if not r.get("error") and r.get("gt_necessary")]
        answers, budgets, idxs = [], [], []
        for r in valid:
            quotes = parse_quotes(r.get("raw_predict", ""))
            budgets.append(max(1, len(quotes)))
            answers.append(parse_answer(r.get("raw_predict", "")) or r.get("answer_text", "") or "")
            idxs.append(r["index"])
        qa_emb = embed([{"text": (bench_by_idx[i]["question"] + "\n" + a)[:4000]} for i, a in zip(idxs, answers)])
        agg = {v: defaultdict(list) for v in ["q-only", "q+answer"]}
        perq_rows = []
        for pos, i in enumerate(idxs):
            meta = bench_by_idx[i]
            gt = [(1, b[0], b[1], b[2], b[3], b[4]) for b in meta["gt_necessary"]]
            if not gt:
                continue
            cached = q_rank_cache.get(i)
            stem = os.path.splitext(os.path.basename(meta["pdf_path"]))[0]
            be = block_emb(stem)
            if be is None or be[0].shape[0] == 0:
                for v in agg:
                    for th in ["page"] + THRESH:
                        agg[v][th].append(0.0)
                continue
            E, pages, boxes = be
            k = budgets[pos]
            if cached is not None:
                order = cached[0]
            else:
                order = np.argsort(-(q_emb[[b["index"] for b in bench].index(i)] @ E.T))
            preds = [(1, int(pages[j]) + 1, *[float(v) for v in boxes[j]]) for j in order[:k]]
            m = rec_multi(gt, preds)
            for th in ["page"] + THRESH:
                agg["q-only"][th].append(m[th])
            sims2 = qa_emb[pos] @ E.T
            order2 = np.argsort(-sims2)
            preds2 = [(1, int(pages[j]) + 1, *[float(v) for v in boxes[j]]) for j in order2[:k]]
            m2 = rec_multi(gt, preds2)
            for th in ["page"] + THRESH:
                agg["q+answer"][th].append(m2[th])
            perq_rows.append({"index": i, "n_gt": len(gt), "k": k,
                              "q_only": m[0.5], "qa": m2[0.5]})
        with open(f"{OUT_DIR}/e1_perq_{dir_pfx}.jsonl", "w") as f:
            for row in perq_rows:
                f.write(json.dumps(row) + "\n")
        for v in ["q-only", "q+answer"]:
            line = f"{name:<26}{v:<12}{100*statistics.mean(agg[v]['page']):>7.1f}"
            for t in THRESH:
                line += f"{100*statistics.mean(agg[v][t]):>7.1f}"
            print(line, flush=True)

    # ---------------- E8 assigned-pair similarity export ----------------
    if os.environ.get("E1_ONLY"):
        print("\nE1_ONLY set; skipping E8 export.", flush=True)
        return
    print("\n=== E8 similarity export for selective-prediction curve ===", flush=True)
    for dir_pfx, name in BACKBONES:
        nl = [json.loads(l) for l in open(f"{MATRIX}/{dir_pfx}_nl/predictions.jsonl")]
        valid = [r for r in nl if not r.get("error") and r.get("gt_necessary")]
        out_rows = []
        for r in valid:
            quotes = parse_quotes(r.get("raw_predict", ""))
            meta = bench_by_idx.get(r["index"])
            if not quotes or not meta:
                continue
            gt = [(1, b[0], b[1], b[2], b[3], b[4]) for b in meta["gt_necessary"]]
            stem = os.path.splitext(os.path.basename(meta["pdf_path"]))[0]
            be = block_emb(stem)
            if be is None or be[0].shape[0] == 0:
                continue
            E, pages, boxes = be
            Q = embed([{"text": q} for q in quotes])
            S = Q @ E.T
            m, n = S.shape
            pairs = []
            if n >= m:
                rows, cols = linear_sum_assignment(-S)
                pairs = [(int(a), int(b)) for a, b in zip(rows, cols)]
            else:
                pairs = [(qi, int(np.argmax(S[qi]))) for qi in range(m)]
            for qi, j in pairs:
                pred = (1, int(pages[j]) + 1, *[float(v) for v in boxes[j]])
                hit = any(iou_hit(g, pred) for g in gt)
                out_rows.append({"index": r["index"], "sim": float(S[qi, j]), "hit": int(hit)})
        fp = f"{OUT_DIR}/sims_{dir_pfx}.jsonl"
        with open(fp, "w") as f:
            for row in out_rows:
                f.write(json.dumps(row) + "\n")
        print(f"{name}: {len(out_rows)} assigned pairs -> {fp}", flush=True)

    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
