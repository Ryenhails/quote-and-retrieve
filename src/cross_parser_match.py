#!/usr/bin/env python3
"""E9 cross-parser check, step 2 (GPU): parser ceiling + language-interface recall with
Docling blocks instead of MinerU blocks. Judge-free by design: only geometric recall is
reported, which is exactly what the annotation-alignment concern is about.

For each NL condition, the model quotes are held fixed; Docling block crops are embedded with
the same encoder (cached under block_emb_docling), and the same whole-document one-to-one
assignment produces citations that are scored at page + IoU 0.1/0.3/0.5/0.7.

Usage (under sbatch): python src/cross_parser_match.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qr_paths  # noqa: F401,E402  sets PROJECT_ROOT / QR_CACHE / QWEN3VL_EMB_REPO defaults
import json, os, re, sys, statistics
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, os.path.expandvars("${QWEN3VL_EMB_REPO}"))
import numpy as np
import torch
import fitz
from PIL import Image
from score_citevqa import iou_hit
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
from scipy.optimize import linear_sum_assignment

ROOT = os.path.expandvars("${PROJECT_ROOT}")
BENCH = f"{ROOT}/data/citevqa/citevqa_singledoc_pub.jsonl"
CACHE = f"{ROOT}/data/citevqa/docling_cache"
MATRIX = f"{ROOT}/outputs/matrix"
EMB_CACHE = os.path.expandvars("${QR_CACHE}/block_emb_docling")
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


def crop(page_img, bbox):
    W, H = page_img.size
    x1, y1, x2, y2 = bbox[0] / 1000 * W, bbox[1] / 1000 * H, bbox[2] / 1000 * W, bbox[3] / 1000 * H
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return page_img.crop((x1, y1, x2, y2))


def doc_blocks(stem):
    cpath = os.path.join(CACHE, f"{stem}.json")
    if not os.path.exists(cpath):
        return None
    cache = json.load(open(cpath))
    blocks = []
    for pidx_s, els in cache.items():
        p = int(pidx_s)
        for b in els:
            bb = b.get("bbox")
            if bb and len(bb) == 4:
                blocks.append((p, bb))
    return blocks


def main():
    os.chdir(ROOT)
    os.makedirs(EMB_CACHE, exist_ok=True)
    bench = [json.loads(l) for l in open(BENCH)]
    bench_by_idx = {r["index"]: r for r in bench}

    # ---------------- ceiling ----------------
    page_rec, thr_rec = [], {t: [] for t in THRESH}
    n_missing = 0
    nblocks = []
    for r in bench:
        gt = [(1, b[0], b[1], b[2], b[3], b[4]) for b in r["gt_necessary"]]
        if not gt:
            continue
        stem = os.path.splitext(os.path.basename(r["pdf_path"]))[0]
        bl = doc_blocks(stem)
        if bl is None:
            n_missing += 1
            page_rec.append(0.0)
            for t in THRESH:
                thr_rec[t].append(0.0)
            continue
        nblocks.append(len(bl))
        tup = [(1, p + 1, *bb) for p, bb in bl]
        pset = {x[1] for x in tup}
        page_rec.append(sum(1 for g in gt if g[1] in pset) / len(gt))
        for t in THRESH:
            thr_rec[t].append(sum(1 for g in gt if any(iou_hit(g, x, t) for x in tup)) / len(gt))
    print(f"=== Docling parser ceiling (n={len(page_rec)}, {n_missing} missing parses -> 0; "
          f"median blocks/doc={int(statistics.median(nblocks)) if nblocks else 0}) ===")
    print(f"page {100*statistics.mean(page_rec):.1f}  " +
          "  ".join(f"@{t} {100*statistics.mean(thr_rec[t]):.1f}" for t in THRESH), flush=True)

    # ---------------- embedder ----------------
    mp = os.path.expandvars("${QR_CACHE}/models--Qwen--Qwen3-VL-Embedding-2B/snapshots")
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

    def block_emb_for(stem, pdf):
        fp = os.path.join(EMB_CACHE, f"{stem}.npz")
        if os.path.exists(fp):
            d = np.load(fp)
            return d["E"], d["pages"], d["bboxes"]
        bl = doc_blocks(stem)
        if not bl:
            np.savez(fp, E=np.zeros((0, 2048), dtype=np.float32), pages=np.zeros(0, int),
                     bboxes=np.zeros((0, 4)))
            return block_emb_for(stem, pdf)
        doc = fitz.open(pdf)
        imgs, pages, boxes = [], [], []
        by_page = defaultdict(list)
        for p, bb in bl:
            by_page[p].append(bb)
        for p, bbs in by_page.items():
            if p >= len(doc):
                continue
            pix = doc[p].get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))
            pim = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            for bb in bbs:
                c = crop(pim, bb)
                if c is None:
                    continue
                imgs.append({"image": c})
                pages.append(p)
                boxes.append(bb)
        doc.close()
        E = embed(imgs)
        pages = np.array(pages, int)
        boxes = np.array(boxes, float) if boxes else np.zeros((0, 4))
        np.savez(fp, E=E, pages=pages, bboxes=boxes)
        return E, pages, boxes

    # ---------------- rematch per condition ----------------
    print("\n=== language-interface recall with Docling blocks ===")
    print(f"{'backbone':<26}{'page':>7}" + "".join(f"{'@'+str(t):>7}" for t in THRESH), flush=True)
    for dir_pfx, name in BACKBONES:
        preds = [json.loads(l) for l in open(f"{MATRIX}/{dir_pfx}_nl/predictions.jsonl")]
        valid = [r for r in preds if not r.get("error") and r.get("gt_necessary")]
        agg = defaultdict(list)
        for i, r in enumerate(valid):
            gt = [(1, b[0], b[1], b[2], b[3], b[4]) for b in r["gt_necessary"]]
            quotes = parse_quotes(r.get("raw_predict", ""))
            meta = bench_by_idx.get(r["index"])
            if not gt:
                continue
            stem = os.path.splitext(os.path.basename(meta["pdf_path"]))[0]
            if not quotes:
                for th in ["page"] + THRESH:
                    agg[th].append(0.0)
                continue
            E, pages, boxes = block_emb_for(stem, meta["pdf_path"])
            if E.shape[0] == 0:
                for th in ["page"] + THRESH:
                    agg[th].append(0.0)
                continue
            Q = embed([{"text": q} for q in quotes])
            S = Q @ E.T
            m, n = S.shape
            if n >= m:
                _, cols = linear_sum_assignment(-S)
                sel = cols
            else:
                sel = [int(np.argmax(row)) for row in S]
            ptup = [(1, int(pages[j]) + 1, *[float(v) for v in boxes[j]]) for j in sel]
            gp = {g[1] for g in gt}
            pp = {x[1] for x in ptup}
            agg["page"].append(len(gp & pp) / len(gp))
            for t in THRESH:
                agg[t].append(sum(1 for g in gt if any(iou_hit(g, x, t) for x in ptup)) / len(gt))
            if (i + 1) % 100 == 0:
                print(f"  [{name} {i+1}/{len(valid)}]", flush=True)
        line = f"{name:<26}{100*statistics.mean(agg['page']):>7.1f}"
        for t in THRESH:
            line += f"{100*statistics.mean(agg[t]):>7.1f}"
        print(line, flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
