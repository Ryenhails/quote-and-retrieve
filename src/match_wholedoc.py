#!/usr/bin/env python3
"""Whole-document image-block embedding matching (NO page-field hard match).

Fixes the dominant remaining-AH cause: the model's stated page is inconsistent
(56% 1-based / 32% 0-based) so page-restricted matching lands on the wrong page.
Here we IGNORE the model's page: embed every MinerU block's IMAGE crop across the
WHOLE document (cached per PDF), then match each quote (text) against the whole-doc
block pool by multimodal cosine + bipartite assignment. The pred page is the matched
block's TRUE page.

Reuses model quotes from eval_nlg (controlled: same quotes/blocks/GT/judge as the
per-page M2-bip run; only candidate SCOPE changes single-page -> whole-doc).
Outputs eval-format predictions for judging.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qr_paths  # noqa: F401,E402  sets PROJECT_ROOT / QR_CACHE / QWEN3VL_EMB_REPO defaults
import sys, json, os, re, argparse
sys.path.insert(0, os.path.expandvars("${QWEN3VL_EMB_REPO}"))
import numpy as np, torch, fitz
from PIL import Image
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
from scipy.optimize import linear_sum_assignment

ROOT=os.path.expandvars("${PROJECT_ROOT}")
EMB_CACHE=os.path.expandvars("${QR_CACHE}/block_emb")
os.makedirs(EMB_CACHE, exist_ok=True)

def parse_quotes(raw):
    t=re.sub(r"^\s*```(?:json)?|```\s*$","",(raw or "").strip(),flags=re.I).strip()
    s,e=t.find("{"),t.rfind("}")
    if s==-1 or e<=s: return []
    try: ev=json.loads(t[s:e+1]).get("evidence") or []
    except: return []
    return [str(x.get("quote","")) for x in ev if isinstance(x,dict) and x.get("quote")]

def crop(page_img, bbox):
    W,H=page_img.size
    x1,y1,x2,y2=bbox[0]/1000*W,bbox[1]/1000*H,bbox[2]/1000*W,bbox[3]/1000*H
    if x2-x1<4 or y2-y1<4: return None
    return page_img.crop((x1,y1,x2,y2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expandvars("${QR_CACHE}/models--Qwen--Qwen3-VL-Embedding-2B/snapshots"))
    ap.add_argument("--pred", default=f"{ROOT}/outputs/eval_nlg/predictions.jsonl")
    ap.add_argument("--bench", default=f"{ROOT}/data/citevqa/citevqa_base_full.jsonl")
    ap.add_argument("--cache", default=f"{ROOT}/data/citevqa/mineru_cache")
    ap.add_argument("--out", default=f"{ROOT}/outputs/eval_nlg_wd/predictions.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args=ap.parse_args()
    mp=args.model
    if os.path.isdir(mp) and os.path.basename(mp)=="snapshots":
        mp=os.path.join(mp, sorted(os.listdir(mp))[0])
    bench={r["index"]:r for r in (json.loads(l) for l in open(args.bench))}
    R_all=[json.loads(l) for l in open(args.pred)]
    R=[r for r in R_all if not r.get("error")]
    if args.limit: R=R[:args.limit]
    print(f"loading embedder {mp}", flush=True)
    emb=Qwen3VLEmbedder(model_name_or_path=mp, torch_dtype=torch.float16, attn_implementation="sdpa")

    @torch.no_grad()
    def embed(items):
        out=[]
        for k in range(0,len(items),48):
            out.append(emb.process(items[k:k+48]).float().cpu().numpy())
        return np.concatenate(out,0) if out else np.zeros((0,2048),dtype=np.float32)

    def block_emb_for_pdf(stem, pdf):
        """-> (E[n,d], pages[n](0-based), bboxes[n,4]); cached on disk."""
        fp=os.path.join(EMB_CACHE, f"{stem}.npz")
        if os.path.exists(fp):
            d=np.load(fp); return d["E"], d["pages"], d["bboxes"]
        cpath=os.path.join(args.cache, f"{stem}.json")
        if not os.path.exists(cpath):
            np.savez(fp, E=np.zeros((0,2048),dtype=np.float32), pages=np.zeros(0,int), bboxes=np.zeros((0,4)))
            return block_emb_for_pdf(stem,pdf)
        cache=json.load(open(cpath))
        doc=fitz.open(pdf); imgs=[]; pages=[]; boxes=[]
        for pidx_s, els in cache.items():
            pidx=int(pidx_s)
            if pidx>=len(doc): continue
            pix=doc[pidx].get_pixmap(matrix=fitz.Matrix(150/72,150/72))
            pim=Image.frombytes("RGB",(pix.width,pix.height),pix.samples)
            for b in els:
                c=crop(pim, b["bbox"])
                if c is None: continue
                imgs.append({"image":c}); pages.append(pidx); boxes.append(b["bbox"])
        doc.close()
        E=embed(imgs) if imgs else np.zeros((0,2048),dtype=np.float32)
        pages=np.array(pages,int); boxes=np.array(boxes,float) if boxes else np.zeros((0,4))
        np.savez(fp, E=E, pages=pages, bboxes=boxes)
        return E, pages, boxes

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dumpmap={}
    for i,r in enumerate(R):
        idx=r["index"]; meta=bench.get(idx)
        quotes=parse_quotes(r.get("raw_predict",""))
        if not meta or not quotes: dumpmap[idx]=[]; continue
        stem=os.path.splitext(os.path.basename(meta["pdf_path"]))[0]
        E,pages,boxes=block_emb_for_pdf(stem, meta["pdf_path"])
        if E.shape[0]==0: dumpmap[idx]=[]; continue
        Q=embed([{"text":q} for q in quotes])         # [m,d]
        S=Q @ E.T                                       # [m,n] cosine (normalized)
        m,n=S.shape
        preds=[]
        if n>=m:
            rows,cols=linear_sum_assignment(-S)         # bipartite, distinct blocks
            for j in cols: preds.append([1, int(pages[j])+1, *[float(v) for v in boxes[j]]])
        else:
            for row in S: j=int(np.argmax(row)); preds.append([1, int(pages[j])+1, *[float(v) for v in boxes[j]]])
        dumpmap[idx]=preds
        if (i+1)%50==0: print(f"[{i+1}/{len(R)}]", flush=True)

    n=0
    with open(args.out,"w") as f:
        for r in R_all:
            if r.get("error"): f.write(json.dumps(r,ensure_ascii=False)+"\n"); continue
            out=dict(r); out["pred_tuples"]=dumpmap.get(r["index"], [])
            f.write(json.dumps(out,ensure_ascii=False)+"\n"); n+=1
    print(f"[dump] whole-doc M2+bip -> {args.out} ({n} items)")

if __name__=="__main__":
    main()
