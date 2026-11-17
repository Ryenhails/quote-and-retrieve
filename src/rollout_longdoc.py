"""GRPO zero-rollout (Part-2): sample G responses per LongDocURL item with the BASE
Qwen3-VL-8B, in the SAME NL-grounding interface as eval (answer + verbatim quotes).

No training here. This produces the rollout set whose reward variance we then measure
(src/reward_longdoc.py) to decide if base behaviour already has enough within-item
signal for a cold-start GRPO on LongDocURL.

Render ONLY the page_window (LongDocURL evidence is 100% inside it); per-page pixel
budget scales down for long windows so G-sampling stays tractable. Output: one row per
item carrying all G raw responses + parsed {answer, evidence:[{page,quote}]}.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qr_paths  # noqa: F401,E402  sets PROJECT_ROOT / QR_CACHE / QWEN3VL_EMB_REPO defaults
import argparse, json, re, random
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent

# identical interface to infer_citevqa_nlground.py (so eval<->train use the same contract)
SYSTEM_PROMPT_NL = """# Document Analysis Assistant

Answer the question based on the provided PDF page images. For EVERY piece of
evidence that supports your answer, QUOTE the exact text span from the document
(verbatim, as it appears), so it can be located in the source.

## Rules
1. Each evidence quote must be a verbatim text span copied from the page (a
   sentence, a table row/cell, a caption, or a note). Do not paraphrase.
2. Quote the evidence at the element level: enough text to identify the region,
   not a single word and not a whole page.
3. If multiple regions support the answer, give multiple quotes.

## Output Format
Return a single JSON object exactly in this form:
{"answer": "<your full answer text>", "evidence": [{"page": p, "quote": "<verbatim text>"}, ...]}
where p is the 0-based page index in the order shown."""


def parse_nl(text):
    """-> (answer, [(shown_page, quote), ...], format_ok)."""
    txt = re.sub(r"^\s*```(?:json)?|```\s*$", "", (text or "").strip(), flags=re.I).strip()
    s, e = txt.find("{"), txt.rfind("}")
    if s == -1 or e <= s:
        return text or "", [], 0
    try:
        obj = json.loads(txt[s:e + 1])
        ans = str(obj.get("answer", ""))
        evs = []
        for ev in obj.get("evidence", []) or []:
            if isinstance(ev, dict) and ev.get("quote"):
                evs.append({"page": int(ev.get("page", 0)), "quote": str(ev["quote"])})
        return ans, evs, 1
    except Exception:
        return text or "", [], 0


def render_window(pdf_path, start_1b, end_1b, max_pixels, dpi=150):
    """render pages [start..end] (1-based inclusive) -> list[PIL], capped to max_pixels."""
    import fitz
    from PIL import Image
    doc = fitz.open(pdf_path); out = []
    npg = len(doc)
    for pid in range(start_1b, min(end_1b, npg) + 1):
        zoom = dpi / 72
        pix = doc[pid - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        if im.width * im.height > max_pixels:
            import math
            s = math.sqrt(max_pixels / (im.width * im.height))
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
        out.append(im)
    doc.close()
    return out


def stratified_sample(recs, n, max_nwin):
    pool = [r for r in recs if r["n_pages_window"] <= max_nwin]
    by = {}
    for r in pool:
        by.setdefault(r["qtype"], []).append(r)
    rng = random.Random(0)
    for v in by.values():
        rng.shuffle(v)
    # proportional allocation across qtypes
    out, qtypes = [], list(by.keys())
    total = sum(len(v) for v in by.values())
    quota = {q: max(1, round(n * len(by[q]) / total)) for q in qtypes}
    for q in qtypes:
        out.extend(by[q][:quota[q]])
    rng.shuffle(out)
    return out[:n]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=f"{ROOT}/data/longdocurl/longdoc_bench.json")
    ap.add_argument("--model", default=os.path.expandvars("${QR_CACHE}/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"))
    ap.add_argument("--out", default=f"{ROOT}/data/longdocurl/rollout/samples.jsonl")
    ap.add_argument("--n_items", type=int, default=160)
    ap.add_argument("--max_nwin", type=int, default=20)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--pixel_budget", type=int, default=10_000_000, help="nwin*max_pixels cap")
    ap.add_argument("--max_pixels_cap", type=int, default=1_000_000)
    ap.add_argument("--min_pixels", type=int, default=65536)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0, help="this shard's index in [0,nshard)")
    ap.add_argument("--nshard", type=int, default=1, help="total shards (split items round-robin)")
    args = ap.parse_args()

    from qwen_vl_utils import process_vision_info
    recs = json.load(open(args.bench))
    items = stratified_sample(recs, args.n_items, args.max_nwin)
    if args.nshard > 1:
        items = items[args.shard::args.nshard]      # deterministic round-robin shard
        print(f"shard {args.shard}/{args.nshard}: {len(items)} items", flush=True)
    if args.limit:
        items = items[: args.limit]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f"sampled {len(items)} items (max_nwin={args.max_nwin}); loading {args.model}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    proc = AutoProcessor.from_pretrained(args.model)

    n_ok = n_err = 0
    with out.open("w") as fout:
        for i, r in enumerate(items):
            try:
                start, end = r["page_window"]
                nwin = r["n_pages_window"]
                max_px = min(args.max_pixels_cap, args.pixel_budget // max(1, nwin))
                imgs = render_window(r["pdf_path"], start, end, max_px)
                content = [{"type": "image", "image": im, "min_pixels": args.min_pixels,
                            "max_pixels": max_px} for im in imgs]
                content.append({"type": "text", "text": f"Question: {r['question']}"})
                messages = [{"role": "system", "content": SYSTEM_PROMPT_NL},
                            {"role": "user", "content": content}]
                text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
                iin, vin = process_vision_info(messages)
                inp = proc(text=[text], images=iin, videos=vin, padding=True, return_tensors="pt").to(model.device)
                gen = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=True,
                                     temperature=args.temperature, top_p=args.top_p,
                                     num_return_sequences=args.G)
                plen = inp["input_ids"].shape[1]
                samples = []
                for g in range(args.G):
                    raw = proc.decode(gen[g, plen:], skip_special_tokens=True).strip()
                    ans, evs, fmt = parse_nl(raw)
                    samples.append({"raw": raw, "answer": ans, "evidence": evs, "format_ok": fmt})
                fout.write(json.dumps({
                    "index": r["index"], "doc_no": r["doc_no"], "pdf_path": r["pdf_path"],
                    "question": r["question"], "gt_answer": r["gt_answer"],
                    "page_window": r["page_window"], "n_pages_window": nwin,
                    "evidence_pages": r["evidence_pages"], "qtype": r["qtype"],
                    "answer_format": r["answer_format"], "n_input_pages": len(imgs),
                    "samples": samples,
                }, ensure_ascii=False) + "\n")
                fout.flush(); n_ok += 1
            except Exception as ex:
                n_err += 1
                fout.write(json.dumps({"index": r["index"], "error": repr(ex)[:300]}, ensure_ascii=False) + "\n")
                fout.flush()
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(items)}] ok={n_ok} err={n_err}", flush=True)
    print(f"DONE rollout ok={n_ok} err={n_err} G={args.G} -> {out}", flush=True)


if __name__ == "__main__":
    main()
