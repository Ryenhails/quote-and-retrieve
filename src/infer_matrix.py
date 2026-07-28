#!/usr/bin/env python3
"""Unified cross-VLM inference client for the P1 matrix (coordinate vs NL grounding).

Talks to a vLLM OpenAI-compatible endpoint, so the SAME client drives every backbone; the
per-model chat template and image handling live inside vLLM. Renders the WHOLE document
(full-page setting) at an adaptive per-page resolution (see docs/protocol_matrix_eval.md):
    per_page_px = min(1e6, BUDGET / n_pages)
Records the resolution used and whether the output parsed. Idempotent: appends per question and
skips indices already written, so a killed job resumes without loss. Does NOT judge; for the NL
interface, run src/match_wholedoc.py afterwards to get bboxes, then score locally + cache for Gemini.
"""
import argparse, json, os, io, base64, re, time
import fitz
from PIL import Image
import requests

# Coordinate interface = CiteVQA official prompt VERBATIM (CiteVQA/prompts/infer_system.txt), so the
# coordinate baseline is the established, published one (with examples, 1-based page, inline <bbox>),
# not a weaker in-house prompt. This reproduces CiteVQA's reported bbox recall for the same models.
SYS_COORD = """# Document Analysis Assistant

Answer the question based on the provided PDF page images, and cite the evidence regions in your answer.

## Evidence Citation Rules

1. Evidence must be at the **element level**: a complete paragraph, a complete table, a complete image, or a complete note. Do not select partial text from a paragraph or a single row from a table, and do not select an entire page or spanning multiple tables/paragraphs. Note: This is very important and will directly affect your score.
2. For **tables and images**, if there are captions or footnotes, they need to be annotated as **separate evidence** with their own bbox, not merged into the table/image bbox.
3. Each piece of cited evidence text should be followed by a `<bbox />` tag indicating the evidence location.
4. When an inference step relies on multiple pieces of evidence, use multiple `<bbox />` tags separately.
5. Pure reasoning/calculation steps do not need `<bbox />`.

## Annotation Format

```
<bbox page="page_number" x1="left" y1="top" x2="right" y2="bottom" />
```

Page numbers start from 1 (note: ignore original page numbers); coordinates are relative coordinates on the page image, range 0-1000.

## Examples

**Question:** What is the net change in the company's precision copper tube production capacity from 2021 to 2024?

**Answer:**

According to the main text, the company's precision copper tube production capacity increased from 798,000 tons in 2021 to 1.31 million tons in 2024:
<bbox page="1" x1="536" y1="65" x2="642" y2="656" />

Therefore, the net change = 1.31 - 0.798 = 0.512 million tons.

Additionally, according to "Table 1: Production line renovation will reduce the company's costs", the per-ton comprehensive cost is expected to decrease by 700 yuan/ton after the production line renovation:
<bbox page="8" x1="584" y1="65" x2="598" y2="371" />
<bbox page="8" x1="598" y1="59" x2="712" y2="670" />

## Final Reminder
Evidence must be a complete paragraph, a complete table, a complete image, or a complete note. Do not select partial rows from a paragraph or a single row from a table, and do not select an entire page or spanning multiple tables/paragraphs. Note: This is very important and will directly affect your score."""

SYS_NL = """# Document Analysis Assistant

Answer the question based on the provided PDF page images. For EVERY piece of evidence that supports your answer, QUOTE the exact text span from the document (verbatim, as it appears), so it can be located in the source.

## Rules
1. Each evidence quote must be a verbatim text span copied from the page (a sentence, a table row/cell, a caption, or a note). Do not paraphrase.
2. Quote the evidence at the element level: enough text to identify the region, not a single word and not a whole page.
3. If multiple regions support the answer, give multiple quotes.

## Output Format
Return a single JSON object exactly in this form:
{"answer": "<your full answer text>", "evidence": [{"page": p, "quote": "<verbatim text>"}, ...]}
where p is the 0-based page index in the order shown."""

# official CiteVQA inline format: <bbox page="N" x1=".." y1=".." x2=".." y2=".."/> (page 1-based, 0-1000)
# STRICT: only <bbox> (matches CiteVQA's parser). A <box> typo / NL prose / HTML is a format failure by
# design — models failing to emit the coordinate format IS the phenomenon we measure, so we do not rescue it.
BBOX_OFFICIAL_RE = re.compile(r'<bbox\s+page="(\d+)"\s+x1="([\d.]+)"\s+y1="([\d.]+)"\s+x2="([\d.]+)"\s+y2="([\d.]+)"\s*/?>', re.S)
CIT_RE = re.compile(r'"bbox_2d"\s*:\s*\[([^\]]+)\][^{}]*?"page"\s*:\s*(\d+)', re.S)  # JSON fallback
# InternVL native grounding format: <box>[[x1,y1,x2,y2]]</box> (0-1000, no explicit page)
BOX_RE = re.compile(r'<box>\s*\[\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)', re.S)

def remove_bbox(text):
    return re.sub(r'<bbox[^>]*/?>', '', text or '').strip()

def render_all_pages(pdf_path, per_page_px):
    """Render EVERY page targeting per_page_px pixels; return list of PIL images."""
    doc = fitz.open(pdf_path)
    imgs = []
    for pg in doc:
        w, h = pg.rect.width, pg.rect.height
        zoom = (per_page_px / (w * h)) ** 0.5 if w * h > 0 else 1.0
        zoom = min(zoom, 300 / 72)  # never upscale beyond 300 dpi
        pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    doc.close()
    return imgs

def to_data_url(img, fmt="JPEG", q=85):
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def call(url, model, sysmsg, images, question, max_tokens=1536, timeout=900):
    content = [{"type": "image_url", "image_url": {"url": to_data_url(im)}} for im in images]
    content.append({"type": "text", "text": question})
    # repetition_penalty breaks the greedy strip-grid loop some VLMs fall into on the coord
    # interface (tiling the page with dozens of identical horizontal strips). Standard for
    # grounding decoding; keeps the output the model's genuine attempt, not a decode pathology.
    body = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
            "repetition_penalty": 1.15,
            "messages": [{"role": "system", "content": sysmsg},
                         {"role": "user", "content": content}]}
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=timeout)
    return r

def parse_coord(text, n_pages):
    """-> pred_tuples [(1, page_1based, x1,y1,x2,y2)], format_ok. Page is 1-based (official prompt);
    coords 0-1000 for all six backbones (verified against each model's grounding docs), no normalization."""
    text = text or ""
    preds = []
    for m in BBOX_OFFICIAL_RE.findall(text):  # official <bbox page="N".../>, 1-based
        try:
            p = int(m[0]); xs = [float(v) for v in m[1:5]]
            if 1 <= p <= n_pages:
                preds.append((1, p, xs[0], xs[1], xs[2], xs[3]))
        except Exception:
            pass
    if not preds:  # JSON fallback (page 1-based, as instructed by the official prompt)
        for m in CIT_RE.findall(text):
            try:
                xs = [float(v) for v in m[0].split(",")]; p = int(m[1])
                if len(xs) == 4 and 1 <= p <= n_pages:
                    preds.append((1, p, xs[0], xs[1], xs[2], xs[3]))
            except Exception:
                pass
    if not preds:  # InternVL native <box>[[x1,y1,x2,y2]]</box> (0-1000, single-image, page 1)
        for m in BOX_RE.findall(text):
            try:
                xs = [float(v) for v in m]
                if len(xs) == 4:
                    preds.append((1, 1, xs[0], xs[1], xs[2], xs[3]))
            except Exception:
                pass
    fmt = 1 if ('<bbox' in text or 'bbox_2d' in text or '<box>' in text) else 0
    return preds, fmt

def get_answer(text):
    m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text or "", re.S)
    return m.group(1) if m else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)          # vLLM OpenAI base, e.g. http://127.0.0.1:8000
    ap.add_argument("--served_model", required=True) # --served-model-name given to vLLM
    ap.add_argument("--interface", required=True, choices=["coord", "nl"])
    ap.add_argument("--bench", default="data/citevqa/citevqa_singledoc_pub.jsonl")
    ap.add_argument("--budget", type=float, required=True)  # BUDGET_b total pixel budget
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=1536,
                    help="output token cap; use 4096 for thinking models (Qwen3.5) so the JSON/bbox is not "
                         "truncated by long <think> (matches CiteVQA's 4096 for the Qwen3.5 series)")
    args = ap.parse_args()

    R = [json.loads(l) for l in open(args.bench)]
    if args.limit:
        R = R[:args.limit]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l)["index"])
            except Exception:
                pass
    sysmsg = SYS_COORD if args.interface == "coord" else SYS_NL
    print(f"[{args.interface}] {len(R)} items, {len(done)} already done, budget={args.budget:.0f}px", flush=True)

    with open(args.out, "a") as f:
        for i, r in enumerate(R):
            if r["index"] in done:
                continue
            npg = r["n_pdf_pages"]
            ppx = min(1_000_000.0, args.budget / max(npg, 1))
            rec = {"index": r["index"], "language": r["language"], "qtype": r["qtype"],
                   "n_pdf_pages": npg, "n_input_pages": npg,
                   "gt_necessary": r["gt_necessary"], "gt_all": r["gt_all"],
                   "standard_answer": r["standard_answer"], "question": r["question"],
                   "pdf_path": r["pdf_path"], "per_page_px": round(ppx)}
            cur = ppx
            for attempt in range(4):
                try:
                    imgs = render_all_pages(r["pdf_path"], cur)
                    resp = call(args.url, args.served_model, sysmsg, imgs, r["question"], max_tokens=args.max_tokens)
                    if resp.status_code == 200:
                        txt = resp.json()["choices"][0]["message"]["content"]
                        rec["raw_predict"] = txt
                        rec["per_page_px"] = round(cur)
                        if args.interface == "coord":
                            rec["answer_text"] = remove_bbox(txt)   # official format: answer = text minus <bbox> tags
                            preds, fmt = parse_coord(txt, npg)
                            rec["pred_tuples"] = preds
                            rec["format_ok"] = fmt
                        else:
                            rec["answer_text"] = get_answer(txt)
                            rec["format_ok"] = 1 if '"evidence"' in txt else 0
                        break
                    elif resp.status_code == 400 and "context" in resp.text.lower():
                        cur = cur / 2.0  # context overflow -> halve resolution, retry
                        continue
                    else:
                        rec["error"] = f"http{resp.status_code}:{resp.text[:120]}"
                        break
                except Exception as e:
                    rec["error"] = str(e)[:150]
                    time.sleep(2)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"[{args.interface}] {i+1}/{len(R)}", flush=True)
    print(f"[{args.interface}] done -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
