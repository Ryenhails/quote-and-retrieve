"""Convert the refined-490 GRPO set into EasyR1 format.

Per item: render the page_window to PNG files (reused across epochs) and emit one
EasyR1 row: {problem (NL-ground system prompt + <image>*N + question), images (paths),
answer (JSON ground_truth the reward service needs), ...metadata}.

The `answer` ground_truth carries everything the reward service needs to score WITHOUT
re-deriving it: index (-> block_emb cache key), question, gt_answer, doc_no, pdf_path,
page_window. Images are rendered at the same per-page pixel budget as rollout.
"""
import json, os, math, argparse
from pathlib import Path
import fitz
from PIL import Image

ROOT = Path("${PROJECT_ROOT}")

# same NL-grounding contract as rollout/eval (answer + verbatim quotes, JSON)
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


def render_window(pdf_path, start_1b, end_1b, outdir, max_pixels, dpi=150):
    doc = fitz.open(pdf_path); npg = len(doc); paths = []
    os.makedirs(outdir, exist_ok=True)
    for k, pid in enumerate(range(start_1b, min(end_1b, npg) + 1)):
        fp = os.path.join(outdir, f"p{k:02d}.png")
        if not os.path.exists(fp):
            pix = doc[pid - 1].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
            im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            if im.width * im.height > max_pixels:
                s = math.sqrt(max_pixels / (im.width * im.height))
                im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
            im.save(fp)
        paths.append(fp)
    doc.close()
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refined", default=str(ROOT / "data/longdocurl/grpo_refined.jsonl"))
    ap.add_argument("--pages_dir", default=str(ROOT / "data/longdocurl/grpo_pages"))
    ap.add_argument("--out", default=str(ROOT / "data/longdocurl/grpo/train.jsonl"))
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--pixel_budget", type=int, default=10_000_000)
    ap.add_argument("--max_pixels_cap", type=int, default=1_000_000)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.refined)]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for i, r in enumerate(items):
        start, end = r["page_window"]; nwin = r["n_pages_window"]
        max_px = min(args.max_pixels_cap, args.pixel_budget // max(1, nwin))
        paths = render_window(r["pdf_path"], start, end, os.path.join(args.pages_dir, r["index"]), max_px)
        problem = "<image>" * len(paths) + "\n" + SYSTEM_PROMPT_NL + f"\n\nQuestion: {r['question']}"
        ground_truth = json.dumps({
            "index": r["index"], "question": r["question"], "gt_answer": r["gt_answer"],
            "doc_no": r["doc_no"], "pdf_path": r["pdf_path"], "page_window": r["page_window"],
        }, ensure_ascii=False)
        rows.append({"problem": problem, "images": paths, "answer": ground_truth,
                     "qtype": r["qtype"], "n_pages": len(paths)})
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(items)}] rendered", flush=True)

    nval = max(1, int(len(rows) * args.val_frac))
    val, train = rows[:nval], rows[nval:]
    with open(args.out, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    valf = args.out.replace("train", "val")
    with open(valf, "w") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tot_imgs = sum(r["n_pages"] for r in rows)
    print(f"DONE: train={len(train)} val={len(val)}  total page-images={tot_imgs}")
    print(f"  -> {args.out}\n  -> {valf}\n  pages -> {args.pages_dir}")


if __name__ == "__main__":
    main()
