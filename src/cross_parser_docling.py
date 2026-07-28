#!/usr/bin/env python3
"""E9 cross-parser check, step 1: parse the 440 pub-719 PDFs with Docling (independent of
MinerU) into the same cache format as data/citevqa/mineru_cache:
  <stem>.json = { "<page_idx_0based>": [ {"type", "bbox"(0-1000 xyxy, top-left origin), "text"}, ... ] }

Purpose: test whether the parser ceiling and the flat language-interface recall curves
survive under a layout parser that shares no lineage with the CiteVQA annotation tooling.
Judge-free: only recall and ceiling are computed downstream.

Usage: python src/docling_parse.py [--shard K --num-shards N]
"""
import argparse, json, os, sys, time

ROOT = "${PROJECT_ROOT}"
BENCH = f"{ROOT}/data/citevqa/citevqa_singledoc_pub.jsonl"
OUT_DIR = f"{ROOT}/data/citevqa/docling_cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opts = PdfPipelineOptions()
    opts.do_ocr = True  # default OCR engine (rapidocr in docling 2.x, covers zh + en)
    opts.do_table_structure = False
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

    pdfs = sorted({json.loads(l)["pdf_path"] for l in open(BENCH)})
    pdfs = [p for i, p in enumerate(pdfs) if i % args.num_shards == args.shard]
    print(f"shard {args.shard}/{args.num_shards}: {len(pdfs)} PDFs", flush=True)

    for i, pdf in enumerate(pdfs):
        stem = os.path.splitext(os.path.basename(pdf))[0]
        out = os.path.join(OUT_DIR, f"{stem}.json")
        if os.path.exists(out):
            continue
        t0 = time.time()
        try:
            res = conv.convert(pdf)
            doc = res.document
            pages = {}
            for item, _lvl in doc.iterate_items():
                for prov in getattr(item, "prov", []) or []:
                    pno = prov.page_no  # 1-based
                    page = doc.pages.get(pno)
                    if page is None:
                        continue
                    W, H = page.size.width, page.size.height
                    bb = prov.bbox.to_top_left_origin(H)
                    x1, y1, x2, y2 = bb.l, bb.t, bb.r, bb.b
                    if x2 < x1:
                        x1, x2 = x2, x1
                    if y2 < y1:
                        y1, y2 = y2, y1
                    if W <= 0 or H <= 0 or x2 - x1 <= 0 or y2 - y1 <= 0:
                        continue
                    rec = {
                        "type": str(getattr(item, "label", "unknown")),
                        "bbox": [1000 * x1 / W, 1000 * y1 / H, 1000 * x2 / W, 1000 * y2 / H],
                        "text": getattr(item, "text", "") or "",
                    }
                    pages.setdefault(str(pno - 1), []).append(rec)
            with open(out, "w") as f:
                json.dump(pages, f, ensure_ascii=False)
            nb = sum(len(v) for v in pages.values())
            print(f"[{i+1}/{len(pdfs)}] {stem}: {nb} blocks, {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(pdfs)}] {stem}: FAILED {type(e).__name__}: {e}", flush=True)
            with open(out + ".fail", "w") as f:
                f.write(str(e))


if __name__ == "__main__":
    main()
