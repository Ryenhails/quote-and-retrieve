#!/usr/bin/env python3
"""Build the PUB version of the CiteVQA single-document evaluation benchmark.

Two labelled benchmarks now exist:
  small version : data/citevqa/citevqa_base_full.jsonl  (498, single-doc, <=48 pages)
  pub version   : data/citevqa/citevqa_singledoc_pub.jsonl  (this script, single-doc, no page cap)

The pub version applies the same record schema and the same quality filters as
build_citevqa_eval.py, namely single-document, a downloadable PDF, at least one necessary
evidence element, all ground-truth pages inside the PDF, and a passing text-alignment check.
The only difference from the small version is that the pub version removes the 48-page cap, so
it covers every usable single-document question, including long documents. This is the realistic
single-document setup that we report on, and the small version remains available for quick checks.
"""
import json, csv
from collections import Counter
import numpy as np
import fitz
from build_citevqa_eval import verify_alignment, to_xyxy, CV, ROOT

def main():
    data = json.load(open(CV / "data/validation/CiteVQA.json"))
    csv_bytes = {}
    with open(CV / "data/download/pdf_source.csv") as f:
        for row in csv.DictReader(f):
            try:
                csv_bytes[row["track_id"] + ".pdf"] = int(row["content_length"])
            except (ValueError, KeyError):
                pass

    pool, n_mismatch, n_unver_drop, n_nopdf, n_nonecess, n_pageoob = [], 0, 0, 0, 0, 0
    ev_types = Counter()
    for d in data:
        if d["dataset_type"] != "Single-Doc":
            continue
        pdf = CV / d["PDF_Source"][0]
        if not (pdf.exists() and open(pdf, "rb").read(4) == b"%PDF"):
            n_nopdf += 1
            continue
        necessary, allboxes, pages, types = [], [], set(), []
        for e in d["Evidence"]:
            box = [e["source_page_id"]] + to_xyxy(e["bbox"])
            allboxes.append(box)
            pages.add(e["source_page_id"])
            types.append(e.get("type", "unknown"))
            if e["necessity"] == "necessary":
                necessary.append(box)
        if not necessary:
            n_nonecess += 1
            continue
        try:
            doc = fitz.open(pdf); n_pdf = len(doc)
        except Exception:
            n_nopdf += 1
            continue
        if max(pages) > n_pdf:
            doc.close(); n_pageoob += 1
            continue
        align = verify_alignment(doc, d)
        doc.close()
        if align == "mismatch":
            n_mismatch += 1
            continue
        if align == "unverifiable" and csv_bytes.get(pdf.name) not in (None, pdf.stat().st_size):
            n_unver_drop += 1
            continue
        for t in types:
            ev_types[t] += 1
        pool.append({
            "alignment": align, "n_pdf_pages": n_pdf, "index": d["index"],
            "question": d["Question"], "standard_answer": d["Standard_Answer"],
            "language": d["language"], "qtype": d["Question_Type"],
            "pdf_path": str(pdf), "gt_pages": sorted(pages),
            "gt_necessary": necessary, "gt_all": allboxes,
        })

    out = ROOT / "data/citevqa/citevqa_singledoc_pub.jsonl"
    with out.open("w") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- composition report ----
    pages = np.array([r["n_pdf_pages"] for r in pool])
    nec = np.array([len(r["gt_necessary"]) for r in pool])
    alln = np.array([len(r["gt_all"]) for r in pool])
    rep = []
    rep.append(f"PUB version single-doc eval bench: {len(pool)} questions -> {out}")
    rep.append(f"dropped: no/invalid PDF={n_nopdf}  no-necessary-evidence={n_nonecess}  "
               f"GT-page-out-of-range={n_pageoob}  text-mismatch={n_mismatch}  "
               f"unverifiable-not-byte-exact={n_unver_drop}")
    rep.append(f"language: {dict(Counter(r['language'] for r in pool))}")
    rep.append(f"question type: {dict(Counter(r['qtype'] for r in pool))}")
    rep.append(f"alignment status: {dict(Counter(r['alignment'] for r in pool))}")
    rep.append(f"pages: median={np.median(pages):.0f} mean={pages.mean():.0f} "
               f"p90={np.percentile(pages,90):.0f} max={pages.max()} | <=48pg={(pages<=48).mean()*100:.0f}%")
    rep.append(f"evidence per Q: necessary mean={nec.mean():.2f} (max {nec.max()}) | "
               f"all mean={alln.mean():.2f} (max {alln.max()})")
    rep.append(f"evidence element types (over all elements): {dict(ev_types.most_common())}")
    rep.append(f"unique source PDFs: {len(set(r['pdf_path'] for r in pool))}")
    report = "\n".join(rep)
    print(report)
    (ROOT / "data/citevqa/citevqa_singledoc_pub_composition.txt").write_text(report + "\n")

if __name__ == "__main__":
    main()
