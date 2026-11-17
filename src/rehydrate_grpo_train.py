#!/usr/bin/env python3
"""Rebuild the GRPO training set from the released question identifiers and LongDocURL.

release/grpo_train_ids.jsonl lists the 1,667 LongDocURL questions the released model was trained
on (1,584 train, 83 validation), in construction order. It carries identifiers only; question
text, gold answers, and PDFs come from the public LongDocURL release (dengchao/LongDocURL on the
Hugging Face Hub, files LongDocURL_public.jsonl and pdf_files.tar.gz).

This script joins the two and writes the intermediate file that src/build_grpo_longdoc.py reads.
Rows are written validation first, so build_grpo_longdoc.py's default split (the first 5 percent
of rows is validation) reproduces the released split exactly.

Usage:
  python src/rehydrate_grpo_train.py \
      --ids release/grpo_train_ids.jsonl \
      --longdocurl /path/to/LongDocURL_public.jsonl \
      --pdf-root /path/to/extracted/pdf_files \
      --out data/longdocurl/grpo_refined.jsonl
  python src/build_grpo_longdoc.py --refined data/longdocurl/grpo_refined.jsonl
"""
import argparse
import glob
import json
import os


def find_pdf(pdf_root, doc_no):
    """LongDocURL ships PDFs as .../<N>000-<N>999/<first four digits>/<doc_no>.pdf"""
    guess = os.path.join(pdf_root, f"{doc_no[0]}000-{doc_no[0]}999", doc_no[:4], f"{doc_no}.pdf")
    if os.path.isfile(guess):
        return guess
    hits = glob.glob(os.path.join(pdf_root, "**", f"{doc_no}.pdf"), recursive=True)
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="release/grpo_train_ids.jsonl")
    ap.add_argument("--longdocurl", required=True, help="LongDocURL_public.jsonl")
    ap.add_argument("--pdf-root", required=True, help="directory holding the extracted LongDocURL PDFs")
    ap.add_argument("--out", default="data/longdocurl/grpo_refined.jsonl")
    args = ap.parse_args()

    src = {}
    for line in open(args.longdocurl, encoding="utf-8"):
        r = json.loads(line)
        src[r["question_id"]] = r
    ids = [json.loads(line) for line in open(args.ids, encoding="utf-8")]
    order = [r for r in ids if r["split"] == "val"] + [r for r in ids if r["split"] == "train"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    missing_q = missing_pdf = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r in order:
            s = src.get(r["question_id"])
            if s is None:
                missing_q += 1
                continue
            pdf = find_pdf(args.pdf_root, s["doc_no"])
            if pdf is None:
                missing_pdf += 1
                continue
            start, end = s["start_end_idx"]
            f.write(json.dumps({
                "index": s["question_id"],
                "question": s["question"],
                "gt_answer": str(s["answer"]),          # the judge compares against this string
                "doc_no": s["doc_no"],
                "pdf_path": pdf,
                "page_window": [start, end],             # 1-based, inclusive
                "n_pages_window": end - start + 1,
                "qtype": s["question_type"],
            }, ensure_ascii=False) + "\n")
    n = len(order) - missing_q - missing_pdf
    print(f"wrote {n} of {len(order)} items -> {args.out}")
    if missing_q or missing_pdf:
        print(f"  not found: {missing_q} question ids, {missing_pdf} PDFs")


if __name__ == "__main__":
    main()
