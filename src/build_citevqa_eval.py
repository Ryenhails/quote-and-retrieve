"""Build CiteVQA eval subsets (Single-Doc) for the G0/G1/G2 grid.

Facts established empirically (2026-06-11):
- Evidence bbox order is [y1, x1, y2, x2], 0-1000 (verified by PDF text
  extraction; the official prompt's x1y1x2y2 description does NOT match the
  released data). We convert to xyxy here.
- source_page_id is 1-based.
- The eval code's `answer_bboxes` field does not exist in the released data;
  we reconstruct GT tuples directly from Evidence.
- necessity: GT for recall = 'necessary' evidence only; precision matches
  against ALL evidence (citing optional evidence is not an error). Applied
  identically across arms.

Outputs (paired subsets, seed 42):
  data/citevqa/citevqa_gt300.jsonl    GT-Pages setting, 300 items
  data/citevqa/citevqa_base120.jsonl  Base setting, 120 items (subset of gt300)

Record fields: index, question, language, qtype, pdf_path, n_pdf_pages,
gt_pages (1-based), gt_necessary [[page,x1,y1,x2,y2],...], gt_all [...].
"""
import argparse, csv, json, random, re
from pathlib import Path

WS = re.compile(r"\s+")


def norm_text(s):
    return WS.sub("", s or "")


def verify_alignment(doc, d):
    """evidence-level check: extract text in claimed page+bbox, match content.
    -> 'verified' | 'mismatch' | 'unverifiable'"""
    import fitz
    n_checked = n_hit = 0
    for e in d["Evidence"]:
        if e.get("type") != "text" or len(e.get("content", "")) < 40:
            continue
        pid = e["source_page_id"]
        if not (1 <= pid <= len(doc)):
            return "mismatch"
        page = doc[pid - 1]
        W, H = page.rect.width, page.rect.height
        y1, x1, y2, x2 = e["bbox"]
        rect = fitz.Rect(x1/1000*W, y1/1000*H, x2/1000*W, y2/1000*H)
        txt = norm_text(page.get_text(clip=rect))
        if len(txt) < 20:
            continue  # likely scanned page, no text layer in region
        n_checked += 1
        gt = norm_text(e["content"])
        if gt[:60] in txt or txt[:60] in gt:
            n_hit += 1
    if n_checked == 0:
        return "unverifiable"
    return "verified" if n_hit > 0 else "mismatch"

ROOT = Path(__file__).resolve().parent.parent
CV = ROOT / "CiteVQA"


def to_xyxy(b):
    y1, x1, y2, x2 = [float(v) for v in b]
    return [x1, y1, x2, y2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_gt", type=int, default=300)
    ap.add_argument("--n_base", type=int, default=120)
    ap.add_argument("--max_pages_base", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import fitz

    data = json.load(open(CV / "data/validation/CiteVQA.json"))
    rng = random.Random(args.seed)

    csv_bytes = {}
    with open(CV / "data/download/pdf_source.csv") as f:
        for row in csv.DictReader(f):
            try:
                csv_bytes[row["track_id"] + ".pdf"] = int(row["content_length"])
            except (ValueError, KeyError):
                pass

    pool = []
    n_mismatch = n_unver_drop = 0
    for d in data:
        if d["dataset_type"] != "Single-Doc":
            continue
        pdf = CV / d["PDF_Source"][0]
        if not pdf.exists():
            continue
        necessary, allboxes, pages = [], [], set()
        for e in d["Evidence"]:
            box = [e["source_page_id"]] + to_xyxy(e["bbox"])
            allboxes.append(box)
            pages.add(e["source_page_id"])
            if e["necessity"] == "necessary":
                necessary.append(box)
        if not necessary:
            continue
        # tier-1: PDF opens and covers all GT pages
        try:
            doc = fitz.open(pdf)
            n_pdf = len(doc)
        except Exception:
            continue
        if max(pages) > n_pdf:
            doc.close()
            continue
        # tier-2: evidence-level alignment (text content matched at claimed bbox);
        # unverifiable (scanned/no text layer) kept only if byte-exact vs csv
        align = verify_alignment(doc, d)
        doc.close()
        if align == "mismatch":
            n_mismatch += 1
            continue
        if align == "unverifiable" and csv_bytes.get(pdf.name) not in (None, pdf.stat().st_size):
            n_unver_drop += 1
            continue
        pool.append({
            "alignment": align,
            "n_pdf_pages": n_pdf,
            "index": d["index"],
            "question": d["Question"],
            "standard_answer": d["Standard_Answer"],
            "language": d["language"],
            "qtype": d["Question_Type"],
            "pdf_path": str(pdf),
            "gt_pages": sorted(pages),
            "gt_necessary": necessary,
            "gt_all": allboxes,
        })

    rng.shuffle(pool)
    # language-balanced GT subset
    en = [r for r in pool if r["language"] == "en"]
    zh = [r for r in pool if r["language"] == "zh"]
    half = args.n_gt // 2
    gt_subset = en[:half] + zh[:half]
    rng.shuffle(gt_subset)

    # base subset: paired (taken from gt_subset), doc length capped
    out_dir = ROOT / "data/citevqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_pages_cache = {}
    base_pool = gt_subset if args.n_base else pool  # n_base=0 -> ALL eligible (final testset mode)
    base_subset = []
    for r in base_pool:
        if args.n_base and len(base_subset) >= args.n_base:
            break
        n = r["n_pdf_pages"]
        if n <= args.max_pages_base:
            base_subset.append(r)

    with (out_dir / f"citevqa_gt{args.n_gt}.jsonl").open("w") as f:
        for r in gt_subset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    base_name = f"citevqa_base{args.n_base}" if args.n_base else "citevqa_base_full"
    with (out_dir / f"{base_name}.jsonl").open("w") as f:
        for r in base_subset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    from collections import Counter as _C
    print(f"pool: {len(pool)} (en {len(en)} / zh {len(zh)})  alignment={dict(_C(r['alignment'] for r in pool))}")
    print(f"dropped: mismatch={n_mismatch}  unverifiable-and-not-byte-exact={n_unver_drop}")
    print(f"gt subset: {len(gt_subset)}  lang={dict(Counter(r['language'] for r in gt_subset))}")
    print(f"base subset: {len(base_subset)}  pages dist={dict(Counter(min(r['n_pdf_pages']//10*10, 40) for r in base_subset))}")


if __name__ == "__main__":
    main()
