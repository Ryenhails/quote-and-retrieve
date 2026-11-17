"""Robust document-level leakage check: LongDocURL (GRPO train) vs CiteVQA (eval).

Both come from ccpdf (CommonCrawl PDFs) but use different file naming, so identity
must be checked by CONTENT. Unlike the earlier evidence-snippet shingle check, this
fingerprints the FULL extracted text of each PDF and estimates document-level Jaccard
via hash-modulo subsampled 5-gram word shingles (unbiased). Identical docs -> ~1.0,
unrelated -> ~0. Flags any LongDocURL doc that matches a CiteVQA doc, and marks whether
the match is inside the base_full(498) EVAL set (the contamination that actually matters).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qr_paths  # noqa: F401,E402  sets PROJECT_ROOT / QR_CACHE / QWEN3VL_EMB_REPO defaults
import json, re, sys, os
from pathlib import Path
import fitz

ROOT = Path(os.path.expandvars("${PROJECT_ROOT}"))
SAMPLE_MOD = 50          # keep shingles with hash%50==0  (~2% subsample, unbiased)
K = 5                    # word-shingle length
WORD = re.compile(r"[a-z0-9]+")


def doc_fingerprint(pdf_path, max_pages=None):
    """full-text -> set of subsampled 5-gram shingle hashes."""
    try:
        d = fitz.open(pdf_path)
    except Exception:
        return None
    toks = []
    for i, pg in enumerate(d):
        if max_pages and i >= max_pages:
            break
        toks.extend(WORD.findall(pg.get_text("text").lower()))
    d.close()
    sh = set()
    for i in range(len(toks) - K + 1):
        h = hash(" ".join(toks[i:i + K])) & 0x7FFFFFFFFFFFFFFF
        if h % SAMPLE_MOD == 0:
            sh.add(h)
    return sh


def jacc(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    # CiteVQA eval docs (base_full 498) + all CiteVQA docs available
    bench = [json.loads(l) for l in open(ROOT / "data/citevqa/citevqa_base_full.jsonl")]
    cv_eval = {}
    for r in bench:
        p = r.get("pdf_path")
        if p and os.path.exists(p):
            cv_eval[os.path.splitext(os.path.basename(p))[0]] = p
    # also include the rest of the CiteVQA pdf pool (superset, catches train-pool overlap too)
    cv_all = dict(cv_eval)
    pdfdir = ROOT / "CiteVQA/data/pdf"
    if pdfdir.exists():
        for p in pdfdir.glob("*.pdf"):
            cv_all.setdefault(p.stem, str(p))
    print(f"CiteVQA docs: eval(base_full)={len(cv_eval)}  all_pool={len(cv_all)}", flush=True)

    # LongDocURL train docs
    d2p = json.load(open(ROOT / "data/longdocurl/doc2pdf.json"))
    print(f"LongDocURL docs: {len(d2p)}", flush=True)

    print("fingerprinting CiteVQA pool ...", flush=True)
    cv_fp = {}
    for i, (stem, p) in enumerate(cv_all.items()):
        fp = doc_fingerprint(p)
        if fp:
            cv_fp[stem] = fp
        if (i + 1) % 100 == 0:
            print(f"  cv {i+1}/{len(cv_all)}", flush=True)
    eval_stems = set(cv_eval.keys())

    print("fingerprinting LongDocURL + comparing ...", flush=True)
    flagged = []
    for j, (did, p) in enumerate(d2p.items()):
        fp = doc_fingerprint(p)
        if not fp:
            continue
        best_s, best_doc = 0.0, None
        for stem, cfp in cv_fp.items():
            s = jacc(fp, cfp)
            if s > best_s:
                best_s, best_doc = s, stem
        if best_s >= 0.1:
            in_eval = best_doc in eval_stems
            flagged.append((did, round(best_s, 3), best_doc, in_eval))
        if (j + 1) % 50 == 0:
            print(f"  ldoc {j+1}/{len(d2p)}", flush=True)

    flagged.sort(key=lambda x: -x[1])
    print("\n==== LEAKAGE RESULTS (Jaccard >= 0.1) ====")
    print(f"{'longdoc':12s} {'jacc':6s} {'best_citevqa_doc':40s} in_EVAL(498)")
    for did, s, doc, in_eval in flagged:
        print(f"{did:12s} {s:<6.3f} {doc[:40]:40s} {'YES <<' if in_eval else 'no'}")
    same = [f for f in flagged if f[1] >= 0.5]
    eval_hits = [f for f in flagged if f[3]]
    print(f"\nsummary: total flagged>=0.1: {len(flagged)} | likely-same(>=0.5): {len(same)} "
          f"| hits inside base_full EVAL: {len(eval_hits)}")
    prev = {"4057121", "4045994", "4153823", "4174181", "4083860"}
    caught = prev & {f[0] for f in flagged}
    print(f"prior-flagged 5 = {sorted(prev)}; re-caught by this stronger check: {sorted(caught)}")
    json.dump([{"longdoc": f[0], "jacc": f[1], "citevqa": f[2], "in_eval": f[3]} for f in flagged],
              open(ROOT / "data/longdocurl/leakage_robust.json", "w"), indent=1)
    print("-> data/longdocurl/leakage_robust.json")


if __name__ == "__main__":
    main()
