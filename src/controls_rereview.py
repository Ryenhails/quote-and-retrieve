#!/usr/bin/env python3
"""Re-review control experiments (offline, CPU): reuse the frozen matrix outputs.

Subcommands
  snap       E3  coordinate-box -> max-IoU MinerU block snapping control, multi-threshold rescoring
  funnel     E4  filtering-funnel disclosure + text-layer analysis of dropped documents
  stats      E5  paired bootstrap CIs + McNemar + intersection-AH for coord/NL and base/GRPO pairs
  precision  E6  citation-level precision (vs gt_all, IoU 0.5) + mean citations per response
  breakdown  E7  per-language (en/zh) and per-evidence-type metrics

All scoring goes through the same code paths as the main table (score_citevqa.iou_hit,
build_main_table recall_at/page_recall, judge parse regexes), so numbers are directly
comparable with Tables 1 and 3.

Usage: python src/controls_rereview.py <subcommand>
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qr_paths  # noqa: F401,E402  sets PROJECT_ROOT / QR_CACHE / QWEN3VL_EMB_REPO defaults
import argparse, json, os, re, sys, random, statistics
from collections import defaultdict, Counter

sys.path.insert(0, "src")
from score_citevqa import iou_hit
from mineru_ceiling import blocks_for

ROOT = os.path.expandvars("${PROJECT_ROOT}")
BENCH = f"{ROOT}/data/citevqa/citevqa_singledoc_pub.jsonl"
CACHE = f"{ROOT}/data/citevqa/mineru_cache"
JUDGE = f"{ROOT}/outputs/judge/batch_results_all.jsonl"
CITE = f"{ROOT}/CiteVQA/data/validation/CiteVQA.json"
MATRIX = f"{ROOT}/outputs/matrix"
THRESH = [0.1, 0.3, 0.5, 0.7]

# (dir prefix, judge tag prefix, display name)
BACKBONES = [
    ("qwen3vl", "qwen3vl", "Qwen3-VL-8B"),
    ("qwen3vl_30b", "qwen3vl_30b", "Qwen3-VL-30B-A3B"),
    ("qwen35_9b", "qwen35_9b", "Qwen3.5-9B"),
    ("qwen35_27b", "qwen35_27b", "Qwen3.5-27B"),
    ("gemma3", "gemma3", "Gemma-3-12B"),
    ("gemma4_31b", "gemma4_31b", "Gemma-4-31B"),
]
GRPO_DIR = "grpo_global_step_144_lr1e5"

QA_RE = re.compile(r'<?qa_acc>\s*(\d+)\s*</qa_acc>')
REL_RE = re.compile(r'<?relevance_score>\s*(\d+)\s*</relevance_score>')


def load_preds(tag_dir):
    path = f"{MATRIX}/{tag_dir}/predictions.jsonl"
    return [json.loads(l) for l in open(path)]


def page_recall(r):
    gp = {b[0] for b in r["gt_necessary"]}
    pp = {p[1] for p in r.get("pred_tuples", [])}
    return len(gp & pp) / len(gp) if gp else None


def recall_at(r, th):
    gt = [(1, b[0], b[1], b[2], b[3], b[4]) for b in r["gt_necessary"]]
    preds = [tuple(t) for t in r.get("pred_tuples", [])]
    if not gt:
        return None
    return sum(1 for g in gt if any(iou_hit(g, p, th) for p in preds)) / len(gt)


def iou_raw(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def resp_text(o):
    r = o.get("response") or {}
    if r.get("text"):
        return r["text"]
    for c in (r.get("candidates") or []):
        for p in ((c.get("content") or {}).get("parts") or []):
            if p.get("text"):
                return p["text"]
    return ""


def load_judge():
    """(tag, idx) -> {'qa': int, 'rel': int}; only fully-parsed pairs are usable."""
    J = {}
    for l in open(JUDGE):
        l = l.strip()
        if not l:
            continue
        o = json.loads(l)
        parts = o.get("key", "").split("|")
        if len(parts) != 3:
            continue
        tag, idx, kind = parts
        m = (QA_RE if kind == "qa" else REL_RE).search(resp_text(o))
        if m:
            J.setdefault((tag, idx), {})[kind] = int(m.group(1))
    return J


def judged_items(preds, J, tag):
    """valid rows with both judge scores; -> list of (r, qa, rel, rec05)."""
    out = []
    for r in preds:
        if r.get("error") or not r.get("gt_necessary"):
            continue
        sc = J.get((tag, r["index"]))
        if not sc or "qa" not in sc or "rel" not in sc:
            continue
        rec = recall_at(r, 0.5)
        if rec is None:
            continue
        out.append((r, sc["qa"], sc["rel"], rec))
    return out


def saa(qa, rel, rec):
    return 1 if (qa >= 4 and (rel >= 4 or rec >= 0.6)) else 0


# ---------------------------------------------------------------- E3 snap
def cmd_snap(_):
    print("=== E3 coordinate -> block snapping control ===")
    print("policy: each predicted box snaps to the max-IoU MinerU block on its predicted page;")
    print("if no overlap, the nearest block by center distance on that page; if the page has no")
    print("parsed blocks, the box is kept unchanged.")
    hdr = f"{'condition':<22}{'variant':<10}{'n':>5} | {'page':>6}" + "".join(f"{'@'+str(t):>7}" for t in THRESH)
    print(hdr)
    print("-" * len(hdr))
    for dir_pfx, _, name in BACKBONES:
        preds = load_preds(f"{dir_pfx}_coord")
        valid = [r for r in preds if not r.get("error") and r.get("gt_necessary")]
        block_cache = {}
        snapped_rows = []
        n_snapped = n_kept = n_boxes = 0
        for r in valid:
            pdf = r["pdf_path"]
            if pdf not in block_cache:
                block_cache[pdf] = blocks_for(pdf, CACHE) or []
            by_page = defaultdict(list)
            for bl in block_cache[pdf]:
                by_page[bl[1]].append(bl)
            new_preds = []
            for t in r.get("pred_tuples", []):
                n_boxes += 1
                flag, pg, x1, y1, x2, y2 = t
                cands = by_page.get(int(pg), [])
                if not cands:
                    new_preds.append(list(t))
                    n_kept += 1
                    continue
                ious = [iou_raw((x1, y1, x2, y2), c[2:]) for c in cands]
                best = max(range(len(cands)), key=lambda j: ious[j])
                if ious[best] <= 0:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    best = min(range(len(cands)), key=lambda j: (
                        ((cands[j][2] + cands[j][4]) / 2 - cx) ** 2 +
                        ((cands[j][3] + cands[j][5]) / 2 - cy) ** 2))
                new_preds.append([1, cands[best][1], *cands[best][2:]])
                n_snapped += 1
            rr = dict(r)
            rr["pred_tuples"] = new_preds
            snapped_rows.append(rr)
        for variant, rows in [("raw", valid), ("snapped", snapped_rows)]:
            pr = [page_recall(r) for r in rows]
            pr = [v for v in pr if v is not None]
            line = f"{name:<22}{variant:<10}{len(pr):>5} | {100*statistics.mean(pr):>6.1f}"
            for t in THRESH:
                vals = [recall_at(r, t) for r in rows]
                vals = [v for v in vals if v is not None]
                line += f"{100*statistics.mean(vals):>7.1f}"
            print(line)
        print(f"{'':<22}snap-stats: boxes={n_boxes} snapped={n_snapped} kept-unsnapped={n_kept}")
    print("interpretation: 'snapped' gives the coordinate interface the same candidate set and the")
    print("same element granularity as the language interface; the residual gap to the NL rows of")
    print("Table 3 is then attributable to the location signal itself, not to box granularity.")


# ---------------------------------------------------------------- E4 funnel
def cmd_funnel(_):
    import fitz
    print("=== E4 filtering funnel + dropped-document text-layer analysis ===")
    print("funnel (from build_citevqa_pub.py, recomputed classification below):")
    print("  987 single-document questions")
    print("   -117 no/invalid PDF   -4 GT page out of range   -131 text mismatch   -16 unverifiable, not byte-exact")
    print("  = 719 retained (72.9%)")

    sys.path.insert(0, "src")
    from build_citevqa_eval import verify_alignment, CV
    import csv as csvmod
    data = json.load(open(CITE))
    csv_bytes = {}
    with open(CV / "data/download/pdf_source.csv") as f:
        for row in csvmod.DictReader(f):
            try:
                csv_bytes[row["track_id"] + ".pdf"] = int(row["content_length"])
            except (ValueError, KeyError):
                pass

    kept_pdfs, drop_mismatch_pdfs = set(), set()
    n = Counter()
    for d in data:
        if d["dataset_type"] != "Single-Doc":
            continue
        n["single_doc"] += 1
        pdf = CV / d["PDF_Source"][0]
        if not (pdf.exists() and open(pdf, "rb").read(4) == b"%PDF"):
            n["no_pdf"] += 1
            continue
        necessary = [e for e in d["Evidence"] if e["necessity"] == "necessary"]
        if not necessary:
            n["no_necessary"] += 1
            continue
        pages = {e["source_page_id"] for e in d["Evidence"]}
        try:
            doc = fitz.open(pdf)
            n_pdf = len(doc)
        except Exception:
            n["no_pdf"] += 1
            continue
        if max(pages) > n_pdf:
            doc.close()
            n["page_oob"] += 1
            continue
        align = verify_alignment(doc, d)
        doc.close()
        if align == "mismatch":
            n["mismatch"] += 1
            drop_mismatch_pdfs.add(str(pdf))
            continue
        if align == "unverifiable" and csv_bytes.get(pdf.name) not in (None, pdf.stat().st_size):
            n["unver_drop"] += 1
            continue
        n["kept"] += 1
        kept_pdfs.add(str(pdf))
    print(f"recomputed: {dict(n)}")

    def text_layer_share(pdf_path, sample=8):
        """fraction of sampled pages with a substantive text layer (>200 chars)."""
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            return None
        pages = list(range(len(doc)))
        if len(pages) > sample:
            step = len(pages) / sample
            pages = [int(i * step) for i in range(sample)]
        rich = 0
        for p in pages:
            try:
                if len(doc[p].get_text()) > 200:
                    rich += 1
            except Exception:
                pass
        doc.close()
        return rich / len(pages) if pages else None

    for label, pdfs in [("retained", kept_pdfs), ("dropped (text mismatch)", drop_mismatch_pdfs)]:
        shares = [text_layer_share(p) for p in sorted(pdfs)]
        shares = [s for s in shares if s is not None]
        scanned = sum(1 for s in shares if s < 0.25)
        print(f"{label}: docs={len(shares)}  mean text-layer share={statistics.mean(shares):.2f}  "
              f"docs with <25% text-layer pages (scan-like)={scanned} ({100*scanned/len(shares):.1f}%)")


# ---------------------------------------------------------------- E5 stats
def boot_ci(vals_a, vals_b, fn, n_boot=10000, seed=0):
    """paired bootstrap CI for fn(sample_a) - fn(sample_b) over question indices."""
    rng = random.Random(seed)
    n = len(vals_a)
    idx = list(range(n))
    deltas = []
    for _ in range(n_boot):
        s = [idx[rng.randrange(n)] for _ in range(n)]
        deltas.append(fn([vals_a[i] for i in s]) - fn([vals_b[i] for i in s]))
    deltas.sort()
    return deltas[int(0.025 * n_boot)], deltas[int(0.975 * n_boot)]


def mcnemar_p(b, c):
    """exact two-sided binomial test on discordant pairs."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cmd_stats(_):
    print("=== E5 paired significance: bootstrap CIs (10k, seed 0) + McNemar + intersection-AH ===")
    J = load_judge()
    tags = sorted({t for t, _ in J})
    print(f"judge tags present: {tags}")

    def get_pair(tag_a, dir_a, tag_b, dir_b):
        A = {r["index"]: x for x in judged_items(load_preds(dir_a), J, tag_a) for r in [x[0]]}
        B = {r["index"]: x for x in judged_items(load_preds(dir_b), J, tag_b) for r in [x[0]]}
        common = sorted(set(A) & set(B))
        return [A[i] for i in common], [B[i] for i in common]

    def summarize(name, X, Y, label_x, label_y):
        n = len(X)
        recs_x = [x[3] for x in X]
        recs_y = [y[3] for y in Y]
        saa_x = [saa(x[1], x[2], x[3]) for x in X]
        saa_y = [saa(y[1], y[2], y[3]) for y in Y]

        def mean(v):
            return statistics.mean(v) if v else 0.0

        d_rec = 100 * (mean(recs_x) - mean(recs_y))
        lo, hi = boot_ci(recs_x, recs_y, mean)
        print(f"\n[{name}]  n(paired)={n}   {label_x} vs {label_y}")
        print(f"  dRec@0.5 = {d_rec:+.1f}  95% CI [{100*lo:+.1f}, {100*hi:+.1f}]")
        d_saa = 100 * (mean(saa_x) - mean(saa_y))
        lo, hi = boot_ci(saa_x, saa_y, mean)
        b = sum(1 for a, c in zip(saa_x, saa_y) if a == 1 and c == 0)
        c = sum(1 for a, c2 in zip(saa_x, saa_y) if a == 0 and c2 == 1)
        print(f"  dSAA     = {d_saa:+.1f}  95% CI [{100*lo:+.1f}, {100*hi:+.1f}]  "
              f"McNemar b={b} c={c} p={mcnemar_p(b, c):.2e}")

        def ah(items_saa, items_qa):
            corr = [s for s, q in zip(items_saa, items_qa) if q >= 4]
            return sum(1 for s in corr if s == 0) / len(corr) if corr else 0.0

        qa_x = [x[1] for x in X]
        qa_y = [y[1] for y in Y]
        ah_x = 100 * ah(saa_x, qa_x)
        ah_y = 100 * ah(saa_y, qa_y)
        pair = list(zip(saa_x, qa_x, saa_y, qa_y))

        def ah_delta(sample):
            cx = [s for s, q, _, _ in sample if q >= 4]
            cy = [s2 for _, _, s2, q2 in sample if q2 >= 4]
            ax = sum(1 for s in cx if s == 0) / len(cx) if cx else 0.0
            ay = sum(1 for s in cy if s == 0) / len(cy) if cy else 0.0
            return ax - ay

        rng = random.Random(0)
        deltas = []
        for _ in range(10000):
            s = [pair[rng.randrange(len(pair))] for _ in range(len(pair))]
            deltas.append(ah_delta(s))
        deltas.sort()
        print(f"  AH: {label_x}={ah_x:.1f} {label_y}={ah_y:.1f}  dAH={ah_x-ah_y:+.1f}  "
              f"95% CI [{100*deltas[250]:+.1f}, {100*deltas[9750]:+.1f}]")
        both = [(sx, sy) for sx, qx, sy, qy in pair if qx >= 4 and qy >= 4]
        if both:
            ax = 100 * sum(1 for sx, _ in both if sx == 0) / len(both)
            ay = 100 * sum(1 for _, sy in both if sy == 0) / len(both)
            print(f"  intersection-AH (both Ans>=4, n={len(both)}): {label_x}={ax:.1f}  {label_y}={ay:.1f}")

    for dir_pfx, tag_pfx, name in BACKBONES:
        X, Y = get_pair(f"{tag_pfx}_nl", f"{dir_pfx}_nl_wd", f"{tag_pfx}_coord", f"{dir_pfx}_coord")
        summarize(name, X, Y, "NL", "coord")

    grpo_tag = next((t for t in tags if t.startswith("grpo")), None)
    if grpo_tag:
        X, Y = get_pair(grpo_tag, f"{GRPO_DIR}_nl_wd", "qwen3vl_nl", "qwen3vl_nl_wd")
        summarize("GRPO vs base (Qwen3-VL-8B NL)", X, Y, "GRPO", "base")


# ---------------------------------------------------------------- E6 precision
def cmd_precision(_):
    print("=== E6 citation-level precision (vs gt_all, IoU>=0.5) + citations per response ===")
    hdr = f"{'condition':<26}{'n':>5}{'prec@0.5':>10}{'cites/resp':>12}{'resp w/ cites %':>16}"
    print(hdr)
    print("-" * len(hdr))
    conds = [(f"{d}_coord", f"{nm} coord") for d, _, nm in BACKBONES]
    conds += [(f"{d}_nl_wd", f"{nm} NL") for d, _, nm in BACKBONES]
    conds += [(f"{GRPO_DIR}_nl_wd", "GRPO (8B) NL")]
    for tag_dir, name in conds:
        preds = load_preds(tag_dir)
        valid = [r for r in preds if not r.get("error") and r.get("gt_necessary")]
        precs, nboxes, has = [], [], 0
        for r in valid:
            gta = [(1, b[0], b[1], b[2], b[3], b[4]) for b in r["gt_all"]]
            pts = [tuple(t) for t in r.get("pred_tuples", [])]
            nboxes.append(len(pts))
            if pts:
                has += 1
                hit = sum(1 for g in gta if any(iou_hit(g, p) for p in pts))
                precs.append(hit / len(pts))
        p = 100 * statistics.mean(precs) if precs else 0.0
        print(f"{name:<26}{len(valid):>5}{p:>10.1f}{statistics.mean(nboxes):>12.2f}{100*has/len(valid):>16.1f}")
    print("precision uses the official loop semantics (GT hits / #pred boxes) against ALL evidence,")
    print("identical to score_citevqa.py; cites/resp counts emitted citations after parsing.")


# ---------------------------------------------------------------- E7 breakdown
def cmd_breakdown(_):
    print("=== E7 per-language and per-evidence-type breakdown ===")
    J = load_judge()
    cite = {it["index"]: it for it in json.load(open(CITE))}

    conds = [(f"{d}_coord", f"{t}_coord", f"{nm} coord") for d, t, nm in BACKBONES]
    conds += [(f"{d}_nl_wd", f"{t}_nl", f"{nm} NL") for d, t, nm in BACKBONES]
    grpo_tag = next((t for t, _ in J if t.startswith("grpo")), None)
    if grpo_tag:
        conds += [(f"{GRPO_DIR}_nl_wd", grpo_tag, "GRPO (8B) NL")]

    print(f"\n-- per language: Rec@0.5 / Ans / SAA / AH --")
    hdr = f"{'condition':<26}" + "".join(f"{lang+' '+m:>12}" for lang in ["en", "zh"] for m in ["Rec", "Ans", "SAA", "AH"])
    print(hdr)
    for tag_dir, jtag, name in conds:
        items = judged_items(load_preds(tag_dir), J, jtag)
        line = f"{name:<26}"
        for lang in ["en", "zh"]:
            sub = [(r, q, rl, rc) for r, q, rl, rc in items if r["language"] == lang]
            if not sub:
                line += "".join(f"{'--':>12}" for _ in range(4))
                continue
            rec = 100 * statistics.mean(rc for _, _, _, rc in sub)
            ans = 20 * statistics.mean(q for _, q, _, _ in sub)
            ss = [saa(q, rl, rc) for _, q, rl, rc in sub]
            sv = 100 * statistics.mean(ss)
            corr = [s for s, (_, q, _, _) in zip(ss, sub) if q >= 4]
            ah = 100 * sum(1 for s in corr if s == 0) / len(corr) if corr else 0.0
            line += f"{rec:>12.1f}{ans:>12.1f}{sv:>12.1f}{ah:>12.1f}"
        print(line)

    print(f"\n-- per evidence-element type: element-level recall@0.5 (NL conditions + coord for contrast) --")
    TYPE_GROUP = {"text": "text", "list": "text", "ref_text": "text", "code": "text",
                  "title": "text", "header": "text", "footer": "text",
                  "table": "table", "table_caption": "table", "table_footnote": "table",
                  "image": "figure", "image_caption": "figure", "image_footnote": "figure",
                  "equation": "other", "page_footnote": "other"}
    for tag_dir, _, name in conds:
        preds = load_preds(tag_dir)
        valid = {r["index"]: r for r in preds if not r.get("error") and r.get("gt_necessary")}
        by_type = defaultdict(lambda: [0, 0])
        for idx, r in valid.items():
            ev = [e for e in cite.get(idx, {}).get("Evidence", []) if e["necessity"] == "necessary"]
            gt = [(1, b[0], b[1], b[2], b[3], b[4]) for b in r["gt_necessary"]]
            if len(ev) != len(gt):
                continue
            pts = [tuple(t) for t in r.get("pred_tuples", [])]
            for e, g in zip(ev, gt):
                grp = TYPE_GROUP.get(e.get("type", "other"), "other")
                by_type[grp][1] += 1
                if any(iou_hit(g, p) for p in pts):
                    by_type[grp][0] += 1
        parts = "  ".join(f"{k}={100*v[0]/v[1]:.1f} (n={v[1]})" for k, v in sorted(by_type.items()))
        print(f"{name:<26}{parts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["snap", "funnel", "stats", "precision", "breakdown"])
    args = ap.parse_args()
    os.chdir(ROOT)
    {"snap": cmd_snap, "funnel": cmd_funnel, "stats": cmd_stats,
     "precision": cmd_precision, "breakdown": cmd_breakdown}[args.cmd](args)


if __name__ == "__main__":
    main()
