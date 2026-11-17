#!/usr/bin/env python3
"""Camera-ready CPU analyses over the frozen outputs.  No new inference.

Subcommands (reviewer item each one answers):
  anls     ANLS / soft normalised-Levenshtein of model answers vs gold + correlation with judge Ans
  qtype    Rec / Prec / Ans / SAA / AH by question type for all 13 conditions
  filter   719 vs 987 composition; verified vs unverifiable alignment groups; scan-like slice
  budget   GRPO vs base at matched citation budget; SAA_rel; split by "cites more than base"
  ansdrop  language-interface answer-score drop: truncation / parse / length diagnosis
  decomp   model-vs-pipeline contribution ladder + quote / q+a complementarity (rewrite of E12)
  buckets  failure buckets by question type (needs outputs/revision/buckets_*.jsonl from the GPU dump)
  union    exact quote ∪ q+a union (needs outputs/revision/qa_boxes_*.jsonl from the GPU dump)

Every subcommand prints markdown and also writes outputs/revision/analysis/<cmd>.md.
Scoring goes through the same helpers as the paper (score_citevqa.iou_hit, build_main_table conventions).
Usage: python src/revision_analysis.py <cmd>
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qr_paths  # noqa: F401,E402  sets PROJECT_ROOT / QR_CACHE / QWEN3VL_EMB_REPO defaults
import argparse, json, math, os, re, statistics, sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")
from score_citevqa import iou_hit
from mineru_ceiling import blocks_for

ROOT = os.path.expandvars("${PROJECT_ROOT}")
BENCH = f"{ROOT}/data/citevqa/citevqa_singledoc_pub.jsonl"
MATRIX = f"{ROOT}/outputs/matrix"
CACHE = f"{ROOT}/data/citevqa/mineru_cache"
JUDGE = f"{ROOT}/outputs/judge/batch_results_all.jsonl"
CITE = f"{ROOT}/CiteVQA/data/validation/CiteVQA.json"
CTRL = f"{ROOT}/outputs/controls"
REB = os.environ.get("QR_REVISION_DIR", f"{ROOT}/outputs/revision")
OUT = os.environ.get("QR_REVISION_OUT", f"{ROOT}/outputs/revision/analysis")

BACKBONES = [
    ("qwen3vl", "qwen3vl", "Qwen3-VL-8B"),
    ("qwen3vl_30b", "qwen3vl_30b", "Qwen3-VL-30B-A3B"),
    ("qwen35_9b", "qwen35_9b", "Qwen3.5-9B"),
    ("qwen35_27b", "qwen35_27b", "Qwen3.5-27B"),
    ("gemma3", "gemma3", "Gemma-3-12B"),
    ("gemma4_31b", "gemma4_31b", "Gemma-4-31B"),
]
GRPO_DIR = "grpo_global_step_144_lr1e5"
# (prediction dir, judge tag, display name)
CONDS = [(f"{d}_coord", f"{t}_coord", f"{nm} coord.") for d, t, nm in BACKBONES] + \
        [(f"{d}_nl_wd", f"{t}_nl", f"{nm} lang.") for d, t, nm in BACKBONES] + \
        [(f"{GRPO_DIR}_nl_wd", "grpo_nl", "8B + GRPO lang.")]
QTYPES = ["Complex Synthesis", "Multimodal Parsing", "Factual Retrieval", "Quantitative Reasoning"]
QA_RE = re.compile(r'<?qa_acc>\s*(\d+)\s*</qa_acc>')
REL_RE = re.compile(r'<?relevance_score>\s*(\d+)\s*</relevance_score>')


# ----------------------------------------------------------------------------- helpers
def load_preds(tag_dir):
    return [json.loads(l) for l in open(f"{MATRIX}/{tag_dir}/predictions.jsonl")]


def valid_rows(rows):
    return [r for r in rows if not r.get("error") and r.get("gt_necessary")]


def gt_nec(r):
    return [(1, b[0], b[1], b[2], b[3], b[4]) for b in r["gt_necessary"]]


def gt_all(r):
    return [(1, b[0], b[1], b[2], b[3], b[4]) for b in r["gt_all"]]


def preds_of(r):
    return [tuple(t) for t in r.get("pred_tuples", [])]


def recall(r, preds=None, th=0.5):
    gt = gt_nec(r)
    preds = preds_of(r) if preds is None else [tuple(p) for p in preds]
    return sum(1 for g in gt if any(iou_hit(g, p, th) for p in preds)) / len(gt)


def page_recall(r, preds=None):
    gp = {b[0] for b in r["gt_necessary"]}
    preds = preds_of(r) if preds is None else preds
    return len(gp & {p[1] for p in preds}) / len(gp)


def precision(r, preds=None):
    """official loop semantics: annotated elements hit / boxes cited; None when nothing cited."""
    preds = preds_of(r) if preds is None else [tuple(p) for p in preds]
    if not preds:
        return None
    return sum(1 for g in gt_all(r) if any(iou_hit(g, p) for p in preds)) / len(preds)


def f1(rc, pr):
    """score_citevqa.score_item semantics: no citations -> precision undefined, recall 0, F1 = 0."""
    if pr is None:
        return 0.0
    return 0.0 if rc + pr == 0 else 2 * rc * pr / (rc + pr)


TYPE_GROUP = {"text": "text", "list": "text", "ref_text": "text", "code": "text",
              "title": "text", "header": "text", "footer": "text",
              "table": "table", "table_caption": "table", "table_footnote": "table",
              "image": "figure", "image_caption": "figure", "image_footnote": "figure",
              "equation": "other", "page_footnote": "other"}


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


def judged(rows, J, tag):
    out = []
    for r in valid_rows(rows):
        sc = J.get((tag, r["index"]))
        if sc and "qa" in sc and "rel" in sc:
            out.append((r, sc["qa"], sc["rel"], recall(r)))
    return out


def saa(qa, rel, rec):
    return 1 if (qa >= 4 and (rel >= 4 or rec >= 0.6)) else 0


def saa_rel(qa, rel):
    return 1 if (qa >= 4 and rel >= 4) else 0


def ah_of(items):
    """items: (saa, qa) -> AH = P(saa=0 | qa>=4)"""
    corr = [s for s, q in items if q >= 4]
    return 100 * sum(1 for s in corr if s == 0) / len(corr) if corr else float("nan")


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else float("nan")


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def mcnemar_p(b, c):
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def fmt(v, d=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v:.{d}f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def emit(cmd, text):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{cmd}.md", "w") as f:
        f.write(text.rstrip() + "\n")
    print(text)
    print(f"\n-> {OUT}/{cmd}.md")


def parse_obj(raw):
    t = re.sub(r"^\s*```(?:json)?|```\s*$", "", (raw or "").strip(), flags=re.I).strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return None


def nl_answer(r):
    """language interface: the parsed answer string (json-unescaped); coord: text minus bbox tags."""
    o = parse_obj(r.get("raw_predict", ""))
    if o and isinstance(o.get("answer"), str):
        return o["answer"]
    a = r.get("answer_text") or ""
    try:
        return json.loads(f'"{a}"') if "\\" in a else a
    except Exception:
        return a


# ----------------------------------------------------------------------------- anls
def cmd_anls(_):
    from rapidfuzz.distance import Levenshtein

    def norm(s):
        return re.sub(r"\s+", " ", (s or "").lower()).strip()

    def nls(a, b):
        a, b = norm(a), norm(b)
        if not a and not b:
            return 1.0
        return Levenshtein.normalized_similarity(a, b)

    J = load_judge()
    bench = [json.loads(l) for l in open(BENCH)]
    gold_len = {lang: statistics.median(len(r["standard_answer"]) for r in bench if r["language"] == lang) for lang in ["en", "zh"]}
    lines = ["# ANLS versus judge answer score", "",
             "ANLS (Biten et al., 2019) = mean over questions of NLS if NLS ≥ 0.5 else 0, where NLS = 1 − normalised Levenshtein "
             "distance between the lower-cased model answer and the gold answer. 'soft NLS' drops the 0.5 threshold. "
             "Ans = Gemini judge score ×20 on the same judged subset. Pearson r is between per-question NLS and the raw 0–5 judge score.",
             f"Gold answers are long: median {gold_len['en']:.0f} characters (English) and {gold_len['zh']:.0f} characters (Chinese); "
             "55% of questions are Complex Synthesis. ANLS was designed for short extractive spans.", ""]
    rows = []
    for d, tag, name in CONDS:
        items = judged(load_preds(d), J, tag)
        if not items:
            continue
        sc = [(nls(nl_answer(r), r["standard_answer"]), q) for r, q, _, _ in items]
        anls = 100 * mean([s if s >= 0.5 else 0.0 for s, _ in sc])
        soft = 100 * mean([s for s, _ in sc])
        nonzero = 100 * mean([1.0 if s >= 0.5 else 0.0 for s, _ in sc])
        r_ = pearson([s for s, _ in sc], [q for _, q in sc])
        ans = 20 * mean([q for _, q in sc])
        rows.append([name, len(items), fmt(anls), fmt(soft), fmt(nonzero), fmt(r_, 2), fmt(ans)])
    lines.append(md_table(["condition", "n", "ANLS", "soft NLS", "% NLS≥0.5", "r(NLS, judge)", "Ans (judge ×20)"], rows))
    lines += ["", "Reading: ANLS collapses on abstractive long answers (most model answers share little character-level overlap with the "
              "gold paragraph even when the judge rates them correct); the soft variant tracks the judge ordering but compresses the scale. "
              "Both interfaces are scored identically, so the interface contrast is unaffected by the choice of answer metric."]
    emit("anls", "\n".join(lines))


# ----------------------------------------------------------------------------- qtype
def cmd_qtype(_):
    J = load_judge()
    lines = ["# Results by question type", "",
             "Rec@0.5 and precision on all valid responses (n per type: 394 / 136 / 122 / 67); Ans, SAA, AH on the judged subset. "
             "Complex Synthesis and Quantitative Reasoning are the types whose evidence is least 'quotable' (cross-element synthesis, numeric computation).", ""]
    tabs = {k: [] for k in ["Rec@0.5", "precision", "AH", "SAA", "Ans"]}
    for d, tag, name in CONDS:
        rows = valid_rows(load_preds(d))
        jt = judged(rows, J, tag)
        by = defaultdict(list); byj = defaultdict(list)
        for r in rows:
            by[r["qtype"]].append(r)
        for r, q, rl, rc in jt:
            byj[r["qtype"]].append((q, rl, rc))
        tabs["Rec@0.5"].append([name] + [fmt(100 * mean([recall(r) for r in by[t]])) for t in QTYPES])
        tabs["precision"].append([name] + [fmt(100 * mean([precision(r) for r in by[t]])) for t in QTYPES])
        tabs["AH"].append([name] + [fmt(ah_of([(saa(q, rl, rc), q) for q, rl, rc in byj[t]])) for t in QTYPES])
        tabs["SAA"].append([name] + [fmt(100 * mean([saa(q, rl, rc) for q, rl, rc in byj[t]])) for t in QTYPES])
        tabs["Ans"].append([name] + [fmt(20 * mean([q for q, _, _ in byj[t]])) for t in QTYPES])
    for k in ["Rec@0.5", "precision", "AH", "SAA", "Ans"]:
        lines += [f"## {k} by question type", "", md_table(["condition"] + QTYPES, tabs[k]), ""]
    emit("qtype", "\n".join(lines))


# ----------------------------------------------------------------------------- filter
def text_layer_shares():
    """fraction of sampled pages (8) with > 200 extractable characters, per PDF; cached."""
    import fitz
    os.makedirs(REB, exist_ok=True)
    fp = f"{REB}/text_layer_share.json"
    if os.path.exists(fp):
        return json.load(open(fp))
    bench = [json.loads(l) for l in open(BENCH)]
    shares = {}
    for pdf in sorted({r["pdf_path"] for r in bench}):
        try:
            doc = fitz.open(pdf)
        except Exception:
            shares[pdf] = None; continue
        pages = list(range(len(doc)))
        if len(pages) > 8:
            step = len(pages) / 8
            pages = [int(i * step) for i in range(8)]
        rich = 0
        for p in pages:
            try:
                if len(doc[p].get_text()) > 200:
                    rich += 1
            except Exception:
                pass
        doc.close()
        shares[pdf] = rich / len(pages) if pages else None
    json.dump(shares, open(fp, "w"))
    return shares


def cmd_filter(_):
    J = load_judge()
    bench = [json.loads(l) for l in open(BENCH)]
    kept = {r["index"] for r in bench}
    cite = [d for d in json.load(open(CITE)) if d["dataset_type"] == "Single-Doc"]
    lines = ["# Does the verification filter favour the language interface?", ""]

    # (1) composition 987 vs 719 vs dropped
    def comp(items):
        n = len(items)
        qt = Counter(d["Question_Type"] for d in items)
        lg = Counter(d["language"] for d in items)
        nec = [e for d in items for e in d["Evidence"] if e["necessity"] == "necessary"]
        nontext = sum(1 for e in nec if e.get("type") in ("table", "image", "table_caption", "image_caption", "table_footnote", "image_footnote"))
        return n, qt, lg, nec, nontext

    rows = []
    for label, items in [("original single-doc (987)", cite), ("verified (719)", [d for d in cite if d["index"] in kept]),
                         ("dropped (268)", [d for d in cite if d["index"] not in kept])]:
        n, qt, lg, nec, nontext = comp(items)
        rows.append([label, n] + [f"{100*qt[t]/n:.0f}%" for t in QTYPES] + [f"{100*lg['zh']/n:.0f}%", f"{len(nec)/n:.2f}", f"{100*nontext/len(nec):.0f}%"])
    lines += ["## 1. Composition before and after filtering", "",
              md_table(["set", "n"] + QTYPES + ["zh share", "necessary elems / q", "non-textual share of necessary elems"], rows),
              "", "The filter inspects only ground-truth annotations against the source PDF; question-type, language and evidence-type "
              "mix are essentially unchanged, i.e. it does not remove the synthesis / numeric / figure questions that would be hardest for quoting.", ""]

    # (2) alignment groups
    align = {r["index"]: r["alignment"] for r in bench}
    cite_by = {d["index"]: d for d in cite}
    comp_rows = []
    for grp_name in ["verified", "unverifiable"]:
        sub = [r for r in bench if r["alignment"] == grp_name]
        nec = [e for r in sub for e in cite_by[r["index"]]["Evidence"] if e["necessity"] == "necessary"]
        tg = Counter(TYPE_GROUP.get(e.get("type", "other"), "other") for e in nec)
        qt = Counter(r["qtype"] for r in sub)
        comp_rows.append([grp_name, len(sub), f"{100*sum(1 for r in sub if r['language']=='zh')/len(sub):.0f}%",
                          f"{statistics.median(r['n_pdf_pages'] for r in sub):.0f}",
                          f"{100*tg['text']/len(nec):.0f}% / {100*tg['table']/len(nec):.0f}% / {100*tg['figure']/len(nec):.0f}%"]
                         + [f"{100*qt[t]/len(sub):.0f}%" for t in QTYPES])
    rows = []
    for d, tag, name in CONDS:
        vr = valid_rows(load_preds(d))
        g = defaultdict(list)
        for r in vr:
            g[align[r["index"]]].append(recall(r))
        jt = judged(vr, J, tag)
        gj = defaultdict(list)
        for r, q, rl, rc in jt:
            gj[align[r["index"]]].append((saa(q, rl, rc), q))
        rows.append([name, len(g["verified"]), fmt(100 * mean(g["verified"])), fmt(ah_of(gj["verified"])),
                     len(g["unverifiable"]), fmt(100 * mean(g["unverifiable"])), fmt(ah_of(gj["unverifiable"]))])
    lines += ["## 2. Text-verified vs. not-text-verifiable questions inside the 719", "",
              "Criterion (iii) could only be *applied* where text extraction under the annotated box succeeded ('verified', 366 questions); "
              "the other 353 questions were kept because the document was byte-identical to the annotated version ('unverifiable'). "
              "Both groups are inside the evaluation set, so this is not a comparison of kept versus dropped questions: it shows how the two "
              "interfaces behave on the questions whose annotated evidence could NOT be text-verified (typically table or figure evidence, or "
              "pages without a text layer), which the filter deliberately retained.", "",
              md_table(["group", "n", "zh share", "median pages", "necessary elems: text / table / figure"] + QTYPES, comp_rows), "",
              md_table(["condition", "n verified", "Rec@0.5", "AH", "n unverifiable", "Rec@0.5", "AH"], rows), "",
              "Reading: the language interface is lower on the not-text-verifiable group (more table and figure evidence), but it is still an order of "
              "magnitude above the coordinate interface there, and the coordinate interface is flat across the two groups. The hard group was kept, not dropped.", ""]

    # (3) scan-like documents
    shares = text_layer_shares()
    share_of = {r["index"]: shares.get(r["pdf_path"]) for r in bench}
    def grp(i):
        s = share_of[i]
        if s is None:
            return "unknown"
        return "scan-like (<25% text pages)" if s < 0.25 else ("partial (25–75%)" if s < 0.75 else "born-digital (≥75%)")
    groups = ["scan-like (<25% text pages)", "partial (25–75%)", "born-digital (≥75%)"]
    ndoc = Counter(grp(r["index"]) for r in bench)
    rows = []
    for d, tag, name in CONDS:
        vr = valid_rows(load_preds(d))
        g = defaultdict(list); gp = defaultdict(list)
        for r in vr:
            g[grp(r["index"])].append(recall(r)); gp[grp(r["index"])].append(page_recall(r))
        rows.append([name] + [f"{fmt(100*mean(g[k]))} / {fmt(100*mean(gp[k]))}" for k in groups])
    lines += ["## 3. Slice by document text layer (Rec@0.5 / page recall)", "",
              "Text-layer share = fraction of 8 sampled pages with more than 200 extractable characters (same statistic as Appendix B). "
              f"Questions per group: {', '.join(f'{k}: {ndoc[k]}' for k in groups)}.", "",
              md_table(["condition"] + groups, rows), "",
              "Reading: on scan-like documents the parser works from OCR, so both the ceiling and the language-interface recall drop, "
              "but the coordinate interface does not recover on them either; the interface gap is present in every slice.", ""]
    emit("filter", "\n".join(lines))


# ----------------------------------------------------------------------------- budget
def cmd_budget(_):
    J = load_judge()
    base = {r["index"]: r for r in valid_rows(load_preds("qwen3vl_nl_wd"))}
    grpo = {r["index"]: r for r in valid_rows(load_preds(f"{GRPO_DIR}_nl_wd"))}
    sims = defaultdict(list)
    for l in open(f"{CTRL}/sims_{GRPO_DIR}.jsonl"):
        o = json.loads(l); sims[o["index"]].append(o)
    # sanity: sims rows align with pred_tuples order
    n_chk = n_bad = 0
    for i, r in grpo.items():
        s = sims.get(i, [])
        if len(s) != len(r.get("pred_tuples", [])):
            n_bad += 1; continue
        for o, p in zip(s, preds_of(r)):
            n_chk += 1
            if int(any(iou_hit(g, p) for g in gt_nec(r))) != o["hit"]:
                n_bad += 1
    common = sorted(set(base) & set(grpo))
    lines = ["# GRPO at the base model's citation budget", "",
             f"Paired on the {len(common)} questions valid under both runs. k(question) = number of citations the base model emitted on that question. "
             "Two truncations of the trained model's citation list: by the model's own output order (first k quotes) and by retrieval similarity (top-k). "
             f"(similarity/pred_tuple alignment check: {n_bad} mismatches over {n_chk} citations.)", ""]

    def scores(rows_preds):
        R = [recall(r, p) for r, p in rows_preds]
        P = [precision(r, p) for r, p in rows_preds]
        F = [f1(rc, pr) for rc, pr in zip(R, P)]
        return 100 * mean(R), 100 * mean(P), 100 * mean(F), sum(1 for p in P if p is not None)

    variants = {"base (k)": [], "GRPO full": [], "GRPO first-k (output order)": [], "GRPO top-k (similarity)": []}
    ncite = {"base": [], "grpo": []}
    for i in common:
        b, g = base[i], grpo[i]
        k = len(b.get("pred_tuples", []))
        gp = preds_of(g)
        ncite["base"].append(k); ncite["grpo"].append(len(gp))
        variants["base (k)"].append((b, preds_of(b)))
        variants["GRPO full"].append((g, gp))
        variants["GRPO first-k (output order)"].append((g, gp[:k]))
        s = [o["sim"] for o in sims.get(i, [])]
        order = sorted(range(len(gp)), key=lambda j: -(s[j] if j < len(s) else -1))
        variants["GRPO top-k (similarity)"].append((g, [gp[j] for j in sorted(order[:k])]))
    rows = []
    for name, rp in variants.items():
        R, P, F, np_ = scores(rp)
        rows.append([name, fmt(R), fmt(P), fmt(F), np_])
    lines += ["## 1. Geometric view", "",
              md_table(["variant", "Rec@0.5", "precision", "F1", "n with citations"], rows),
              f"\nMean citations per response: base {mean(ncite['base']):.2f}, GRPO {mean(ncite['grpo']):.2f}.", "",
              "Reading: at the base budget the trained model's recall advantage disappears (as stated in Appendix H); the gain is coverage, "
              "i.e. it comes from quoting evidence the base model did not quote at all.", ""]

    # judge view
    jb = {r["index"]: (q, rl, rc) for r, q, rl, rc in judged(list(base.values()), J, "qwen3vl_nl")}
    jg = {r["index"]: (q, rl, rc) for r, q, rl, rc in judged(list(grpo.values()), J, "grpo_nl")}
    cj = sorted(set(jb) & set(jg))
    def stat(sel):
        sb = [saa_rel(*jb[i][:2]) for i in sel]; sg = [saa_rel(*jg[i][:2]) for i in sel]
        b = sum(1 for x, y in zip(sg, sb) if x == 1 and y == 0); c = sum(1 for x, y in zip(sg, sb) if x == 0 and y == 1)
        ahb = ah_of([(s, jb[i][0]) for s, i in zip(sb, sel)]); ahg = ah_of([(s, jg[i][0]) for s, i in zip(sg, sel)])
        relb = 20 * mean([jb[i][1] for i in sel]); relg = 20 * mean([jg[i][1] for i in sel])
        return len(sel), 100 * mean(sb), 100 * mean(sg), b, c, mcnemar_p(b, c), ahb, ahg, relb, relg
    n, sb, sg, b, c, p, ahb, ahg, relb, relg = stat(cj)
    lines += ["## 2. Judge view, independent of the number of citations", "",
              "SAA_rel = 1[Ans ≥ 4 ∧ Rel ≥ 4] drops the recall disjunct, so it cannot be inflated by citing more; the relevance judge reads the crops "
              "(≤ 6) and penalises irrelevant ones.", "",
              md_table(["n paired", "SAA_rel base", "SAA_rel GRPO", "McNemar b / c", "p", "AH_rel base", "AH_rel GRPO", "Rel base", "Rel GRPO"],
                       [[n, fmt(sb), fmt(sg), f"{b} / {c}", f"{p:.1e}", fmt(ahb), fmt(ahg), fmt(relb), fmt(relg)]]), ""]
    # split by whether GRPO cites more than base on that question
    more = [i for i in cj if len(grpo[i].get("pred_tuples", [])) > len(base[i].get("pred_tuples", []))]
    same = [i for i in cj if i not in set(more)]
    rows = []
    for label, sel in [("GRPO cites MORE than base", more), ("GRPO cites the same or fewer", same)]:
        if not sel:
            continue
        n, sb, sg, b, c, p, ahb, ahg, relb, relg = stat(sel)
        rb = 100 * mean([recall(base[i]) for i in sel]); rg = 100 * mean([recall(grpo[i]) for i in sel])
        rows.append([label, n, f"{fmt(rb)} → {fmt(rg)}", f"{fmt(relb)} → {fmt(relg)}", f"{fmt(sb)} → {fmt(sg)}", f"{b} / {c}", f"{p:.1e}"])
    lines += ["## 3. Split by whether the trained model cites more than the base on the same question", "",
              md_table(["subset", "n", "Rec@0.5 base → GRPO", "Rel base → GRPO", "SAA_rel base → GRPO", "McNemar b / c", "p"], rows), "",
              "If the extra citations were noise, judged relevance should fall in the 'cites more' subset; if it rises there, the extra citations land on supporting material.", ""]
    # relevance by citation count bins
    bins = [(1, 2), (3, 4), (5, 99)]
    rows = []
    for lo, hi in bins:
        cb = [jb[i][1] for i in cj if lo <= len(base[i].get("pred_tuples", [])) <= hi]
        cg = [jg[i][1] for i in cj if lo <= len(grpo[i].get("pred_tuples", [])) <= hi]
        rows.append([f"{lo}–{hi if hi < 99 else '+'} citations", len(cb), fmt(20 * mean(cb)), len(cg), fmt(20 * mean(cg))])
    lines += ["## 4. Judged relevance as a function of how many regions a response cites", "",
              md_table(["citations in response", "n base", "Rel base", "n GRPO", "Rel GRPO"], rows), ""]
    emit("budget", "\n".join(lines))


# ----------------------------------------------------------------------------- ansdrop
def cmd_ansdrop(_):
    J = load_judge()
    lines = ["# Why does the language interface lower Qwen3-VL-8B's answer score?", "",
             "Hypothesis in the paper: the quoting obligation strains the 1536-token output budget. Proxies available without re-inference: "
             "a language-interface response is 'truncated' if its JSON object is never closed (no parseable object), and its raw length in characters.", ""]
    rows, rows2 = [], []
    for d, t, name in BACKBONES:
        co = {r["index"]: r for r in valid_rows(load_preds(f"{d}_coord"))}
        nl = {r["index"]: r for r in valid_rows(load_preds(f"{d}_nl_wd"))}
        jc = {r["index"]: q for r, q, _, _ in judged(list(co.values()), J, f"{t}_coord")}
        jn = {r["index"]: q for r, q, _, _ in judged(list(nl.values()), J, f"{t}_nl")}
        common = sorted(set(jc) & set(jn))
        trunc = {i: parse_obj(nl[i].get("raw_predict", "")) is None for i in common}
        n_tr = sum(trunc.values())
        ok = [i for i in common if not trunc[i]]; bad = [i for i in common if trunc[i]]
        raw_len = [len(nl[i].get("raw_predict", "")) for i in common]
        ans_len_nl = [len(nl_answer(nl[i])) for i in ok]
        ans_len_co = [len(co[i].get("answer_text", "")) for i in common]
        rows.append([name, len(common), fmt(20 * mean([jc[i] for i in common])), fmt(20 * mean([jn[i] for i in common])),
                     f"{n_tr} ({100*n_tr/len(common):.1f}%)",
                     fmt(20 * mean([jn[i] for i in bad])) if bad else "--", fmt(20 * mean([jc[i] for i in bad])) if bad else "--",
                     fmt(20 * mean([jn[i] for i in ok])), fmt(20 * mean([jc[i] for i in ok])),
                     f"{int(statistics.median(raw_len))}", f"{int(statistics.median(ans_len_nl)) if ans_len_nl else '--'} / {int(statistics.median(ans_len_co))}"])
        # per-question paired delta by NL answer length quartile
        if d == "qwen3vl":
            pairs = sorted([(len(nl_answer(nl[i])), jn[i] - jc[i]) for i in ok])
            qn = len(pairs) // 4
            for qi in range(4):
                seg = pairs[qi * qn:(qi + 1) * qn] if qi < 3 else pairs[3 * qn:]
                rows2.append([f"Q{qi+1} (answer length {seg[0][0]}–{seg[-1][0]} chars)", len(seg), fmt(20 * mean([dlt for _, dlt in seg]), 1)])
    # paired delta by question type, all backbones; and "does the coordinate answer contain worked reasoning?"
    CALC = re.compile(r"\d\s*[=＝]\s*|[=＝]\s*\d|\btherefore\b|\bthus\b|\bhence\b|所以|因此|计算")
    rows3, rows4 = [], []
    for d, t, name in BACKBONES:
        co = {r["index"]: r for r in valid_rows(load_preds(f"{d}_coord"))}
        nl = {r["index"]: r for r in valid_rows(load_preds(f"{d}_nl_wd"))}
        jc = {r["index"]: q for r, q, _, _ in judged(list(co.values()), J, f"{t}_coord")}
        jn = {r["index"]: q for r, q, _, _ in judged(list(nl.values()), J, f"{t}_nl")}
        common = sorted(set(jc) & set(jn))
        cells = []
        for qt in QTYPES:
            sel = [i for i in common if co[i]["qtype"] == qt]
            cells.append(f"{fmt(20*mean([jn[i]-jc[i] for i in sel]))} (n={len(sel)})")
        rows3.append([name] + cells)
        for qt in ["Quantitative Reasoning", "Complex Synthesis"]:
            sel = [i for i in common if co[i]["qtype"] == qt]
            calc_co = 100 * mean([1.0 if CALC.search(co[i].get("answer_text", "")) else 0.0 for i in sel])
            calc_nl = 100 * mean([1.0 if CALC.search(nl_answer(nl[i])) else 0.0 for i in sel])
            rows4.append([name, qt, len(sel), fmt(calc_co), fmt(calc_nl),
                          f"{int(statistics.median(len(co[i].get('answer_text','')) for i in sel))} / {int(statistics.median(len(nl_answer(nl[i])) for i in sel))}"])
    lines += [md_table(["backbone", "n paired", "Ans coord.", "Ans lang.", "lang. responses with no parseable JSON (truncated)",
                        "Ans lang. on truncated", "Ans coord. on same", "Ans lang. on parsed", "Ans coord. on same",
                        "median raw chars (lang.)", "median answer chars lang. / coord."], rows), "",
              "## Paired Ans(lang.) − Ans(coord.) by question type (judged intersection, ×20 scale)", "",
              md_table(["backbone"] + QTYPES, rows3), "",
              "## Does the answer contain worked reasoning? (share of answers with an '=' between digits or a 'therefore/thus/所以/因此/计算' marker)", "",
              "Under the coordinate prompt the official in-context example shows a step-by-step calculation with inline boxes, so models write out their "
              "working in the answer; under the language prompt the JSON contract asks for a bare answer string.", "",
              md_table(["backbone", "question type", "n", "% coord. answers with working", "% lang. answers with working", "median answer chars coord. / lang."], rows4), "",
              "## Qwen3-VL-8B: paired Ans(lang.) − Ans(coord.) by language-interface answer length (parsed responses only)", "",
              md_table(["quartile", "n", "mean Δ Ans (×20)"], rows2), "",
              "Reading: if the drop is concentrated in truncated responses and disappears on parsed ones, the budget explanation holds and a 4096-token "
              "rerun (in progress) should close the gap; if the drop persists on parsed responses, part of it is the quoting burden itself "
              "(shorter, more clipped answers under the JSON contract).", ""]
    emit("ansdrop", "\n".join(lines))


# ----------------------------------------------------------------------------- decomp
def iou_raw(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1); ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def snap_preds(r, blocks):
    by_page = defaultdict(list)
    for bl in blocks:
        by_page[bl[1]].append(bl)
    out = []
    for t in r.get("pred_tuples", []):
        _, pg, x1, y1, x2, y2 = t
        cands = by_page.get(int(pg), [])
        if not cands:
            out.append(tuple(t)); continue
        ious = [iou_raw((x1, y1, x2, y2), c[2:]) for c in cands]
        best = max(range(len(cands)), key=lambda j: ious[j])
        if ious[best] <= 0:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            best = min(range(len(cands)), key=lambda j: (((cands[j][2] + cands[j][4]) / 2 - cx) ** 2 + ((cands[j][3] + cands[j][5]) / 2 - cy) ** 2))
        out.append((1, cands[best][1], *cands[best][2:]))
    return out


def cmd_decomp(_):
    lines = ["# Model contribution versus pipeline contribution", "",
             "Every row is Rec@0.5 on all 719 valid responses with the same candidate set (MinerU blocks), the same encoder and the same per-question "
             "citation budget. The ladder adds one ingredient at a time. 'union' columns score a question by the better of two readings of the same "
             "model output (a lower bound on the true union; exact union follows once the GPU dump has run, see `union`).", ""]
    ceil = {}
    bench = {json.loads(l)["index"]: json.loads(l) for l in open(BENCH)}
    blk = {}
    for i, r in bench.items():
        if r["pdf_path"] not in blk:
            blk[r["pdf_path"]] = blocks_for(r["pdf_path"], CACHE) or []
        gt = gt_nec(r)
        ceil[i] = sum(1 for g in gt if any(iou_hit(g, b) for b in blk[r["pdf_path"]])) / len(gt)
    rows = []
    pooled = defaultdict(list)
    for d, t, name in BACKBONES + [(GRPO_DIR, "grpo", "8B + GRPO")]:
        raw = snap = None
        if d != GRPO_DIR:
            co = valid_rows(load_preds(f"{d}_coord"))
            raw = 100 * mean([recall(r) for r in co])
            snap = 100 * mean([recall(r, snap_preds(r, blk[r["pdf_path"]])) for r in co])
        nl = {r["index"]: r for r in valid_rows(load_preds(f"{d}_nl_wd"))}
        e1 = {json.loads(l)["index"]: json.loads(l) for l in open(f"{CTRL}/e1_perq_{d}.jsonl")}
        idx = sorted(set(nl) & set(e1))
        rq = [e1[i]["q_only"] for i in idx]; rqa = [e1[i]["qa"] for i in idx]; rquo = [recall(nl[i]) for i in idx]
        u_quo_qa = [max(a, b) for a, b in zip(rquo, rqa)]; u_q_qa = [max(a, b) for a, b in zip(rq, rqa)]
        for k, v in [("q", rq), ("qa", rqa), ("quotes", rquo), ("quotes∪qa", u_quo_qa), ("q∪qa", u_q_qa)]:
            pooled[k].extend(v)
        r_ = pearson(rquo, rqa)
        diff = 100 * mean([1.0 if a != b else 0.0 for a, b in zip(rquo, rqa)])
        qb = 100 * mean([1.0 if a > b else 0.0 for a, b in zip(rquo, rqa)]); ab = 100 * mean([1.0 if b > a else 0.0 for a, b in zip(rquo, rqa)])
        rows.append([name, fmt(raw), fmt(snap), fmt(100 * mean(rq)), fmt(100 * mean(rqa)), fmt(100 * mean(rquo)),
                     fmt(100 * mean(u_quo_qa)), fmt(100 * mean(u_q_qa)), fmt(100 * mean([ceil[i] for i in idx])),
                     fmt(r_, 2), f"{fmt(diff)} ({fmt(qb)} / {fmt(ab)})"])
    lines += [md_table(["backbone", "coord raw", "coord +snap", "retriever: q", "retriever: q+a", "quotes (ours)", "quotes ∪ q+a", "q ∪ q+a",
                        "parser ceiling", "r(quotes, q+a)", "% questions differing (quotes better / q+a better)"], rows), "",
              f"Pooled over the seven language runs: q {100*mean(pooled['q']):.1f}, q+a {100*mean(pooled['qa']):.1f}, quotes {100*mean(pooled['quotes']):.1f}, "
              f"quotes ∪ q+a {100*mean(pooled['quotes∪qa']):.1f} (+{100*(mean(pooled['quotes∪qa'])-mean(pooled['qa'])):.1f} over q+a), "
              f"q ∪ q+a {100*mean(pooled['q∪qa']):.1f} (+{100*(mean(pooled['q∪qa'])-mean(pooled['qa'])):.1f} over q+a, the query-diversity control).", "",
              "Ladder reading. coord raw → +snap: giving the coordinate stream the same candidate set and granularity recovers little, so the deficit is not granularity. "
              "q → q+a: the retriever alone is far below the language runs; adding the model's generated answer closes most of the gap, so the load-bearing signal is "
              "model-generated language. q+a → quotes ∪ q+a: the quotes reach 11 points of evidence that question-plus-answer retrieval never reaches, against 2–3 points "
              "for a second generic query, so the quote channel is not redundant even where its mean equals q+a. All language rows sit far below the parser ceiling, "
              "so the ceiling is not what limits the comparison.", ""]
    emit("decomp", "\n".join(lines))


# ----------------------------------------------------------------------------- buckets (post-GPU)
def cmd_buckets(_):
    lines = ["# Where the language interface loses evidence, by question type", "",
             "Per necessary evidence element (N = 1,034). recovered = matched by the produced citation; rank-limited = correct block in some quote's top-5 but "
             "lost at assignment; not retrieved = in no quote's top-5 (the model did not quote it); parser-unreachable = no block overlaps the annotation at IoU 0.5 "
             "('contain' = share of the annotation covered by the best block).", ""]
    for tag, name in [("base", "Qwen3-VL-8B lang."), ("grpo", "8B + GRPO lang.")]:
        fp = f"{REB}/buckets_{tag}.jsonl"
        if not os.path.exists(fp):
            lines.append(f"({name}: {fp} not found; run slurm/revision_gpu_dump.sh first)"); continue
        rows = [json.loads(l) for l in open(fp)]
        for key, groups in [("qtype", QTYPES), ("elem_type", ["text", "table", "figure"]), ("language", ["en", "zh"])]:
            tab = []
            for g in groups:
                sub = [r for r in rows if r[key] == g]
                if not sub:
                    continue
                c = Counter(r["bucket"] for r in sub); n = len(sub)
                unreach = [r for r in sub if r["bucket"] == "parser_unreachable"]
                cont = 100 * mean([1.0 if r["contain"] >= 0.5 else 0.0 for r in unreach]) if unreach else float("nan")
                tab.append([g, n] + [f"{100*c[b]/n:.1f}" for b in ["matched@1", "recoverable@5", "not_retrieved", "parser_unreachable"]] + [fmt(cont)])
            lines += [f"## {name} — by {key}", "",
                      md_table([key, "elements", "recovered", "rank-limited", "not retrieved", "parser-unreachable", "% of unreachable with contain ≥ 0.5"], tab), ""]
    emit("buckets", "\n".join(lines))


# ----------------------------------------------------------------------------- union (post-GPU)
def cmd_union(_):
    lines = ["# Exact union of quote retrieval and question+answer retrieval", "",
             "Unlike the oracle per-question max in the paper, here the two citation sets are actually merged before scoring, so the union is exact "
             "for multi-element questions too. Same encoder, candidates and per-question budget k; the merged set has up to 2k citations, "
             "so precision is reported alongside.", ""]
    rows = []
    for d, t, name in BACKBONES + [(GRPO_DIR, "grpo", "8B + GRPO")]:
        fp = f"{REB}/qa_boxes_{d}.jsonl"
        if not os.path.exists(fp):
            lines.append(f"({name}: {fp} not found; run slurm/revision_gpu_dump.sh first)"); continue
        qa = {json.loads(l)["index"]: json.loads(l) for l in open(fp)}
        nl = {r["index"]: r for r in valid_rows(load_preds(f"{d}_nl_wd"))}
        idx = sorted(set(nl) & set(qa))
        R = defaultdict(list); P = defaultdict(list)
        for i in idx:
            r = nl[i]
            sets = {"quotes": preds_of(r), "q+a": [tuple(x) for x in qa[i]["qa_tuples"]], "q": [tuple(x) for x in qa[i]["q_tuples"]]}
            sets["quotes ∪ q+a"] = sets["quotes"] + sets["q+a"]
            sets["q ∪ q+a"] = sets["q"] + sets["q+a"]
            for k, p in sets.items():
                R[k].append(recall(r, p)); P[k].append(precision(r, p))
        rows.append([name, len(idx)] + [f"{fmt(100*mean(R[k]))} / {fmt(100*mean(P[k]))}" for k in ["q", "q+a", "quotes", "quotes ∪ q+a", "q ∪ q+a"]])
    lines += [md_table(["backbone", "n", "q (R / P)", "q+a (R / P)", "quotes (R / P)", "quotes ∪ q+a (R / P)", "q ∪ q+a (R / P)"], rows), ""]
    emit("union", "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["anls", "qtype", "filter", "budget", "ansdrop", "decomp", "buckets", "union"])
    args = ap.parse_args()
    os.chdir(ROOT)
    {"anls": cmd_anls, "qtype": cmd_qtype, "filter": cmd_filter, "budget": cmd_budget,
     "ansdrop": cmd_ansdrop, "decomp": cmd_decomp, "buckets": cmd_buckets, "union": cmd_union}[args.cmd](args)


if __name__ == "__main__":
    main()
