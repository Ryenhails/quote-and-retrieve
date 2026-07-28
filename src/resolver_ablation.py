#!/usr/bin/env python3
"""E2 quote-fidelity + lexical resolver ablation (CPU, whole-document scope).

Holds the model quotes fixed (parsed from each NL condition's raw_predict) and swaps only the
quote -> block resolver, always over the whole-document MinerU candidate set with the same
one-to-one assignment as the production pipeline:
  M0-lex   token-F1 between quote text and block text
  BM25     Okapi BM25 over block texts (k1=1.5, b=0.75), per document
  (M1 text-emb and M2 image-emb are produced by the GPU job / production run.)

Also measures quote fidelity per backbone: the fraction of quotes that appear verbatim
(normalized exact substring) in some parsed block, the fraction with a fuzzy match
(token-F1 >= 0.8), and question-level Rec@0.5 split by quote fidelity.

Usage: python src/resolver_ablation.py [--conds qwen3vl,gemma3,...]
"""
import argparse, json, math, os, re, sys, statistics
from collections import defaultdict

sys.path.insert(0, "src")
from score_citevqa import iou_hit

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

ROOT = "${PROJECT_ROOT}"
BENCH = f"{ROOT}/data/citevqa/citevqa_singledoc_pub.jsonl"
CACHE = f"{ROOT}/data/citevqa/mineru_cache"
MATRIX = f"{ROOT}/outputs/matrix"
THRESH = [0.1, 0.3, 0.5, 0.7]

BACKBONES = [
    ("qwen3vl", "Qwen3-VL-8B"),
    ("qwen3vl_30b", "Qwen3-VL-30B-A3B"),
    ("qwen35_9b", "Qwen3.5-9B"),
    ("qwen35_27b", "Qwen3.5-27B"),
    ("gemma3", "Gemma-3-12B"),
    ("gemma4_31b", "Gemma-4-31B"),
    ("grpo_global_step_144_lr1e5", "GRPO (8B) NL"),
]

_TOK = re.compile(r"[a-z0-9]+|[一-鿿]")


def toks(s):
    return _TOK.findall((s or "").lower())


def tokf1(q_set, b_set):
    if not q_set or not b_set:
        return 0.0
    inter = len(q_set & b_set)
    if not inter:
        return 0.0
    prec, rec = inter / len(q_set), inter / len(b_set)
    return 0.7 * prec + 0.3 * (2 * prec * rec / (prec + rec))


def norm_text(s):
    return re.sub(r"[\s　]+", "", (s or "").lower())


def parse_quotes(raw):
    t = re.sub(r"^\s*```(?:json)?|```\s*$", "", (raw or "").strip(), flags=re.I).strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e <= s:
        return []
    try:
        ev = json.loads(t[s:e + 1]).get("evidence") or []
    except Exception:
        return []
    return [str(x.get("quote", "")) for x in ev if isinstance(x, dict) and x.get("quote")]


class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.N = len(docs_tokens)
        self.avgdl = sum(len(d) for d in docs_tokens) / max(1, self.N)
        self.df = defaultdict(int)
        self.tf = []
        for d in docs_tokens:
            c = defaultdict(int)
            for t in d:
                c[t] += 1
            self.tf.append(c)
            for t in c:
                self.df[t] += 1

    def score(self, query_tokens, j):
        s, dl = 0.0, len(self.docs[j])
        for t in set(query_tokens):
            f = self.tf[j].get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


def assign(S_rows, blocks):
    """one-to-one assignment identical to the production matcher."""
    import numpy as np
    S = np.array(S_rows)
    if S.size == 0:
        return []
    m, n = S.shape
    preds = []
    if HAVE_SCIPY and n >= m:
        _, cols = linear_sum_assignment(-S)
        for j in cols:
            preds.append(blocks[int(j)])
    else:
        for row in S:
            preds.append(blocks[int(row.argmax())])
    return preds


def doc_blocks(pdf_path):
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    cpath = os.path.join(CACHE, f"{stem}.json")
    if not os.path.exists(cpath):
        return [], []
    cache = json.load(open(cpath))
    blocks, texts = [], []
    for pidx_s, els in cache.items():
        p = int(pidx_s) + 1
        for b in els:
            bb = b.get("bbox")
            if bb and len(bb) == 4:
                blocks.append((1, p, bb[0], bb[1], bb[2], bb[3]))
                texts.append(b.get("text") or "")
    return blocks, texts


def rec_multi(gt, preds):
    out = {}
    gp = {g[1] for g in gt}
    pp = {p[1] for p in preds}
    out["page"] = len(gp & pp) / len({g[1] for g in gt}) if gt else None
    for t in THRESH:
        out[t] = sum(1 for g in gt if any(iou_hit(g, p, t) for p in preds)) / len(gt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="")
    args = ap.parse_args()
    os.chdir(ROOT)
    conds = BACKBONES
    if args.conds:
        want = set(args.conds.split(","))
        conds = [c for c in BACKBONES if c[0] in want]

    print("=== E2 quote fidelity + lexical resolver ablation (whole-document scope) ===")
    for dir_pfx, name in conds:
        preds_nl = [json.loads(l) for l in open(f"{MATRIX}/{dir_pfx}_nl/predictions.jsonl")]
        valid = [r for r in preds_nl if not r.get("error") and r.get("gt_necessary")]

        n_quotes = n_exact = n_fuzzy = 0
        agg = {k: defaultdict(list) for k in ["M0-lex", "BM25"]}
        rec_verbatim, rec_nonverb = [], []
        doc_cache = {}
        for r in valid:
            quotes = parse_quotes(r.get("raw_predict", ""))
            gt = [(1, b[0], b[1], b[2], b[3], b[4]) for b in r["gt_necessary"]]
            if not gt:
                continue
            pdf = r["pdf_path"]
            if pdf not in doc_cache:
                blocks, texts = doc_blocks(pdf)
                btoks = [toks(t) for t in texts]
                bsets = [set(t) for t in btoks]
                bnorm = [norm_text(t) for t in texts]
                doc_cache[pdf] = (blocks, texts, btoks, bsets, bnorm, BM25(btoks) if blocks else None)
            blocks, texts, btoks, bsets, bnorm, bm25 = doc_cache[pdf]
            if not quotes or not blocks:
                for k in agg:
                    for th in ["page"] + THRESH:
                        agg[k][th].append(0.0)
                continue

            q_is_verbatim = []
            for q in quotes:
                n_quotes += 1
                qn = norm_text(q)
                qs = set(toks(q))
                exact = bool(qn) and len(qn) >= 6 and any(qn in bn for bn in bnorm)
                fuzzy = exact or max((tokf1(qs, bs) for bs in bsets), default=0.0) >= 0.8
                n_exact += exact
                n_fuzzy += fuzzy
                q_is_verbatim.append(exact)

            S_lex = [[tokf1(set(toks(q)), bs) for bs in bsets] for q in quotes]
            S_bm = [[bm25.score(toks(q), j) for j in range(len(blocks))] for q in quotes]
            for key, S in [("M0-lex", S_lex), ("BM25", S_bm)]:
                p = assign(S, blocks)
                m = rec_multi(gt, p)
                for th in ["page"] + THRESH:
                    agg[key][th].append(m[th])

            # production (image-emb) recall from the _nl_wd file is joined later; here split
            # question-level fidelity: all-verbatim vs any-non-verbatim
            m_prod = None
            (rec_verbatim if all(q_is_verbatim) else rec_nonverb).append(r["index"])

        print(f"\n[{name}]  n={len(valid)}  quotes={n_quotes}  "
              f"verbatim(exact)={100*n_exact/max(1,n_quotes):.1f}%  fuzzy(F1>=0.8)={100*n_fuzzy/max(1,n_quotes):.1f}%")
        hdr = f"  {'resolver':<10}{'page':>7}" + "".join(f"{'@'+str(t):>7}" for t in THRESH)
        print(hdr)
        for key in ["M0-lex", "BM25"]:
            line = f"  {key:<10}{100*statistics.mean(agg[key]['page']):>7.1f}"
            for t in THRESH:
                line += f"{100*statistics.mean(agg[key][t]):>7.1f}"
            print(line)

        # production Rec@0.5 split by quote fidelity (from the _nl_wd matched file)
        wd = {r["index"]: r for r in (json.loads(l) for l in open(f"{MATRIX}/{dir_pfx}_nl_wd/predictions.jsonl"))
              if not r.get("error") and r.get("gt_necessary")}

        def prod_rec(idxs):
            vals = []
            for i in idxs:
                r = wd.get(i)
                if not r:
                    continue
                gt = [(1, b[0], b[1], b[2], b[3], b[4]) for b in r["gt_necessary"]]
                p = [tuple(t) for t in r.get("pred_tuples", [])]
                vals.append(sum(1 for g in gt if any(iou_hit(g, x) for x in p)) / len(gt))
            return 100 * statistics.mean(vals) if vals else 0.0

        print(f"  production Rec@0.5 | all-quotes-verbatim (n={len(rec_verbatim)}): {prod_rec(rec_verbatim):.1f}"
              f"   any-non-verbatim (n={len(rec_nonverb)}): {prod_rec(rec_nonverb):.1f}")


if __name__ == "__main__":
    main()
