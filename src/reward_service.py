"""Persistent GRPO reward service (decoupled from the EasyR1 actor).

Holds the embedder (Qwen3-VL-Embedding-2B) + the precomputed block-emb cache, and calls
a co-located vLLM VISION judge (Qwen3.5-9B, OpenAI API) for ac/rel/cov. The EasyR1 reward
function POSTs a batch of (index, response, ground_truth) and gets back fine-grained rewards.

Pipeline per response (SAME as reward_longdoc / eval):
  parse {answer, evidence:[quote]} -> embed quotes -> bipartite-match to cached block_emb
  of the item's window (IGNORE model page) -> crop matched blocks -> vision judge -> reward.

Batched 3-pass for throughput: embed ALL quotes once, match (CPU), judge ALL in parallel.
reward = FLOOR + (1-FLOOR) * gate(ac) * (rel/5) * (cov/5)   [product combiner; user choice]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qr_paths  # noqa: F401,E402  sets PROJECT_ROOT / QR_CACHE / QWEN3VL_EMB_REPO defaults
import sys, os, json, re, io, base64, threading
sys.path.insert(0, os.path.expandvars("${QWEN3VL_EMB_REPO}"))
sys.path.insert(0, os.path.expandvars("${PROJECT_ROOT}/src"))
import numpy as np, torch, fitz
from PIL import Image
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from scipy.optimize import linear_sum_assignment
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
from reward_longdoc import JUDGE_SYS, parse_judge, crop, parse_quotes

EMB_CACHE = os.path.expandvars("${QR_CACHE}/ldoc_block_emb")
EMB_MODEL = os.environ.get("EMB_MODEL", os.path.expandvars("${QR_CACHE}/models--Qwen--Qwen3-VL-Embedding-2B/snapshots"))
JUDGE_URL = os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "judge")
JUDGE_WORKERS = int(os.environ.get("JUDGE_WORKERS", "24"))
FLOOR = float(os.environ.get("REWARD_FLOOR", "0.0"))
CROP_MAXPX = 600_000

def _snap(p):
    return os.path.join(p, sorted(os.listdir(p))[0]) if os.path.isdir(p) and os.path.basename(p) == "snapshots" else p

print(f"[reward_service] loading embedder {EMB_MODEL}", flush=True)
_emb = Qwen3VLEmbedder(model_name_or_path=_snap(EMB_MODEL), torch_dtype=torch.float16, attn_implementation="sdpa")
_emb_lock = threading.Lock()
_page_cache = OrderedDict()       # index -> {pidx: PIL}; LRU
_PAGE_CAP = 64

@torch.no_grad()
def embed(items):
    with _emb_lock:
        out = [_emb.process(items[k:k+48]).float().cpu().numpy() for k in range(0, len(items), 48)]
    return np.concatenate(out, 0) if out else np.zeros((0, 2048), np.float32)

def window_pages(index, pdf_path, start, end):
    if index in _page_cache:
        _page_cache.move_to_end(index); return _page_cache[index]
    doc = fitz.open(pdf_path); pages = {}
    for pidx in range(start - 1, min(end, len(doc))):
        pix = doc[pidx].get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
        pages[pidx] = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    _page_cache[index] = pages
    if len(_page_cache) > _PAGE_CAP:
        _page_cache.popitem(last=False)
    return pages

def load_block_emb(doc_no, start, end):
    fp = os.path.join(EMB_CACHE, f"{doc_no}_{start}_{end}.npz")
    if not os.path.exists(fp):
        return np.zeros((0, 2048), np.float32), np.zeros(0, int), np.zeros((0, 4))
    d = np.load(fp); return d["E"], d["pages"], d["bboxes"]

def match(Q, E, pages, boxes):
    """Q[m,d] quotes vs E[n,d] blocks -> list[(pidx, bbox)] (bipartite, eval-identical)."""
    if Q.shape[0] == 0 or E.shape[0] == 0:
        return []
    S = Q @ E.T; m, n = S.shape; out = []
    if n >= m:
        _, cols = linear_sum_assignment(-S)
        for j in cols: out.append((int(pages[j]), boxes[j]))
    else:
        for row in S: j = int(np.argmax(row)); out.append((int(pages[j]), boxes[j]))
    return out

def _b64(im):
    b = io.BytesIO(); im.save(b, format="PNG"); return base64.b64encode(b.getvalue()).decode()

def judge_http(question, gt, answer, crops):
    import urllib.request
    head = f"Question: {question}\nReference answer: {gt}\nSystem answer: {answer}\n" + (
        f"Cited evidence ({len(crops)} cropped regions), shown below:" if crops
        else "Cited evidence: No evidence was provided.")
    content = [{"type": "text", "text": head}]
    for c in crops:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(c)}"}})
    body = json.dumps({"model": JUDGE_MODEL, "temperature": 0, "max_tokens": 512,
                       "chat_template_kwargs": {"enable_thinking": False},   # Qwen3.5 is a thinking model; off -> direct JSON
                       "messages": [{"role": "system", "content": JUDGE_SYS},
                                    {"role": "user", "content": content}]}).encode()
    try:
        req = urllib.request.Request(JUDGE_URL + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        content = r["choices"][0]["message"]["content"]
        if os.environ.get("REWARD_DEBUG"):
            print(f"[judge raw | {len(crops)} crops] {content[:400]!r}", flush=True)
        return parse_judge(content)
    except Exception as ex:
        if os.environ.get("REWARD_DEBUG"):
            print(f"[judge ERROR] {repr(ex)[:200]}", flush=True)
        return None

def score_batch(items):                          # [{index, response, ground_truth}]
    # pass 1: parse + load blocks/pages, collect all quotes
    state = []                                    # per item
    all_quotes = []; spans = []
    for it in items:
        gt = json.loads(it["ground_truth"]) if isinstance(it["ground_truth"], str) else it["ground_truth"]
        smp_quotes = parse_quotes_from_response(it["response"])[:5]
        ans = parse_answer(it["response"])
        E, pages, boxes = load_block_emb(gt["doc_no"], gt["page_window"][0], gt["page_window"][1])
        s0 = len(all_quotes); all_quotes.extend(smp_quotes); spans.append((s0, len(all_quotes)))
        state.append({"gt": gt, "ans": ans, "quotes": smp_quotes, "E": E, "pages": pages, "boxes": boxes})
    # pass 2: embed ALL quotes once, match per item, crop
    QE = embed([{"text": q} for q in all_quotes]) if all_quotes else np.zeros((0, 2048), np.float32)
    judge_args = []
    for st, (a, b) in zip(state, spans):
        Q = QE[a:b]
        matched = match(Q, st["E"], st["pages"], st["boxes"])[:5]
        crops = []
        if matched:
            gt = st["gt"]; pg = window_pages(gt["index"], gt["pdf_path"], *gt["page_window"])
            for pidx, bb in matched:
                if pidx in pg:
                    c = crop(pg[pidx], bb)
                    if c is not None: crops.append(c)
        judge_args.append((st["gt"]["question"], st["gt"]["gt_answer"], st["ans"], crops, len(st["quotes"]), len(crops)))
    # pass 3: judge ALL in parallel
    def run(arg):
        q, gt, ans, crops, nq, nm = arg
        J = judge_http(q, gt, ans, crops)
        if J is None: J = {"ac": 0, "rel": 0, "cov": 0, "reason": "JUDGE_FAIL"}
        ac, rel, cov = J["ac"], J["rel"], J["cov"]
        # C2 reward: soft answer-gate x ADDITIVE evidence -> long-tail (denser within-item
        # gradient; many more prompts land in DAPO's [filter_low, filter_high] keep band).
        grounded = (ac / 5.0) * (rel + cov) / 10.0
        overall = FLOOR + (1.0 - FLOOR) * grounded
        return {"overall": round(overall, 4), "answer_correctness": ac, "evidence_relevance": rel,
                "evidence_coverage": cov, "n_quotes": nq, "n_matched": nm}
    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
        out = list(ex.map(run, judge_args))
    return {"rewards": out}

# response parsing (NL-grounding grammar)
_FENCE = re.compile(r"^\s*```(?:json)?|```\s*$", re.I)
def _obj(text):
    t = _FENCE.sub("", (text or "").strip()).strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e <= s: return None
    try: return json.loads(t[s:e+1])
    except Exception: return None
def parse_quotes_from_response(text):
    o = _obj(text)
    if not o: return []
    return [str(x.get("quote", "")) for x in (o.get("evidence") or []) if isinstance(x, dict) and x.get("quote")]
def parse_answer(text):
    o = _obj(text); return str(o.get("answer", "")) if o else ""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/score":
            self._send(404, {"error": "not found"}); return
        n = int(self.headers.get("Content-Length", 0))
        try:
            items = json.loads(self.rfile.read(n))["items"]
            self._send(200, score_batch(items))
        except Exception as ex:
            self._send(500, {"error": repr(ex)[:300]})


if __name__ == "__main__":
    port = int(os.environ.get("REWARD_PORT", "8001"))
    print(f"[reward_service] ready on :{port} (judge={JUDGE_URL})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
