#!/usr/bin/env python3
"""Side-by-side evidence boxes: quote retrieval vs question+answer retrieval (camera-ready, Sec. 5.3).

Two real Qwen3-VL-8B questions from the verified set, same encoder / candidate blocks / budget;
only the query differs. Case 1 (table evidence): the verbatim quote hits, q+a drifts to another
page. Case 2 (figure evidence): the figure cannot be quoted, q+a hits the chart block.
Inputs: outputs/matrix/qwen3vl_nl_wd/predictions.jsonl (quote boxes, gold boxes) and
outputs/revision/qa_boxes_qwen3vl.jsonl (q+a boxes). CPU only.
Usage: python src/make_fig_sidebyside.py  ->  paper/latex/figs/fig_sidebyside.pdf
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qr_paths  # noqa: F401,E402  sets PROJECT_ROOT / QR_CACHE / QWEN3VL_EMB_REPO defaults
import json, sys
import numpy as np
import fitz
import PIL.Image as PILImage
sys.path.insert(0, os.path.expandvars("${PROJECT_ROOT}/src"))
from make_figures import plt, Rectangle, ROOT, FIGS, C_LANG, C_GT, INK, iou

C_QA = "#CC79A7"      # reddish purple (Okabe-Ito) = question+answer retrieval
CASES = [("2ee04fb175cd", "Case 1: table evidence"), ("398abf59-469", "Case 2: figure evidence")]
ASPECT = 1.15         # crop height / width, shared by all panels


def render(pdf, page1, dpi=150):
    doc = fitz.open(pdf)
    pix = doc[page1 - 1].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    img = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return np.asarray(img)


def best(tuples, gold):
    """the retrieved box that best overlaps the gold element (same page), else the first one."""
    gp, gb = gold[0], gold[1:5]
    same = [t for t in tuples if t[1] == gp]
    if same:
        t = max(same, key=lambda t: iou(tuple(t[2:6]), tuple(gb)))
        return t, iou(tuple(t[2:6]), tuple(gb))
    return tuples[0], 0.0


def panel(ax, pdf, box_t, gold, color, name):
    page = box_t[1]
    img = render(pdf, page)
    H, W = img.shape[:2]
    px = lambda b: (b[0] / 1000 * W, b[1] / 1000 * H, b[2] / 1000 * W, b[3] / 1000 * H)
    b = px(box_t[2:6])
    boxes = [b]
    g = px(gold[1:5]) if gold[0] == page else None
    if g:
        boxes.append(g)
    x1 = max(0, min(v[0] for v in boxes) - 0.06 * W); x2 = min(W, max(v[2] for v in boxes) + 0.06 * W)
    y1 = max(0, min(v[1] for v in boxes) - 0.07 * H); y2 = min(H, max(v[3] for v in boxes) + 0.07 * H)
    cw, ch = x2 - x1, y2 - y1
    if ch / cw < ASPECT:
        need = ASPECT * cw - ch
        y1 -= need / 2; y2 += need / 2
        if y1 < 0: y2 -= y1; y1 = 0
        if y2 > H: y1 -= (y2 - H); y2 = H
        y1 = max(0, y1)
    else:
        need = ch / ASPECT - cw
        x1 = max(0, x1 - need / 2); x2 = min(W, x2 + need / 2)
    ax.imshow(img); ax.set_xlim(x1, x2); ax.set_ylim(y2, y1)
    ax.set_xticks([]); ax.set_yticks([])
    if g:
        ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec=C_GT, lw=2.2))
    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=color, lw=1.6,
                           ls=(0, (4, 2))))
    hit = g is not None and iou(tuple(box_t[2:6]), tuple(gold[1:5])) >= 0.5
    tag = f"{name}: hit (IoU {iou(tuple(box_t[2:6]), tuple(gold[1:5])):.2f})" if hit else f"{name}: miss"
    ax.text(0.03, 0.97, tag, transform=ax.transAxes, fontsize=6.6, color=color, fontweight="bold",
            va="top", ha="left", bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9))
    note = f"page {page}" + ("" if g else f"  (gold element is on page {gold[0]})")
    ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=6.2, color=INK, va="bottom", ha="left",
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9))
    return hit


def main():
    P = {}
    for l in open(f"{ROOT}/outputs/matrix/qwen3vl_nl_wd/predictions.jsonl"):
        r = json.loads(l); P[r["index"]] = r
    B = {}
    for l in open(f"{ROOT}/outputs/revision/qa_boxes_qwen3vl.jsonl"):
        r = json.loads(l); B[r["index"]] = r
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.35), dpi=300)
    for ci, (prefix, title) in enumerate(CASES):
        idx = [k for k in P if k.startswith(prefix) and len(P[k]["gt_necessary"]) == 1][0]
        r, qa = P[idx], B[idx]
        gold = r["gt_necessary"][0]
        qt, _ = best(r["pred_tuples"], gold)
        at, _ = best(qa["qa_tuples"], gold)
        h1 = panel(axes[2 * ci], r["pdf_path"], qt, gold, C_LANG, "quotes")
        h2 = panel(axes[2 * ci + 1], r["pdf_path"], at, gold, C_QA, "question+answer")
        axes[2 * ci].set_title(title, fontsize=7.6, pad=4, loc="left")
        print(f"{title}: {idx} | gold {gold} | quote box {qt} hit={h1} | q+a box {at} hit={h2}")
        print("   Q:", r["question"])
    fig.tight_layout(pad=0.4, w_pad=0.6)
    out = f"{FIGS}/fig_sidebyside.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print("->", out)


if __name__ == "__main__":
    main()
