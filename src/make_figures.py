#!/usr/bin/env python3
"""Paper figures (matplotlib, Okabe-Ito CVD-safe palette, one axis, direct labels).

fig3  threshold-decay curves  -> paper/latex/figs/fig_threshold.pdf   (single column)
figa1 GRPO validation curve   -> paper/latex/figs/fig_valcurve.pdf    (single column, appendix)
figa2 residual decomposition  -> paper/latex/figs/fig_residual.pdf    (single column, appendix)
fig5  qualitative 3-panel     -> paper/latex/figs/fig_cases.pdf       (full width)
      (a) coarse coord box vs element GT   (b) prose-location output   (c) NL quote -> retrieved block
      Cases are mined by rules over pub-719 outputs (English-only, since no CJK font on this host).

Usage: python src/make_figures.py [fig3|figa1|figa2|mine5|fig5|all]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qr_paths  # noqa: F401,E402  sets PROJECT_ROOT / QR_CACHE / QWEN3VL_EMB_REPO defaults
import json, re, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
plt.style.use(["science", "no-latex"])
from matplotlib.patches import Rectangle, FancyBboxPatch

ROOT = os.path.expandvars("${PROJECT_ROOT}")
FIGS = f"{ROOT}/paper/latex/figs"
os.makedirs(FIGS, exist_ok=True)

# Okabe-Ito (CVD-safe)
C_COORD = "#D55E00"   # vermillion  = coordinate interface
C_LANG  = "#0072B2"   # blue        = language interface
C_GRPO  = "#0072B2"
C_CEIL  = "#8C8C8C"   # gray
C_GT    = "#009E73"   # bluish green = ground truth
INK     = "#2B2B2B"

plt.rcParams.update({
    # serif/STIX to match the Times body font of the ACL template
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.linewidth": 0.7,
    "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
    "axes.edgecolor": INK, "pdf.fonttype": 42,
})

def panel_title(ax, color, text):
    """panel title: small colored square + bold left-aligned serif text above the axes."""
    ax.text(-0.002, 1.05, "\u25a0", transform=ax.transAxes, fontsize=7.5, color=color,
            va="bottom", ha="left")
    ax.text(0.032, 1.045, text, transform=ax.transAxes, fontsize=9.5, color=INK,
            va="bottom", ha="left", fontweight="bold")

XT = ["page", r"IoU$\geq$0.1", r"$\geq$0.3", r"$\geq$0.5", r"$\geq$0.7"]
CEIL = [100.0, 95.2, 90.3, 88.2, 86.4]
NL = {  # per-model language-interface recall (valid n=719, matches tab:threshold)
    "Qwen3-VL-8B": [63.2, 42.3, 40.7, 39.9, 39.2],
    "Gemma-4-31B": [59.0, 47.3, 45.6, 44.5, 43.8],
    "Qwen3.5-27B": [56.1, 47.3, 45.6, 44.5, 43.8],
    "Qwen3-VL-30B": [50.2, 37.1, 35.6, 34.8, 34.0],
    "Gemma-3-12B": [49.8, 32.0, 30.5, 29.9, 29.3],
    "Qwen3.5-9B": [32.7, 26.5, 25.6, 25.2, 24.9],
}
CO = {  # per-model coordinate-interface recall (valid n=719)
    "Gemma-4-31B": [45.5, 19.3, 12.2, 7.8, 3.7],
    "Qwen3.5-27B": [45.5, 22.0, 14.1, 7.8, 3.3],
    "Qwen3.5-9B": [33.6, 13.0, 7.0, 3.9, 1.8],
    "Qwen3-VL-8B": [30.7, 5.8, 1.2, 0.3, 0.0],
    "Gemma-3-12B": [20.6, 4.8, 1.0, 0.3, 0.0],
    "Qwen3-VL-30B": [14.3, 4.5, 1.9, 1.3, 1.1],
}
GRPO = [73.8, 54.2, 52.2, 51.2, 50.1]


def fig3():
    """Full-width two-panel threshold figure: model = color+marker (consistent), panel = interface."""
    x = np.arange(5)
    MODELS = [  # (name, color, marker)  family-grouped hues, Okabe-Ito
        ("Qwen3-VL-8B",      "#0072B2", "o"),
        ("Qwen3-VL-30B-A3B", "#56B4E9", "s"),
        ("Qwen3.5-9B",       "#E69F00", "^"),
        ("Qwen3.5-27B",      "#D55E00", "D"),
        ("Gemma-3-12B",      "#009E73", "v"),
        ("Gemma-4-31B",      "#CC79A7", "P"),
    ]
    KEY = {"Qwen3-VL-30B-A3B": "Qwen3-VL-30B"}
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 2.95), dpi=300, sharex=True)
    for ax, data, sq, title in [(axes[0], CO, C_COORD, "(a) coordinate interface"),
                                (axes[1], NL, C_LANG, "(b) language interface")]:
        panel_title(ax, sq, title)
        for name, c, mk in MODELS:
            v = data[KEY.get(name, name)]
            ax.plot(x, v, lw=1.6, color=c, marker=mk, ms=4.5, mew=0.8, mfc="white", zorder=4)
        ax.set_xticks(x); ax.set_xticklabels(XT)
        ax.set_ylim(0, 103)
        ax.grid(axis="y", lw=0.4, color="#E3E3E3", zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    # ceiling + GRPO only where they belong (bounds on retrieval / trained model)
    for ax in axes:
        ax.plot(x, CEIL, ls="--", lw=1.3, color=C_CEIL, zorder=2)
    axes[1].annotate("parser ceiling", (x[-1], CEIL[-1]), xytext=(-2, 5),
                     textcoords="offset points", ha="right", fontsize=7, color=C_CEIL)
    axes[1].plot(x, GRPO, lw=2.2, color="#111111", marker="*", ms=8, mfc="white",
                 mew=1.2, zorder=6)
    axes[1].annotate("Qwen3-VL-8B + GRPO", (x[3], GRPO[3]), xytext=(0, 7),
                     textcoords="offset points", ha="center", fontsize=7,
                     color="#111111", fontweight="bold")
    for ax in axes:
        ax.set_ylabel("recall (%)")
    handles = [plt.Line2D([], [], color=c, marker=mk, ms=4.5, mfc="white", lw=1.6,
                          label=name) for name, c, mk in MODELS]
    handles.append(plt.Line2D([], [], color="#111111", marker="*", ms=8, mfc="white",
                              lw=2.0, label="Qwen3-VL-8B + GRPO"))
    handles.append(plt.Line2D([], [], color=C_CEIL, ls="--", lw=1.3, label="parser ceiling"))
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=6.4,
               bbox_to_anchor=(0.53, 1.17), columnspacing=0.9, handlelength=1.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{FIGS}/fig_threshold.pdf", bbox_inches="tight")
    print("fig3 ->", f"{FIGS}/fig_threshold.pdf")


def figa1():
    steps = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    val = [0.287, 0.292, 0.304, 0.292, 0.310, 0.338, 0.310, 0.378, 0.346, 0.367]
    fig, ax = plt.subplots(figsize=(3.35, 2.0), dpi=300)
    ax.plot(steps, val, lw=1.8, color=C_LANG, marker="o", ms=3.5)
    ax.axvline(144, ls=":", lw=1.0, color=C_CEIL)
    ax.annotate("final ckpt (144)", (144, 0.30), rotation=90, fontsize=7,
                color=C_CEIL, ha="right", va="bottom", xytext=(-4, 0), textcoords="offset points")
    ax.set_xlabel("training step"); ax.set_ylabel("held-out val reward")
    ax.set_xlim(45, 150); ax.grid(axis="y", lw=0.4, color="#E3E3E3")
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{FIGS}/fig_valcurve.pdf")
    print("figa1 ->", f"{FIGS}/fig_valcurve.pdf")


def figa2():
    # box-level 4-way decomposition of the 1,034 necessary evidence elements (pub-719)
    labels = ["recovered (matched@1)", "rank-limited (in top-5)",
              "not retrieved (coverage)", "parser-unreachable"]
    colors = [C_LANG, "#9BC7E4", "#F0C29B", C_CEIL]
    rows = [("base NL", [35.0, 16.4, 34.6, 13.9]),
            ("+ GRPO", [46.9, 15.0, 24.2, 13.9])]
    fig, ax = plt.subplots(figsize=(3.35, 1.9), dpi=300)
    for yi, (name, vals) in enumerate(rows):
        left = 0.0
        y = 1 - yi
        for v, c in zip(vals, colors):
            ax.barh(y, v, left=left, height=0.52, color=c, edgecolor="white", lw=1.5)
            ax.text(left + v/2, y, f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if c in (C_LANG, C_CEIL) else INK, fontweight="bold")
            left += v
        ax.text(-1.5, y, name, ha="right", va="center", fontsize=7.5, fontweight="bold")
    ax.set_xlim(0, 100); ax.set_ylim(-0.55, 1.62); ax.set_yticks([])
    ax.set_xlabel("share of necessary evidence elements (%)", fontsize=7.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, labels, loc="upper center", frameon=False, fontsize=6.2, ncol=2,
              bbox_to_anchor=(0.5, 1.42), columnspacing=0.9, handlelength=1.2)
    ax.spines["left"].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{FIGS}/fig_residual.pdf", bbox_inches="tight")
    print("figa2 ->", f"{FIGS}/fig_residual.pdf")


# ---------------- fig5: rule-based case mining ----------------
sys.path.insert(0, f"{ROOT}/src")
from score_citevqa import iou_hit  # noqa: E402


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def load_qa_scores():
    """(tag, idx) -> qa score, lenient regex, from merged judge results."""
    QA = re.compile(r'<?qa_acc>\s*(\d+)\s*</qa_acc>')
    out = {}
    for l in open(f"{ROOT}/outputs/judge/batch_results_all.jsonl"):
        o = json.loads(l)
        k = o.get("key", "")
        if not k.endswith("|qa"):
            continue
        rr = o.get("response") or {}
        txt = rr.get("text") or ""
        if not txt:
            for c in (rr.get("candidates") or []):
                for p in ((c.get("content") or {}).get("parts") or []):
                    txt = txt or p.get("text", "")
        m = QA.search(txt)
        if m:
            tag, idx, _ = k.split("|")
            out[(tag, idx)] = int(m.group(1))
    return out


def mine5(render=False):
    qa = load_qa_scores()
    co = {r["index"]: r for r in map(json.loads, open(f"{ROOT}/outputs/matrix/qwen3vl_coord/predictions.jsonl"))}
    nl = {r["index"]: r for r in map(json.loads, open(f"{ROOT}/outputs/matrix/qwen3vl_nl_wd/predictions.jsonl"))}

    # (a)+(c): en, answer correct on BOTH interfaces, coord box on right page but IoU<0.15,
    # NL matched block IoU>=0.6, small-ish GT element, few GT boxes (clean visual)
    pairs = []
    for idx, r in co.items():
        if r.get("language") != "en" or r.get("error") or not r.get("gt_necessary"):
            continue
        if qa.get(("qwen3vl_coord", idx), 0) < 4 or qa.get(("qwen3vl_nl", idx), 0) < 4:
            continue
        gt = r["gt_necessary"]
        if len(gt) != 1:
            continue
        g = gt[0]  # [page,x1,y1,x2,y2]
        garea = (g[3]-g[1]) * (g[4]-g[2])
        preds = r.get("pred_tuples", [])
        on_page = [t for t in preds if t[1] == g[0]]
        if not on_page:
            continue
        best_iou = max(iou((t[2], t[3], t[4], t[5]), (g[1], g[2], g[3], g[4])) for t in on_page)
        n = nl.get(idx)
        if not n:
            continue
        npreds = [t for t in n.get("pred_tuples", []) if t[1] == g[0]]
        if not npreds:
            continue
        nl_iou = max(iou((t[2], t[3], t[4], t[5]), (g[1], g[2], g[3], g[4])) for t in npreds)
        if best_iou < 0.2 and nl_iou >= 0.6 and garea < 200_000:
            pairs.append((idx, g, best_iou, nl_iou, garea, on_page, npreds, r["pdf_path"]))
    pairs.sort(key=lambda p: (p[2], -p[3]))  # lowest coord IoU, highest NL IoU
    print(f"(a)/(c) candidates: {len(pairs)}")
    for p in pairs[:8]:
        print(f"  idx={p[0][:14]} coordIoU={p[2]:.2f} nlIoU={p[3]:.2f} gtArea={p[4]:.0f} page={p[1][0]}")

    # (b): 30B en, no parseable box, prose location markers, complete ending
    tf = [json.loads(l) for l in open(f"{ROOT}/outputs/matrix/qwen3vl_30b_coord/predictions.jsonl")]
    LOC = re.compile(r"\b(?:on\s+)?[Pp]age\s+\d+|\b[Tt]able\s+[\dA-Z.]+|\b[Ss]ection\s+[\dA-Z.]+|\b[Ff]igure\s+\d+")
    proses = []
    for r in tf:
        if r.get("language") != "en" or r.get("error") or r.get("format_ok"):
            continue
        t = r.get("raw_predict", "")
        hits = LOC.findall(t)
        endsok = t.rstrip()[-1:] in "”.。!?\"')]"
        if len(hits) >= 2 and endsok and 300 < len(t) < 1400:
            proses.append((len(hits), r["index"], t, hits))
    proses.sort(reverse=True)
    print(f"(b) candidates: {len(proses)}")
    for h, idx, t, hits in proses[:6]:
        print(f"  idx={idx[:14]} locRefs={h} :: {hits[:4]} :: {t[:90]!r}")
    return pairs, proses


def fig5(pair_idx=0, prose_idx=0, ac_index=None):
    import fitz
    pairs, proses = mine5()
    if not pairs or not proses:
        print("no candidates; adjust rules"); return
    if ac_index:
        pick = [p for p in pairs if p[0].startswith(ac_index)]
        if pick:
            pairs = pick
            pair_idx = 0
    idx, g, ciou, niou, _, on_page, npreds, pdf = pairs[pair_idx]
    _, pidx, ptxt, phits = proses[prose_idx]

    def render_page(pdf, page1, dpi=150):
        doc = fitz.open(pdf)
        pg = doc[page1 - 1]
        pix = pg.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        import PIL.Image as im
        img = im.frombytes("RGB", (pix.width, pix.height), pix.samples)
        doc.close()
        return np.asarray(img)

    img = render_page(pdf, g[0])
    H, W = img.shape[:2]
    def px(box):  # 0-1000 -> pixel (x1,y1,x2,y2)
        return (box[0]/1000*W, box[1]/1000*H, box[2]/1000*W, box[3]/1000*H)
    gt_px = px((g[1], g[2], g[3], g[4]))
    gtb = (g[1], g[2], g[3], g[4])
    co_best = max(on_page, key=lambda t: iou((t[2], t[3], t[4], t[5]), gtb))
    nl_best = max(npreds, key=lambda t: iou((t[2], t[3], t[4], t[5]), gtb))
    co_px = px((co_best[2], co_best[3], co_best[4], co_best[5]))
    nl_px = px((nl_best[2], nl_best[3], nl_best[4], nl_best[5]))
    # crop window: union of GT + coord box + margin, then pad to a FIXED aspect ratio so all
    # image panels share the same axes shape (keeps the three titles aligned)
    ASPECT = 1.25   # height / width of the crop
    x1 = max(0, min(gt_px[0], co_px[0]) - 0.05*W); x2 = min(W, max(gt_px[2], co_px[2]) + 0.05*W)
    y1 = max(0, min(gt_px[1], co_px[1]) - 0.05*H); y2 = min(H, max(gt_px[3], co_px[3]) + 0.05*H)
    cw, ch = x2 - x1, y2 - y1
    if ch / cw < ASPECT:   # too wide -> grow height around center
        need = ASPECT * cw - ch
        y1 = max(0, y1 - need/2); y2 = min(H, y2 + need/2)
    else:                  # too tall -> grow width
        need = ch / ASPECT - cw
        x1 = max(0, x1 - need/2); x2 = min(W, x2 + need/2)

    def label(ax, x, y, s, color, va="bottom"):
        """white-backed label clamped inside the crop window"""
        x = min(max(x, x1 + 4), x2 - 4)
        y = min(max(y, y1 + 12), y2 - 6)
        ax.text(x, y, s, fontsize=6.3, color=color, fontweight="bold", va=va,
                bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none", alpha=0.85))

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75), dpi=300,
                             gridspec_kw={"width_ratios": [1.0, 1.18, 1.0]})
    titles = ["(a) coordinate interface: coarse box",
              "(b) coordinate interface: prose location",
              "(c) language interface (ours): quote hits"]
    for ax, title in zip(axes, titles):
        ax.set_anchor("N")
        ax.set_title(title, fontsize=7.4, pad=5)
        ax.set_xticks([]); ax.set_yticks([])
    # (a)
    axes[0].imshow(img); axes[0].set_xlim(x1, x2); axes[0].set_ylim(y2, y1)
    axes[0].add_patch(Rectangle((gt_px[0], gt_px[1]), gt_px[2]-gt_px[0], gt_px[3]-gt_px[1],
                                fill=False, ec=C_GT, lw=1.8))
    axes[0].add_patch(Rectangle((co_px[0], co_px[1]), co_px[2]-co_px[0], co_px[3]-co_px[1],
                                fill=False, ec=C_COORD, lw=1.8))
    label(axes[0], gt_px[0], gt_px[3] + 14, "ground-truth element", C_GT, va="top")
    label(axes[0], co_px[0], co_px[1] - 5, f"predicted box  (IoU {ciou:.2f})", C_COORD)
    # (b) text card
    axes[1].set_facecolor("#FAFAF7")
    for sp in axes[1].spines.values():
        sp.set_edgecolor("#D8D8D2"); sp.set_linewidth(0.8)
    wrapped = []
    for line in ptxt.split("\n"):
        while len(line) > 56:
            wrapped.append(line[:56]); line = line[56:]
        wrapped.append(line)
    show = "\n".join(wrapped[:15]) + ("\n…" if len(wrapped) > 15 else "")
    axes[1].text(0.04, 0.96, show, fontsize=5.0, family="DejaVu Sans Mono",
                 va="top", ha="left", transform=axes[1].transAxes)
    axes[1].text(0.5, 0.06, "NO PARSEABLE BOX", transform=axes[1].transAxes, fontsize=8.5,
                 color=C_COORD, fontweight="bold", ha="center", rotation=3,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_COORD, lw=1.4))
    # (c)
    axes[2].imshow(img); axes[2].set_xlim(x1, x2); axes[2].set_ylim(y2, y1)
    axes[2].add_patch(Rectangle((gt_px[0], gt_px[1]), gt_px[2]-gt_px[0], gt_px[3]-gt_px[1],
                                fill=False, ec=C_GT, lw=1.8))
    axes[2].add_patch(Rectangle((nl_px[0], nl_px[1]), nl_px[2]-nl_px[0], nl_px[3]-nl_px[1],
                                fill=False, ec=C_LANG, lw=1.8, ls=(0, (4, 2))))
    label(axes[2], nl_px[0], nl_px[1] - 5, f"retrieved block  (IoU {niou:.2f})", C_LANG)
    fig.tight_layout(pad=0.5)
    fig.savefig(f"{FIGS}/fig_cases.pdf", bbox_inches="tight")
    print(f"fig5 -> {FIGS}/fig_cases.pdf  [a/c idx={idx[:14]} page={g[0]} coordIoU={ciou:.2f} nlIoU={niou:.2f} | b idx={pidx[:14]}]")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("fig3", "all"): fig3()
    if what in ("figa1", "all"): figa1()
    if what in ("figa2", "all"): figa2()
    if what == "mine5": mine5()
    if what in ("fig5", "all"): fig5()


def figa3():
    """Reward-component and citing-behavior dynamics during GRPO (appendix)."""
    import json
    rows = [json.loads(l) for l in open(f"{ROOT}/outputs/grpo_longdoc/metrics_master.jsonl")]
    rows = [r for r in rows if r.get("reward") and r["step"] >= 45]  # steps 1-44 lost to a requeue overwrite
    rows.sort(key=lambda r: r["step"])
    steps = np.array([r["step"] for r in rows], dtype=float)
    def series(k):
        v = np.array([r["reward"][k] for r in rows], dtype=float)
        # break the line across logging gaps (>5 steps) instead of drawing a straight segment
        gap = np.where(np.diff(steps) > 5)[0]
        v2 = v.copy()
        for gi in gap:
            v2[gi] = np.nan
        return v2
    def smooth(v, w=7):
        out = np.full_like(v, np.nan)
        for i in range(len(v)):
            seg = v[max(0, i-w+1):i+1]
            seg = seg[~np.isnan(seg)]
            if len(seg):
                out[i] = seg.mean()
        gap = np.where(np.diff(steps) > 5)[0]
        for gi in gap:
            out[gi] = np.nan
        return out
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 2.95), dpi=300, sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1.0]})
    comp = [("answer_correctness", "$a$ (answer corr.)", "#333333"),
            ("evidence_relevance", r"$e_{\mathrm{rel}}$ (relevance)", C_LANG),
            ("evidence_coverage", r"$e_{\mathrm{cov}}$ (coverage)", C_COORD)]
    for k, lab, c in comp:
        v = series(k)
        axes[0].plot(steps, v, lw=0.6, color=c, alpha=0.25)
        axes[0].plot(steps, smooth(v), lw=1.8, color=c, label=lab)
    axes[0].set_ylabel("judge score (0-5)")
    axes[0].set_ylim(0.8, 4.2)
    axes[0].legend(frameon=False, fontsize=6.8, loc="lower right", ncol=1)
    axes[0].grid(axis="y", lw=0.4, color="#E3E3E3")
    nq = series("n_quotes")
    axes[1].plot(steps, nq, lw=0.6, color=C_LANG, alpha=0.25)
    axes[1].plot(steps, smooth(nq), lw=1.8, color=C_LANG)
    axes[1].set_ylabel("quotes / response")
    axes[1].set_xlabel("training step")
    axes[1].set_ylim(2.5, 5.0)
    axes[1].grid(axis="y", lw=0.4, color="#E3E3E3")
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{FIGS}/fig_rewarddyn.pdf", bbox_inches="tight")
    print("figa3 ->", f"{FIGS}/fig_rewarddyn.pdf")


def figa4():
    """Selective prediction on citation similarity (appendix): risk-coverage + reliability.
    Data: outputs/controls/sims_*.jsonl (assigned-pair cosine sim + necessary-GT hit at IoU 0.5)."""
    import glob
    NAMES = {"qwen3vl": "Qwen3-VL-8B", "qwen3vl_30b": "Qwen3-VL-30B",
             "qwen35_9b": "Qwen3.5-9B", "qwen35_27b": "Qwen3.5-27B",
             "gemma3": "Gemma-3-12B", "gemma4_31b": "Gemma-4-31B",
             "grpo_global_step_144_lr1e5": "8B + GRPO"}
    COLORS = {"Qwen3-VL-8B": "#0072B2", "Qwen3-VL-30B": "#56B4E9",
              "Qwen3.5-9B": "#009E73", "Qwen3.5-27B": "#E69F00",
              "Gemma-3-12B": "#CC79A7", "Gemma-4-31B": "#D55E00", "8B + GRPO": "#2B2B2B"}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3))
    for fp in sorted(glob.glob(f"{ROOT}/outputs/controls/sims_*.jsonl")):
        key = os.path.basename(fp)[5:-6]
        name = NAMES.get(key)
        if not name:
            continue
        rows = [json.loads(l) for l in open(fp)]
        sims = np.array([r["sim"] for r in rows])
        hits = np.array([r["hit"] for r in rows], dtype=float)
        order = np.argsort(-sims)
        fracs = np.linspace(0.1, 1.0, 19)
        prec = [100 * hits[order[:max(1, int(len(rows) * f))]].mean() for f in fracs]
        lw = 1.9 if name == "8B + GRPO" else 1.3
        axes[0].plot(100 * fracs, prec, color=COLORS[name], lw=lw, label=name)
        qs = np.quantile(sims, np.linspace(0, 1, 6))
        xs, ys = [], []
        for i in range(5):
            m = (sims >= qs[i]) & (sims < qs[i + 1]) if i < 4 else (sims >= qs[i])
            xs.append(sims[m].mean())
            ys.append(100 * hits[m].mean())
        axes[1].plot(xs, ys, color=COLORS[name], lw=lw, marker="o", ms=2.6)
    panel_title(axes[0], C_LANG, "(a) precision vs. retention")
    axes[0].set_xlabel("citations retained (%)")
    axes[0].set_ylabel("citation precision (%)")
    panel_title(axes[1], C_GT, "(b) reliability by similarity")
    axes[1].set_xlabel("assigned cosine similarity (quintile mean)")
    axes[1].set_ylabel("hit rate (%)")
    for ax in axes:
        ax.grid(axis="y", lw=0.4, color="#E3E3E3")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.8, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{FIGS}/fig_abstention.pdf", bbox_inches="tight")
    print("figa4 ->", f"{FIGS}/fig_abstention.pdf")
