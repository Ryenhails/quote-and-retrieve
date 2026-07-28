# Scoring protocol

Every experiment in the paper is scored by a single code path, `src/score_citevqa.py`, so all
conditions are measured identically. This document states exactly what that code computes, what
it inherits from the benchmark, and what we added. `tests/test_scoring.py` asserts each of these
properties, so the document and the code cannot drift apart silently.

## Matching rule

A cited box matches an annotated element when both of the following hold:

1. they lie on the same page, and
2. their intersection over union is at least 0.5.

Page identity is required first, so a geometrically plausible box on the wrong page never
counts. This is why the paper reports page-level recall separately: it isolates how often a
model finds the right page at all, independently of how well it places a box on it.

## Two levels of ground truth

The benchmark annotates two kinds of evidence per question:

| kind | meaning | used for |
|---|---|---|
| `gt_necessary` | elements every correct answer must cite | recall |
| `gt_all` | necessary elements plus additional supporting elements the annotators marked optional | precision |

## Recall

Per question:

```
recall = |{ e in gt_necessary : some citation matches e }| / |gt_necessary|
```

then macro-averaged over questions. Every question contributes equally regardless of how many
elements it has, which is why a question with one necessary element and a question with four
carry the same weight in the reported number.

## Precision

Precision replicates the loop of the scorer released with the benchmark verbatim:

```
precision = |{ e in gt_all : some citation matches e }| / (number of boxes cited)
```

Note that the numerator and the denominator range over **different objects**: the numerator
counts annotated elements, the denominator counts predicted boxes. A single box covering two
annotated elements therefore contributes two to the numerator and one to the denominator, so
this quantity is not a standard precision and can exceed one in rare cases. We keep the released
semantics rather than substituting a cleaner definition, so that our reproduction of the
coordinate interface is comparable to published scores.

When a response cites nothing, precision is undefined and that response is excluded from the
precision average, while its recall is zero and is included. The reported `n` therefore differs
between the two columns.

## F1

Per-question harmonic mean of the two quantities above.

## Two choices that are ours, not the benchmark's

Both are applied identically to every condition, so no arm is advantaged.

**Recall against necessary evidence, precision against all annotated evidence.** Citing
supporting material that the annotators marked optional is not counted as an error. If anything
this disadvantages the language interface, which cites more regions per response and therefore
has more opportunity to land outside the annotation.

**No page-index correction.** Citations are scored against whichever page the model names. We
never repair an off-by-one page index or search neighbouring pages for a better match.

## Why precision must be read alongside recall

Recall is monotonic in the number of citations: a system that cited every parsed block on every
page would reach the parser ceiling by construction, at a precision below one percent. Neither
metric is meaningful alone, which is why every table in the paper reports both, and why the
judge-based metrics of the paper score the retrieved crops directly rather than counting boxes.

## Verifying this document

```bash
pip install pytest
pytest tests/ -q
```

The tests assert the same-page requirement, the IoU threshold behaviour, the recall denominator,
the precision denominator, the treatment of optional evidence, the undefined-precision case, and
the separation between page-level and element-level recall.
