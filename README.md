# Evidence Attribution without Coordinates or Region Labels

[![arXiv](https://img.shields.io/badge/arXiv-2607.24651-b31b1b.svg)](https://arxiv.org/abs/2607.24651)
[![ci](https://github.com/Ryenhails/quote-and-retrieve/actions/workflows/ci.yml/badge.svg)](https://github.com/Ryenhails/quote-and-retrieve/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)

Code for the paper [**"Evidence Attribution in Visual Document Understanding without
Coordinates or Region Labels"**](https://arxiv.org/abs/2607.24651)
(Zhuchenyang Liu, Yao Zhang, Yu Xiao).

When a vision-language model answers a question about a document, we usually ask it to point at
the evidence by generating bounding-box coordinates. Under that interface open models very often
answer correctly while citing the wrong region, a failure known as **Attribution Hallucination**,
and this has been read as a missing grounding capability.

This repository contains the code for a controlled comparison showing that the failure is
largely an artifact of the **output interface** rather than of the models. Keep the model, the
input pages, the questions, and the scoring fixed, and change only how the model expresses
evidence:

| interface | what the model outputs | who produces the box |
|---|---|---|
| coordinate | box coordinates, token by token | the model |
| language | verbatim quotes, as text | a layout parser and a retriever, outside the model |

Across six open models from four families, evidence recall rises from **at most 8** under
coordinates to **26 to 47** under the language interface, and the hallucination rate roughly
halves, with little change in answer quality. Building on that comparison, a **region-label-free
GRPO recipe**, whose reward reads only the gold answer and crops of the retrieved regions, raises
an 8B model's strict attributed accuracy from **22.4 to 33.8** without a single region label.

---

## See the contrast in 30 seconds

No documents, no models, no GPU:

```bash
git clone https://github.com/Ryenhails/quote-and-retrieve.git
cd quote-and-retrieve
python src/score_citevqa.py examples/sample_predictions.jsonl
python src/score_citevqa.py examples/sample_predictions_coordinate.jsonl
```

```
examples/sample_predictions.jsonl              recall 62.5   precision 83.3   page_recall 75.0
examples/sample_predictions_coordinate.jsonl   recall 25.0   precision 33.3   page_recall 62.5
```

Four synthetic questions, the same correct answers in both files, only the evidence interface
differs. One of the coordinate predictions lands on the right page but misses the element, and
one produces no parseable box at all. That is the paper's central finding in miniature.

---

## Install

```bash
pip install -r requirements.txt
```

Scoring, the control experiments, and the tests need only `numpy` and `scipy`. Inference,
retrieval, and training additionally need `torch`, `transformers`, and the parsing and serving
stack listed in `requirements.txt`.

```bash
pip install pytest && pytest tests/ -q     # 11 tests, no downloads, under a second
```

---

## Repository layout

```
src/
  infer_matrix.py             run one backbone under one interface over the evaluation set
  match_wholedoc.py           quote -> semantic block retrieval and one-to-one assignment
  prebuild_block_emb.py       cache one crop embedding per parsed block, per document
  mineru_ceiling.py           parser ceiling: recall attainable with a perfect retriever
  score_citevqa.py            the single scoring path: recall, precision, F1, page recall
  build_citevqa_pub.py        build the verified evaluation set (the filtering funnel)
  build_citevqa_eval.py       assemble per-question ground-truth tuples
  build_judge_batch.py        format judge requests (answer quality, evidence relevance)
  submit_judge_batch.py       submit the judge batch
  score_judge_batch.py        parse judge scores into SAA and AH
  build_main_table.py         reproduce the main results table
  controls_rereview.py        snapping, funnel, significance, precision, breakdowns
  resolver_ablation.py        lexical, BM25, text-embedding, crop-embedding resolvers
  retrieval_baselines.py      retrieval-only baselines and the abstention similarity export
  cross_parser_docling.py     re-parse the corpus with an independent layout parser
  cross_parser_match.py       re-run matching against the independent parser's blocks
  table_multithreshold.py     multi-threshold recall tables
  build_grpo_longdoc.py       training-set construction from the long-document corpus
  rollout_longdoc.py          rollout: parse, retrieve, crop, judge
  reward_longdoc_grpo.py      the gated reward
  reward_service.py           reward server used during training
  make_figures.py             all figures in the paper
configs/    rollout prompt template
slurm/      one batch script per stage; paths are placeholders, see below
examples/   four synthetic questions under both interfaces
tests/      scoring-protocol tests, runnable without any data
docs/       scoring protocol and record format
```

The batch scripts use placeholders (`${PROJECT_ROOT}`, `${ENV_MAIN}`, `<PARTITION>`,
`<ACCOUNT>`) instead of absolute paths, so set them for your own cluster before submitting.

---

## Method

**Language interface.** The model reads all pages and returns an answer together with verbatim
quotes of its evidence. A layout parser segments every page into typed semantic blocks. One
multimodal encoder embeds both the quotes and the rendered block crops, and a one-to-one
assignment maps each quote to a block, whose page and box become the citation. Crops rather than
extracted text are embedded, because about thirty percent of the evidence is non-textual and
rendered tables match plain-language quotes poorly in text space. Candidates come from the whole
document and the model's own page guess is deliberately ignored, so a page error stays
recoverable.

**Region-label-free training.** For each question the policy samples several responses. Each is
parsed into an answer and quotes, resolved to block crops by the same retrieval step, and scored
by a vision-language judge on answer correctness, evidence relevance, and evidence coverage. The
reward multiplies the answer term by the evidence terms, so evidence only counts when the answer
is already correct, which is exactly the condition Attribution Hallucination concerns. The judge
reads the retrieved crops rather than the raw quotes, so a fluent quote assigned to the wrong
block scores low. Nothing in this signal is a region label: the gold answer ships with the
dataset, the retriever is unsupervised, and the judge sees only crops.

---

## Reproducing the paper

Steps 1 and 2 need the public document corpora cited in the paper; steps 3 onward need GPUs.

```bash
# 1. Build the verified evaluation set (987 -> 719 questions)
python src/build_citevqa_pub.py
python src/build_citevqa_eval.py

# 2. Parse documents and cache block embeddings (once per document)
python src/prebuild_block_emb.py

# 3. Run the interface comparison, one job per backbone and interface
sbatch slurm/matrix_backbone.sh
python  src/match_wholedoc.py                                     # language interface only
python  src/score_citevqa.py outputs/matrix/<condition>/predictions.jsonl

# 4. Judge-based metrics (SAA, AH) and the main table
python src/build_judge_batch.py && sbatch slurm/judge_batch.sh
python src/score_judge_batch.py
python src/build_main_table.py

# 5. Control experiments; these reuse the frozen outputs of step 3
python src/controls_rereview.py snap        # coordinate boxes snapped to blocks
python src/controls_rereview.py funnel      # filtering funnel, dropped-document analysis
python src/controls_rereview.py stats       # paired bootstrap and McNemar
python src/controls_rereview.py precision   # citation precision, citations per response
python src/controls_rereview.py breakdown   # per-language and per-evidence-type
sbatch slurm/controls_embed.sh              # retrieval-only baselines, abstention curve
python src/resolver_ablation.py             # lexical and BM25 resolvers, quote fidelity
sbatch slurm/docling_parse.sh && sbatch slurm/docling_match.sh    # independent parser

# 6. Region-label-free training
python src/build_grpo_longdoc.py
sbatch slurm/serve_judge.sh
sbatch slurm/grpo_longdoc_chain.sh
```

Compute reported in the paper: about **85 GPU-hours** for training, about **60 single-GPU-hours**
for the thirteen-condition evaluation matrix, and under two GPU-hours for the diagnostics.

---

## Scoring

All conditions share one scorer, so no arm can drift. A citation matches an annotated element
when both lie on the same page and their IoU is at least 0.5. Recall is computed per question
over the **necessary** elements and macro-averaged over questions. Precision replicates the
released scorer's loop, annotated elements hit divided by boxes cited, and is matched against
**all** annotated evidence so that citing optional supporting material is not an error.

Recall alone is not meaningful, since it is monotonic in the number of citations: citing every
parsed block would reach the parser ceiling at a precision below one percent. Every table
reports both.

Full details, including the two protocol choices that are ours rather than the benchmark's, are
in [`docs/scoring_protocol.md`](docs/scoring_protocol.md); the record format is in
[`docs/data_format.md`](docs/data_format.md). Both are asserted by the test suite.

---

## What this repository does not contain

Model weights, the source PDF corpora, and cached embeddings. These come from the public
releases cited in the paper and are regenerated by the scripts above. The evaluation set is
distributed as question identifiers plus the verification filter, not as redistributed document
content.

---

## Citation

```bibtex
@article{liu2026evidence,
  title         = {Evidence Attribution in Visual Document Understanding without Coordinates or Region Labels},
  author        = {Liu, Zhuchenyang and Zhang, Yao and Xiao, Yu},
  journal       = {arXiv preprint arXiv:2607.24651},
  year          = {2026},
  eprint        = {2607.24651},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2607.24651}
}
```

## License

MIT, see [LICENSE](LICENSE). The benchmarks, parsers, and models this code builds on are covered
by their own licenses.
