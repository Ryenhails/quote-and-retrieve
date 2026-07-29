# Using the released model and evaluation set

Three artifacts accompany the paper, and each one is useless on its own. This page is the map
between them.

| artifact | where | what it is |
|---|---|---|
| code | [github.com/Ryenhails/quote-and-retrieve](https://github.com/Ryenhails/quote-and-retrieve) | the two interfaces, the retrieval step, the scorer, the controls, the reward |
| evaluation set | [Ryenhails/quote-and-retrieve-eval](https://huggingface.co/datasets/Ryenhails/quote-and-retrieve-eval) | 719 verified question identifiers and normalised ground-truth boxes |
| model | [Ryenhails/quote-and-retrieve-8b-grpo](https://huggingface.co/Ryenhails/quote-and-retrieve-8b-grpo) | Qwen3-VL-8B trained with the region-label-free GRPO recipe |

The reason they are separate is the division of labour the paper studies. **The model only writes
text.** It answers a question and quotes its evidence verbatim; it never emits a coordinate. The
code turns those quotes into page regions. The evaluation set says which regions were correct.
Attribution is a property of the three together, which is exactly the point of the paper.

```
question + page images
        |
        v
   [ model ]  ---->  {"answer": ..., "evidence": [{"quote": ...}, ...]}      (HF model)
        |
        v
   [ layout parser ] -> semantic blocks -> [ encoder ] -> [ assignment ]     (this repo)
        |
        v
   citations: (page, box)  ---->  [ scorer ]  <----  ground truth            (HF dataset)
```

---

## 1. Get everything

```bash
# code
git clone https://github.com/Ryenhails/quote-and-retrieve.git
cd quote-and-retrieve
pip install -r requirements.txt

# evaluation set (identifiers and ground-truth geometry)
huggingface-cli download Ryenhails/quote-and-retrieve-eval \
    --repo-type dataset --local-dir data/release

# model
huggingface-cli download Ryenhails/quote-and-retrieve-8b-grpo \
    --local-dir models/quote-and-retrieve-8b-grpo
```

The evaluation set ships identifiers and geometry but not the benchmark's question text, answers,
or PDFs, which stay under CiteVQA's terms. Download CiteVQA yourself, then join:

```bash
python data/release/rehydrate.py \
    --release data/release/verified_eval_set.jsonl \
    --citevqa /path/to/CiteVQA/data/validation/CiteVQA.json \
    --pdf-dir /path/to/CiteVQA/data/pdf \
    --out data/citevqa/citevqa_singledoc_pub.jsonl
```

That file is what every script in this repository reads. It is schema-identical to what
`src/build_citevqa_pub.py` produces, so you can also rebuild it from scratch instead of
downloading, and get the same 719 records.

---

## 2. Reproduce the trained model's row of the main table

```bash
# once per document: parse pages into semantic blocks and embed each block crop
python src/prebuild_block_emb.py

# generate answers and quotes with the trained model
python src/infer_matrix.py \
    --model models/quote-and-retrieve-8b-grpo \
    --interface language \
    --bench data/citevqa/citevqa_singledoc_pub.jsonl \
    --out outputs/matrix/grpo_nl

# turn quotes into regions, then score
python src/match_wholedoc.py --pred outputs/matrix/grpo_nl/predictions.jsonl
python src/score_citevqa.py outputs/matrix/grpo_nl/predictions.jsonl
```

Expected, matching the paper: evidence recall 51.3 at IoU 0.5, same-page recall 73.8, citation
precision 26.5, averaging 4.6 citations per response. SAA (33.8) and the hallucination rate (28.4)
additionally need the judge pass in step 4 of the main README, because they depend on a judge
scoring answers and retrieved crops.

Swap `--model` for `Qwen/Qwen3-VL-8B-Instruct` to get the untrained language-interface baseline
(recall 39.9), and add `--interface coordinate` to get the coordinate condition (recall 0.3). Those
three runs are the paper's central comparison.

---

## 3. Use the model on your own documents

The model is standalone for quoting, but you need the retrieval step for regions.

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = "Ryenhails/quote-and-retrieve-8b-grpo"
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto").eval()

SYSTEM = open("configs/language_interface_prompt.txt").read()
messages = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
    {"role": "user", "content": [
        *[{"type": "image", "image": im} for im in page_images],   # all pages, in order
        {"type": "text", "text": question},
    ]},
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=1536, do_sample=False)
print(processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Render each page at up to one megapixel and pass them in document order; that is the protocol the
model was trained and evaluated under. Greedy decoding with a 1536-token budget reproduces the
reported numbers.

To resolve the quotes to regions on your own documents, point the parser and the matcher at your
PDFs:

```bash
python src/prebuild_block_emb.py --pdf-dir /path/to/your/pdfs
python src/match_wholedoc.py --pred your_predictions.jsonl
```

Each citation comes back as `(page, x1, y1, x2, y2)` in rendered-page pixels, together with the
retrieval similarity that produced it. That similarity is a usable abstention signal: dropping
low-similarity citations trades coverage for reliability, which matters because the assignment step
always returns some block for every quote, including for a quote that matches nothing.

---

## 4. Retrain the recipe on your own data

The reward needs questions with gold answers and the documents they concern. It does **not** need
region labels.

```bash
python src/build_grpo_longdoc.py --corpus /path/to/your/qa_corpus
sbatch slurm/serve_judge.sh          # vision-language judge that scores retrieved crops
sbatch slurm/grpo_longdoc_chain.sh   # chained GRPO jobs
```

The judge reads the question, the gold answer, the model's answer, and crops of the cited regions,
and returns answer correctness, evidence relevance, and evidence coverage. The reward multiplies
the answer term by the evidence terms, so evidence only counts when the answer is already correct.
See `src/reward_longdoc_grpo.py`.

---

## Common mistakes

**Using the raw CiteVQA box order.** The released evidence boxes are `[y1, x1, y2, x2]` on a
0-1000 scale, which contradicts the `x1y1x2y2` description in the benchmark's own prompt. The
dataset release is already converted to `x1 y1 x2 y2` in page pixels. Feeding raw released boxes
into the scorer silently produces near-zero recall.

**Scoring quotes without running the matcher.** A predictions file straight from the model has no
`pred_tuples`, so the scorer reports zero. Run `src/match_wholedoc.py` first.

**Comparing across different evaluation subsets.** The paper's numbers are on the 719-question
verified set with a specific judge. Numbers from the full CiteVQA split or a different judge are
not comparable; the closed-source rows in the paper's appendix are reference points, not
head-to-head results.

**Expecting the model to output coordinates.** It will not, by design. If you need boxes, you need
the retrieval step.
