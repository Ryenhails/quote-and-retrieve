# Reproduction details

Settings that the scripts need and the paper has no room for. Everything here was read from the
job logs and configuration files of the runs reported in the paper.

## Paths

Every script imports `src/qr_paths.py`, which reads three environment variables. Each has a
default, so a plain clone runs without editing any file.

| variable | meaning | default |
|---|---|---|
| `PROJECT_ROOT` | repository root; `data/`, `outputs/`, `logs/` live under it | the clone |
| `QR_CACHE` | model snapshots and block-embedding caches | `<PROJECT_ROOT>/../cache` |
| `QWEN3VL_EMB_REPO` | directory containing `scripts/qwen3_vl_embedding.py` | `<QR_CACHE>/Qwen3-VL-Embedding-2B` |

The batch scripts in `slurm/` additionally use `${ENV_MAIN}`, `${ENV_INFER}`, `${ENV_TRAIN}`,
`${ENV_PARSE}` (conda environments), `<PARTITION>`, and `<cluster-toolchain>`. Set them for your
cluster before submitting.

The retrieval encoder's wrapper class is distributed inside its model repository, so download the
model once and point `QWEN3VL_EMB_REPO` at it:

```bash
huggingface-cli download Qwen/Qwen3-VL-Embedding-2B \
    --revision 2a50926d213628c727f38025982a76f655673f54 \
    --local-dir "$QR_CACHE/Qwen3-VL-Embedding-2B"
# pass the same directory as --model to prebuild_block_emb.py and match_wholedoc.py
```

## Models

All backbones are served with vLLM 0.22.1 (`--gpu-memory-utilization 0.90`,
`--limit-mm-per-prompt image=200`, `--trust-remote-code`), temperature 0, and
`repetition_penalty` 1.15 under both interfaces (set in `src/infer_matrix.py`; it stops the
coordinate interface from looping on repeated boxes).

| backbone | `BACKBONE=` | checkpoint | revision | context | `MAXTOK` |
|---|---|---|---|---|---|
| Qwen3-VL-8B | `qwen3vl` | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd0` | 262,144 | 1536 |
| Qwen3-VL-30B-A3B | `qwen3vl_30b` | `Qwen/Qwen3-VL-30B-A3B-Instruct` | `9c4b90e1` | 262,144 | 1536 |
| Qwen3.5-9B | `qwen35_9b` | `Qwen/Qwen3.5-9B` | `c2022362` | 262,144 | 4096 |
| Qwen3.5-27B | `qwen35_27b` | `Qwen/Qwen3.5-27B` | `b7ca741b` | 262,144 | 4096 |
| Gemma-3-12B | `gemma3` | `google/gemma-3-12b-it` | `96b6f1ec` | 131,072 | 1536 |
| Gemma-4-31B | `gemma4_31b` | `google/gemma-4-31B-it` | `35487898` | 262,144 | 4096 |
| Qwen3-VL-8B + GRPO | `slurm/matrix_grpo.sh` | base model with the LoRA adapter | | 262,144 | 1536 |

`MAXTOK` defaults to 1536, so the three 4096 runs need it on the command line:

```bash
sbatch --export=ALL,BACKBONE=qwen3vl                 slurm/matrix_backbone.sh
sbatch --export=ALL,BACKBONE=qwen35_9b,MAXTOK=4096   slurm/matrix_backbone.sh
sbatch --export=ALL,BACKBONE=gemma4_31b,MAXTOK=4096  slurm/matrix_backbone.sh
```

Pages are rendered with PyMuPDF at up to one megapixel each and sent as JPEG (quality 85). The
system message holds the prompt; the user message holds all page images in order, then the
question.

## Layout parser

MinerU 3.2.3, `pipeline` backend, table and formula recognition off, OCR language chosen per
document (`en` or `ch`):

```bash
mineru -p <pdf_dir> -o <raw_out> -b pipeline -m auto -t False -f False -l <en|ch>
python src/mineru_extract_cache.py <raw_out> data/citevqa/mineru_cache
```

`slurm/mineru_parse.sh` runs both steps for the evaluation PDFs and
`slurm/mineru_parse_longdoc.sh` for the training PDFs. Model weights come from
`opendatalab/PDF-Extract-Kit-1.0` (revision `ed6b654c`). Every entry of `content_list.json` with
a page index and a box is kept as a candidate block; blocks whose crop is under 4 pixels on a side
are dropped at embedding time.

## Retrieval

Qwen3-VL-Embedding-2B (revision `2a50926d`), float16, the wrapper's default instruction for both
quotes and block crops, 2048-dimensional L2-normalised embeddings. Block crops are cut from
150 dpi page renders. One-to-one assignment uses `scipy.optimize.linear_sum_assignment`.

## Judge

`gemini-3.5-flash` through the Batch API, temperature 0, 1024 output tokens, thinking budget 0.
The relevance judge sees the crops of the first six citations of a response (JPEG, quality 80).
Requests whose score tag could not be parsed were resubmitted once; `src/build_main_table.py`
reads the merged results and reports how many remain unparsed.

`src/build_main_table.py` takes one `tag=path` pair per condition:

```bash
python src/build_main_table.py --results outputs/judge/batch_results_all.jsonl --preds \
    qwen3vl_coord=outputs/matrix/qwen3vl_coord/predictions.jsonl \
    qwen3vl_nl=outputs/matrix/qwen3vl_nl_wd/predictions.jsonl \
    grpo_nl=outputs/matrix/grpo_global_step_144_lr1e5_nl_wd/predictions.jsonl
```

`src/score_citevqa.py` averages over all 719 questions (the trained model: recall 51.2,
precision 26.4). The main table averages over the questions whose two judge scores both parsed
(716 for the trained model: recall 51.3, precision 26.5).

## Training

The training set is fixed by `release/grpo_train_ids.jsonl`; see `docs/using_the_release.md`,
section 4.

- Framework: [EasyR1](https://github.com/hiyouga/EasyR1) at commit `dd71bbd`.
- LoRA rank 64, alpha 64, on all linear layers of the language model; vision tower frozen.
- AdamW, learning rate 1e-5, betas (0.9, 0.999), weight decay 0.01, gradient clipping 1.0, no
  warmup. 144 training iterations (three epochs); each iteration samples 8 responses for each of
  32 questions at temperature 1.0.
- Page windows are rendered at ten megapixels per window with a one-megapixel cap per page, so a
  30-page window is about a third of a megapixel per page.
- Reward judge: Qwen3.5-9B served with vLLM, temperature 0, 512 output tokens, thinking disabled.
  The reward service resolves the first five quotes of a response and shows their crops to the
  judge.
- The reported model is the final checkpoint (iteration 144), with no checkpoint selection.

`slurm/grpo_longdoc_chain.sh` starts the judge server, the reward service, and the trainer in one
job and resubmits itself until training finishes.

## Benchmark versions

- CiteVQA: `opendatalab/CiteVQA` at commit `039f6a3b`, `data/validation/CiteVQA.json`. Its
  evidence boxes are stored as `[y1, x1, y2, x2]` on a 0 to 1000 scale and its page numbers are
  1-based; `src/build_citevqa_eval.py` converts both.
- LongDocURL: `dengchao/LongDocURL` at revision `5c96dbe9`.
- Cross-parser check: Docling with OCR on and table-structure recognition off.

## Tables added in the camera-ready version

All run on CPU from the frozen predictions and judge results, except the two files produced by
`slurm/revision_gpu_dump.sh` (one small GPU, about an hour).

| table in the paper | command |
|---|---|
| benchmark composition before and after filtering; scan-like slice | `python src/revision_analysis.py filter` |
| recall by question type | `python src/revision_analysis.py qtype` |
| trained model at the base model's citation budget | `python src/revision_analysis.py budget` |
| residual by question type and evidence type | `sbatch slurm/revision_gpu_dump.sh`, then `python src/revision_analysis.py buckets` |
| ANLS | `python src/revision_analysis.py anls` |
| quotes and question-plus-answer retrieval side by side | `sbatch slurm/revision_gpu_dump.sh`, then `python src/make_fig_sidebyside.py` |

Each command prints markdown and writes it to `outputs/revision/analysis/`.
