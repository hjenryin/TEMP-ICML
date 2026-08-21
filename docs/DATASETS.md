# Datasets

TEMP expects **ShareGPT-style** JSONL training files (same format used by LLaMA-Factory in this repo):

```json
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "source": "...", "domain": "..."}
```

Place them as:

```
data/
  m23k-prd-sharegpt/train.jsonl          # 23,493 medical reasoning traces
  op114kmath-prd-sharegpt/train.jsonl    # 56,370 OpenThoughts-Math traces
  m1_eval_data.json                      # medical eval suite (shipped in this repo under data/)
```

Override the root with `export TEMP_DATA_ROOT=/path/to/data`.

Build the training JSONLs with the converters below (from public Hugging Face sources).


## M23k (medical)

**Paper:** Huang et al., *m1: Unleash the Potential of Test-Time Scaling for Medical Reasoning* ([arXiv:2504.00869](https://arxiv.org/abs/2504.00869)).

**HF source:** [`UCSC-VLAA/m23k-tokenized`](https://huggingface.co/datasets/UCSC-VLAA/m23k-tokenized)  
(fields: `prompt`, `reasoning`, `distilled_answer_string`, `source`, …)

Related: [`UCSC-VLAA/m1k-tokenized`](https://huggingface.co/datasets/UCSC-VLAA/m1k-tokenized) (M1K baseline subset).

```bash
python TEMP/src/tokenize_data/m23k_to_sharegpt.py \
  --output data/m23k-prd-sharegpt/train.jsonl
```

Optional: `--input /path/to/local.jsonl` if you already have the same columns locally.

**Expected size:** 23,493 examples.

### Medical evaluation data

Shipped at `data/m1_eval_data.json` (from [`UCSC-VLAA/m1_eval_data`](https://huggingface.co/datasets/UCSC-VLAA/m1_eval_data)).

To re-download:

```bash
huggingface-cli download --repo-type dataset UCSC-VLAA/m1_eval_data --local-dir /tmp/m1_eval
# place m1_eval_data.json at data/m1_eval_data.json
```

Benchmarks: MedMCQA, MedQA-USMLE, PubMedQA, MMLU-Pro (Medical), GPQA (Medical), Lancet, NEJM, MedBullets, MedXpertQA, etc. (10 sets; average accuracy reported).


## OpenThoughts-Math

**Paper / data:** Guha et al., *OpenThoughts*; Open R1 release.

**HF source:** [`open-r1/OpenThoughts-114k-math`](https://huggingface.co/datasets/open-r1/OpenThoughts-114k-math)  
(keep `correct=True` → 56,370 rows; rename `problem`→`prompt`; normalize `conversations` from `from`/`value` to `role`/`content`; set `domain=math`).

```bash
python TEMP/src/tokenize_data/openthoughts_to_sharegpt.py \
  --output data/op114kmath-prd-sharegpt/train.jsonl
```

**Expected size:** 56,370 examples.

**$\theta_f$ for math:** [`OpenThinker-7B`](https://huggingface.co/open-thoughts/OpenThinker-7B). Ablations may use undertrained checkpoints or SynLogic-7B.

### Math evaluation

Requires [evalchemy](https://github.com/mlfoundations/evalchemy). Set:

```bash
export EVALCHEMY_DIR=/path/to/evalchemy
```

Tasks: **AMC23, AIME24, AIME25, MATH500** (temperature 0.7, up to 28k new tokens).


## Models

| Role | Default |
|------|---------|
| Student / $\theta_0$ | `Qwen/Qwen2.5-7B-Instruct` (or `meta-llama/Llama-3.1-8B-Instruct` via `--llama`) |
| $\theta_f$ (M23k) | Full-data SFT of $\theta_0$ on M23k (this repo) |
| $\theta_f$ (math) | `open-thoughts/OpenThinker-7B` |

Download with Hugging Face (`huggingface-cli download ...`) into a cache visible to transformers / LLaMA-Factory.
