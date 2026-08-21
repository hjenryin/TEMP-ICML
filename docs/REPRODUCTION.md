# Reproduction guide

This document documents the main experiments to commands in this repo.

First, after setting up the env, please set up the env variables.

```bash
cd /path/to/repo
export TEMP_DATA_ROOT=$PWD/data
export TEMP_OUTPUT_ROOT=$PWD/outputs
# Optional for SFT logging:
# export WANDB_API_KEY=...
```

Datasets laid out as in [DATASETS.md](DATASETS.md).


## A. Train $\theta_f$ (M23k)

Full-data SFT (no subset):

```bash
cd TEMP
python src/train/llama_factory.py
# Llama-3.1-8B:
# python src/train/llama_factory.py --llama
```

Checkpoint is written under `${TEMP_OUTPUT_ROOT}/m23k-llamafactory/...`. Use the final checkpoint as `--direction_end_checkpoint` below.

**OpenThoughts-Math:** download [`OpenThinker-7B`](https://huggingface.co/open-thoughts/OpenThinker-7B) and pass it as `--direction_end_checkpoint` (paper default). Ablations may use undertrained / SynLogic endpoints instead (see §E).


## B. TEMP selection (step by step)

### B1. Random perturbation (difficulty, $\theta_{\mathrm{rnd}}$)

```bash
python TEMP/src/perturb/perturb_model_gaussian.py \
  --base_model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir $TEMP_OUTPUT_ROOT/m23k_random \
  --start_l2 2e-3 \
  --end_l2 8e-3 \
  --num_checkpoints 8
```

Writes under `$TEMP_OUTPUT_ROOT/m23k_random/perturb_all_except_embed_cumulative/`. Then compute 100-token losses:

```bash
RND_DIR=$TEMP_OUTPUT_ROOT/m23k_random/perturb_all_except_embed_cumulative

python TEMP/src/process_loss/get-tokenwise-loss.py \
  --model_dir $RND_DIR \
  --dataset_path $TEMP_DATA_ROOT/m23k-prd-sharegpt/train.jsonl \
  --max_tokens 100

python TEMP/src/process_loss/process_tokenwise_loss.py \
  --base_dir $RND_DIR \
  --chunk_sizes 100 \
  --output_prefix m23k-prd \
  --input_filename m23k-prd-tokenwise-100token-23k.pt
```

### B2. Directional cumulative perturbations (diversity, $\Theta$)

```bash
python TEMP/src/perturb/perturb_model_directional.py \
  --base_model_name Qwen/Qwen2.5-7B-Instruct \
  --direction_start_checkpoint Qwen/Qwen2.5-7B-Instruct \
  --direction_end_checkpoint /path/to/theta_f \
  --output_dir $TEMP_OUTPUT_ROOT/m23k_directional \
  --num_checkpoints 8 \
  --min_radius 1 \
  --final_radius 10 \
  --skip_eval --skip_distance --skip_existing
```

### B3. Tokenwise losses on directional checkpoints (100 / 1000)

```bash
python TEMP/src/process_loss/get-tokenwise-loss.py \
  --model_dir $TEMP_OUTPUT_ROOT/m23k_directional \
  --dataset_path $TEMP_DATA_ROOT/m23k-prd-sharegpt/train.jsonl \
  --max_tokens 1000

python TEMP/src/process_loss/process_tokenwise_loss.py \
  --base_dir $TEMP_OUTPUT_ROOT/m23k_directional \
  --chunk_sizes 100,1000 \
  --output_prefix m23k-prd \
  --input_filename m23k-prd-tokenwise-1000token-23k.pt
```

### B4. Difficulty filter (high-loss cluster)

Compute 100-token means on $\theta_0$ and on each random ckpt, pick the ckpt whose dataset-mean ratio falls in $[2,3]$, then:

```bash
bash TEMP/select-high-loss.sh $RND_DIR \
  --use_ckpts <rnd-ckpt-matching-ratio-2-to-3> \
  --full_data_path $TEMP_DATA_ROOT/m23k-prd-sharegpt/train.jsonl \
  --loss_file_name m23k-prd-100token-r-losses.pt
```

(`select-high-loss.sh` already sets `n_clusters=2`, `sampling_method=high_loss_clusters`, and `--all_samples`, and stratifies by `source`.)

This writes `labeled_idx.npy` under `$TEMP_OUTPUT_ROOT/selection/...` (the difficult subset $V^d_s$).

### B5. Diversity + brittle selection

```bash
bash TEMP/select-custom-path.sh $TEMP_OUTPUT_ROOT/m23k_directional \
  --use_ckpts 1-8 \
  --full_data_path $TEMP_DATA_ROOT/m23k-prd-sharegpt/train.jsonl \
  --loss_file_name m23k-prd-1000token-r-losses.pt \
  --difficulty_loss_file_name m23k-prd-100token-r-losses.pt \
  --subset_indices_path /path/to/difficulty/labeled_idx.npy \
  --softmax \
  --n_sample_per_cluster 4 \
  --sampling_method learnability \
  --init_label_num 1000
```

### B6. SFT + medical eval

```bash
# <run_name> = basename of the selection result directory
bash TEMP/run-m23k.sh <run_name>
# Llama:
# bash TEMP/run-m23k.sh <run_name> --llama
```


## C. Baselines

```bash
bash TEMP/select-baseline.sh middle_perplexity /path/to/ckpt_dir --use_ckpts <id>
bash TEMP/select-baseline.sh embedding /path/to/ckpt_dir --use_ckpts <id>
bash TEMP/select-baseline.sh learnability_baseline /path/to/ckpt_dir --use_ckpts 0,N

python TEMP/src/process_loss/get_embedding.py --help
```

Then train/eval with `TEMP/run-m23k.sh` / `TEMP/run-openthoughts-math.sh` on the resulting run directory name.


## D. Paper hyperparameters (SFT / eval)

From the paper appendix (implemented in `configs/sft_*.yaml` + eval scripts):

| Item | Value |
|------|-------|
| Epochs | 5 |
| LR | 1e-5, cosine, 5% warmup |
| AdamW β | (0.9, 0.95) |
| Effective batch | 16 |
| M23k cutoff | 8192 |
| OpenThoughts cutoff | 32768 |
| Medical eval temperature | 0.7, max new tokens 8192 |
| Math eval temperature | 0.7, max new tokens ~28k |


## E. Ablations

Vary `--num_checkpoints`, `--n_sample_per_cluster`, token chunk sizes (100 / 1000 loss windows), or $\theta_f$ quality (undertrained checkpoints / SynLogic) as in the paper figures. Core selection code paths stay the same.
