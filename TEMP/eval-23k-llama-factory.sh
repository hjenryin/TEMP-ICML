#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_DATA_ROOT="${TEMP_DATA_ROOT:-$REPO_ROOT/data}"
TEMP_OUTPUT_ROOT="${TEMP_OUTPUT_ROOT:-$REPO_ROOT/outputs}"

# conda activate <your-env>

# Parse arguments
dataset_name=${1}
shift  # Remove first argument

# Check for --llama flag
use_llama=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --llama)
            use_llama=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

# 23k dataset uses different directory structure
if [ "$use_llama" = true ]; then
    SPLIT_PATH=m23k-llamafactory-llama
    model_family="llama"
else
    SPLIT_PATH=m23k-llamafactory
    model_family="qwen"
fi

# Determine model prefix for run name
if [ "$model_family" = "llama" ]; then
    MODEL_RUN_PREFIX="llama3.1"
    MODEL_EVAL_PREFIX="llama3.1"
else
    MODEL_RUN_PREFIX="qwen"
    MODEL_EVAL_PREFIX="qwen2.5"
fi

MODEL_PATH=$SPLIT_PATH/$dataset_name/

mkdir -p "${TEMP_OUTPUT_ROOT}/${MODEL_PATH}"

# Kill any existing sglang processes
pgrep -f "sglang" | xargs -r kill -9

# Eval output directory for 23k
if [ "$use_llama" = true ]; then
    EVAL_OUTPUT_DIR=${TEMP_OUTPUT_ROOT}/eval/m23k-llama-factory-llama/
else
    EVAL_OUTPUT_DIR=${TEMP_OUTPUT_ROOT}/eval/m23k-llama-factory/
fi

# Detect number of GPUs and set parallelism
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    echo "CUDA_VISIBLE_DEVICES is set: $CUDA_VISIBLE_DEVICES"
    echo "Using $NUM_GPUS GPUs from CUDA_VISIBLE_DEVICES"
else
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "CUDA_VISIBLE_DEVICES not set, detected $NUM_GPUS GPUs"
fi

if [ "$NUM_GPUS" -eq 8 ]; then
    TP=2
    DP=4
elif [ "$NUM_GPUS" -eq 6 ]; then
    TP=2
    DP=3
elif [ "$NUM_GPUS" -eq 4 ]; then
    TP=1
    DP=4
else
    TP=1
    DP="$NUM_GPUS"
fi
TEMPERATURE=0.7
make -f exp/250318-eval-medical_llm/template.makefile \
  model_path=${TEMP_OUTPUT_ROOT}/${MODEL_PATH} \
  tp=$TP dp=$DP  \
  seed=42 \
  temperature=$TEMPERATURE \
  limit=-1 \
  overwrite=True \
  exp_name=$dataset_name \
  output_dir=$EVAL_OUTPUT_DIR \
  max_new_tokens=8192 \
  port=13146 \
  eval_llm
