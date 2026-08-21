#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMP_DATA_ROOT="${TEMP_DATA_ROOT:-$REPO_ROOT/data}"
TEMP_OUTPUT_ROOT="${TEMP_OUTPUT_ROOT:-$REPO_ROOT/outputs}"

# Parse arguments
dataset=$1
shift  # Remove first argument

# Check for --llama and --seed flags
use_llama=false
seed_value=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --llama)
            use_llama=true
            shift
            ;;
        --seed)
            seed_value="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

RES_DIR="${TEMP_OUTPUT_ROOT:-$REPO_ROOT/outputs}/selection"

LLAMA_TAG=""
if [ "$use_llama" = true ]; then
    LLAMA_TAG="-llama"
fi

MODEL_DIR="${TEMP_OUTPUT_ROOT}/m23k-llamafactory${LLAMA_TAG}/$dataset"

cd "$SCRIPT_DIR"
# Build python command with seed parameter
seed_arg=""
if [ -n "$seed_value" ]; then
    seed_arg="--seed $seed_value"
fi

if [ -d "$MODEL_DIR" ]; then
    echo "Trained model exists at $MODEL_DIR. Skipping training."
else
    # Build python command
    if [ "$use_llama" = true ]; then
        DISABLE_VERSION_CHECK=1 python src/train/llama_factory.py --subset_file "$RES_DIR/$dataset/data/labeled_idx.npy" --llama $seed_arg
    else
        DISABLE_VERSION_CHECK=1 python src/train/llama_factory.py --subset_file "$RES_DIR/$dataset/data/labeled_idx.npy" $seed_arg
        echo "done"
    fi
fi

# Evaluate
if [ "$use_llama" = true ]; then
    bash eval-23k-llama-factory.sh $dataset --llama
else
    bash eval-23k-llama-factory.sh $dataset
fi
