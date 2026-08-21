#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMP_DATA_ROOT="${TEMP_DATA_ROOT:-$REPO_ROOT/data}"
TEMP_OUTPUT_ROOT="${TEMP_OUTPUT_ROOT:-$REPO_ROOT/outputs}"

# Capture the full command for logging
FULL_COMMAND="$0 $@"

# Usage: ./select-baseline.sh <sampling_method> <ref_model_path> [options]
# Example: ./select-baseline.sh middle_perplexity /path/to/ckpts --use_ckpts 10
# Example: ./select-baseline.sh embedding /path/to/ckpts --use_ckpts 10
# Example: ./select-baseline.sh longest_reasoning /path/to/ckpts
# Example: ./select-baseline.sh learnability_baseline /path/to/ckpts --use_ckpts 5,10
# Example: ./select-baseline.sh longest_reasoning /path/to/ckpts
# Example: ./select-baseline.sh learnability_baseline /path/to/ckpts --use_ckpts "1-10"

set -e  # Exit on error

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <sampling_method> <ref_model_path> [options]"
    echo ""
    echo "Positional arguments:"
    echo "  sampling_method:    Baseline method to use ('middle_perplexity', 'embedding', 'longest_reasoning', 'learnability_baseline')"
    echo "  ref_model_path:     Path to directory containing checkpoints"
    echo ""
    echo "Required options:"
    echo "  --use_ckpts <num>:           Checkpoint numbers to use (e.g., '10' for single, '1-10' for range)"
    echo "                               NOTE: middle_perplexity and embedding require exactly one checkpoint"
    echo "                               NOTE: learnability_baseline requires at least two checkpoints"
    echo "                               NOTE: longest_reasoning does not use checkpoints"
    echo ""
    echo "Optional arguments:"
    echo "  --full_data_path <path>:     Path to the training data jsonl file (default: data/m23k-prd-sharegpt/train.jsonl)"
    echo "  --result_dir_name <name>:    Name for the result directory (default: derived from ref_model_path)"
    echo "  --loss_file_name <name>:     Name of the loss file (default: m23k-prd-1000token-r-losses.pt)"
    echo "                               Note: Only used for middle_perplexity, not for embedding"
    echo "  --embedding_file_name <name>: Name of the embedding file (default: embedding.pt)"
    echo "                               Note: Only used for embedding, not for middle_perplexity"
    echo "  --subset_indices_path <path>: Path to .npy file with subset indices"
    echo "  --init_label_num <num>:      Number of samples to select (default: 1000)"
    echo ""
    echo "Note: Parameters like --n_clusters, --softmax, --cluster_features, --all_samples, etc."
    echo "      are NOT allowed or used. These are simple baseline methods."
    exit 1
fi

# Parse positional arguments
SAMPLING_METHOD=$1
REF_MODEL_PATH=$2
shift 2

# Validate sampling method
if [[ "$SAMPLING_METHOD" != "middle_perplexity" && "$SAMPLING_METHOD" != "embedding" && "$SAMPLING_METHOD" != "longest_reasoning" && "$SAMPLING_METHOD" != "learnability_baseline" ]]; then
    echo "Error: sampling_method must be one of 'middle_perplexity', 'embedding', 'longest_reasoning', 'learnability_baseline'"
    echo "Got: $SAMPLING_METHOD"
    exit 1
fi

# Default values
# Use OpenThoughts dataset if REF_MODEL_PATH contains "openthoughts"
if [[ "$REF_MODEL_PATH" == *"openthoughts"* ]]; then
    FULL_DATA_PATH="${TEMP_DATA_ROOT:-$REPO_ROOT/data}/op114kmath-prd-sharegpt/train.jsonl"
else
    FULL_DATA_PATH="${TEMP_DATA_ROOT:-$REPO_ROOT/data}/m23k-prd-sharegpt/train.jsonl"
fi
RESULT_DIR_NAME=""
LOSS_FILE_NAME="m23k-prd-1000token-r-losses.pt"
EMBEDDING_FILE_NAME="embedding.pt"
USE_CKPTS=""
SUBSET_INDICES_PATH=""
INIT_LABEL_NUM=1000
CONFIG_FILE="${REPO_ROOT}/configs/default-selection-args.yml"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full_data_path)
            FULL_DATA_PATH="$2"
            shift 2
            ;;
        --result_dir_name)
            RESULT_DIR_NAME="$2"
            shift 2
            ;;
        --loss_file_name)
            LOSS_FILE_NAME="$2"
            shift 2
            ;;
        --embedding_file_name)
            EMBEDDING_FILE_NAME="$2"
            shift 2
            ;;
        --use_ckpts)
            USE_CKPTS="$2"
            shift 2
            ;;
        --subset_indices_path)
            SUBSET_INDICES_PATH="$2"
            shift 2
            ;;
        --init_label_num)
            INIT_LABEL_NUM="$2"
            shift 2
            ;;

        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate use_ckpts based on sampling method
if [[ "$SAMPLING_METHOD" == "longest_reasoning" ]]; then
    if [[ -n "$USE_CKPTS" ]]; then
        echo "Warning: --use_ckpts is ignored for longest_reasoning method"
    fi
    USE_CKPTS=""  # Clear it
elif [[ "$SAMPLING_METHOD" == "middle_perplexity" || "$SAMPLING_METHOD" == "embedding" ]]; then
    if [[ -z "$USE_CKPTS" ]]; then
        echo "Error: --use_ckpts is required for $SAMPLING_METHOD baseline selection"
        echo "Please specify a single checkpoint number (e.g., --use_ckpts 10)"
        exit 1
    fi
    # Validate that use_ckpts is a single number (no ranges or commas)
    if [[ "$USE_CKPTS" =~ [,-] ]]; then
        echo "Error: --use_ckpts must be a single checkpoint number for $SAMPLING_METHOD selection"
        echo "Got: $USE_CKPTS"
        echo "Expected format: --use_ckpts 10 (not ranges like '1-10' or lists like '1,3,5')"
        exit 1
    fi
elif [[ "$SAMPLING_METHOD" == "learnability_baseline" ]]; then
    if [[ -z "$USE_CKPTS" ]]; then
        echo "Error: --use_ckpts is required for learnability_baseline selection"
        echo "Please specify checkpoint numbers (e.g., --use_ckpts '1-10' for a range)"
        exit 1
    fi
    # Allow ranges and lists for learnability_baseline
fi

# Replace commas with underscores for result directory naming
USE_CKPTS_DIR="${USE_CKPTS//,/_}"

# Function to abbreviate directory names by taking first character of each word
abbreviate() {
    echo "$1" | tr '_-' '\n' | cut -c1 | tr -d '\n'
}

# Derive result_dir_name from ref_model_path if not specified
# Use res-openthoughts/ directory if REF_MODEL_PATH contains "openthoughts"
if [[ "$REF_MODEL_PATH" == *"openthoughts"* ]]; then
    RES_DIR="${TEMP_OUTPUT_ROOT:-$REPO_ROOT/outputs}/selection-openthoughts-math"
else
    RES_DIR="${TEMP_OUTPUT_ROOT:-$REPO_ROOT/outputs}/selection"
fi

if [[ -z "$RESULT_DIR_NAME" ]]; then
    RESULT_DIR_NAME="${RES_DIR}/$(abbreviate "$(basename "$REF_MODEL_PATH")")"
else
    # If user provides a relative path, convert to absolute under res/ or res-openthoughts/
    if [[ "$RESULT_DIR_NAME" != /* ]]; then
        RESULT_DIR_NAME="${RES_DIR}/${RESULT_DIR_NAME}"
    fi
fi

# Append sampling method to result dir name
RESULT_DIR_NAME="${RESULT_DIR_NAME}-${SAMPLING_METHOD}"

# Append checkpoint spec to result dir (just the number for single checkpoint)
RESULT_DIR_NAME="${RESULT_DIR_NAME}-ckpt${USE_CKPTS_DIR}"

# Append subset indicator to result dir if specified
if [[ -n "$SUBSET_INDICES_PATH" ]]; then
    SUBSET_DIR=$(dirname "$SUBSET_INDICES_PATH")
    SUBSET_PARENT=$(dirname "$SUBSET_DIR")
    SUBSET_BASENAME=$(basename "$SUBSET_PARENT")
    RESULT_DIR_NAME="${RESULT_DIR_NAME}-subset"
fi

# Append number of selected samples to result dir name
RESULT_DIR_NAME="${RESULT_DIR_NAME}-${INIT_LABEL_NUM}selected"

# Append loss file name to result dir if not default
LOSS_BASE="${LOSS_FILE_NAME%.pt}"
if [[ "$LOSS_FILE_NAME" != "m23k-prd-1000token-r-losses.pt" ]]; then
    RESULT_DIR_NAME="${RESULT_DIR_NAME}-${LOSS_BASE}"
fi

# Check if directory exists and find next available number (only when using subset)
if [[ -n "$SUBSET_INDICES_PATH" ]]; then
    ORIGINAL_RESULT_DIR_NAME="$RESULT_DIR_NAME"
    COUNTER=2
    while [ -d "$RESULT_DIR_NAME/data" ]; do
        echo "Directory $RESULT_DIR_NAME/data already exists, trying -${COUNTER}"
        RESULT_DIR_NAME="${ORIGINAL_RESULT_DIR_NAME}-${COUNTER}"
        COUNTER=$((COUNTER + 1))
    done
fi

# Print configuration
# Print configuration
echo "============================================"
echo "Running baseline $SAMPLING_METHOD selection with the following config:"
echo "============================================"
echo "Ref Model Path:   $REF_MODEL_PATH"
echo "Full Data Path:   $FULL_DATA_PATH"
echo "Result Dir Name:  $RESULT_DIR_NAME"
echo "Config File:      $CONFIG_FILE"
echo "Sampling Method:  $SAMPLING_METHOD (FIXED - no clustering, no sources)"
if [[ "$SAMPLING_METHOD" == "middle_perplexity" ]]; then
    echo "Loss File Name:   $LOSS_FILE_NAME"
elif [[ "$SAMPLING_METHOD" == "embedding" ]]; then
    echo "Embedding File:   $EMBEDDING_FILE_NAME"
elif [[ "$SAMPLING_METHOD" == "learnability_baseline" ]]; then
    echo "Loss File Name:   $LOSS_FILE_NAME"
fi
if [[ "$SAMPLING_METHOD" != "longest_reasoning" ]]; then
    echo "Use Ckpt:         $USE_CKPTS"
fi
echo "Subset Indices:   $SUBSET_INDICES_PATH"
echo "Init Label Num:   $INIT_LABEL_NUM"
echo "============================================"
echo ""

# Check if ref model path exists
if [ ! -d "$REF_MODEL_PATH" ]; then
    echo "Error: Ref model path not found at $REF_MODEL_PATH"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Check if data file exists
if [ ! -f "$FULL_DATA_PATH" ]; then
    echo "Warning: Data file not found at $FULL_DATA_PATH"
    echo "Proceeding anyway..."
fi

# Run baseline selection
echo "Starting baseline $SAMPLING_METHOD selection..."

# Build command
CMD="python \"$SCRIPT_DIR/src/selection/selection.py\" \
    --config_file \"$CONFIG_FILE\" \
    --full_data_path \"$FULL_DATA_PATH\" \
    --ref_model_path \"$REF_MODEL_PATH\" \
    --result_dir_name \"$RESULT_DIR_NAME\" \
    --sampling_method \"$SAMPLING_METHOD\" \
    --init_label_num $INIT_LABEL_NUM \
    --loss_file_name \"$LOSS_FILE_NAME\""

if [[ "$SAMPLING_METHOD" != "longest_reasoning" ]]; then
    CMD="$CMD --use_ckpts \"$USE_CKPTS\""
else
    CMD="$CMD --use_ckpts \"None\""
fi

# Add method-specific parameters
if [[ "$SAMPLING_METHOD" == "middle_perplexity" || "$SAMPLING_METHOD" == "learnability_baseline" ]]; then
    CMD="$CMD --loss_file_name \"$LOSS_FILE_NAME\""
elif [[ "$SAMPLING_METHOD" == "embedding" ]]; then
    CMD="$CMD --embedding_file_name \"$EMBEDDING_FILE_NAME\""
fi

if [[ -n "$SUBSET_INDICES_PATH" ]]; then
    CMD="$CMD --subset_indices_path \"$SUBSET_INDICES_PATH\""
fi

eval $CMD
echo ""
if [[ -n "$SUBSET_INDICES_PATH" ]]; then
    echo "Subset directory: $SUBSET_INDICES_PATH" >> "${RESULT_DIR_NAME}/data_selection_info.txt"
fi
echo "Run command: $FULL_COMMAND" >> "${RESULT_DIR_NAME}/data_selection_info.txt"
echo "Baseline $SAMPLING_METHOD selection completed successfully!"
