#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMP_DATA_ROOT="${TEMP_DATA_ROOT:-$REPO_ROOT/data}"
TEMP_OUTPUT_ROOT="${TEMP_OUTPUT_ROOT:-$REPO_ROOT/outputs}"

# Capture the full command for logging
FULL_COMMAND="$0 $@"

# Usage: ./select-high-loss.sh <ref_model_path> --use_ckpts <spec> [options]
# Example: ./select-high-loss.sh /path/to/ckpts --use_ckpts 8-15 --full_data_path /path/to/data.jsonl

set -e  # Exit on error

# Check if required arguments are provided
if [ $# -lt 1 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <ref_model_path> --use_ckpts <spec> [options]"
    echo ""
    echo "Positional arguments:"
    echo "  ref_model_path:     Path to directory containing checkpoints"
    echo ""
    echo "Required arguments:"
    echo "  --use_ckpts <spec>:          Checkpoint numbers (e.g., '8-15' for checkpoint8 through checkpoint15)"
    echo "                               Examples: '1-10', '1,3,5,7', '1-5,8,10-12'"
    echo ""
    echo "Optional arguments:"
    echo "  --full_data_path <path>:     Path to the training data jsonl file (default: data/m23k-prd-sharegpt/train.jsonl)"
    echo "  --result_dir_name <name>:    Name for the result directory (default: derived from ref_model_path)"
    echo "  --n_clusters <num>:        Number of k-means clusters (default: 2)"
    echo "  --loss_file_name <name>:     Name of the loss file (default: m23k-prd-100token-r-losses.pt)"
    echo "  --subset_indices_path <path>: Path to .npy file with subset indices"
    exit 1
fi

# Parse positional argument
REF_MODEL_PATH=$1
shift

# Use OpenThoughts dataset if REF_MODEL_PATH contains "openthoughts"
if [[ "$REF_MODEL_PATH" == *"openthoughts"* ]]; then
    FULL_DATA_PATH=""
    RES_DIR="${TEMP_OUTPUT_ROOT:-$REPO_ROOT/outputs}/selection-openthoughts-math"
else
    FULL_DATA_PATH="${TEMP_DATA_ROOT:-$REPO_ROOT/data}/m23k-prd-sharegpt/train.jsonl"
    RES_DIR="${TEMP_OUTPUT_ROOT:-$REPO_ROOT/outputs}/selection"
fi

# Default values
RESULT_DIR_NAME=""
N_CLUSTERS=2
SAMPLING_METHOD="high_loss_clusters"
LOSS_FILE_NAME="m23k-prd-100token-r-losses.pt"
USE_CKPTS=""
SUBSET_INDICES_PATH=""
ALL_SAMPLES_FLAG="--all_samples"
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
        --use_ckpts)
            USE_CKPTS="$2"
            shift 2
            ;;
        --subset_indices_path)
            SUBSET_INDICES_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check that use_ckpts is provided
if [[ -z "$USE_CKPTS" ]]; then
    echo "Error: --use_ckpts is required"
    exit 1
fi

# Derive result_dir_name from ref_model_path if not specified
if [[ -z "$RESULT_DIR_NAME" ]]; then
    RESULT_DIR_NAME="${RES_DIR}/$(basename "$REF_MODEL_PATH")"
else
    # If user provides a relative path, convert to absolute under res/ or res-openthoughts/
    if [[ "$RESULT_DIR_NAME" != /* ]]; then
        RESULT_DIR_NAME="${RES_DIR}/${RESULT_DIR_NAME}"
    fi
fi
# Append sampling method
RESULT_DIR_NAME="${RESULT_DIR_NAME}-${SAMPLING_METHOD}"
# If n_clusters is not default, append to result dir name
if [[ "$N_CLUSTERS" != "2" ]]; then 
    RESULT_DIR_NAME="${RESULT_DIR_NAME}-nclusters${N_CLUSTERS}"
fi
# Append checkpoint spec to result dir if specified
if [[ -n "$USE_CKPTS" ]]; then
    # Replace commas and dashes with underscores for filename safety
    CKPTS_SUFFIX=$(echo "$USE_CKPTS" | tr ',' '_' | tr '-' 'to')
    RESULT_DIR_NAME="${RESULT_DIR_NAME}-ckpts_${CKPTS_SUFFIX}"
fi

# Append subset indicator to result dir if specified
if [[ -n "$SUBSET_INDICES_PATH" ]]; then
    SUBSET_DIR=$(dirname "$SUBSET_INDICES_PATH")
    SUBSET_PARENT=$(dirname "$SUBSET_DIR")
    SUBSET_BASENAME=$(basename "$SUBSET_PARENT")
    RESULT_DIR_NAME="${RESULT_DIR_NAME}-subset"
fi

# Append all_samples indicator if flag is set
if [[ -n "$ALL_SAMPLES_FLAG" ]]; then
    RESULT_DIR_NAME="${RESULT_DIR_NAME}-all_samples"
fi

# Append loss file name to result dir if not default
LOSS_BASE="${LOSS_FILE_NAME%.pt}"
if [[ "$LOSS_FILE_NAME" != "m23k-prd-100token-r-losses.pt" ]]; then
    RESULT_DIR_NAME="${RESULT_DIR_NAME}-${LOSS_BASE}"
fi

# Print configuration
echo "============================================"
echo "Running training with the following config:"
echo "============================================"
echo "Ref Model Path:   $REF_MODEL_PATH"
echo "Full Data Path:   $FULL_DATA_PATH"
echo "Result Dir Name:  $RESULT_DIR_NAME"
echo "Config File:      $CONFIG_FILE"
echo "N Clusters:        $N_CLUSTERS"
echo "Sampling Method:  $SAMPLING_METHOD"
echo "Loss File Name:   $LOSS_FILE_NAME"
echo "Use Ckpts:        $USE_CKPTS"
echo "Subset Indices:   $SUBSET_INDICES_PATH"
echo "All Samples:      $ALL_SAMPLES_FLAG"
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

# Run selection
echo "Starting high-loss selection..."

# Build command
CMD="python \"$SCRIPT_DIR/src/selection/selection.py\" \
    --config_file \"$CONFIG_FILE\" \
    --full_data_path \"$FULL_DATA_PATH\" \
    --ref_model_path \"$REF_MODEL_PATH\" \
    --result_dir_name \"$RESULT_DIR_NAME\" \
    --n_clusters $N_CLUSTERS \
    --sampling_method \"$SAMPLING_METHOD\" \
    --loss_file_name \"$LOSS_FILE_NAME\""

if [[ -n "$USE_CKPTS" ]]; then
    CMD="$CMD --use_ckpts \"$USE_CKPTS\""
fi

if [[ -n "$SUBSET_INDICES_PATH" ]]; then
    CMD="$CMD --subset_indices_path \"$SUBSET_INDICES_PATH\""
fi

if [[ -n "$ALL_SAMPLES_FLAG" ]]; then
    CMD="$CMD $ALL_SAMPLES_FLAG"
fi

eval $CMD

echo ""
if [[ -n "$SUBSET_INDICES_PATH" ]]; then
    echo "Subset directory: $SUBSET_INDICES_PATH" >> "${RESULT_DIR_NAME}/data_selection_info.txt"
fi
echo "Run command: $FULL_COMMAND" >> "${RESULT_DIR_NAME}/data_selection_info.txt"
echo "Training completed successfully!"
