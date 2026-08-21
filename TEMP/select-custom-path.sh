#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMP_DATA_ROOT="${TEMP_DATA_ROOT:-$REPO_ROOT/data}"
TEMP_OUTPUT_ROOT="${TEMP_OUTPUT_ROOT:-$REPO_ROOT/outputs}"

# Capture the full command for logging
FULL_COMMAND="$0 $@"

# Usage: ./select-custom-path.sh <ref_model_path> [options]
# Example: ./select-custom-path.sh /path/to/ckpts --full_data_path /path/to/data.jsonl

set -e  # Exit on error

# Function to abbreviate directory names by taking first character of each word
abbreviate() {
    echo "$1" | tr '_-' '\n' | cut -c1 | tr -d '\n'
}
# Check if required arguments are provided
if [ $# -lt 1 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <ref_model_path> [options]"
    echo ""
    echo "Positional arguments:"
    echo "  ref_model_path:     Path to directory containing checkpoints"
    echo ""
    echo "Optional arguments:"
    echo "  --full_data_path <path>:     Path to the training data jsonl file (default: data/m23k-prd-sharegpt/train.jsonl)"
    echo "  --result_dir_name <name>:    Name for the result directory (default: derived from ref_model_path)"
    echo "  --n_clusters <num>:        Number of k-means clusters (default: 100)"
    echo "  --sampling_method <method>:  Sampling method (default: learnability)"
    echo "                               Note: high_loss_clusters requires --use_ckpts and auto-sets n_clusters=2 and --all_samples"
    echo "  --loss_file_name <name>:     Name of the loss file (default: m23k-prd-1000token-r-losses.pt)"
    echo "  --use_ckpts <spec>:          Checkpoint numbers (e.g., '8-15' for checkpoint8 through checkpoint15)"
    echo "                               Examples: '1-10', '1,3,5,7', '1-5,8,10-12'"
    echo "                               Required when using --sampling_method high_loss_clusters"
    echo "  --subset_indices_path <path>: Path to .npy file with subset indices"
    echo "  --all_samples:               For cluster methods, take all samples from selected clusters"
    echo "  --cluster_features <type>:   Whether to cluster on raw or standardized losses (default: raw)"
    echo "                               Options: raw, standardized"
    echo "  --softmax:                   Hierarchical softmax source allocation (paper Eq. 4; T=1 fixed)."
    echo "                               Exclusive with --n_clusters."
    echo "  --n_sample_per_cluster <num>: Number of samples per cluster for softmax allocation (default: 4 when --softmax is set)"
    echo "                               Only used when --softmax is set."
    echo "  --difficulty_loss_file_name <name>: Loss file for softmax difficulty calculation (default: m23k-prd-100token-r-losses.pt)"
    echo "                               Only used when --softmax is set."
    echo "  --seed <num>:                Random seed for reproducibility (default: 42)"
    exit 1
fi

# Parse positional argument
REF_MODEL_PATH=$1
shift

# Default values
# Use OpenThoughts dataset if REF_MODEL_PATH contains "openthoughts"
if [[ "$REF_MODEL_PATH" == *"openthoughts"* ]]; then
    FULL_DATA_PATH=""
else
    FULL_DATA_PATH="${TEMP_DATA_ROOT:-$REPO_ROOT/data}/m23k-prd-sharegpt/train.jsonl"
fi
RESULT_DIR_NAME=""
N_CLUSTERS=100
SAMPLING_METHOD="learnability"
LOSS_FILE_NAME="m23k-prd-1000token-r-losses.pt"
USE_CKPTS=""
SUBSET_INDICES_PATH=""
ALL_SAMPLES_FLAG=""
INIT_LABEL_NUM=1000
CONFIG_FILE="${REPO_ROOT}/configs/default-selection-args.yml"
CLUSTER_FEATURES="raw"
USE_SOFTMAX=false
N_SAMPLE_PER_CLUSTER=""
DIFFICULTY_LOSS_FILE_NAME="m23k-prd-100token-r-losses.pt"
SEED=""

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
        --n_clusters)
            N_CLUSTERS="$2"
            shift 2
            ;;
        --sampling_method)
            SAMPLING_METHOD="$2"
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
        --all_samples)
            ALL_SAMPLES_FLAG="--all_samples"
            shift
            ;;
        --init_label_num)
            INIT_LABEL_NUM="$2"
            shift 2
            ;;
        --cluster_features)
            CLUSTER_FEATURES="$2"
            shift 2
            ;;
        --softmax)
            USE_SOFTMAX=true
            shift
            ;;
        --n_sample_per_cluster)
            N_SAMPLE_PER_CLUSTER="$2"
            shift 2
            ;;
        --difficulty_loss_file_name)
            DIFFICULTY_LOSS_FILE_NAME="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If sampling method is high_loss_clusters, override n_clusters and all_samples
if [[ "$SAMPLING_METHOD" == "high_loss_clusters" ]]; then
    N_CLUSTERS=2
    ALL_SAMPLES_FLAG="--all_samples"
    
    # Ensure use_ckpts is set for high_loss_clusters
    if [[ -z "$USE_CKPTS" ]]; then
        echo "Error: --use_ckpts must be specified when using --sampling_method high_loss_clusters"
        exit 1
    fi
fi

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
# If sampling method is not default learnability, append to result dir name
if [[ "$SAMPLING_METHOD" != "learnability" ]]; then 
    RESULT_DIR_NAME="${RESULT_DIR_NAME}-${SAMPLING_METHOD}"
fi
# If n_clusters is not default, append to result dir name (but not for high_loss_clusters)
if [[ "$N_CLUSTERS" != "100" && "$SAMPLING_METHOD" != "high_loss_clusters" && "$USE_SOFTMAX" != true ]]; then 
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

# Append all_samples indicator if flag is set (but not for high_loss_clusters)
if [[ -n "$ALL_SAMPLES_FLAG" && "$SAMPLING_METHOD" != "high_loss_clusters" ]]; then
    RESULT_DIR_NAME="${RESULT_DIR_NAME}-all_samples"
fi

# Append cluster_features indicator if not default
if [[ "$CLUSTER_FEATURES" != "raw" ]]; then
    RESULT_DIR_NAME="${RESULT_DIR_NAME}-std_loss_cluster"
fi

# Append softmax indicator if enabled (temperature is always 1)
if [[ "$USE_SOFTMAX" == true ]]; then
    RESULT_DIR_NAME="${RESULT_DIR_NAME}-softmax_geo_mean"
fi

# Append n_sample_per_cluster indicator if specified
if [[ -n "$N_SAMPLE_PER_CLUSTER" ]]; then
    RESULT_DIR_NAME="${RESULT_DIR_NAME}-${N_SAMPLE_PER_CLUSTER}_per_cluster"
fi

# Append number of selected samples to result dir name
RESULT_DIR_NAME="${RESULT_DIR_NAME}-${INIT_LABEL_NUM}selected"

# Append loss file name to result dir if not default
LOSS_BASE="${LOSS_FILE_NAME%.pt}"
if [[ "$LOSS_FILE_NAME" != "m23k-prd-1000token-r-losses.pt" ]]; then
    RESULT_DIR_NAME="${RESULT_DIR_NAME}-${LOSS_BASE}"
fi

# Append seed if specified and not default (42)
if [[ -n "$SEED" && "$SEED" != "42" ]]; then
    RESULT_DIR_NAME="${RESULT_DIR_NAME}-seed${SEED}"
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
echo "Init Label Num:   $INIT_LABEL_NUM"
echo "Cluster Features: $CLUSTER_FEATURES"
echo "Softmax:          $USE_SOFTMAX"
echo "N Sample Per Cluster: $N_SAMPLE_PER_CLUSTER"
echo "Difficulty Loss File: $DIFFICULTY_LOSS_FILE_NAME"
echo "Seed:              $SEED"
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
echo "Starting selection..."

# Build command
CMD="python \"$SCRIPT_DIR/src/selection/selection.py\" \
    --config_file \"$CONFIG_FILE\" \
    --full_data_path \"$FULL_DATA_PATH\" \
    --ref_model_path \"$REF_MODEL_PATH\" \
    --result_dir_name \"$RESULT_DIR_NAME\" \
    --sampling_method \"$SAMPLING_METHOD\" \
    --loss_file_name \"$LOSS_FILE_NAME\""

if [[ "$USE_SOFTMAX" != true ]]; then
    CMD="$CMD --n_clusters $N_CLUSTERS"
fi


if [[ -n "$USE_CKPTS" ]]; then
    CMD="$CMD --use_ckpts \"$USE_CKPTS\""
fi

if [[ -n "$SUBSET_INDICES_PATH" ]]; then
    CMD="$CMD --subset_indices_path \"$SUBSET_INDICES_PATH\""
fi

if [[ -n "$ALL_SAMPLES_FLAG" ]]; then
    CMD="$CMD $ALL_SAMPLES_FLAG"
fi

if [[ -n "$INIT_LABEL_NUM" ]]; then
    CMD="$CMD --init_label_num $INIT_LABEL_NUM"
fi

if [[ -n "$SEED" ]]; then
    CMD="$CMD --seed $SEED"
fi

if [[ "$CLUSTER_FEATURES" != "raw" ]]; then
    CMD="$CMD --cluster_features \"$CLUSTER_FEATURES\""
fi

if [[ "$USE_SOFTMAX" == true ]]; then
    CMD="$CMD --softmax"
    if [[ -n "$N_SAMPLE_PER_CLUSTER" ]]; then
        CMD="$CMD --n_sample_per_cluster $N_SAMPLE_PER_CLUSTER"
    fi
    CMD="$CMD --difficulty_loss_file_name \"$DIFFICULTY_LOSS_FILE_NAME\""
fi

eval $CMD
echo ""
if [[ -n "$SUBSET_INDICES_PATH" ]]; then
    echo "Subset directory: $SUBSET_INDICES_PATH" >> "${RESULT_DIR_NAME}/data_selection_info.txt"
fi
echo "Run command: $FULL_COMMAND" >> "${RESULT_DIR_NAME}/data_selection_info.txt"
echo "Training completed successfully!"
