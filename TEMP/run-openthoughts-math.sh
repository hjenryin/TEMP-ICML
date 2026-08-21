SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMP_DATA_ROOT="${TEMP_DATA_ROOT:-$REPO_ROOT/data}"
TEMP_OUTPUT_ROOT="${TEMP_OUTPUT_ROOT:-$REPO_ROOT/outputs}"
EVALCHEMY_DIR="${EVALCHEMY_DIR:-$REPO_ROOT/../evalchemy}"

# set -e

# Parse arguments
SEED=""
ARG=""
SUBSET_FILE=""
LLAMA_FLAG=""
LLAMA_TAG=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --subset_file)
      SUBSET_FILE="$2"
      shift 2
      ;;
    --llama)
      LLAMA_FLAG="--llama"
      LLAMA_TAG="-llama"
      shift
      ;;
    *)
      ARG="$1"
      shift
      ;;
  esac
done

RES_DIR="${TEMP_OUTPUT_ROOT:-$REPO_ROOT/outputs}/selection-openthoughts-math"

if [ -z "$ARG" ]; then
  echo "Usage: $0 <arg> [--seed <seed_value>] [--subset_file <path_to_labeled_idx.npy>] [--llama]"
  exit 1
fi

# Determine subset file path
if [ -z "$SUBSET_FILE" ]; then
  SUBSET_FILE="$RES_DIR/$ARG/data/labeled_idx.npy"
fi

MODEL_DIR="${TEMP_OUTPUT_ROOT}/openthoughts-llamafactory${LLAMA_TAG}/$ARG"
DATASET_TYPE="openthoughts"
EVAL_OUT="${TEMP_OUTPUT_ROOT}/eval/openthoughts-math/$ARG"
# OpenThoughts-Math suite (paper math eval)
MATH_TASKS="MATH500,AMC23,AIME24,AIME25"
MAX_TOKENS=27768
MAX_MODEL_LEN=32768

has_model_artifacts() {
    local dir="$1"
    [ -f "$dir/config.json" ] || return 1
    [ -f "$dir/tokenizer_config.json" ] || return 1
    find "$dir" -maxdepth 1 \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) | grep -q .
}

cd "$SCRIPT_DIR"
run_training() {
    local -a train_cmd=(
        python src/train/llama_factory.py
        --dataset_type "$DATASET_TYPE"
        --subset_file "$SUBSET_FILE"
    )
    if [ -n "$SEED" ]; then
        train_cmd+=(--seed "$SEED")
    fi
    if [ -n "$LLAMA_FLAG" ]; then
        train_cmd+=($LLAMA_FLAG)
    fi
    DISABLE_VERSION_CHECK=1 "${train_cmd[@]}"
}

if has_model_artifacts "$MODEL_DIR"; then
    echo "Trained model exists at $MODEL_DIR. Skipping training."
else
    if [ -d "$MODEL_DIR" ]; then
      echo "Found $MODEL_DIR, but it does not contain complete model artifacts. Training will run."
    fi
    run_training
fi

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    DATA_PARALLEL_SIZE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    DATA_PARALLEL_SIZE=$(nvidia-smi --list-gpus | wc -l)
fi

if [ ! -d "$EVALCHEMY_DIR" ]; then
    echo "Error: EVALCHEMY_DIR not found: $EVALCHEMY_DIR"
    echo "Set it to your evalchemy checkout, e.g. export EVALCHEMY_DIR=/path/to/evalchemy"
    exit 1
fi

mkdir -p "$EVAL_OUT"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Evaluating with evalchemy (tasks=$MATH_TASKS, data_parallel_size=$DATA_PARALLEL_SIZE)"
echo "  model:  $MODEL_DIR"
echo "  output: $EVAL_OUT"
echo "  cwd:    $EVALCHEMY_DIR"

(
  cd "$EVALCHEMY_DIR"
  python -m eval.eval \
    --model vllm \
    --tasks "$MATH_TASKS" \
    --model_args "pretrained=${MODEL_DIR},tensor_parallel_size=1,data_parallel_size=${DATA_PARALLEL_SIZE},max_model_len=${MAX_MODEL_LEN}" \
    --batch_size 2048 \
    --output_path "$EVAL_OUT" \
    --max_tokens "$MAX_TOKENS"
)

python "${REPO_ROOT}/TEMP/src/train/extract_accuracies-math.py" "$EVAL_OUT"

if [ -f "$RES_DIR/$ARG/data_selection_info.txt" ]; then
    mkdir -p "$EVAL_OUT"
    cp "$RES_DIR/$ARG/data_selection_info.txt" "$EVAL_OUT/data_selection_info-$(date +%Y%m%d-%H%M%S).txt"
fi
