#!/usr/bin/env python3
"""
LLaMA-Factory based TEMP training script
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

_REPO_ROOT = Path(__file__).resolve().parents[3]  # TEMP/src/train -> repo root
_DATA_ROOT = Path(os.environ.get("TEMP_DATA_ROOT", _REPO_ROOT / "data"))
_OUTPUT_ROOT = Path(os.environ.get("TEMP_OUTPUT_ROOT", _REPO_ROOT / "outputs"))


def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def extract_name_from_npy_path(npy_path):
    """Extract a run name from a selection indices .npy path."""
    path_parts = Path(npy_path)
    basename = path_parts.stem  # filename without extension

    if basename == "labeled_idx":
        # e.g., .../selection/run-name/data/labeled_idx.npy -> run-name
        grandparent_dir = path_parts.parent.parent.name
        return grandparent_dir
    else:
        parent_dir = path_parts.parent.name
        return f"{parent_dir}-{basename}"


def create_subset_dataset(npy_path, source_jsonl, output_dir, name, dataset_type='m23k'):
    """
    Create a subset dataset by selecting specific indices from the source JSONL

    Args:
        npy_path: Path to .npy file containing indices (or None for full data)
        source_jsonl: Path to source train.jsonl file
        output_dir: Output directory for the subset
        name: Name for the dataset
        dataset_type: Type of dataset ('m23k' or 'openthoughts') for correct formatting
    """
    log(f"Creating subset dataset: {name}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load indices if provided, otherwise use all
    if npy_path:
        log(f"Loading indices from: {npy_path}")
        indices = np.load(npy_path)
        log(f"Loaded {len(indices)} indices")
        indices_set = set(indices.tolist())
    else:
        log("No subset file provided, using full dataset")
        indices_set = None

    # Read source JSONL and filter by indices
    log(f"Reading source data from: {source_jsonl}")
    output_jsonl = output_path / "train.jsonl"

    selected_count = 0
    with open(source_jsonl, 'r') as f_in, open(output_jsonl, 'w') as f_out:
        for idx, line in enumerate(f_in):
            if indices_set is None or idx in indices_set:
                f_out.write(line)
                selected_count += 1

    log(f"Selected {selected_count} samples, saved to: {output_jsonl}")

    # Create dataset_info.json with format based on dataset type
    if dataset_type == 'openthoughts':
        dataset_info = {
            name: {
                "file_name": "train.jsonl",
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations"
                },
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant"
                },
                "system": "You are a helpful assistant."
            }
        }
    else:  # m23k
        dataset_info = {
            name: {
                "file_name": "train.jsonl",
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations"
                },
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant"
                },
                "system": "You are a helpful assistant."
            }
        }

    dataset_info_path = output_path / "dataset_info.json"
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)

    log(f"Created dataset_info.json at: {dataset_info_path}")

    return str(output_path), name


def get_gpu_count():
    """
    Get the number of GPUs to use based on CUDA_VISIBLE_DEVICES
    Returns the count and the actual device IDs
    """
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    
    if cuda_visible is None:
        # Not set, use all available GPUs
        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True,
                check=True
            )
            gpu_count = len(result.stdout.strip().split('\n'))
            log(f"CUDA_VISIBLE_DEVICES not set, using all {gpu_count} GPUs")
            return gpu_count
        except Exception as e:
            log(f"Error detecting GPUs: {e}, defaulting to 1 GPU")
            return 1
    else:
        # Parse CUDA_VISIBLE_DEVICES
        gpu_ids = [x.strip() for x in cuda_visible.split(',') if x.strip()]
        gpu_count = len(gpu_ids)
        log(f"CUDA_VISIBLE_DEVICES={cuda_visible}, using {gpu_count} GPU(s)")
        return gpu_count


def calculate_batch_size(gpu_count, dataset_type='m23k'):
    """
    Calculate per_device_batch_size based on GPU count and dataset type

    For m23k: 16 / gpu_count, with special case for 1 GPU
    For openthoughts: Always 1
    """
    if dataset_type == 'openthoughts':
        return 1

    assert 16%gpu_count == 0, f"16 must be divisible by gpu_count ({gpu_count})"
    return 2

def calculate_gradient_accumulation_steps(gpu_count, per_device_batch_size, target_batch_size=16):
    """
    Calculate gradient_accumulation_steps to maintain target effective batch size

    Effective batch size = gpu_count * per_device_batch_size * gradient_accumulation_steps
    We want effective_batch_size = target_batch_size
    So: gradient_accumulation_steps = target_batch_size / (gpu_count * per_device_batch_size)
    """
    effective_per_step = gpu_count * per_device_batch_size
    grad_accum_steps = target_batch_size // effective_per_step
    assert grad_accum_steps >= 1, f"Gradient accumulation steps must be >= 1, got {grad_accum_steps}"
    return grad_accum_steps


def launch_training(config_path, dataset_dir, dataset_name, output_dir, gpu_count, run_name, dataset_type='m23k', target_batch_size=16, model_name_or_path=None, seed=None):
    """
    Launch LLaMA-Factory training with the specified configuration
    """
    # Calculate batch size and gradient accumulation steps
    batch_size = calculate_batch_size(gpu_count, dataset_type)
    grad_accum_steps = calculate_gradient_accumulation_steps(gpu_count, batch_size, target_batch_size)
    effective_batch_size = gpu_count * batch_size * grad_accum_steps
    
    log(f"Calculated per_device_train_batch_size: {batch_size}")
    log(f"Calculated gradient_accumulation_steps: {grad_accum_steps}")
    log(f"Effective batch size: {effective_batch_size}")
    assert effective_batch_size == target_batch_size, f"Effective batch size {effective_batch_size} != target {target_batch_size}"
    
    # Build command
    cmd = [
        "llamafactory-cli", "train",
        config_path,
        f"dataset_dir={dataset_dir}",
        f"dataset={dataset_name}",
        f"output_dir={output_dir}",
        f"per_device_train_batch_size={batch_size}",
        f"gradient_accumulation_steps={grad_accum_steps}",
        f"run_name={run_name}",
    ]
    
    # Override model if specified
    if model_name_or_path:
        cmd.append(f"model_name_or_path={model_name_or_path}")
    
    # Override seed if specified
    if seed is not None:
        cmd.append(f"seed={seed}")
    
    log("=" * 60)
    log("Launching LLaMA-Factory training")
    log("=" * 60)
    log(f"Command: {' '.join(cmd)}")
    log(f"Config: {config_path}")
    log(f"Dataset: {dataset_name} (from {dataset_dir})")
    log(f"Output: {output_dir}")
    log(f"GPUs: {gpu_count}")
    log(f"Per-device batch size: {batch_size}")
    log(f"Gradient accumulation steps: {grad_accum_steps}")
    log(f"Effective batch size: {effective_batch_size}")
    log(f"Run name: {run_name}")
    if model_name_or_path:
        log(f"Model: {model_name_or_path}")
    if seed is not None:
        log(f"Seed: {seed}")
    log("=" * 60)
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        log("Training completed successfully")
        return result.returncode
    except subprocess.CalledProcessError as e:
        log(f"Training failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        log("Training interrupted by user")
        return 130
    except Exception as e:
        log(f"Error during training: {e}")
        return 1


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='LLaMA-Factory TEMP Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                                    # Train on full m23k dataset
  %(prog)s --subset_file /path/to/indices.npy  # Train on m23k subset
  %(prog)s --dataset_type openthoughts        # Train on OpenThoughts dataset
        '''
    )
    parser.add_argument('--subset_file', type=str, default=None,
                        help='Path to .npy file containing subset indices (0-indexed). If not provided, uses full dataset.')
    parser.add_argument('--dataset_type', type=str, default='m23k', choices=['m23k', 'openthoughts'],
                        help='Dataset type to train on (default: m23k)')
    parser.add_argument('--llama', action='store_true',
                        help='Use meta-llama/Llama-3.1-8B-Instruct instead of default Qwen model')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for training (default: uses value from config, typically 42)')
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Dataset-specific configuration
    MODEL_SIZE = "7b"
    
    # Determine model based on --llama flag
    if args.llama:
        MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
        MODEL_PREFIX = "llama"
        MODEL_TAG = "-llama"
    else:
        MODEL_NAME = None  # Will use default from config (Qwen)
        MODEL_PREFIX = "qwen"
        MODEL_TAG = ""

    if args.dataset_type == 'm23k':
        DATA_TYPE = "prd"
        SOURCE_JSONL = str(_DATA_ROOT / "m23k-prd-sharegpt" / "train.jsonl")
        if args.llama:
            CONFIG_YAML = str(_REPO_ROOT / "configs" / "sft_m23k_llama.yaml")
        else:
            CONFIG_YAML = str(_REPO_ROOT / "configs" / "sft_m23k.yaml")
        SUBSET_BASE_DIR = str(_OUTPUT_ROOT / "23k_subset")
        TARGET_BATCH_SIZE = 16
    elif args.dataset_type == 'openthoughts':
        DATA_TYPE = "openthoughts"
        SOURCE_JSONL = str(_DATA_ROOT / "op114kmath-prd-sharegpt" / "train.jsonl")
        CONFIG_YAML = str(_REPO_ROOT / "configs" / "sft_openthoughts.yaml")
        SUBSET_BASE_DIR = str(_OUTPUT_ROOT / "openthoughts-math_subset")
        TARGET_BATCH_SIZE = 16

    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")

    log("=" * 60)
    log("LLaMA-Factory Training Manager Started")
    log("=" * 60)
    log(f"Dataset type: {args.dataset_type}")
    log(f"Model: {MODEL_NAME if MODEL_NAME else 'Qwen2.5-7B-Instruct (default)'}")
    log(f"Model size: {MODEL_SIZE}")
    log(f"Data type: {DATA_TYPE}")
    log(f"Source data: {SOURCE_JSONL}")
    log(f"Config: {CONFIG_YAML}")
    log(f"Target batch size: {TARGET_BATCH_SIZE}")
    log("=" * 60)
    
    # Determine dataset name and create subset if needed
    if args.subset_file:
        name = extract_name_from_npy_path(args.subset_file)
        dataset_dir, dataset_name = create_subset_dataset(
            args.subset_file,
            SOURCE_JSONL,
            os.path.join(SUBSET_BASE_DIR, name),
            name,
            dataset_type=args.dataset_type
        )
        if args.dataset_type == 'm23k':
            output_dir = str(_OUTPUT_ROOT / f"m23k-llamafactory{MODEL_TAG}" / name)
            run_name = f"{MODEL_PREFIX}_{MODEL_SIZE}_selection_1k_{name}"
        else:  # openthoughts
            output_dir = str(_OUTPUT_ROOT / f"openthoughts-llamafactory{MODEL_TAG}" / name)
            run_name = f"{MODEL_PREFIX}_{MODEL_SIZE}_openthoughts_{name}"
    else:
        name = "full"
        dataset_dir, dataset_name = create_subset_dataset(
            None,
            SOURCE_JSONL,
            os.path.join(SUBSET_BASE_DIR, name),
            name,
            dataset_type=args.dataset_type
        )
        if args.dataset_type == 'm23k':
            output_dir = str(_OUTPUT_ROOT / f"m23k-llamafactory{MODEL_TAG}" / name)
            run_name = f"{MODEL_PREFIX}_{MODEL_SIZE}_selection_full"
        else:  # openthoughts
            output_dir = str(_OUTPUT_ROOT / f"openthoughts-llamafactory{MODEL_TAG}" / name)
            run_name = f"{MODEL_PREFIX}_{MODEL_SIZE}_openthoughts_full"

    log(f"Dataset name: {dataset_name}")
    log(f"Output directory: {output_dir}")
    log(f"Run name: {run_name}")

    # Get GPU count
    gpu_count = get_gpu_count()

    # Launch training
    exit_code = launch_training(
        CONFIG_YAML,
        dataset_dir,
        dataset_name,
        output_dir,
        gpu_count,
        run_name,
        dataset_type=args.dataset_type,
        target_batch_size=TARGET_BATCH_SIZE,
        model_name_or_path=MODEL_NAME,
        seed=args.seed
    )
    
    log(f"Script exited with code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
