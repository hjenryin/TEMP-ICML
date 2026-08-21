"""
Process token-wise losses for fixed-token chunks on m23k dataset.

This script processes token-wise losses from checkpoint files, cutting reasoning chains
at fixed token chunk sizes and computing average losses for REASONING ONLY.

Input (from get-tokenwise-loss.py):
    - checkpoint-*/m23k-prd-tokenwise-{N}token-23k.pt
      (or op114kmath-prd-tokenwise-…); list of per-token loss tensors

Outputs for each checkpoint (mean of first N reasoning tokens):
    - m23k-prd-100token-r-losses.pt
    - m23k-prd-1000token-r-losses.pt
    - m23k-prd-full-r-losses.pt  (when chunk_size=-1)
    
    Note: Only reasoning tokens, no answer included.

Algorithm:
    1. For each chunk size in chunk_sizes:
       - Take first min(chunk_size, len(reasoning)) tokens
       - Compute mean loss of these tokens (reasoning only, no answer)
"""

import torch
from pathlib import Path
import argparse
import os
from tqdm import tqdm


def process_fixed_chunks(tokenwise_losses, output_dir, 
                                       output_prefix, chunk_sizes=[100, 200, 300, 500, 1000]):
    """
    Process losses for fixed token chunk sizes (REASONING ONLY, no answer).

    For each chunk size in chunk_sizes:
    - Take first min(chunk_size, len(reasoning_tokens)) tokens from reasoning
    - Compute mean loss of reasoning tokens only (no answer included)
    - If chunk_size == -1, take all available tokens (full)

    Args:
        tokenwise_losses: List of dicts with 'reasoning' and 'answer' tensors, or list of tensors.
                         If dict, concatenates reasoning and answer tokens.
                         If tensor, uses directly.
        output_dir: Directory to save output files
        output_prefix: Prefix for output files (e.g., 'm23k-prd')
        chunk_sizes: List of chunk sizes in tokens (default [100, 200, 300, 500, 1000]).
                     Use -1 to include all available tokens.
    """
    output_dir = Path(output_dir)
    n_samples = len(tokenwise_losses)

    for chunk_size in chunk_sizes:
        # Handle -1 as "full" for all available tokens
        if chunk_size == -1:
            chunk_label = "full"
            print(f"\nProcessing all available tokens (full)...")
        else:
            chunk_label = f"{chunk_size}token"
            print(f"\nProcessing chunk size {chunk_size} tokens...")

        r_losses = []   # reasoning only losses

        for i in tqdm(range(n_samples), desc=f"Chunk {chunk_label}"):
            sample = tokenwise_losses[i]
            
            # Handle both dict format and direct tensor format
            if isinstance(sample, dict):
                # Concatenate reasoning and answer tokens
                reasoning_tokens = torch.cat([sample['reasoning'], sample['answer']])
            else:
                # Assume it's already a tensor
                reasoning_tokens = sample

            # Take first min(chunk_size, len(reasoning_tokens)) tokens from reasoning
            # If chunk_size == -1, take all tokens
            if chunk_size == -1:
                reasoning_segment = reasoning_tokens
            else:
                n_tokens = min(chunk_size, len(reasoning_tokens))
                reasoning_segment = reasoning_tokens[:n_tokens]

            # Compute r loss (reasoning segment only)
            if len(reasoning_segment) > 0:
                r_loss = reasoning_segment.mean().item()
            else:
                r_loss = 0.0  # Use 0 if no tokens taken
            r_losses.append(r_loss)

        # Save r losses
        r_tensor = torch.tensor(r_losses)
        print(r_tensor.shape)
        r_path = output_dir / f"{output_prefix}-{chunk_label}-r-losses.pt"
        torch.save(r_tensor, r_path)
        print(f"  Saved r losses to {r_path}")
        print(f"  Mean r loss: {r_tensor.mean().item():.4f}")


def process_checkpoint(checkpoint_dir, input_filename, output_prefix, chunk_sizes=[100, 200, 300, 500, 1000]):
    """Process a single checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    print(f"\n{'='*80}")
    print(f"Processing checkpoint: {checkpoint_dir.name}")
    print(f"{'='*80}")

    # Load tokenwise losses
    tokenwise_path = checkpoint_dir / input_filename
    if not tokenwise_path.exists():
        print(f"ERROR: Tokenwise losses not found at {tokenwise_path}")
        return

    print(f"Loading tokenwise losses from {tokenwise_path}...")
    tokenwise_losses = torch.load(tokenwise_path)
    print(f"Loaded {len(tokenwise_losses)} samples")

    # Process fixed chunk sizes (reasoning only)
    process_fixed_chunks(
        tokenwise_losses, checkpoint_dir, output_prefix, 
        chunk_sizes=chunk_sizes
    )

    print(f"\n{'='*80}")
    print(f"Completed processing {checkpoint_dir.name}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process token-wise losses with fixed token chunks (reasoning only) for m23k"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoint-*",
        help="Path to checkpoint directory (can use glob pattern like 'checkpoint-*')"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=os.environ.get("TEMP_OUTPUT_ROOT", "outputs"),
        help="Base directory containing checkpoint folders"
    )
    parser.add_argument(
        "--input_filename",
        type=str,
        default="m23k-prd-tokenwise-1000token-23k.pt",
        help="Raw tokenwise file from get-tokenwise-loss.py (e.g. m23k-prd-tokenwise-100token-23k.pt)"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="m23k-prd",
        help="Prefix for output files (e.g., 'm23k-prd' -> 'm23k-prd-100token-r-losses.pt')"
    )
    parser.add_argument(
        "--chunk_sizes",
        type=str,
        default="100,200,300,500,1000",
        help="Comma-separated list of chunk sizes in tokens (default: '100,200,300,500,1000'). Use -1 to include all available tokens."
    )

    args = parser.parse_args()

    # Parse chunk sizes
    try:
        chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(',')]
    except ValueError:
        print(f"ERROR: Invalid chunk_sizes format: {args.chunk_sizes}. Expected comma-separated integers.")
        return

    print(f"Using chunk sizes: {chunk_sizes}")

    # Handle glob patterns in checkpoint_dir
    base_dir = Path(args.base_dir)
    checkpoint_pattern = args.checkpoint_dir

    # If checkpoint_dir contains wildcards, expand it
    if '*' in checkpoint_pattern:
        checkpoint_dirs = sorted(base_dir.glob(checkpoint_pattern))
    else:
        # Single checkpoint directory
        checkpoint_path = base_dir / checkpoint_pattern if not Path(checkpoint_pattern).is_absolute() else Path(checkpoint_pattern)
        checkpoint_dirs = [checkpoint_path]

    if not checkpoint_dirs:
        print(f"ERROR: No checkpoint directories found matching pattern: {checkpoint_pattern}")
        return

    print(f"Found {len(checkpoint_dirs)} checkpoint(s) to process:")
    for cp in checkpoint_dirs:
        print(f"  - {cp}")

    # Process each checkpoint
    for checkpoint_dir in checkpoint_dirs:
        try:
            process_checkpoint(
                checkpoint_dir,
                args.input_filename,
                args.output_prefix,
                chunk_sizes
            )
        except Exception as e:
            print(f"ERROR processing {checkpoint_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
