"""
Generate cumulative Gaussian-perturbed checkpoints from a base model.

Perturbs all parameters except embeddings. Independent Gaussian increments
add in quadrature: ||total||² = Σ||noise_i||². To go from L2/√d = a to b in
n checkpoints (n-1 increments):

    Δ = sqrt((b² - a²) / (n - 1))
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name, device="cuda"):
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def should_perturb_param(param_name: str) -> bool:
    return "embed" not in param_name.lower()


def add_gaussian_perturbation_inplace(model, target_l2_per_sqrt_d: float) -> int:
    num_perturbed = 0
    num_skipped = 0
    with torch.no_grad():
        for name, param in tqdm(
            model.named_parameters(),
            desc=f"Adding perturbation (L2/√d={target_l2_per_sqrt_d:.4e})",
        ):
            if should_perturb_param(name):
                noise = torch.randn_like(param) * target_l2_per_sqrt_d
                param.add_(noise)
                num_perturbed += 1
            else:
                num_skipped += 1
    print(f"Perturbed {num_perturbed} parameters, skipped {num_skipped} parameters")
    return num_perturbed


def find_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None
    checkpoint_nums = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-"):
            try:
                checkpoint_nums.append(int(item.split("-")[1]))
            except (ValueError, IndexError):
                continue
    return max(checkpoint_nums) if checkpoint_nums else None


def main():
    parser = argparse.ArgumentParser(
        description="Generate cumulative Gaussian perturbed model checkpoints"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Base output directory (writes under perturb_all_except_embed_cumulative/)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--start_l2", type=float, default=2e-3)
    parser.add_argument("--end_l2", type=float, default=8e-3)
    parser.add_argument("--num_checkpoints", type=int, default=8)
    args = parser.parse_args()

    a, b, n = args.start_l2, args.end_l2, args.num_checkpoints
    if n < 1:
        raise ValueError("num_checkpoints must be at least 1")

    args.output_dir = os.path.join(
        args.output_dir, "perturb_all_except_embed_cumulative"
    )

    if n == 1:
        delta_l2_per_sqrt_d = a
        l2_norms = [a]
    else:
        delta_l2_per_sqrt_d = np.sqrt((b**2 - a**2) / (n - 1))
        l2_norms = [np.sqrt(a**2 + i * delta_l2_per_sqrt_d**2) for i in range(n)]

    print("=" * 80)
    print("Cumulative Gaussian Perturbation")
    print("=" * 80)
    print(f"Start L2/√d: {a:.4e}")
    print(f"End L2/√d: {b:.4e}")
    print(f"Number of checkpoints: {n}")
    print(f"Increment per step (Δ): {delta_l2_per_sqrt_d:.4e}")
    print("Parameter filter: all parameters except embeddings")
    print(f"Output directory: {args.output_dir}")
    for i, norm in enumerate(l2_norms, 1):
        print(f"  Checkpoint {i}: {norm:.4e}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)
    latest_checkpoint = find_latest_checkpoint(args.output_dir)
    if latest_checkpoint is not None:
        print(f"\nFound existing checkpoint-{latest_checkpoint} in {args.output_dir}")
        print("Cumulative perturbation cannot resume. Remove checkpoints or change --output_dir.")
        return

    print(f"\nLoading tokenizer/model from: {args.base_model_name}")
    model, tokenizer = load_model(args.base_model_name, device=args.device)

    for i in range(n):
        checkpoint_num = i + 1
        target_l2 = l2_norms[i]
        incremental_l2 = a if i == 0 else delta_l2_per_sqrt_d
        print(f"\n=== Generating checkpoint {checkpoint_num}/{n} ===")
        print(f"Incremental L2/√d: {incremental_l2:.4e}")
        print(f"Cumulative L2/√d: {target_l2:.4e}")
        add_gaussian_perturbation_inplace(model, target_l2_per_sqrt_d=incremental_l2)

        save_path = f"{args.output_dir}/checkpoint-{checkpoint_num}"
        print(f"Saving to {save_path}...")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Saved checkpoint-{checkpoint_num}")
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print(f"Generated {n} checkpoints in: {args.output_dir}")
    print(f"L2/√d range: {a:.4e} → {b:.4e}")
    print(f"Increment per step: {delta_l2_per_sqrt_d:.4e}")
    print("=" * 80)


if __name__ == "__main__":
    main()
