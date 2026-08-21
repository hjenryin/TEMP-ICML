#!/usr/bin/env python3
"""Generate, optionally evaluate, and distance-check one perturbed checkpoint on one device."""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import perturb_model_directional as perturb


def parse_args():
    parser = argparse.ArgumentParser(description="Perturb one checkpoint on one GPU/CPU worker process.")
    parser.add_argument("--checkpoint_idx", type=int, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--direction_path", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, required=True)
    parser.add_argument("--lambdas_json", type=str, required=True)
    parser.add_argument("--target_radius", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    parser.add_argument("--exclude_embeddings", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_distance", action="store_true")
    parser.add_argument("--dataset_path", type=str, default=perturb.DEFAULT_DATASET)
    parser.add_argument("--eval_max_samples", type=int, default=100)
    parser.add_argument("--eval_max_tokens", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    return parser.parse_args()


def load_direction(path):
    payload = torch.load(path, map_location="cpu")
    return payload["direction"], float(payload["direction_norm"])


def load_worker_model(path, args):
    print(f"Loading model on {args.device}: {path}")
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=perturb.torch_dtype(args.torch_dtype),
        device_map=args.device,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )


def evaluate_checkpoint_loss(checkpoint_path, args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    processed = perturb.build_processed_dataset(
        args.dataset_path,
        tokenizer,
        max_tokens=args.eval_max_tokens,
        max_samples=args.eval_max_samples,
    )
    eval_args = argparse.Namespace(**vars(args))
    eval_args.device_map = args.device
    eval_args.eval_device_map = args.device
    return perturb.evaluate_model_loss(str(checkpoint_path), processed, tokenizer, eval_args)


def distance_checkpoint(checkpoint_path, args):
    distance_args = argparse.Namespace(**vars(args))
    distance_args.device_map = args.device
    return perturb.compute_model_distance(args.base_model_name, str(checkpoint_path), distance_args)


def main():
    args = parse_args()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))

    lambdas = json.loads(args.lambdas_json)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    direction, direction_norm = load_direction(args.direction_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=args.trust_remote_code)
    model = load_worker_model(args.base_model_name, args)

    step_norms = []
    total_touched = 0
    print(
        f"\n=== Worker checkpoint-{args.checkpoint_idx} on {args.device}: "
        f"{len(lambdas)} cumulative step(s) ==="
    )
    for step_idx, lam in enumerate(lambdas, start=1):
        touched, step_norm = perturb.add_concentrated_step(model, direction, direction_norm, lam, step_idx, args)
        total_touched = max(total_touched, touched)
        step_norms.append(step_norm)
        print(
            f"checkpoint-{args.checkpoint_idx}: applied step {step_idx}/{len(lambdas)} "
            f"to {touched} tensors; sampled step norm={step_norm:.8e}"
        )

    print(f"checkpoint-{args.checkpoint_idx}: saving to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    del model, tokenizer, direction
    gc.collect()
    torch.cuda.empty_cache()

    metrics = {
        "checkpoint": args.checkpoint_idx,
        "target_radius": args.target_radius,
        "touched_tensors": total_touched,
        "step_norms": step_norms,
    }

    if not args.skip_eval:
        loss, tokens = evaluate_checkpoint_loss(save_path, args)
        metrics.update({"loss": loss, "tokens": tokens})
        print(f"checkpoint-{args.checkpoint_idx}: loss={loss:.6f} over {tokens} labeled tokens")

    if not args.skip_distance:
        dist, num_params = distance_checkpoint(save_path, args)
        metrics.update({
            "actual_distance": dist,
            "actual_over_target": dist / args.target_radius if args.target_radius > 0 else float("inf"),
            "l2_per_sqrt_d": dist / math.sqrt(num_params) if num_params > 0 else 0.0,
            "num_params": num_params,
        })
        print(
            f"checkpoint-{args.checkpoint_idx}: actual={dist:.8e}, "
            f"target={args.target_radius:.8e}, ratio={metrics['actual_over_target']:.6f}"
        )

    metrics_path = save_path / "worker_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"checkpoint-{args.checkpoint_idx}: wrote metrics {metrics_path}")


if __name__ == "__main__":
    main()
