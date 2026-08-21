#!/usr/bin/env python3
"""
Generate cumulative perturbed checkpoints along a signed training direction.

The perturbation follows

    theta_j = theta_{j-1} + lambda_j * (1 + xi_j) odot e

where e is the globally unit-normalized training direction and xi_j is
standard Gaussian noise sampled independently at each step.  The lambdas are
computed without sampling xi_j, using the high-dimensional concentration
approximation:

    lambda_j = (-m_{j-1} + sqrt(m_{j-1}^2 + 2(r_j^2 - r_{j-1}^2))) / 2

with m_j = sum_i lambda_i and r_j = j * R / n.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq


DEFAULT_BASE_DIR = os.environ.get("TEMP_OUTPUT_ROOT", "outputs")
DEFAULT_DATASET = os.path.join(os.environ.get("TEMP_DATA_ROOT", "data"), "m23k-prd-sharegpt", "train.jsonl")
DEFAULT_FINAL_RADIUS = 10.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate concentrated-lambda cumulative perturbation checkpoints."
    )
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--direction_start_checkpoint", type=str, required=True)
    parser.add_argument("--direction_end_checkpoint", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_checkpoints", type=int, default=8)
    parser.add_argument(
        "--final_radius",
        type=float,
        default=DEFAULT_FINAL_RADIUS,
        help="Final target L2 distance from the base model. Default: %(default)s.",
    )
    parser.add_argument(
        "--min_radius",
        type=float,
        default=None,
        help="Optional lower target L2 distance for checkpoint-1. If omitted, uses final_radius / num_checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Transformers device_map for model loading. With CUDA_VISIBLE_DEVICES=2,3, 'auto' uses GPUs 2 and 3.",
    )
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    parser.add_argument("--exclude_embeddings", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_distance", action="store_true")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--eval_max_samples", type=int, default=100)
    parser.add_argument("--eval_max_tokens", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_device_map", type=str, default=None)
    parser.add_argument(
        "--checkpoint_workers",
        type=int,
        default=None,
        help="Number of checkpoint-generation workers. Defaults to one worker per visible CUDA GPU, or 1 on CPU.",
    )
    parser.add_argument(
        "--checkpoint_devices",
        type=str,
        default=None,
        help="Comma-separated logical CUDA device ids for checkpoint workers, e.g. 0,1,2,3. Defaults to all visible GPUs.",
    )
    return parser.parse_args()


def normalize_path_arg(value):
    if value is None:
        return None
    return os.path.normpath(os.path.expanduser(value))


def normalize_args(args):
    args.base_dir = normalize_path_arg(args.base_dir)
    args.direction_start_checkpoint = normalize_path_arg(args.direction_start_checkpoint)
    args.direction_end_checkpoint = normalize_path_arg(args.direction_end_checkpoint)
    args.base_model_name = normalize_path_arg(args.base_model_name)
    args.output_dir = normalize_path_arg(args.output_dir)
    args.dataset_path = normalize_path_arg(args.dataset_path)


def torch_dtype(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def is_embedding_param(name):
    return "embed" in name.lower()


def should_perturb(name, args):
    return not (args.exclude_embeddings and is_embedding_param(name))


def load_model(path, args, device_map=None):
    print(f"Loading model: {path}")
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch_dtype(args.torch_dtype),
        device_map=args.device_map if device_map is None else device_map,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )



def checkpoint_worker_devices(args):
    if not torch.cuda.is_available():
        return [None]
    if args.checkpoint_devices:
        devices = []
        for raw_device in args.checkpoint_devices.split(","):
            raw_device = raw_device.strip()
            if not raw_device:
                continue
            if raw_device.startswith("cuda:"):
                raw_device = raw_device.split(":", 1)[1]
            devices.append(int(raw_device))
    else:
        devices = list(range(torch.cuda.device_count()))
    if not devices:
        raise ValueError("--checkpoint_devices did not contain any CUDA device ids.")
    return devices


def checkpoint_worker_count(args, devices):
    default_workers = len(devices) if devices != [None] else 1
    workers = args.checkpoint_workers if args.checkpoint_workers is not None else default_workers
    if workers < 1:
        raise ValueError("--checkpoint_workers must be at least 1.")
    return min(workers, default_workers)


def per_tensor_step_seed(seed, step_idx, name):
    digest = hashlib.blake2b(f"{seed}:{step_idx}:{name}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") & ((1 << 63) - 1)


def direction_cache_path(args):
    return Path(args.output_dir) / "training_direction.pt"


def save_direction_cache(path, direction, direction_norm):
    torch.save({"direction": direction, "direction_norm": direction_norm}, path)
    print(f"Wrote direction cache: {path}")


def worker_script_path():
    return Path(__file__).with_name("perturb_checkpoint_worker.py")


def run_checkpoint_worker(idx, save_path, device_idx, args, direction_path, lambdas, target_radius):
    worker = worker_script_path()
    device_arg = "cpu" if device_idx is None else "cuda:0"
    cmd = [
        sys.executable,
        str(worker),
        "--checkpoint_idx",
        str(idx),
        "--save_path",
        str(save_path),
        "--direction_path",
        str(direction_path),
        "--base_model_name",
        args.base_model_name,
        "--lambdas_json",
        json.dumps(lambdas[:idx]),
        "--target_radius",
        str(target_radius),
        "--seed",
        str(args.seed),
        "--torch_dtype",
        args.torch_dtype,
        "--device",
        device_arg,
        "--dataset_path",
        args.dataset_path,
        "--eval_max_samples",
        str(args.eval_max_samples),
        "--eval_max_tokens",
        str(args.eval_max_tokens),
        "--eval_batch_size",
        str(args.eval_batch_size),
    ]
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")
    if args.exclude_embeddings:
        cmd.append("--exclude_embeddings")
    if args.skip_eval:
        cmd.append("--skip_eval")
    if args.skip_distance:
        cmd.append("--skip_distance")

    env = os.environ.copy()
    if device_idx is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device_idx)
    print(f"Launching checkpoint-{idx} on {'cpu' if device_idx is None else f'cuda:{device_idx}'}")
    subprocess.run(cmd, check=True, env=env)
    return idx, save_path


def write_worker_metric_summaries(args, checkpoint_dirs):
    eval_rows = []
    distance_rows = []
    for idx, ckpt in checkpoint_dirs:
        metrics_path = Path(ckpt) / "worker_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        if "loss" in metrics:
            eval_rows.append({
                "checkpoint": idx,
                "loss": metrics["loss"],
                "tokens": metrics["tokens"],
            })
        if "actual_distance" in metrics:
            distance_rows.append({
                "checkpoint": idx,
                "actual_distance": metrics["actual_distance"],
                "target_radius": metrics["target_radius"],
                "actual_over_target": metrics["actual_over_target"],
                "l2_per_sqrt_d": metrics["l2_per_sqrt_d"],
                "num_params": metrics["num_params"],
            })

    if eval_rows and not args.skip_eval:
        print("\nPreparing base loss for worker eval summary...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        processed = build_processed_dataset(
            args.dataset_path,
            tokenizer,
            max_tokens=args.eval_max_tokens,
            max_samples=args.eval_max_samples,
        )
        base_loss, base_tokens = evaluate_model_loss(args.base_model_name, processed, tokenizer, args)
        rows = [{"checkpoint": "base", "loss": base_loss, "tokens": base_tokens, "delta": 0.0, "ratio": 1.0}]
        print(f"BASE loss: {base_loss:.6f} over {base_tokens} labeled tokens")
        for row in sorted(eval_rows, key=lambda item: item["checkpoint"]):
            row["delta"] = row["loss"] - base_loss
            row["ratio"] = row["loss"] / base_loss if base_loss > 0 else float("inf")
            rows.append(row)
            print(
                f"checkpoint-{row['checkpoint']}: loss={row['loss']:.6f}, "
                f"delta={row['delta']:.6f}, ratio={row['ratio']:.6f}"
            )
        out_path = Path(args.output_dir) / "quick_eval_losses.json"
        out_path.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"Wrote loss eval: {out_path}")
    if distance_rows and not args.skip_distance:
        out_path = Path(args.output_dir) / "distance_validation.json"
        out_path.write_text(json.dumps(distance_rows, indent=2) + "\n")
        print(f"Wrote distance validation: {out_path}")


def default_direction_paths(args):
    if args.output_dir is None:
        args.output_dir = f"{args.base_dir}/perturb_concentrated_lambda"


def compute_signed_unit_direction(args):
    """Return raw direction tensors parked on CPU; tensor math is done on GPUs in parallel."""
    start_model = load_model(args.direction_start_checkpoint, args)
    end_model = load_model(args.direction_end_checkpoint, args)

    start_state = start_model.state_dict()
    end_state = end_model.state_dict()
    cuda_devices = [torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    names_by_device = defaultdict(list)
    assign_idx = 0
    for name, start_tensor in start_state.items():
        if name not in end_state or not should_perturb(name, args):
            continue
        if start_tensor.device.type == "cuda":
            compute_device = start_tensor.device
        elif cuda_devices:
            compute_device = cuda_devices[assign_idx % len(cuda_devices)]
            assign_idx += 1
        else:
            compute_device = start_tensor.device
        names_by_device[str(compute_device)].append(name)

    def compute_device_direction(device_name, names):
        device = torch.device(device_name)
        use_stream = device.type == "cuda"
        stream = torch.cuda.Stream(device=device) if use_stream else None
        local_direction = {}
        local_sq = torch.zeros((), device=device, dtype=torch.float32)

        def run():
            nonlocal local_sq
            with torch.no_grad():
                for name in tqdm(names, desc=f"Direction {device_name}", leave=False):
                    start = start_state[name].detach().to(device=device, dtype=torch.float32, non_blocking=True)
                    end = end_state[name].detach().to(device=device, dtype=torch.float32, non_blocking=True)
                    delta = end - start
                    local_sq = local_sq + torch.sum(delta * delta)
                    local_direction[name] = delta.to(device="cpu", non_blocking=False)
                    del start, end, delta

        if use_stream:
            with torch.cuda.device(device), torch.cuda.stream(stream):
                run()
            stream.synchronize()
        else:
            run()
        return local_direction, float(local_sq.item())

    direction = {}
    total_sq = 0.0
    print("Computing signed training direction across devices...")
    with ThreadPoolExecutor(max_workers=max(1, len(names_by_device))) as executor:
        futures = [
            executor.submit(compute_device_direction, device_name, names)
            for device_name, names in names_by_device.items()
        ]
        for future in as_completed(futures):
            local_direction, local_sq = future.result()
            direction.update(local_direction)
            total_sq += local_sq

    direction_norm = math.sqrt(total_sq)
    if direction_norm == 0.0:
        raise RuntimeError("Training direction has zero norm.")

    print(f"Raw direction norm: {direction_norm:.8e}")

    del start_model, end_model, start_state, end_state
    gc.collect()
    torch.cuda.empty_cache()
    return direction, direction_norm


def compute_lambdas(num_checkpoints, final_radius, min_radius=None):
    if final_radius <= 0.0:
        raise ValueError("--final_radius must be positive.")
    if min_radius is not None:
        if min_radius < 0.0:
            raise ValueError("--min_radius must be non-negative.")
        if min_radius > final_radius:
            raise ValueError("--min_radius must be <= --final_radius.")

    lambdas = []
    radii = []
    m = 0.0
    prev_r = 0.0
    for j in range(1, num_checkpoints + 1):
        if min_radius is None:
            r = j * final_radius / num_checkpoints
        elif num_checkpoints == 1:
            r = final_radius
        else:
            r = min_radius + (j - 1) * (final_radius - min_radius) / (num_checkpoints - 1)
        lam = (-m + math.sqrt(m * m + 2.0 * (r * r - prev_r * prev_r))) / 2.0
        lambdas.append(lam)
        radii.append(r)
        m += lam
        prev_r = r
    return lambdas, radii


def add_concentrated_step(model, direction, direction_norm, lam, step_idx, args):
    params_by_device = defaultdict(list)
    for name, param in model.named_parameters():
        if name not in direction:
            continue
        if torch.cuda.is_available() and param.device.type != "cuda":
            raise RuntimeError(
                f"Perturbed parameter {name} is on {param.device}; use a GPU-only device_map so sampling "
                "and parameter updates stay on GPU."
            )
        params_by_device[str(param.device)].append((name, param))

    def perturb_device_params(device_name, items):
        device = torch.device(device_name)
        use_stream = device.type == "cuda"
        stream = torch.cuda.Stream(device=device) if use_stream else None
        local_sq = torch.zeros((), device=device, dtype=torch.float32)
        touched = 0

        def run():
            nonlocal local_sq, touched
            with torch.no_grad():
                for name, param in tqdm(items, desc=f"Adding {device_name} lambda={lam:.4e}", leave=False):
                    e = direction[name].to(device=param.device, dtype=torch.float32, non_blocking=True)
                    e.mul_(1.0 / direction_norm)
                    generator = torch.Generator(device=param.device)
                    generator.manual_seed(per_tensor_step_seed(args.seed, step_idx, name))
                    noise = torch.randn(e.shape, device=param.device, dtype=torch.float32, generator=generator)
                    step = lam * (1.0 + noise) * e
                    local_sq = local_sq + torch.sum(step * step)
                    param.add_(step.to(dtype=param.dtype))
                    touched += 1
                    del e, noise, step

        if use_stream:
            with torch.cuda.device(device), torch.cuda.stream(stream):
                run()
            stream.synchronize()
        else:
            run()
        return touched, float(local_sq.item())

    total_touched = 0
    total_sq = 0.0
    with ThreadPoolExecutor(max_workers=max(1, len(params_by_device))) as executor:
        futures = [
            executor.submit(perturb_device_params, device_name, items)
            for device_name, items in params_by_device.items()
        ]
        for future in as_completed(futures):
            touched, step_sq = future.result()
            total_touched += touched
            total_sq += step_sq
    return total_touched, math.sqrt(total_sq)


def save_run_metadata(args, direction_norm, final_radius, min_radius, lambdas, radii, out_dir):
    meta = {
        "base_model_name": args.base_model_name,
        "direction_start_checkpoint": args.direction_start_checkpoint,
        "direction_end_checkpoint": args.direction_end_checkpoint,
        "raw_direction_norm": direction_norm,
        "final_radius": final_radius,
        "min_radius": min_radius,
        "num_checkpoints": args.num_checkpoints,
        "lambdas": lambdas,
        "target_radii": radii,
        "exclude_embeddings": args.exclude_embeddings,
        "seed": args.seed,
    }
    path = Path(out_dir) / "perturbation_metadata.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote metadata: {path}")


def build_processed_dataset(dataset_path, tokenizer, max_tokens, max_samples):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def process_data(examples, indices):
        if "conversations" in examples:
            convos = examples["conversations"]
            prompt_msgs = []
            responses = []
            for chat in convos:
                msgs = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}]
                user_key = "value" if "value" in chat[0] else "content"
                assistant_key = "value" if "value" in chat[1] else "content"
                msgs.append({"role": "user", "content": chat[0][user_key]})
                prompt_msgs.append(msgs)
                responses.append(chat[1][assistant_key])
            prompts_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            full_texts = [p + r + tokenizer.eos_token for p, r in zip(prompts_text, responses)]
        elif "text" in examples:
            full_texts = examples["text"]
            prompts_text = ["" for _ in full_texts]
        elif "prompt" in examples and "reasoning" in examples:
            prompt_msgs = [
                [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
                for prompt in examples["prompt"]
            ]
            prompts_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            responses = examples["reasoning"]
            full_texts = [p + r + tokenizer.eos_token for p, r in zip(prompts_text, responses)]
        else:
            raise ValueError(f"Unsupported dataset columns: {list(examples.keys())}")

        prompt_ids = tokenizer(prompts_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_texts, add_special_tokens=False)["input_ids"]

        labels_list = []
        input_ids_list = []
        expected_lengths = []
        for full, prompt in zip(full_ids, prompt_ids):
            p_len = len(prompt)
            response_len = len(full) - p_len if p_len > 0 else len(full)
            response_len = response_len if max_tokens == -1 else min(response_len, max_tokens)
            total_len = p_len + response_len
            labels = list(full)
            if p_len > 0:
                labels[:p_len] = [-100] * p_len
            input_ids_list.append(full[:total_len])
            labels_list.append(labels[:total_len])
            expected_lengths.append(response_len)

        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "expected_length": expected_lengths,
            "index": indices,
        }

    return dataset.map(
        process_data,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        with_indices=True,
        desc="Tokenizing eval data",
    )


def evaluate_model_loss(model_path, processed_dataset, tokenizer, args):
    device_map = args.eval_device_map or args.device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype(args.torch_dtype),
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    model.eval()
    dataloader = DataLoader(
        processed_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc=f"Evaluating {Path(str(model_path)).name}", leave=False):
        batch.pop("expected_length")
        batch.pop("index")
        first_device = next(model.parameters()).device
        batch = {k: v.to(first_device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].to(logits.device).contiguous()
            labels_flat = shift_labels.reshape(-1)
            losses = torch.nn.functional.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                labels_flat,
                reduction="none",
                ignore_index=-100,
            )
            valid = labels_flat != -100
            finite = torch.isfinite(losses)
            valid = valid & finite
            total_loss += float(losses[valid].sum().item())
            total_tokens += int(valid.sum().item())
        del batch, outputs, logits, shift_labels, labels_flat, losses

    mean_loss = total_loss / max(total_tokens, 1)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return mean_loss, total_tokens


def quick_eval_losses(args, checkpoint_dirs):
    print("\nPreparing quick loss evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    processed = build_processed_dataset(
        args.dataset_path,
        tokenizer,
        max_tokens=args.eval_max_tokens,
        max_samples=args.eval_max_samples,
    )

    rows = []
    base_loss, base_tokens = evaluate_model_loss(args.base_model_name, processed, tokenizer, args)
    rows.append({"checkpoint": "base", "loss": base_loss, "tokens": base_tokens, "delta": 0.0, "ratio": 1.0})
    print(f"BASE loss: {base_loss:.6f} over {base_tokens} labeled tokens")

    for idx, ckpt in checkpoint_dirs:
        loss, tokens = evaluate_model_loss(str(ckpt), processed, tokenizer, args)
        rows.append(
            {
                "checkpoint": idx,
                "loss": loss,
                "tokens": tokens,
                "delta": loss - base_loss,
                "ratio": loss / base_loss if base_loss > 0 else float("inf"),
            }
        )
        print(f"checkpoint-{idx}: loss={loss:.6f}, delta={loss - base_loss:.6f}, ratio={rows[-1]['ratio']:.6f}")

    out_path = Path(args.output_dir) / "quick_eval_losses.json"
    out_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote loss eval: {out_path}")
    return rows


def compute_model_distance(base_model_path, checkpoint_path, args):
    if torch.cuda.is_available():
        distance_devices = [torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())]
    else:
        distance_devices = [torch.device("cpu")]

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype(args.torch_dtype),
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    ckpt = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch_dtype(args.torch_dtype),
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    base_state = base.state_dict()
    ckpt_state = ckpt.state_dict()
    names_by_device = defaultdict(list)
    assign_idx = 0
    for name in base_state.keys():
        if name not in ckpt_state or not should_perturb(name, args):
            continue
        device = distance_devices[assign_idx % len(distance_devices)]
        names_by_device[str(device)].append(name)
        assign_idx += 1

    def distance_device_worker(device_name, names):
        device = torch.device(device_name)
        use_stream = device.type == "cuda"
        stream = torch.cuda.Stream(device=device) if use_stream else None
        local_sq = torch.zeros((), device=device, dtype=torch.float32)
        local_params = 0

        def run():
            nonlocal local_sq, local_params
            with torch.no_grad():
                for name in tqdm(names, desc=f"Distance {Path(checkpoint_path).name} {device_name}", leave=False):
                    base_gpu = base_state[name].to(device=device, dtype=torch.float32, non_blocking=True)
                    ckpt_gpu = ckpt_state[name].to(device=device, dtype=torch.float32, non_blocking=True)
                    diff = ckpt_gpu - base_gpu
                    local_sq = local_sq + torch.sum(diff * diff)
                    local_params += diff.numel()
                    del base_gpu, ckpt_gpu, diff

        if use_stream:
            with torch.cuda.device(device), torch.cuda.stream(stream):
                run()
            stream.synchronize()
        else:
            run()
        return float(local_sq.item()), local_params

    total_sq = 0.0
    total_params = 0
    with ThreadPoolExecutor(max_workers=max(1, len(names_by_device))) as executor:
        futures = [
            executor.submit(distance_device_worker, device_name, names)
            for device_name, names in names_by_device.items()
        ]
        for future in as_completed(futures):
            local_sq, local_params = future.result()
            total_sq += local_sq
            total_params += local_params

    del base, ckpt, base_state, ckpt_state
    gc.collect()
    torch.cuda.empty_cache()
    return math.sqrt(total_sq), total_params


def validate_distances(args, checkpoint_dirs, target_radii):
    print("\nComputing actual distances from base model...")
    rows = []
    for (idx, ckpt), target in zip(checkpoint_dirs, target_radii):
        dist, num_params = compute_model_distance(args.base_model_name, str(ckpt), args)
        rows.append(
            {
                "checkpoint": idx,
                "actual_distance": dist,
                "target_radius": target,
                "actual_over_target": dist / target if target > 0 else float("inf"),
                "l2_per_sqrt_d": dist / math.sqrt(num_params) if num_params > 0 else 0.0,
                "num_params": num_params,
            }
        )
        print(
            f"checkpoint-{idx}: actual={dist:.8e}, target={target:.8e}, "
            f"ratio={rows[-1]['actual_over_target']:.6f}"
        )

    out_path = Path(args.output_dir) / "distance_validation.json"
    out_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote distance validation: {out_path}")
    return rows


def main():
    args = parse_args()
    default_direction_paths(args)
    normalize_args(args)
    if args.num_checkpoints < 1:
        raise ValueError("--num_checkpoints must be at least 1")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    final_radius = args.final_radius
    min_radius = args.min_radius
    lambdas, target_radii = compute_lambdas(args.num_checkpoints, final_radius, min_radius)

    requested_checkpoint_dirs = [
        (idx, Path(args.output_dir) / f"checkpoint-{idx}")
        for idx in range(1, args.num_checkpoints + 1)
    ]
    all_checkpoints_exist = all(path.exists() for _, path in requested_checkpoint_dirs)
    if args.skip_existing and all_checkpoints_exist:
        print("\nAll requested checkpoints already exist; skipping perturbation generation.")
        print("Skipping direction endpoint loads and base model load for generation.")
        print("\nConcentrated lambda schedule:")
        for idx, (lam, radius) in enumerate(zip(lambdas, target_radii), start=1):
            print(f"  checkpoint-{idx}: lambda={lam:.8e}, target_radius={radius:.8e}")

        if not args.skip_eval:
            quick_eval_losses(args, requested_checkpoint_dirs)
        if not args.skip_distance:
            validate_distances(args, requested_checkpoint_dirs, target_radii)
        print("\nDone.")
        return

    pending_checkpoints = []
    checkpoint_dirs = []
    for idx, save_path in requested_checkpoint_dirs:
        if save_path.exists() and args.skip_existing:
            print(f"Skipping existing {save_path}")
            checkpoint_dirs.append((idx, save_path))
        else:
            pending_checkpoints.append((idx, save_path))

    if pending_checkpoints:
        direction, direction_norm = compute_signed_unit_direction(args)
        direction_path = direction_cache_path(args)
        save_direction_cache(direction_path, direction, direction_norm)
        save_run_metadata(args, direction_norm, final_radius, min_radius, lambdas, target_radii, args.output_dir)
        del direction
        gc.collect()
        torch.cuda.empty_cache()

        print("\nConcentrated lambda schedule:")
        for idx, (lam, radius) in enumerate(zip(lambdas, target_radii), start=1):
            print(f"  checkpoint-{idx}: lambda={lam:.8e}, target_radius={radius:.8e}")

        devices = checkpoint_worker_devices(args)
        workers = checkpoint_worker_count(args, devices)
        device_labels = ["cpu" if device is None else f"cuda:{device}" for device in devices[:workers]]
        print(
            f"\nRunning {len(pending_checkpoints)} checkpoint worker subprocess(es) with {workers} slot(s) "
            f"on {', '.join(device_labels)}"
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for work_idx, (idx, save_path) in enumerate(pending_checkpoints):
                device_idx = devices[work_idx % workers]
                futures.append(
                    executor.submit(
                        run_checkpoint_worker,
                        idx,
                        save_path,
                        device_idx,
                        args,
                        direction_path,
                        lambdas,
                        target_radii[idx - 1],
                    )
                )
            for future in as_completed(futures):
                idx, save_path = future.result()
                print(f"Finished checkpoint-{idx}: {save_path}")
                checkpoint_dirs.append((idx, save_path))
    else:
        print("\nNo checkpoint workers needed.")

    checkpoint_dirs.sort(key=lambda item: item[0])
    write_worker_metric_summaries(args, checkpoint_dirs)
    gc.collect()
    torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
