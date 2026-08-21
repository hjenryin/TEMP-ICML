import subprocess
import os
import sys
import time
import argparse
from fnmatch import fnmatch
from datasets import load_dataset

def get_allowed_physical_gpus():
    """Return physical GPU IDs from CUDA_VISIBLE_DEVICES, or None if unset."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_visible:
        return None
    try:
        return [int(x.strip()) for x in cuda_visible.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain integer GPU IDs for this launcher. "
            f"Got: {cuda_visible!r}"
        ) from exc


def resolve_requested_gpus(requested_gpus):
    """Resolve --force_gpus against CUDA_VISIBLE_DEVICES.

    If CUDA_VISIBLE_DEVICES is set, prefer interpreting requested IDs as logical
    visible IDs. For example, with CUDA_VISIBLE_DEVICES=2,3, --force_gpus 0,1
    resolves to physical GPUs 2,3. If the requested IDs are not logical but are
    all explicitly present in CUDA_VISIBLE_DEVICES, keep them as physical IDs.
    """
    allowed = get_allowed_physical_gpus()
    if allowed is None:
        return requested_gpus

    logical_to_physical = {logical: physical for logical, physical in enumerate(allowed)}
    if all(gpu in logical_to_physical for gpu in requested_gpus):
        resolved = [logical_to_physical[gpu] for gpu in requested_gpus]
        print(
            "Resolved --force_gpus as logical visible IDs: "
            f"{requested_gpus} -> physical {resolved} "
            f"under CUDA_VISIBLE_DEVICES={allowed}"
        )
        return resolved

    if all(gpu in allowed for gpu in requested_gpus):
        print(
            "Using --force_gpus as physical IDs already present in "
            f"CUDA_VISIBLE_DEVICES={allowed}: {requested_gpus}"
        )
        return requested_gpus

    raise ValueError(
        f"--force_gpus {requested_gpus} is incompatible with CUDA_VISIBLE_DEVICES={allowed}. "
        f"Use logical IDs 0..{len(allowed) - 1} or physical IDs from {allowed}."
    )


def get_idle_gpus(util_threshold=10, mem_threshold=10):
    try:
        allowed_gpus = get_allowed_physical_gpus()

        # Check GPU status using nvidia-smi
        cmd = "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        result = subprocess.check_output(cmd.split(), encoding='utf-8')
        idle_gpus = []
        rows = result.strip().split('\n')
        print("\n[GPU Status Check]")
        for row in rows:
            if not row.strip(): continue
            vals = [int(x.strip()) for x in row.split(',')]
            idx, gpu_util, mem_used, mem_total = vals
            mem_util_percent = (mem_used / mem_total) * 100
            
            status = f"GPU {idx}: Util={gpu_util}%, Mem={mem_util_percent:.1f}%"
            
            if allowed_gpus is not None and idx not in allowed_gpus:
                print(f"{status} -> [SKIPPED (not in CUDA_VISIBLE_DEVICES)]")
                continue

            if gpu_util < util_threshold and mem_util_percent < mem_threshold:
                print(f"{status} -> [ACCEPTED]")
                idle_gpus.append(idx)
            else:
                print(f"{status} -> [SKIPPED]")
                
        return idle_gpus
    except Exception as e:
        print(f"Error querying GPUs: {e}")
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="open-thoughts/OpenThoughts-114k")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--prompt_only", action="store_true",
                        help="Only load prompts (no answers), evaluate loss on prompt tokens only")
    parser.add_argument("--ckpt_pattern", type=str, default="checkpoint-*",
                        help="Pattern to match checkpoint directories (e.g., 'checkpoint-*')")
    parser.add_argument(
        "--force_gpus",
        type=str,
        default=None,
        help=(
            "Comma-separated GPUs to use, skipping nvidia-smi idle check. "
            "With CUDA_VISIBLE_DEVICES set, values are logical visible IDs first "
            "(e.g. CUDA_VISIBLE_DEVICES=2,3 and --force_gpus 0,1 uses physical 2,3)."
        ),
    )
    args = parser.parse_args()
    # Resolve paths relative to this launcher so workers work from any cwd
    launcher_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the worker script
    # This ensures it works regardless of where you launch the command from
    worker_script_path = os.path.join(launcher_dir, "_get-tokenwise-loss_ckpt.py")
    
    if not os.path.exists(worker_script_path):
        print(f"FATAL ERROR: Could not find worker script at: {worker_script_path}")
        print("Please ensure '_get-tokenwise-loss_ckpt.py' is in the same directory as this launcher.")
        sys.exit(1)

    # Resolve model_dir to absolute path just in case
    abs_model_dir = os.path.abspath(args.model_dir)

    # --- 2. Calculate Filename ---
    if "json" in args.dataset_path:
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    else:
        dataset = load_dataset(args.dataset_path, split="train")

    
    if args.debug: 
        dataset = dataset.select(range(99))
    elif args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        
    dataset_size = len(dataset)
    if dataset_size >= 1000:
        size_suffix = f"{dataset_size//1000}k"
    else:
        size_suffix = str(dataset_size)
    
    # Handle max_tokens in filename (matches process_tokenwise_loss aggregates)
    if args.max_tokens == -1:
        token_suffix = "full"
    else:
        token_suffix = f"{args.max_tokens}token"

    prompt_marker = "-prompt" if args.prompt_only else ""
    path_l = args.dataset_path.lower()

    if "m23k" in path_l:
        prefix = "m23k-prd"
    elif "openthoughts" in path_l or "op114" in path_l:
        prefix = "op114kmath-prd"
    else:
        prefix = "tokenwise"

    if prefix == "tokenwise":
        output_filename = f"tokenwise{prompt_marker}-{token_suffix}-{size_suffix}.pt"
    else:
        output_filename = f"{prefix}-tokenwise{prompt_marker}-{token_suffix}-{size_suffix}.pt"
    
    print(f"Target Output Filename: {output_filename}")

    # --- 3. Setup Queue ---
    # Use the absolute model dir path
    if not os.path.exists(abs_model_dir):
        print(f"Model directory not found: {abs_model_dir}")
        sys.exit(1)

    # If ckpt_pattern is empty, use model_dir itself as the only model
    if args.ckpt_pattern == "":
        subdirs = [abs_model_dir]
    else:
        subdirs = sorted([f.path for f in os.scandir(abs_model_dir)
                          if f.is_dir() and fnmatch(f.name, args.ckpt_pattern)])
    task_queue = subdirs[:]
    
    if args.force_gpus is not None:
        requested_gpus = [int(x.strip()) for x in args.force_gpus.split(",") if x.strip() != ""]
        if not requested_gpus:
            print("--force_gpus was empty after parsing. Exiting.")
            sys.exit(1)
        idle_gpus = resolve_requested_gpus(requested_gpus)
        print(f"Using physical GPUs from --force_gpus (skipping idle check): {idle_gpus}")
    else:
        idle_gpus = get_idle_gpus()
        if not idle_gpus:
            print("No GPUs found. Exiting.")
            sys.exit(1)
        
    print(f"Starting execution on {len(idle_gpus)} idle GPUs: {idle_gpus}")
    
    # Track processes: { gpu_id: subprocess.Popen_object }
    gpu_processes = {gpu: None for gpu in idle_gpus}
    
    # --- 4. Main Loop ---
    while len(task_queue) > 0 or any(p is not None for p in gpu_processes.values()):
        
        for gpu in idle_gpus:
            proc = gpu_processes[gpu]

            # Check if finished
            if proc is not None:
                status = proc.poll()
                if status is not None:
                    if status != 0:
                        print(f"[Launcher] ⚠️  GPU {gpu} task FAILED with code {status}")
                    else:
                        print(f"[Launcher] ✅ GPU {gpu} task COMPLETED")
                    gpu_processes[gpu] = None
                    proc = None

            # Assign new task
            if proc is None and len(task_queue) > 0:
                model_path = task_queue.pop(0)
                
                # Check for existence before launching
                final_path = os.path.join(model_path, output_filename)
                if os.path.exists(final_path):
                    print(f"[Launcher] Skipping {os.path.basename(model_path)} (Output exists)")
                    continue

                cmd = [
                    sys.executable,  # Use the same python interpreter running this script
                    worker_script_path, # Use the ABSOLUTE path to the worker
                    "--gpu_id", "0",
                    "--model_path", model_path,
                    "--output_filename", output_filename,
                    "--dataset_path", args.dataset_path,
                    "--batch_size", str(args.batch_size),
                    "--max_tokens", str(args.max_tokens)
                ]
                
                if args.max_samples:
                    cmd.extend(["--max_samples", str(args.max_samples)])
                if args.debug:
                    cmd.append("--debug")
                if args.prompt_only:
                    cmd.append("--prompt_only")
                
                print(f"[Launcher] 🚀 Launching {os.path.basename(model_path)} on physical GPU {gpu} (as logical GPU 0)")
                
                # Start independent process
                # Set CUDA_VISIBLE_DEVICES so the worker only sees this specific GPU
                new_env = os.environ.copy()
                new_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                new_proc = subprocess.Popen(cmd, env=new_env)
                gpu_processes[gpu] = new_proc

        time.sleep(1)

    print("All jobs completed.")

if __name__ == "__main__":
    main()