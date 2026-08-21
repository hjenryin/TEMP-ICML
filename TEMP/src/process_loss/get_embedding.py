import subprocess
import os
import sys
import time
import argparse
from datasets import load_dataset

def get_idle_gpus(util_threshold=10, mem_threshold=10):
    try:
        # Check GPU status using nvidia-smi
        cmd = "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        result = subprocess.check_output(cmd.split(), encoding='utf-8')
        idle_gpus = []
        rows = result.strip().split('\n')
        print("\n[GPU Status Check]")
        for row in rows:
            vals = [int(x.strip()) for x in row.split(',')]
            idx, gpu_util, mem_used, mem_total = vals
            mem_util_percent = (mem_used / mem_total) * 100
            
            status = f"GPU {idx}: Util={gpu_util}%, Mem={mem_util_percent:.1f}%"
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
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset (HF dataset name or local JSON)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_name", type=str, default="embedding.pt",
                        help="Name of the final output file")
    parser.add_argument("--max_length", type=int, default=16384,
                        help="Maximum sequence length for tokenization (default: 16384)")
    args = parser.parse_args()
    
    # --- 1. RESOLVE ABSOLUTE PATHS ---
    launcher_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script_path = os.path.join(launcher_dir, "_get_embedding_worker.py")
    
    if not os.path.exists(worker_script_path):
        print(f"FATAL ERROR: Could not find worker script at: {worker_script_path}")
        print("Please ensure '_get_embedding_worker.py' is in the same directory as this launcher.")
        sys.exit(1)

    abs_model_path = os.path.abspath(args.model_path)
    if not os.path.exists(abs_model_path):
        print(f"Model path not found: {abs_model_path}")
        sys.exit(1)

    # --- 2. Load Dataset to Calculate Size ---
    print(f"[Launcher] Loading dataset to determine size...")
    if "json" in args.dataset_path:
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    else:
        dataset = load_dataset(args.dataset_path, split="train")
    
    if args.debug: 
        dataset = dataset.select(range(99))
    elif args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    dataset_size = len(dataset)
    print(f"[Launcher] Dataset size: {dataset_size} samples")

    # --- 3. Get Idle GPUs ---
    idle_gpus = get_idle_gpus()
    if not idle_gpus:
        print("No idle GPUs found. Exiting.")
        sys.exit(1)
    
    num_processes = len(idle_gpus)
    print(f"\n[Launcher] Starting execution on {num_processes} idle GPUs: {idle_gpus}")

    # --- 4. Calculate Dataset Splits ---
    # Each GPU processes a contiguous portion of the dataset
    samples_per_gpu = dataset_size // num_processes
    remainder = dataset_size % num_processes
    
    splits = []
    start_idx = 0
    for i in range(num_processes):
        # Distribute remainder samples to first few GPUs
        end_idx = start_idx + samples_per_gpu + (1 if i < remainder else 0)
        splits.append((start_idx, end_idx))
        print(f"[Launcher] GPU {idle_gpus[i]}: samples {start_idx} to {end_idx-1} ({end_idx - start_idx} samples)")
        start_idx = end_idx

    # --- 5. Launch Workers ---
    processes = []
    for gpu_idx, (start, end) in zip(idle_gpus, splits):
        output_filename = f"embedding-{gpu_idx+1}-of-{num_processes}.pt"
        
        cmd = [
            sys.executable,
            worker_script_path,
            "--gpu_id", str(gpu_idx),
            "--model_path", abs_model_path,
            "--output_filename", output_filename,
            "--dataset_path", args.dataset_path,
            "--batch_size", str(args.batch_size),
            "--start_idx", str(start),
            "--end_idx", str(end),
            "--process_id", str(gpu_idx + 1),
            "--total_processes", str(num_processes),
            "--max_length", str(args.max_length)
        ]
        
        if args.max_samples:
            cmd.extend(["--max_samples", str(args.max_samples)])
        if args.debug:
            cmd.append("--debug")
        
        print(f"[Launcher] 🚀 Launching worker {gpu_idx+1}/{num_processes} on GPU {gpu_idx}")
        proc = subprocess.Popen(cmd)
        processes.append((gpu_idx, proc, output_filename))

    # --- 6. Monitor Processes ---
    print("\n[Launcher] Waiting for all workers to complete...")
    all_success = True
    for gpu_idx, proc, output_file in processes:
        proc.wait()
        if proc.returncode != 0:
            print(f"[Launcher] ⚠️  GPU {gpu_idx} worker FAILED with code {proc.returncode}")
            all_success = False
        else:
            print(f"[Launcher] ✅ GPU {gpu_idx} worker COMPLETED")
    
    if not all_success:
        print("\n[Launcher] ❌ Some workers failed. Aborting concatenation.")
        sys.exit(1)

    # --- 7. Concatenate Results ---
    print("\n[Launcher] Concatenating embeddings...")
    import torch
    
    all_embeddings = []
    for i in range(num_processes):
        partial_file = os.path.join(abs_model_path, f"embedding-{i+1}-of-{num_processes}.pt")
        if not os.path.exists(partial_file):
            print(f"[Launcher] ⚠️  Missing partial file: {partial_file}")
            sys.exit(1)
        
        partial_embeddings = torch.load(partial_file, map_location='cpu')
        all_embeddings.extend(partial_embeddings)
        print(f"[Launcher] Loaded {len(partial_embeddings)} embeddings from worker {i+1}")
    
    # Verify total count
    if len(all_embeddings) != dataset_size:
        print(f"[Launcher] ⚠️  Expected {dataset_size} embeddings, got {len(all_embeddings)}")
        sys.exit(1)
    
    # Stack into tensor
    final_embeddings = torch.stack(all_embeddings)
    print(f"[Launcher] Final embedding shape: {final_embeddings.shape}")
    
    # Save final result
    final_output_path = os.path.join(abs_model_path, args.output_name)
    torch.save(final_embeddings, final_output_path)
    print(f"[Launcher] ✅ Saved final embeddings to: {final_output_path}")
    
    # Clean up partial files
    print("\n[Launcher] Cleaning up partial files...")
    for i in range(num_processes):
        partial_file = os.path.join(abs_model_path, f"embedding-{i+1}-of-{num_processes}.pt")
        if os.path.exists(partial_file):
            os.remove(partial_file)
            print(f"[Launcher] Removed {partial_file}")
    
    print("\n[Launcher] 🎉 All done!")

if __name__ == "__main__":
    main()
