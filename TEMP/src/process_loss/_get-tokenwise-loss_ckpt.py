import os
import argparse
import torch
import warnings
import gc
import sys
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from tqdm import tqdm
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

# --- 1. Apply Patch Globally (Safe because this process is isolated) ---
apply_liger_kernel_to_qwen2(
    rope=True,
    swiglu=True,
    cross_entropy=False,
    fused_linear_cross_entropy=True,
    rms_norm=True
)

warnings.filterwarnings("ignore")

def process_checkpoint(model_path, dataset, tokenizer, output_filename, device_id, batch_size):
    """
    Worker function to handle a single checkpoint on a specific GPU.
    """
    import time
    device = f"cuda:{device_id}"
    output_file = os.path.join(model_path, output_filename)

    # Double check existence (Launcher checks too, but safety first)
    if os.path.exists(output_file):
        return

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            use_cache=False,
            device_map=device 
        )
        model.eval()

        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=DataCollatorForSeq2Seq(tokenizer, padding=True),
            pin_memory=True
        )

        results = []

        # Progress bar only if not in debug mode to avoid log spam
        iterator = tqdm(dataloader, desc=f"GPU {device_id} | {os.path.basename(model_path)}", position=device_id, leave=False)
        
        for batch in iterator:
            # Extract metadata (stay on CPU)
            expected_lens = batch.pop("expected_length")
            indices = batch.pop("index")
            
            # Move input tensors to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = model.model(**batch)
                hidden_states = outputs.last_hidden_state

                # Shift states and labels for Causal LM training
                shift_hidden_states = hidden_states[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()

                # Flatten for the kernel
                shift_hidden_states_flat = shift_hidden_states.view(-1, shift_hidden_states.size(-1))
                shift_labels_flat = shift_labels.view(-1)

                # Calculate Loss using Liger's Fused Kernel
                token_losses = liger_fused_linear_cross_entropy(
                    shift_hidden_states_flat, 
                    model.lm_head.weight, 
                    shift_labels_flat, 
                    reduction='none' 
                )

                # Reshape back to (Batch, Seq_Len)
                token_losses = token_losses.view(shift_labels.shape)
                
                # --- STEP 1: Extract valid losses for all samples ---
                batch_valid_losses = []
                for i in range(len(indices)):
                    valid_mask = (shift_labels[i] != -100)
                    
                    if valid_mask.sum() > 0:
                        valid_losses = token_losses[i][valid_mask].cpu()
                    else:
                        valid_losses = torch.tensor([], dtype=torch.float)
                    
                    batch_valid_losses.append(valid_losses)
                
                # --- STEP 2: Perform assertions ---
                for i, valid_losses in enumerate(batch_valid_losses):
                    expected_len = expected_lens[i].item()
                    if len(valid_losses) != expected_len:
                        raise ValueError(
                            f"FATAL: Token count mismatch for sample {indices[i].item()}! "
                            f"Expected {expected_len}, got {len(valid_losses)}."
                        )
                
                # --- STEP 3: Store results ---
                for i, valid_losses in enumerate(batch_valid_losses):
                    results.append({
                        'index': indices[i].item(),
                        'losses': valid_losses
                    })

            # Clean up batch-level tensors
            del outputs, hidden_states, shift_hidden_states, shift_labels, token_losses, batch
        
        # --- ASSERTION 3: Total Dataset Size Verification ---
        if len(results) != len(dataset):
            raise ValueError(f"FATAL: Processed {len(results)} samples, expected {len(dataset)}!")
        
        # Sort results by index to restore original order
        results.sort(key=lambda x: x['index'])
        final_results = [r['losses'] for r in results]
        
        # --- ASSERTION 4: Final Token Count Verification (Pre-Save) ---
        print(f"[GPU {device_id}] Performing final token count verification...")
        for idx, (losses, expected_len) in enumerate(zip(final_results, dataset['expected_length'])):
            assert len(losses) == expected_len, (
                f"FATAL: Final verification failed at index {idx}! "
                f"Expected {expected_len} tokens, got {len(losses)} tokens."
            )
        
        print(f"All assertions passed for {os.path.basename(model_path)} on GPU {device_id}")
        torch.save(final_results, output_file)

    except Exception as e:
        print(f"[Error] Failed processing {model_path} on GPU {device_id}: {e}")
        # Exit with error code so launcher knows
        sys.exit(1)
    finally:
        if 'model' in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--prompt_only", action="store_true",
                        help="Only load prompts (no answers), evaluate loss on prompt tokens only")
    args = parser.parse_args()

    print(f"[Worker] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    if "json" in args.dataset_path:
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    else:
        dataset = load_dataset(args.dataset_path, split="train")
        
    if args.debug: 
        dataset = dataset.select(range(99))
    elif args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    def process_data(examples, indices):
        """Process data with index tracking for order preservation"""
        if "conversations" in examples:
            convos = examples['conversations']
            
            prompt_msgs = []
            for chat in convos:
                assert len(chat) == 2, "Each conversation must have exactly 2 messages (user and assistant)."
                msgs = []
                # Always use Qwen's default system prompt
                msgs.append({"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."})
                if "value" in chat[0]:
                    msgs.append({"role": "user", "content": chat[0]['value']})
                    label_content="value"
                elif "content" in chat[0]:
                    msgs.append({"role": "user", "content": chat[0]['content']})
                    label_content="content"
                else:
                    raise ValueError("User message must contain either 'value' or 'content' field.")
                prompt_msgs.append(msgs)
            if label_content == "content":
                responses = [chat[1]['content'] for chat in convos]
            else:
                responses = [chat[1]['value'] for chat in convos]
        elif "prompt" in examples and "completion" in examples:
            prompt_msgs = examples['prompt']
            responses = [comp[0]['content'] for comp in examples['completion']]
        else:
            raise ValueError("Dataset must contain either 'conversations' with 'system' or 'prompt' with 'completion' fields.")

        prompts_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        
        # In prompt_only mode, only use the prompt (no response)
        if args.prompt_only:
            full_texts = prompts_text
        else:
            full_texts = [p + r + tokenizer.eos_token for p, r in zip(prompts_text, responses)]

        prompt_ids = tokenizer(prompts_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_texts, add_special_tokens=False)["input_ids"]
        
        labels_list = []
        truncate_full_ids = []
        expected_lengths = []
        
        for full, prompt in zip(full_ids, prompt_ids):
            p_len = len(prompt)
            
            if args.prompt_only:
                # In prompt_only mode, no masking of prompt, only evaluate prompt tokens
                # Truncate prompt if needed
                if args.max_tokens == -1:
                    total_len = p_len
                else:
                    total_len = min(p_len, args.max_tokens)
                
                lab = list(full[:total_len])
                truncate_full_ids.append(full[:total_len])
                labels_list.append(lab)
                # Account for the shift operation in causal LM: we lose 1 token
                expected_lengths.append(total_len - 1)
            else:
                # Original mode: mask prompt, evaluate on response
                # Calculate response length (excluding prompt)
                response_len = len(full) - p_len
                
                # Truncate response to max_tokens (unless max_tokens == -1, then use full response)
                if args.max_tokens == -1:
                    truncated_response_len = response_len
                else:
                    truncated_response_len = min(response_len, args.max_tokens)
                total_len = p_len + truncated_response_len

                lab = list(full)
                if p_len < len(lab):
                    lab[:p_len] = [-100] * p_len
                
                # Truncate to include prompt + max_tokens of response
                truncated_lab = lab[:total_len]
                labels_list.append(truncated_lab)
                truncate_full_ids.append(full[:total_len])
                
                # Expected valid tokens = truncated response length
                expected_lengths.append(truncated_response_len)

        return {
            "input_ids": truncate_full_ids, 
            "labels": labels_list, 
            "expected_length": expected_lengths,
            "index": indices  # Add dataset indices for order preservation
        }

    print("[Worker] Tokenizing dataset...")
    # NOTE: load_from_cache_file=True is default in map, so this is efficient across processes
    processed_dataset = dataset.map(
        process_data, 
        batched=True, 
        batch_size=1000,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        with_indices=True 
    )

    process_checkpoint(
        args.model_path, 
        processed_dataset, 
        tokenizer, 
        args.output_filename, 
        args.gpu_id, 
        args.batch_size
    )

if __name__ == "__main__":
    main()