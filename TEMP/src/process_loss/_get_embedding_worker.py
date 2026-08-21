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

warnings.filterwarnings("ignore")

def process_embeddings(model, dataset, tokenizer, device_id, batch_size, output_file):
    """
    Extract hidden state embeddings from the model for a portion of the dataset.
    Returns mean-pooled last layer hidden states (excluding padding tokens).
    """
    device = f"cuda:{device_id}"
    
    model.eval()
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=DataCollatorForSeq2Seq(tokenizer, padding=True),
        pin_memory=True
    )
    
    results = []
    
    iterator = tqdm(dataloader, desc=f"GPU {device_id} | Extracting Embeddings", position=device_id)
    
    for batch in iterator:
        # Extract indices (stay on CPU)
        indices = batch.pop("index")
        
        # Move input tensors to GPU
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with torch.no_grad():
            # Get model outputs with hidden states
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract last layer hidden states
            hidden_states = outputs.hidden_states  # Tuple of all layers
            last_layer_hidden_states = hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
            
            # Process each sample in the batch
            for i in range(len(indices)):
                # Get the attention mask for this sample (1 for real tokens, 0 for padding)
                sample_attention_mask = attention_mask[i]  # (seq_len,)
                sample_hidden_states = last_layer_hidden_states[i]  # (seq_len, hidden_dim)
                
                # Mask out padding tokens
                valid_token_mask = sample_attention_mask.bool()
                valid_hidden_states = sample_hidden_states[valid_token_mask]  # (num_valid_tokens, hidden_dim)
                
                # Compute mean across valid tokens
                if valid_hidden_states.size(0) > 0:
                    mean_embedding = torch.mean(valid_hidden_states, dim=0)  # (hidden_dim,)
                else:
                    # Edge case: if somehow all tokens are padding (shouldn't happen)
                    mean_embedding = torch.zeros(sample_hidden_states.size(-1), device=device)
                
                results.append({
                    'index': indices[i].item(),
                    'embedding': mean_embedding.cpu()  # Move to CPU for storage
                })
        
        # Clean up batch-level tensors
        del outputs, hidden_states, last_layer_hidden_states, input_ids, attention_mask, batch
        torch.cuda.empty_cache()
    
    # Sort by index to preserve original order
    results.sort(key=lambda x: x['index'])
    final_embeddings = [r['embedding'] for r in results]
    
    # Save partial results
    torch.save(final_embeddings, output_file)
    print(f"[GPU {device_id}] Saved {len(final_embeddings)} embeddings to {output_file}")
    
    return len(final_embeddings)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    parser.add_argument("--process_id", type=int, required=True)
    parser.add_argument("--total_processes", type=int, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_length", type=int, default=16384,
                        help="Maximum sequence length for tokenization")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    output_file = os.path.join(args.model_path, args.output_filename)

    try:
        print(f"[Worker {args.process_id}] Loading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        if tokenizer.pad_token is None: 
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        print(f"[Worker {args.process_id}] Loading Dataset...")
        if "json" in args.dataset_path:
            dataset = load_dataset("json", data_files=args.dataset_path, split="train")
        else:
            dataset = load_dataset(args.dataset_path, split="train")
            
        if args.debug: 
            dataset = dataset.select(range(99))
        elif args.max_samples is not None:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        # Select the portion for this worker
        dataset_portion = dataset.select(range(args.start_idx, args.end_idx))
        print(f"[Worker {args.process_id}] Processing samples {args.start_idx} to {args.end_idx-1} ({len(dataset_portion)} samples)")

        # Process data for tokenization
        def process_data(examples, indices):
            """Process data with index tracking for order preservation"""
            # Adjust indices to be global (relative to full dataset)
            global_indices = [idx + args.start_idx for idx in indices]
            
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
                        label_content = "value"
                    elif "content" in chat[0]:
                        msgs.append({"role": "user", "content": chat[0]['content']})
                        label_content = "content"
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
            full_texts = [p + r + tokenizer.eos_token for p, r in zip(prompts_text, responses)]

            # Tokenize full texts with max_length truncation
            full_ids = tokenizer(
                full_texts, 
                add_special_tokens=False,
                max_length=args.max_length,
                truncation=True
            )["input_ids"]
            
            return {
                "input_ids": full_ids,
                "index": global_indices  # Use global indices for proper ordering
            }

        print(f"[Worker {args.process_id}] Tokenizing dataset portion...")
        processed_dataset = dataset_portion.map(
            process_data, 
            batched=True, 
            batch_size=1000,
            remove_columns=dataset_portion.column_names,
            desc=f"Tokenizing Worker {args.process_id}",
            with_indices=True 
        )

        print(f"[Worker {args.process_id}] Loading Model from {args.model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            use_cache=False,
            device_map=device 
        )

        print(f"[Worker {args.process_id}] Extracting embeddings...")
        num_processed = process_embeddings(
            model,
            processed_dataset, 
            tokenizer, 
            args.gpu_id, 
            args.batch_size,
            output_file
        )

        # Verify count
        expected_count = args.end_idx - args.start_idx
        if num_processed != expected_count:
            raise ValueError(
                f"Worker {args.process_id} expected to process {expected_count} samples, "
                f"but processed {num_processed} samples!"
            )

        print(f"[Worker {args.process_id}] ✅ Successfully completed!")

    except Exception as e:
        print(f"[Worker {args.process_id}] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'model' in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
