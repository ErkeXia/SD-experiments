import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

# Set Seed
set_seed(42)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Create an 8-bit quantization configuration
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# --- Load the tokenizer as before ---
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")

# --- Load the big model with the new config ---
big_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b-it",
    device_map="auto",
    attn_implementation="flash_attention_2",
    quantization_config=quantization_config
)

# --- Load the small model as before ---
small_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)

# Set padding token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def normal_inference(model, tokenizer, prompt, max_new_tokens=50):
    """Standard autoregressive generation."""
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    
    num_prompt_tokens = inputs.input_ids.shape[1]
    num_generated_tokens = outputs.shape[1] - num_prompt_tokens
    
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output, num_generated_tokens


def speculative_decoding_loop(small_model, big_model, tokenizer, prompt, max_new_tokens=50, gamma=5):
    """Correctly implements the speculative decoding loop with detailed timing."""
    prompt_inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = prompt_inputs.input_ids
    num_prompt_tokens = input_ids.shape[1]
    
    total_accepted_tokens = 0
    num_draft_cycles = 0

    # NEW: Add detailed timers for bottleneck analysis
    total_draft_time = 0
    total_verify_time = 0
    
    n_generated = 0
    print("\n--- Starting Speculative Generation ---")
    while n_generated < max_new_tokens:
        num_draft_cycles += 1
        
        # 1. DRAFT - Time this section
        torch.cuda.synchronize() # Ensures previous GPU operations are complete for accurate timing
        draft_start_time = time.time()
        
        draft_outputs = small_model.generate(
            input_ids, max_new_tokens=gamma, return_dict_in_generate=True,
            output_scores=True, pad_token_id=tokenizer.pad_token_id
        )
        draft_ids = draft_outputs.sequences[:, input_ids.shape[-1]:]
        
        torch.cuda.synchronize() # Wait for the generate call to finish
        draft_time = time.time() - draft_start_time
        total_draft_time += draft_time
        
        # 2. VERIFY - Time this section
        torch.cuda.synchronize()
        verify_start_time = time.time()
        
        with torch.no_grad():
            combined_ids = torch.cat([input_ids, draft_ids], dim=-1)
            big_model_outputs = big_model(combined_ids)
            big_model_logits = big_model_outputs.logits[:, input_ids.shape[-1]-1:-1, :]

        torch.cuda.synchronize()
        verify_time = time.time() - verify_start_time
        total_verify_time += verify_time

        # 3. ACCEPT/REJECT LOOP
        accepted_len_this_cycle = 0
        all_accepted = True
        for i in range(draft_ids.shape[-1]):
            big_model_pred_id = torch.argmax(big_model_logits[:, i, :], dim=-1)
            draft_token_id = draft_ids[:, i]

            if big_model_pred_id != draft_token_id:
                total_accepted_tokens += i
                accepted_len_this_cycle = i
                
                accepted_ids = draft_ids[:, :i]
                corrected_id = big_model_pred_id.unsqueeze(0)
                input_ids = torch.cat([input_ids, accepted_ids, corrected_id], dim=-1)
                n_generated += i + 1
                all_accepted = False
                break
        
        if all_accepted:
            # NEW: The whole draft was accepted.
            total_accepted_tokens += draft_ids.shape[-1]
            accepted_len_this_cycle = draft_ids.shape[-1]
            
            # OPTIMIZATION: No second call to big_model.
            # We already have the logits from the first verification pass.
            # The last logit in that sequence predicts the token AFTER the draft.
            next_token_logits = big_model_outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            input_ids = torch.cat([input_ids, draft_ids, next_token], dim=-1)
            n_generated += gamma + 1

        # NEW: Print a detailed breakdown of each generation cycle
        print(f"  Cycle {num_draft_cycles:2d}: Draft Time={draft_time:.4f}s, Verify Time={verify_time:.4f}s, Accepted={accepted_len_this_cycle}/{gamma}")

        if input_ids[0, -1] == tokenizer.eos_token_id:
            break
            
    num_generated_tokens = input_ids.shape[1] - num_prompt_tokens
    decoded_output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    avg_accepted_len = total_accepted_tokens / num_draft_cycles if num_draft_cycles > 0 else 0
    
    # NEW: Print a final summary of where time was spent
    print("--- Finished Speculative Generation ---")
    print(f"Total time spent drafting: {total_draft_time:.4f}s")
    print(f"Total time spent verifying: {total_verify_time:.4f}s")
    
    return decoded_output, num_generated_tokens, avg_accepted_len


def measure_latency(small_model, big_model, tokenizer, prompt, max_new_tokens=50):
    # This function is largely the same, but the speculative_decoding_loop now prints its own details
    start_time = time.time(); normal_output, normal_tokens = normal_inference(big_model, tokenizer, prompt, max_new_tokens); normal_inference_latency = time.time() - start_time
    start_time = time.time(); small_output, small_tokens = normal_inference(small_model, tokenizer, prompt, max_new_tokens); small_inference_latency = time.time() - start_time
    normal_tps = normal_tokens / normal_inference_latency if normal_inference_latency > 0 else 0
    small_tps = small_tokens / small_inference_latency if small_inference_latency > 0 else 0
    
    print(f"Normal Inference Throughput: {normal_tps:.2f} tokens/sec\n" + "-"*20)
    print(f"Small Inference Throughput: {small_tps:.2f} tokens/sec\n" + "-"*20)

    start_time = time.time()
    speculative_output, speculative_tokens, avg_accepted_len = speculative_decoding_loop(
        small_model, big_model, tokenizer, prompt, max_new_tokens
    )
    speculative_decoding_latency = time.time() - start_time
    speculative_tps = speculative_tokens / speculative_decoding_latency if speculative_decoding_latency > 0 else 0
    
    print(f"\nSpeculative Decoding Summary:")
    print(f"  Latency: {speculative_decoding_latency:.4f} seconds")
    print(f"  Tokens: {speculative_tokens}")
    print(f"  Throughput: {speculative_tps:.2f} tokens/sec")
    print(f"  Avg. Accepted Tokens per Cycle: {avg_accepted_len:.2f}")
    print("="*40 + "\n")

    return small_tps, normal_tps, speculative_tps


# Main execution loop
# (No changes needed in the main loop, it will just display the new detailed output)
prompts = [
    "The future of artificial intelligence is ",
    "Machine learning is transforming the world by ",
]
max_new_tokens = 100
total_normal_tps, total_small_tps, total_speculative_tps = 0, 0, 0

for prompt in prompts:
    print(f"Processing prompt: '{prompt}'")
    small_tps, normal_tps, speculative_tps = measure_latency(
        small_model, big_model, tokenizer, prompt, max_new_tokens
    )
    total_normal_tps += normal_tps
    total_small_tps += small_tps
    total_speculative_tps += speculative_tps

# Calculate averages
average_normal_tps = total_normal_tps / len(prompts)
average_small_tps = total_small_tps / len(prompts)
average_speculative_tps = total_speculative_tps / len(prompts)

print(f"\n--- Final Averages ---")
print(f"Average Normal Inference Throughput: {average_normal_tps:.2f} tokens/sec")
print(f"Average Small Inference Throughput: {average_small_tps:.2f} tokens/sec")
print(f"Average Speculative Decoding Throughput: {average_speculative_tps:.2f} tokens/sec")

speedup = average_speculative_tps / average_normal_tps if average_normal_tps > 0 else 0
print(f"\nAchieved Throughput Speedup: {speedup:.2f}x")