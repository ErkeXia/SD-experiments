import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import warnings

# --- fast matmul on Ampere ---
torch.backends.cuda.matmul.allow_tf32 = True

# Suppress warnings
warnings.filterwarnings("ignore")

# Seed
set_seed(42)

def get_device():
    """Checks for CUDA and returns the device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device != "cuda":
        print("WARNING: CUDA not available. This will be extremely slow.")
    return device

def load_tokenizer(tokenizer_name="gpt2"):
    """Loads the tokenizer for a CausalLM."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # For CausalLM, padding on the left is standard for batch generation
        tokenizer.padding_side = "left" 
    return tokenizer

def load_big_model(device):
    """Loads the large target model (GPT2-Medium or fallback GPT2) in bfloat16."""
    
    model_dtype = torch.bfloat16
    
    try:
        big_model_name = "gpt2-medium" # 355M params
        print(f"Attempting to load {big_model_name} in {model_dtype}...")
        big_model = AutoModelForCausalLM.from_pretrained(
            big_model_name,
            device_map="auto",
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
        print("\n" + "="*30)
        print(f"Successfully loaded big model: {big_model_name}")
        print("="*30 + "\n")
    except Exception as e:
        print(f"Error loading {big_model_name}: {e}")
        print(f"Falling back to gpt2 (124M) in {model_dtype}...")
        big_model_name = "gpt2" # 124M params
        big_model = AutoModelForCausalLM.from_pretrained(
            big_model_name,
            device_map="auto",
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
        print("\n" + "="*30)
        print(f"Successfully loaded big model: {big_model_name}")
        print("="*30 + "\n")
    return big_model, big_model_name

def load_small_model(model_name, device):
    """Loads a small approximation model (DistilGPT2) in bf16."""
    print(f"Loading small model: {model_name}...")
    small_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print(f"Successfully loaded {model_name}\n")
    return small_model

def normal_inference(model, tokenizer, prompt, max_new_tokens=50):
    """Runs standard autoregressive inference for a CausalLM."""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_ids_len = inputs.input_ids.shape[1]
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False  # Disabling cache for fair TPS comparison
        )
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    latency = time.time() - start_time
    
    num_generated_tokens = outputs.shape[1] - input_ids_len
    
    # Decode only the newly generated tokens
    decoded_output = tokenizer.decode(outputs[0, input_ids_len:], skip_special_tokens=True)
    tps = num_generated_tokens / latency if latency > 0 else 0
    return decoded_output, num_generated_tokens, latency, tps

def speculative_decoding_loop(small_model, big_model, tokenizer, prompt, max_new_tokens=50, gamma=5):
    """
    Runs greedy speculative decoding for a CausalLM (GPT-style).
    """
    device = big_model.device
    
    # --- Prompt Encoding ---
    prompt_inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = prompt_inputs.input_ids # This is our main sequence, it will grow
    prompt_len = input_ids.shape[1]

    total_accepted_tokens = 0
    num_draft_cycles = 0
    total_draft_time = 0
    total_verify_time = 0
    n_generated = 0
    
    eos_id = tokenizer.eos_token_id
    
    overall_start_time = time.time()

    while n_generated < max_new_tokens:
        num_draft_cycles += 1

        # --- 1. Drafting (Small Model) ---
        if torch.cuda.is_available(): torch.cuda.synchronize()
        draft_start_time = time.time()
        with torch.no_grad():
            draft_outputs = small_model.generate(
                input_ids=input_ids.to(small_model.device), # Pass the *entire* current sequence
                max_new_tokens=gamma,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Slice to get *only* the new draft tokens
        draft_ids = draft_outputs.sequences[:, input_ids.shape[-1]:]
        
        # Check if EOS was generated in the draft
        if draft_ids.numel() > 0:
            eos_pos = (draft_ids == eos_id).nonzero(as_tuple=False)
            if eos_pos.numel() > 0:
                first_eos_idx = int(eos_pos[0, 1])
                draft_ids = draft_ids[:, :first_eos_idx + 1]
                
        if torch.cuda.is_available(): torch.cuda.synchronize()
        draft_time = time.time() - draft_start_time
        total_draft_time += draft_time

        current_draft_len = draft_ids.shape[-1]
        if current_draft_len == 0: # Draft model generated nothing (e.g., EOS)
            break

        # --- 2. Verification (Big Model) ---
        if torch.cuda.is_available(): torch.cuda.synchronize()
        verify_start_time = time.time()
        with torch.no_grad():
            # Pass the *entire* sequence (prefix + draft) to the big model
            combined_ids = torch.cat([input_ids, draft_ids.to(big_model.device)], dim=-1)
            
            # NO ENCODER. Just pass the combined_ids.
            big_model_outputs = big_model(combined_ids)
            
            # Get logits for the newly generated tokens
            # We need to check (G) draft tokens + (1) correction token
            start_index = input_ids.shape[-1] - 1
            big_model_logits = big_model_outputs.logits[:, start_index:, :]

        if torch.cuda.is_available(): torch.cuda.synchronize()
        verify_time = time.time() - verify_start_time
        total_verify_time += verify_time

        # --- 3. Acceptance/Rejection (Greedy) ---
        accepted_len_this_cycle = 0
        all_accepted = True

        for i in range(current_draft_len):
            big_model_pred_id = torch.argmax(big_model_logits[:, i, :], dim=-1)
            draft_token_id = draft_ids[:, i].to(big_model.device)

            if big_model_pred_id != draft_token_id:
                # REJECT: Mismatch found
                total_accepted_tokens += i
                accepted_len_this_cycle = i
                accepted_ids = draft_ids[:, :i].to(big_model.device)
                
                # Append the *corrected* token from the big model
                corrected_id = big_model_pred_id.unsqueeze(0)
                
                # Update the main 'input_ids'
                input_ids = torch.cat([input_ids, accepted_ids, corrected_id], dim=-1)
                n_generated += i + 1
                
                if int(corrected_id.item()) == eos_id:
                    break # Stop immediately if big model ended
                
                all_accepted = False
                break
        
        if all_accepted:
            # ACCEPT all gamma tokens
            total_accepted_tokens += current_draft_len
            accepted_len_this_cycle = current_draft_len
            
            # Check if the draft *ended* with EOS
            if int(draft_ids[0, -1].item()) == eos_id:
                input_ids = torch.cat([input_ids, draft_ids.to(big_model.device)], dim=-1)
                n_generated += current_draft_len
                break # Stop, we are done
            
            # Sample one more token from the big model's *last* logit
            next_token_logits = big_model_logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Update the main 'input_ids'
            input_ids = torch.cat([input_ids, draft_ids.to(big_model.device), next_token], dim=-1)
            n_generated += current_draft_len + 1
            
            if int(next_token.item()) == eos_id:
                break # Stop, we are done

        # print(f"   Cycle {num_draft_cycles:2d}: Draft Time={draft_time:.4f}s, Verify Time={verify_time:.4f}s, Accepted={accepted_len_this_cycle}/{current_draft_len}")

        if input_ids[0, -1] == eos_id or n_generated >= max_new_tokens:
            break
            
    if torch.cuda.is_available(): torch.cuda.synchronize()
    total_latency = time.time() - overall_start_time

    num_generated_tokens = input_ids.shape[1] - prompt_len
    
    # Decode only the newly generated tokens
    decoded_output = tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
    avg_accepted_len = total_accepted_tokens / num_draft_cycles if num_draft_cycles > 0 else 0
    tps = num_generated_tokens / total_latency if total_latency > 0 else 0

    return decoded_output, num_generated_tokens, total_latency, tps, avg_accepted_len, total_draft_time, total_verify_time

# --- Example Usage ---
if __name__ == "__main__":
    
    device = get_device()
    
    if device == "cuda":
        # --- Config ---
        SMALL_MODEL_NAME = "distilgpt2" # Draft model (82M)
        MAX_NEW_TOKENS = 100
        GAMMA = 4 # Number of tokens to draft
        PROMPT = "The future of artificial intelligence is"

        # --- Load Models ---
        tokenizer = load_tokenizer(SMALL_MODEL_NAME)
        small_model = load_small_model(SMALL_MODEL_NAME, device)
        big_model, big_model_name = load_big_model(device) # Will be gpt2-medium or gpt2

        print("--- Running Standard Inference (Big Model) ---")
        # Warmup
        normal_inference(big_model, tokenizer, PROMPT, max_new_tokens=10)
        
        # Real run
        norm_output, norm_tokens, norm_lat, norm_tps = normal_inference(
            big_model, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS
        )
        print(f"Output: ...{norm_output}")
        print(f"Generated Tokens: {norm_tokens}")
        print(f"Latency (s): {norm_lat:.4f}")
        print(f"Throughput (tok/s): {norm_tps:.2f}\n")


        print("--- Running Speculative Decoding ---")
        # Warmup
        speculative_decoding_loop(
            small_model, big_model, tokenizer, PROMPT, 
            max_new_tokens=10, gamma=GAMMA
        )
        
        # Real run
        spec_output, spec_tokens, spec_lat, spec_tps, avg_accept, draft_time, verify_time = speculative_decoding_loop(
            small_model, big_model, tokenizer, PROMPT, 
            max_new_tokens=MAX_NEW_TOKENS, gamma=GAMMA
        )
        
        print(f"Output: ...{spec_output}")
        print(f"Generated Tokens: {spec_tokens}")
        print(f"Latency (s): {spec_lat:.4f}")
        print(f"Throughput (tok/s): {spec_tps:.2f}")
        print(f"Avg. Accepted Tokens: {avg_accept:.2f}")
        print(f"Total Draft Time (s): {draft_time:.4f}")
        print(f"Total Verify Time (s): {verify_time:.4f}\n")
        
        # --- Comparison ---
        print("--- Comparison ---")
        print(f"Prompt: '{PROMPT}...'")
        print(f"Big Model: {big_model_name}, Small Model: {SMALL_MODEL_NAME}")
        print(f"Speedup (Speculative vs Standard): {norm_lat / spec_lat:.2f}x")
        print(f"TPS (Standard): {norm_tps:.2f}")
        print(f"TPS (Speculative): {spec_tps:.2f}")
    
    else:
        print("Not running example usage because CUDA is not available.")