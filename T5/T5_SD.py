import torch
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, set_seed
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

def load_tokenizer(tokenizer_name="google/flan-t5-small"):
    """Loads the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_big_model(device):
    """Loads the large target model (T5-XXL or fallback T5-XL) in 4-bit."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    try:
        big_model_name = "google/flan-t5-xxl"
        print(f"Attempting to load {big_model_name} in 4-bit...")
        big_model = AutoModelForSeq2SeqLM.from_pretrained(
            big_model_name,
            device_map="auto",
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
        )
        print("\n" + "="*30)
        print(f"Successfully loaded big model: {big_model_name}")
        print("="*30 + "\n")
    except Exception as e:
        print(f"Error loading {big_model_name}: {e}")
        print("Falling back to flan-t5-xl (3B) in 4-bit...")
        big_model_name = "google/flan-t5-xl"
        big_model = AutoModelForSeq2SeqLM.from_pretrained(
            big_model_name,
            device_map="auto",
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
        )
        print("\n" + "="*30)
        print(f"Successfully loaded big model: {big_model_name}")
        print("="*30 + "\n")
    return big_model, big_model_name

def load_small_model(model_name, device):
    """Loads a small approximation model in bf16."""
    print(f"Loading small model: {model_name}...")
    small_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print(f"Successfully loaded {model_name}\n")
    return small_model

def normal_inference(model, tokenizer, prompt, max_new_tokens=50):
    """Runs standard autoregressive inference."""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False  # Disabling cache for fair TPS comparison
    )
    if torch.cuda.is_available(): torch.cuda.synchronize()
    latency = time.time() - start_time
    
    # num_generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    num_generated_tokens = outputs.shape[1]
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tps = num_generated_tokens / latency if latency > 0 else 0
    return decoded_output, num_generated_tokens, latency, tps

def speculative_decoding_loop(small_model, big_model, tokenizer, prompt, max_new_tokens=50, gamma=5):
    """
    Runs greedy speculative decoding.
    Note: This is the GREEDY implementation (temp=0 in the paper).
    """
    device = big_model.device # Assume big model device is the primary
    
    # --- Prompt Encoding ---
    prompt_inputs = tokenizer(prompt, return_tensors='pt').to(device)
    prompt_input_ids = prompt_inputs.input_ids
    
    # --- Encoder Pass (Once per prompt) ---
    with torch.no_grad():
        small_encoder_outputs = small_model.encoder(input_ids=prompt_input_ids.to(small_model.device))
        big_encoder_outputs = big_model.encoder(input_ids=prompt_input_ids.to(big_model.device))

    # --- Decoder Initialization ---
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long, device=big_model.device)

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
                decoder_input_ids=decoder_input_ids.to(small_model.device),
                encoder_outputs=small_encoder_outputs,
                max_new_tokens=gamma,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        draft_ids = draft_outputs.sequences[:, decoder_input_ids.shape[-1]:]
        
        if draft_ids.numel() > 0:
            eos_pos = (draft_ids == eos_id).nonzero(as_tuple=False)
            if eos_pos.numel() > 0:
                first = int(eos_pos[0, 1])
                draft_ids = draft_ids[:, :first + 1]
                
        if torch.cuda.is_available(): torch.cuda.synchronize()
        draft_time = time.time() - draft_start_time
        total_draft_time += draft_time

        current_draft_len = draft_ids.shape[-1]
        if current_draft_len == 0:
            break

        # --- 2. Verification (Big Model) ---
        if torch.cuda.is_available(): torch.cuda.synchronize()
        verify_start_time = time.time()
        with torch.no_grad():
            combined_ids = torch.cat([decoder_input_ids, draft_ids.to(big_model.device)], dim=-1)
            big_model_outputs = big_model(decoder_input_ids=combined_ids, encoder_outputs=big_encoder_outputs)
            # Get logits for the newly generated tokens
            start_index = decoder_input_ids.shape[-1] - 1
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
                decoder_input_ids = torch.cat([decoder_input_ids, accepted_ids, corrected_id], dim=-1)
                n_generated += i + 1
                
                if int(corrected_id.item()) == eos_id:
                    # stop immediately if big model ended
                    break
                
                all_accepted = False
                break
        
        if all_accepted:
            # ACCEPT all gamma tokens
            total_accepted_tokens += current_draft_len
            accepted_len_this_cycle = current_draft_len
            
            if int(draft_ids[0, -1].item()) == eos_id:
                decoder_input_ids = torch.cat([decoder_input_ids, draft_ids.to(big_model.device)], dim=-1)
                n_generated += current_draft_len
                break
            
            # Sample one more token from the big model
            next_token_logits = big_model_logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            decoder_input_ids = torch.cat([decoder_input_ids, draft_ids.to(big_model.device), next_token], dim=-1)
            n_generated += current_draft_len + 1
            
            if int(next_token.item()) == eos_id:
                break

        # print(f"  Cycle {num_draft_cycles:2d}: Draft Time={draft_time:.4f}s, Verify Time={verify_time:.4f}s, Accepted={accepted_len_this_cycle}/{current_draft_len}")

        if decoder_input_ids[0, -1] == eos_id or n_generated >= max_new_tokens:
            break
        
    if torch.cuda.is_available(): torch.cuda.synchronize()
    total_latency = time.time() - overall_start_time

    # Final generated tokens (excluding initial pad_token)
    num_generated_tokens = decoder_input_ids.shape[1] - 1
    decoded_output = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
    avg_accepted_len = total_accepted_tokens / num_draft_cycles if num_draft_cycles > 0 else 0
    tps = num_generated_tokens / total_latency if total_latency > 0 else 0

    return decoded_output, num_generated_tokens, total_latency, tps, avg_accepted_len, total_draft_time, total_verify_time
