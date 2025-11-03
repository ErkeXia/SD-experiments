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

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device != "cuda":
    print("WARNING: CUDA not available. This will be extremely slow.")

# --- Quantization: 4-bit NF4 for everything that's quantized ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# --- Tokenizer (FLAN) ---
tokenizer_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

print("Loading models... This may take a moment.")
print("Models will be downloaded to the cache set by your HF_HOME/TRANSFORMERS_CACHE environment variables.")

# --- BIG MODEL: flan-t5-xxl in 4-bit, fallback to flan-t5-xl ---
try:
    big_model_name = "google/flan-t5-xxl"
    print(f"Attempting to load {big_model_name} in 4-bit...")
    big_model = AutoModelForSeq2SeqLM.from_pretrained(
        big_model_name,
        device_map="auto",                 # shard + CPU offload as needed
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

# --- SMALL MODEL: flan-t5-small (bf16) ---
small_model_name = "google/flan-t5-small"
small_model = AutoModelForSeq2SeqLM.from_pretrained(
    small_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
print("\n" + "="*30)
print(f"Successfully loaded small model: {small_model_name}")
print("="*30 + "\n")

# Ensure pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def normal_inference(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False
    )
    num_generated_tokens = outputs.shape[1]
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output, num_generated_tokens

def speculative_decoding_loop(small_model, big_model, tokenizer, prompt, max_new_tokens=50, gamma=5):
    prompt_inputs = tokenizer(prompt, return_tensors='pt').to(device)
    prompt_input_ids = prompt_inputs.input_ids.to(big_model.device)

    with torch.no_grad():
        small_encoder_outputs = small_model.encoder(input_ids=prompt_input_ids.to(small_model.device))
        big_encoder_outputs   = big_model.encoder(input_ids=prompt_input_ids.to(big_model.device))

    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long, device=big_model.device)

    total_accepted_tokens = 0
    num_draft_cycles = 0
    total_draft_time = 0
    total_verify_time = 0
    n_generated = 0
    print("\n--- Starting Speculative Generation (Greedy) ---")

    while n_generated < max_new_tokens:
        num_draft_cycles += 1

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
        if torch.cuda.is_available(): torch.cuda.synchronize()
        draft_time = time.time() - draft_start_time
        total_draft_time += draft_time

        if torch.cuda.is_available(): torch.cuda.synchronize()
        verify_start_time = time.time()
        with torch.no_grad():
            combined_ids = torch.cat([decoder_input_ids, draft_ids.to(big_model.device)], dim=-1)
            big_model_outputs = big_model(decoder_input_ids=combined_ids, encoder_outputs=big_encoder_outputs)
            start_index = decoder_input_ids.shape[-1] - 1
            big_model_logits = big_model_outputs.logits[:, start_index:, :]
        if torch.cuda.is_available(): torch.cuda.synchronize()
        verify_time = time.time() - verify_start_time
        total_verify_time += verify_time

        accepted_len_this_cycle = 0
        all_accepted = True
        current_draft_len = draft_ids.shape[-1]
        if current_draft_len == 0:
            print(f"   Cycle {num_draft_cycles:2d}: Draft model produced 0 tokens. Stopping.")
            break

        for i in range(current_draft_len):
            big_model_pred_id = torch.argmax(big_model_logits[:, i, :], dim=-1)
            draft_token_id = draft_ids[:, i].to(big_model.device)

            if big_model_pred_id != draft_token_id:
                total_accepted_tokens += i
                accepted_len_this_cycle = i
                accepted_ids = draft_ids[:, :i].to(big_model.device)
                corrected_id = big_model_pred_id.unsqueeze(0)
                decoder_input_ids = torch.cat([decoder_input_ids, accepted_ids, corrected_id], dim=-1)
                n_generated += i + 1
                all_accepted = False
                break

        if all_accepted:
            total_accepted_tokens += current_draft_len
            accepted_len_this_cycle = current_draft_len
            next_token_logits = big_model_logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            decoder_input_ids = torch.cat([decoder_input_ids, draft_ids.to(big_model.device), next_token], dim=-1)
            n_generated += current_draft_len + 1

        print(f"   Cycle {num_draft_cycles:2d}: Draft Time={draft_time:.4f}s, Verify Time={verify_time:.4f}s, Accepted={accepted_len_this_cycle}/{current_draft_len}")

        if decoder_input_ids[0, -1] == tokenizer.eos_token_id or n_generated >= max_new_tokens:
            break

    num_generated_tokens = decoder_input_ids.shape[1] - 1
    decoded_output = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
    avg_accepted_len = total_accepted_tokens / num_draft_cycles if num_draft_cycles > 0 else 0

    print("--- Finished Speculative Generation ---")
    print(f"Total time spent drafting: {total_draft_time:.4f}s")
    print(f"Total time spent verifying: {total_verify_time:.4f}s")

    return decoded_output, num_generated_tokens, avg_accepted_len

def measure_latency(small_model, big_model, tokenizer, prompt, max_new_tokens=50, gamma=5):
    print("\n" + "="*50)
    print(f"Processing prompt: '{prompt[:50]}...'")
    print(f"Max new tokens: {max_new_tokens}, Gamma: {gamma}")
    print("="*50)

    print("Running Normal Inference (Baseline)...")
    start_time = time.time()
    normal_output, normal_tokens = normal_inference(big_model, tokenizer, prompt, max_new_tokens)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    normal_inference_latency = time.time() - start_time
    normal_tps = normal_tokens / normal_inference_latency if normal_inference_latency > 0 else 0
    print(f"   -> Latency: {normal_inference_latency:.4f}s, Tokens: {normal_tokens}, TPS: {normal_tps:.2f}")

    print("\nRunning Small Model Inference (Reference)...")
    start_time = time.time()
    small_output, small_tokens = normal_inference(small_model, tokenizer, prompt, max_new_tokens)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    small_inference_latency = time.time() - start_time
    small_tps = small_tokens / small_inference_latency if small_inference_latency > 0 else 0
    print(f"   -> Latency: {small_inference_latency:.4f}s, Tokens: {small_tokens}, TPS: {small_tps:.2f}")

    print("\nRunning Speculative Decoding...")
    start_time = time.time()
    speculative_output, speculative_tokens, avg_accepted_len = speculative_decoding_loop(
        small_model, big_model, tokenizer, prompt, max_new_tokens, gamma=gamma
    )
    if torch.cuda.is_available(): torch.cuda.synchronize()
    speculative_decoding_latency = time.time() - start_time
    speculative_tps = speculative_tokens / speculative_decoding_latency if speculative_decoding_latency > 0 else 0

    print(f"\nSpeculative Decoding Summary:")
    print(f"   -> Latency: {speculative_decoding_latency:.4f} seconds")
    print(f"   -> Tokens: {speculative_tokens}")
    print(f"   -> Throughput: {speculative_tps:.2f} tokens/sec")
    print(f"   -> Avg. Accepted Tokens per Cycle: {avg_accepted_len:.2f}")

    print("\n--- Verification of Outputs ---")
    print(f"Baseline Output:    {normal_output}")
    print(f"Speculative Output: {speculative_output}")
    if normal_output == speculative_output:
        print("SUCCESS: Outputs match!")
    else:
        print("WARNING: Outputs differ. (Greedy may stop at different steps)")

    print("="*50 + "\n")

    return small_tps, normal_tps, speculative_tps

# --- Prompts ---
prompts = [
    "translate English to German: The future of artificial intelligence is rapidly evolving, with new models and capabilities being developed at an unprecedented pace.",
    "summarize: Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'..."
]
max_new_tokens = 50
gamma_value = 5
total_normal_tps, total_small_tps, total_speculative_tps = 0, 0, 0

for prompt in prompts:
    small_tps, normal_tps, speculative_tps = measure_latency(
        small_model, big_model, tokenizer, prompt, max_new_tokens, gamma=gamma_value
    )
    total_normal_tps += normal_tps
    total_small_tps += small_tps
    total_speculative_tps += speculative_tps

average_normal_tps = total_normal_tps / len(prompts)
average_small_tps = total_small_tps / len(prompts)
average_speculative_tps = total_speculative_tps / len(prompts)

print(f"\n--- Final Averages ---")
print(f"Average Normal Inference (Baseline) Throughput: {average_normal_tps:.2f} tokens/sec")
print(f"Average Small Inference (Reference) Throughput: {average_small_tps:.2f} tokens/sec")
print(f"Average Speculative Decoding Throughput: {average_speculative_tps:.2f} tokens/sec")

speedup = average_speculative_tps / average_normal_tps if average_normal_tps > 0 else 0
print(f"\nAchieved Throughput Speedup (Speculative vs Baseline): {speedup:.2f}x")
