import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

import T5_SD as core


device = core.get_device()
tokenizer = core.load_tokenizer()

big_model, big_model_name = core.load_big_model(device)

small_model = core.load_small_model("google/flan-t5-small", device)

prompt = "translate English to German: Climate change is one of the most pressing issues of our time, requiring global cooperation and innovative solutions."
max_new_tokens = 60

print("Running Normal Inference (Baseline)...")
normal_output, normal_tokens, normal_latency, normal_tps = core.normal_inference(
    big_model, tokenizer, prompt, max_new_tokens
)
print(f"Baseline: Latency={normal_latency:.4f}s, Tokens={normal_tokens}, TPS={normal_tps:.2f}")
print(f"Big output {normal_output}")

# --- Run Small Model Inference (Reference) ---
print("\nRunning Small Model Inference (Reference)...")
small_output, small_tokens, small_latency, small_tps = core.normal_inference(
    small_model, tokenizer, prompt, max_new_tokens
)
print(f"Small Model: Latency={small_latency:.4f}s, Tokens={small_tokens}, TPS={small_tps:.2f}")
print(f"Small output  {small_output}")

# --- Run Speculative Decoding ---
print("\nRunning Speculative Decoding...")
spec_output, spec_tokens, spec_latency, spec_tps, avg_accepted, _, _ = core.speculative_decoding_loop(
    small_model, big_model, tokenizer, prompt, max_new_tokens, gamma=5
)
print(f"Speculative: Latency={spec_latency:.4f}s, Tokens={spec_tokens}, TPS={spec_tps:.2f}")
print(f"Spec output {spec_output}")

print(f"Avg. Accepted Tokens per Cycle: {avg_accepted:.2f}")