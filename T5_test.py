import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

model_name = "google/flan-t5-xxl"

# 1. quantization config: 4-bit nf4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model (this may take a while)...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",              # let accelerate split GPU/CPU if needed
    low_cpu_mem_usage=True,
)

# quick sanity check of where layers got placed
print(model.hf_device_map)

# 2. prepare a tiny prompt
prompt = "translate English to German: Hello, how are you?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 3. run 1 short generation
with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=32,
    )

# 4. decode and print result
print("=== GENERATED TEXT ===")
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
