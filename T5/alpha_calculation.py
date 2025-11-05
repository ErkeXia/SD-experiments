import torch
import torch.nn.functional as F
import time
from transformers import set_seed
import warnings
from tqdm import tqdm # Using tqdm to show progress

# Import the refactored helper functions
from T5_SD_std import get_device, load_tokenizer, load_big_model, load_small_model

# --- fast matmul on Ampere ---
torch.backends.cuda.matmul.allow_tf32 = True

# Suppress warnings
warnings.filterwarnings("ignore")

# Seed
set_seed(42)

def calculate_alpha_for_prompts(model_p, model_q, tokenizer, prompts, max_length=50, sampling_temp=1.0):
    """
    Calculates the alpha value (acceptance rate) as defined in the paper.
    
    Alpha = E[sum(min(p, q))]
    
    For T=0 (greedy), this is equivalent to the accuracy of model_q's
    greedy choice vs. model_p's greedy choice.
    
    For T=1, this is the expected overlap of the probability distributions.
    
    Args:
        model_p: The large target model (Mp).
        model_q: The small approximation model (Mq).
        tokenizer: The tokenizer.
        prompts: A list of strings to use for calculation.
        max_length: The max number of tokens to generate/check per prompt.
        sampling_temp: The temperature to use. 0.0 means greedy (argmax).
    """
    
    all_step_alphas = [] # Stores the alpha (beta_t) for each individual token step
    device_p = model_p.device
    device_q = model_q.device

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Processing prompts"):
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device_p)
            
            # Get encoder outputs once per prompt
            encoder_outputs_p = model_p.encoder(input_ids=input_ids)
            encoder_outputs_q = model_q.encoder(input_ids=input_ids.to(device_q))

            # Start decoder with pad token
            decoder_input_ids = torch.tensor(
                [[tokenizer.pad_token_id]], dtype=torch.long, device=device_p
            )

            for _ in range(max_length):
                # Get logits from the target model (p)
                outputs_p = model_p(
                    encoder_outputs=encoder_outputs_p,
                    decoder_input_ids=decoder_input_ids
                )
                logits_p = outputs_p.logits[:, -1, :] # Get last token's logits

                # Get logits from the approximation model (q)
                outputs_q = model_q(
                    encoder_outputs=encoder_outputs_q,
                    decoder_input_ids=decoder_input_ids.to(device_q)
                )
                logits_q = outputs_q.logits[:, -1, :] # Get last token's logits
                
                next_token = None
                
                if sampling_temp == 0.0:
                    # Greedy case (T=0 in the paper)
                    # Alpha is the accuracy of Mq's argmax vs Mp's argmax
                    p_token = torch.argmax(logits_p, dim=-1)
                    q_token = torch.argmax(logits_q, dim=-1)
                    
                    beta_t = (p_token == q_token.to(p_token.device)).float()
                    all_step_alphas.append(beta_t.item())
                    
                    # Use the target model's token for the next step
                    next_token = p_token.unsqueeze(0)
                    
                else:
                    # Stochastic case (T=1 in the paper)
                    # Alpha is E[sum(min(p, q))]
                    dist_p = F.softmax(logits_p / sampling_temp, dim=-1)
                    dist_q = F.softmax(logits_q / sampling_temp, dim=-1)
                    
                    # Calculate sum(min(p, q)) for this step
                    min_probs = torch.min(dist_p, dist_q.to(dist_p.device))
                    beta_t = torch.sum(min_probs, dim=-1)
                    all_step_alphas.append(beta_t.item())
                    
                    # Sample from the target model's distribution for the next step
                    # (as done in the paper's alpha calculation, Sec 4.2)
                    next_token = torch.multinomial(dist_p, num_samples=1)

                # Append the chosen token to continue the sequence
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
                
                # Stop if EOS is generated
                if next_token.item() == tokenizer.eos_token_id:
                    break
    
    # The final alpha is the mean of all step-alphas (E[beta_t])
    if not all_step_alphas:
        print("Warning: No tokens were processed.")
        return 0.0
        
    alpha = sum(all_step_alphas) / len(all_step_alphas)
    return alpha


# --- Example Usage ---
if __name__ == "__main__":
    
    device = get_device()
    
    if device == "cuda":
        # --- Config ---
        SMALL_MODEL_NAME = "google/flan-t5-small" # Draft model (Mq)
        MAX_LENGTH_PER_PROMPT = 50 # Max tokens to check per prompt
        
        # Using a few example prompts. For a more robust alpha,
        # you would use a large dataset (e.g., 10k samples as in the paper).
        PROMPTS = [
            "Translate to German: My name is Wolfgang and I live in Berlin.",
            "Translate to German: The quick brown fox jumps over the lazy dog.",
            "summarize: Scientists have discovered a new species of glowing frog in the Amazon rainforest. The frog, which has translucent skin, was found during a nighttime expedition. Researchers believe its unique glow may be used for communication or camouflage.",
            "summarize: The James Webb Space Telescope has captured stunning new images of the Pillars of Creation, revealing intricate details of star formation within the dense clouds of gas and dust.",
            "Answer the question: What is the capital of France?"
        ]

        # --- Load Models ---
        tokenizer = load_tokenizer(SMALL_MODEL_NAME)
        small_model = load_small_model(SMALL_MODEL_NAME, device)
        big_model, big_model_name = load_big_model(device) # Target model (Mp)

        print("--- Calculating Alpha (α) ---")
        print(f"Target Model (p):   {big_model_name}")
        print(f"Draft Model (q):    {SMALL_MODEL_NAME}")
        print(f"Processing {len(PROMPTS)} prompts up to {MAX_LENGTH_PER_PROMPT} tokens each...")
        
        # --- Calculate for T=1.0 (Standard Sampling) ---
        print("\nCalculating for T=1.0 (Standard Sampling)...")
        alpha_t1 = calculate_alpha_for_prompts(
            big_model, small_model, tokenizer, PROMPTS, 
            max_length=MAX_LENGTH_PER_PROMPT, 
            sampling_temp=1.0
        )
        print(f"Calculated Alpha (α) for T=1.0: {alpha_t1:.4f}")
        
        # --- Calculate for T=0.0 (Greedy/Argmax) ---
        print("\nCalculating for T=0.0 (Greedy Sampling)...")
        alpha_t0 = calculate_alpha_for_prompts(
            big_model, small_model, tokenizer, PROMPTS, 
            max_length=MAX_LENGTH_PER_PROMPT, 
            sampling_temp=0.0
        )
        print(f"Calculated Alpha (α) for T=0.0: {alpha_t0:.4f}")
        
        print("\n--- Done ---")
        print(f"(Note: These values are estimates based on a small sample of {len(PROMPTS)} prompts.)")

    else:
        print("Not running calculation because CUDA is not available.")