import torch
import pandas as pd
from rich.console import Console
from rich.table import Table
import time

# Import the core logic from your gpt_experiment.py file
# Make sure your file is named gpt_experiment.py
try:
    import gpt_SD as core
except ImportError:
    print("Error: Could not import gpt_experiment.py. Make sure the file is in the same directory.")
    exit(1)


# --- 1. DEFINE EXPERIMENTS (GPT-2) ---

# Prompts for unconditional generation (Imlb task mentioned in paper)
PROMPTS = {
    "Unconditional Generation": [
        "The future of artificial intelligence is",
        "Machine learning is a field of inquiry devoted to",
        "In a surprising turn of events, the researchers found that",
        "The quick brown fox jumps over the lazy dog because",
        "Once upon a time, in a land far, far away,"
    ]
}

# Experiment matrix
# We will test distilgpt2 (82M) and gpt2 (124M) as approximation models
# against the larger gpt2-medium (355M) model.
EXPERIMENT_CONFIG = [
    {
        "task_name": "Unconditional Generation (GPT-2)",
        "small_model_names": ["distilgpt2", "gpt2"],
        "gamma_values": [3, 5, 7],
        "prompts": PROMPTS["Unconditional Generation"],
        "max_new_tokens": 100
    }
]

def measure_latency(small_model, big_model, tokenizer, prompt, max_new_tokens=50, gamma=5):
    """
    Runs both baseline and speculative inference and returns a dict of results.
    (Adapted from your T5 runner)
    """
    print("\n" + "="*50)
    print(f"Processing prompt: '{prompt}...'")
    print(f"Max new tokens: {max_new_tokens}, Gamma: {gamma}")
    print("="*50)

    # --- Run Normal Inference (Baseline) ---
    print("Running Normal Inference (Baseline)...")
    normal_output, normal_tokens, normal_latency, normal_tps = core.normal_inference(
        big_model, tokenizer, prompt, max_new_tokens
    )
    print(f" ... Baseline: Latency={normal_latency:.4f}s, Tokens={normal_tokens}, TPS={normal_tps:.2f}")

    # --- Run Small Model Inference (Reference) ---
    print("\nRunning Small Model Inference (Reference)...")
    small_output, small_tokens, small_latency, small_tps = core.normal_inference(
        small_model, tokenizer, prompt, max_new_tokens
    )
    print(f" ... Small Model: Latency={small_latency:.4f}s, Tokens={small_tokens}, TPS={small_tps:.2f}")

    # --- Run Speculative Decoding ---
    print("\nRunning Speculative Decoding...")
    spec_output, spec_tokens, spec_latency, spec_tps, avg_accepted, _, _ = core.speculative_decoding_loop(
        small_model, big_model, tokenizer, prompt, max_new_tokens, gamma=gamma
    )
    print(f" ... Speculative: Latency={spec_latency:.4f}s, Tokens={spec_tokens}, TPS={spec_tps:.2f}")
    print(f" ... Avg. Accepted Tokens per Cycle: {avg_accepted:.2f}")

    # --- Verification ---
    print("\n--- Verification of Outputs ---")
    # We compare the starts since outputs can diverge slightly but should match greedily
    baseline_start = normal_output.strip()
    speculative_start = spec_output.strip()
    
    # Greedy outputs should match exactly
    outputs_match = (baseline_start == speculative_start)
    
    if outputs_match:
        print("SUCCESS: Outputs match!")
    else:
        print("WARNING: Outputs differ.")
        print(f"   Baseline:    ...{baseline_start[:50]}...")
        print(f"   Speculative: ...{speculative_start[:50]}...")
        
    
    speedup = normal_latency / spec_latency if spec_latency > 0 else 0
    
    print(f"\nSpeedup (Speculative vs Baseline): {speedup:.2f}x")
    print("="*50 + "\n")

    return {
        "prompt": prompt[:50],
        "max_new_tokens": max_new_tokens,
        "gamma": gamma,
        "small_tps": small_tps,
        "normal_tps": normal_tps,
        "speculative_tps": spec_tps,
        "avg_accepted_len": avg_accepted,
        "speedup": speedup,
        "normal_latency": normal_latency,
        "speculative_latency": spec_latency,
        "outputs_match": outputs_match
    }

def run_experiment(big_model, small_model_dict, tokenizer, config):
    """
    Runs a full experiment configuration.
    (Adapted from your T5 runner)
    """
    all_results = []
    
    for experiment in config:
        task_name = experiment["task_name"]
        max_tokens = experiment["max_new_tokens"]
        
        for small_model_name in experiment["small_model_names"]:
            small_model = small_model_dict[small_model_name]
            
            for gamma in experiment["gamma_values"]:
                
                # Run warmup pass
                print(f"--- Warming up {small_model_name} (gamma={gamma}) ---")
                _ = core.speculative_decoding_loop(
                    small_model, big_model, tokenizer, 
                    experiment["prompts"][0], max_new_tokens=10, gamma=gamma
                )
                print("--- Warmup complete ---")
                
                for prompt in experiment["prompts"]:
                    result = measure_latency(
                        small_model, big_model, tokenizer, prompt, max_tokens, gamma
                    )
                    result["task_name"] = task_name
                    result["small_model_name"] = small_model_name # Already clean name
                    all_results.append(result)
                    
                    # Give GPU a quick break
                    time.sleep(1)
                    
    return all_results

def print_summary_report(all_results, big_model_name):
    """
    Prints a final summary table of all results using Pandas.
    (Adapted from your T5 runner)
    """
    if not all_results:
        print("No results to display.")
        return
        
    df = pd.DataFrame(all_results)
    
    # Calculate mean values, grouped by our experiment parameters
    summary = df.groupby(['task_name', 'small_model_name', 'gamma']).agg(
        avg_speedup=pd.NamedAgg(column='speedup', aggfunc='mean'),
        avg_spec_tps=pd.NamedAgg(column='speculative_tps', aggfunc='mean'),
        avg_normal_tps=pd.NamedAgg(column='normal_tps', aggfunc='mean'),
        avg_accepted=pd.NamedAgg(column='avg_accepted_len', aggfunc='mean'),
        outputs_match=pd.NamedAgg(column='outputs_match', aggfunc='all')
    ).reset_index()
    
    # Sort for readability
    summary = summary.sort_values(by=['task_name', 'small_model_name', 'avg_speedup'], ascending=[True, True, False])
    
    console = Console()
    
    for task in summary['task_name'].unique():
        task_df = summary[summary['task_name'] == task]
        
        table = Table(title=f"Experiment Summary: {task} (Target: {big_model_name} / Greedy)", show_header=True, header_style="bold magenta")
        table.add_column("Small Model", style="dim", width=18)
        table.add_column("Gamma (γ)", justify="right")
        table.add_column("Avg. Speedup", justify="right")
        table.add_column("Avg. Spec TPS", justify="right")
        table.add_column("Avg. Normal TPS", justify="right")
        table.add_column("Avg. Accepted", justify="right")
        table.add_column("Outputs Match?", justify="center")
        
        for _, row in task_df.iterrows():
            speedup_str = f"[bold green]{row['avg_speedup']:.2f}x[/bold green]" if row['avg_speedup'] > 1 else f"{row['avg_speedup']:.2f}x"
            match_str = "[green]✔[/green]" if row['outputs_match'] else "[red]✘[/red]"
            
            table.add_row(
                row['small_model_name'],
                f"{row['gamma']}",
                speedup_str,
                f"{row['avg_spec_tps']:.2f}",
                f"{row['avg_normal_tps']:.2f}",
                f"{row['avg_accepted']:.2f}",
                match_str
            )
        
        console.print(table)

def main():
    """
    Main function to load models and run the experiment suite.
    (Adapted from your T5 runner)
    """
    try:
        device = core.get_device()
        
        # 1. Load All Small Models
        # We define all small models we want to test here
        small_model_names_to_test = set()
        for exp in EXPERIMENT_CONFIG:
            small_model_names_to_test.update(exp['small_model_names'])
            
        # We need a tokenizer. We'll use the smallest model's tokenizer.
        # 'gpt2' and 'distilgpt2' share a tokenizer, so this is safe.
        tokenizer_name = sorted(list(small_model_names_to_test))[0]
        tokenizer = core.load_tokenizer(tokenizer_name)
        
        # Load the small models into a dictionary
        small_model_dict = {}
        for name in sorted(list(small_model_names_to_test)):
            small_model_dict[name] = core.load_small_model(name, device)
            
        # 2. Load Big Model (Once)
        big_model, big_model_name = core.load_big_model(device)
            
        # 3. Run Experiments
        print("\n\n" + "*"*80)
        print("STARTING GPT-2 EXPERIMENT RUN")
        print("Note: This will take a significant amount of time.")
        print("*"*80 + "\n\n")
        
        all_results = run_experiment(big_model, small_model_dict, tokenizer, EXPERIMENT_CONFIG)
        
        # 4. Print Summary
        print("\n\n" + "*"*80)
        print("EXPERIMENT RUN COMPLETE")
        print("*"*80 + "\n\n")
        print_summary_report(all_results, big_model_name)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nExperiment script finished.")
        print("Cleaning up models from memory (if possible)...")
        # Help Python's garbage collector
        big_model = None
        small_model_dict = None
        torch.cuda.empty_cache()
        print("Cleanup attempted.")


if __name__ == "__main__":
    # We need to install pandas and rich for the summary report
    try:
        import pandas
        import rich
    except ImportError:
        print("Missing dependencies. Installing pandas and rich...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "rich"])
        print("Installation complete. Please run the script again.")
        sys.exit(1)
        
    main()