import torch
import pandas as pd
from rich.console import Console
from rich.table import Table
import time

# Import the core logic from your original file
import T5_SD as core

# --- 1. DEFINE EXPERIMENTS ---

# Prompts based on paper's tasks (WMT EnDe and CNN/DM)
PROMPTS = {
    "Translation (En->De)": [
        "translate English to German: The future of artificial intelligence is rapidly evolving, with new models and capabilities being developed at an unprecedented pace.",
        "translate English to German: Climate change is one of the most pressing issues of our time, requiring global cooperation and innovative solutions.",
        "translate English to German: The quick brown fox jumps over the lazy dog."
    ],
    "Summarization (CNN/DM)": [
        "summarize: (CNN) -- A new study suggests that drinking coffee may help reduce the risk of developing certain types of cancer. The research, published in the journal 'Clinical Gastroenterology and Hepatology', found that coffee drinkers had a lower risk of hepatocellular carcinoma, the most common type of liver cancer. The study's authors, from the University of Milan, analyzed data from 16 previous studies involving more than 3,150 participants. 'Our research confirms past findings that coffee is good for your health, and particularly your liver,' said study co-author Dr. Carlo La Vecchia. The researchers found that people who drank coffee regularly had a 40% reduced risk of liver cancer compared to those who did not drink coffee. The risk reduction was even more significant for those who drank three or more cups a day.",
        "summarize: (Reuters) - Global stocks edged higher on Monday as investors weighed the prospects of a solid economic recovery against concerns about rising inflation. The MSCI All-Country World Index gained 0.2%, led by strong performances in Europe and Asia. In the United States, the S&P 500 and the Nasdaq Composite opened modestly higher. Investors are awaiting key inflation data later this week, which could influence the Federal Reserve's timeline for tapering its asset purchase program. Despite the inflation fears, market sentiment remains largely positive, supported by ongoing vaccination efforts and strong corporate earnings."
    ]
}

# Experiment matrix
# We will test T5-small, T5-base, and T5-large as approximation models
# We will test gamma values [3, 5, 7] for each
EXPERIMENT_CONFIG = [
    {
        "task_name": "Translation (En->De)",
        "small_model_names": ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"],
        "gamma_values": [3, 5, 7],
        "prompts": PROMPTS["Translation (En->De)"],
        "max_new_tokens": 60
    },
    {
        "task_name": "Summarization (CNN/DM)",
        "small_model_names": ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"],
        "gamma_values": [3, 5, 7],
        "prompts": PROMPTS["Summarization (CNN/DM)"],
        "max_new_tokens": 100
    }
]

def measure_latency(small_model, big_model, tokenizer, prompt, max_new_tokens=50, gamma=5):
    """
    Runs both baseline and speculative inference and returns a dict of results.
    """
    print("\n" + "="*50)
    print(f"Processing prompt: '{prompt}'")
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
    print(f"Baseline Output:    {normal_output}")
    print(f"Speculative Output: {spec_output}")
    outputs_match = (normal_output == spec_output)
    if outputs_match:
        print("SUCCESS: Outputs match!")
    else:
        print("WARNING: Outputs differ.")
        
    speedup = spec_tps / normal_tps if normal_tps > 0 else 0
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
                    result["small_model_name"] = small_model_name.split('/')[-1] # Clean name
                    all_results.append(result)
                    
                    # Give GPU a quick break
                    time.sleep(1)
                    
    return all_results

def print_summary_report(all_results):
    """
    Prints a final summary table of all results using Pandas.
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
        
        table = Table(title=f"Experiment Summary: {task} (Greedy / temp=0)", show_header=True, header_style="bold magenta")
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
    """
    try:
        device = core.get_device()
        tokenizer = core.load_tokenizer()
        
        # 1. Load Big Model (Once)
        big_model, big_model_name = core.load_big_model(device)
        
        # 2. Load All Small Models
        small_model_names = set()
        for exp in EXPERIMENT_CONFIG:
            small_model_names.update(exp['small_model_names'])
            
        small_model_dict = {}
        for name in sorted(list(small_model_names)):
            small_model_dict[name] = core.load_small_model(name, device)
            
        # 3. Run Experiments
        print("\n\n" + "*"*80)
        print("STARTING EXPERIMENT RUN")
        print("Note: This will take a significant amount of time.")
        print("*"*80 + "\n\n")
        
        all_results = run_experiment(big_model, small_model_dict, tokenizer, EXPERIMENT_CONFIG)
        
        # 4. Print Summary
        print("\n\n" + "*"*80)
        print("EXPERIMENT RUN COMPLETE")
        print("*"*80 + "\n\n")
        print_summary_report(all_results)
        
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
