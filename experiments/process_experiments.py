import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict


WANDB_ENTITY = "thom-mousseau" 
WANDB_PROJECT = "GFlowNet-Experiments" 
METRIC_NAME = "time_train"

# Mapping of Experiment Folder Name -> WandB Tag to filter by
EXPERIMENT_GROUPS = {
    "neural-nets": "mlp",
    "environment": "environment",
    "hyperparameters": "hyperparameters",
    "hardware": "hardware"
}

TAG_JAX = "JAX"
TAG_TORCH = "PyTorch"

def get_runs(api, project_path, group_tag):
    """
    Fetch runs from WandB that match the group tag.
    """
    filters = {
        "tags": {"$in": [group_tag]},
        "state": "finished" 
    }
    runs = api.runs(project_path, filters=filters)
    return runs

def parse_run_info(run, param_prefix):
    """
    Extract relevant info from a run based on specific tag prefixes.
    Expected tags:
    - framework_<name>
    - <param_prefix><value>
    """
    tags = run.tags
    
    framework = None
    param_value = None
    
    for tag in tags:
        if tag.startswith("framework_"):
            framework = tag.replace("framework_", "")
        elif tag.startswith(param_prefix):
            param_value = tag.replace(param_prefix, "")
    
    if not framework or param_value is None:
        return None

    history = run.history(keys=[METRIC_NAME, "_step"])
    
    if METRIC_NAME not in history:
        return None

    return {
        "id": run.id,
        "name": run.name,
        "framework": framework,
        "parameter": param_value,
        "history": history,
        "total_time": history[METRIC_NAME].sum()
    }

def process_experiment_group(group_name, exp_tag, param_prefix):
    print(f"Processing group: {group_name} (Tag: {exp_tag}, Param: {param_prefix})")
    api = wandb.Api()
    project_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    
    runs = get_runs(api, project_path, exp_tag)
    
    data = []
    for run in runs:
        info = parse_run_info(run, param_prefix)
        if info:
            data.append(info)
            
    if not data:
        print(f"No valid runs found for {group_name}")
        return

    os.makedirs(os.path.join("experiments", group_name), exist_ok=True)
    
    filename_suffix = param_prefix.rstrip("_")

    df_rows = []
    grouped_data = defaultdict(lambda: defaultdict(list))
    
    for item in data:
        grouped_data[item["parameter"]][item["framework"]].append(item["total_time"])

    try:
        sorted_params = sorted(grouped_data.keys(), key=lambda x: float(x))
    except ValueError:
        sorted_params = sorted(grouped_data.keys())

    all_frameworks = set()
    for item in data:
        all_frameworks.add(item["framework"])
    all_frameworks = sorted(list(all_frameworks))

    for param in sorted_params:
        frameworks = grouped_data[param]
        row = {"Parameter": param}
        
        for fw in all_frameworks:
            times = frameworks.get(fw, [])
            if times:
                mean = np.mean(times)
                var = np.var(times)
                row[f"{fw} Mean"] = mean
                row[f"{fw} Var"] = var
            else:
                row[f"{fw} Mean"] = np.nan
                row[f"{fw} Var"] = np.nan
        
        df_rows.append(row)
        
    df_table = pd.DataFrame(df_rows)
    output_csv_path = os.path.join("experiments", group_name, f"results_{filename_suffix}_table.csv")
    df_table.to_csv(output_csv_path, index=False)
    print(f"Saved table to {output_csv_path}")

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    all_frames = []
    for item in data:
        h = item["history"].copy()
        h["Framework"] = item["framework"]
        h["Parameter"] = item["parameter"]
        all_frames.append(h)
        
    if not all_frames:
        return

    full_df = pd.concat(all_frames)
    
    try:
        full_df["Parameter"] = pd.to_numeric(full_df["Parameter"])
    except ValueError:
        pass

    sns.lineplot(
        data=full_df,
        x="_step",
        y=METRIC_NAME,
        hue="Parameter",
        style="Framework",
        markers=False,
        palette="viridis"
    )
    
    plt.title(f"Training Speed Comparison: {group_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Time per Iteration (s)")
    
    output_plot_path = os.path.join("experiments", group_name, f"training_curve_{filename_suffix}.png")
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Saved plot to {output_plot_path}")

if __name__ == "__main__":

    #! Neural Networks Experiments
    process_experiment_group("neural-nets", "exp_mlp", "nhid_")
    # process_experiment_group("environment", "exp_grid", "ndim_")
    
    #! Environment Experiments
    # process_experiment_group("environment", "exp_environment", "length_")
    # process_experiment_group("environment", "exp_environment", "ndim_")
    
    #! Hyperparameters Experiments
    #process_experiment_group("hyperparameters", "exp_hyperparam", "batchsize_")
    #process_experiment_group("hyperparameters", "exp_hyperparam", "trajectory_")
    
    #! Hardware Experiments
    #process_experiment_group("hardware", "exp_hardware", "gpu_")
    
    
