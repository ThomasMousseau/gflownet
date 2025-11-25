import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict


WANDB_ENTITY = "fatty_data" 
WANDB_PROJECT = "GFlowNet-Experiments" 
METRIC_NAME = "time_train"
METRIC_NAME_SMOOTH = "time_train_smooth"
TARGET_STEPS = 5000 
WINDOW = 20

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

def get_run_partially_failed(api, project_path, group_tag):
    filters = {
        "tags": {"$in": [group_tag]},
        "state": {"$in": ["finished", "crashed"]}
    }
    runs = api.runs(project_path, filters=filters)
    return runs


def estimate_total_time(history, target_steps=5000, window=500):
    """
    Estimate total training time based on logged history.
    If the run has not reached target_steps, estimate the remaining time
    using the average time per iteration from the last `window` logged points.
    Returns:
        estimated_total_time (float): Estimated total training time.
        actual_total_time (float): Actual logged training time.
        max_step (int): Maximum step reached in the run.
        was_estimated (bool): Whether the total time was estimated due to crash.
    """
    if history.empty or '_step' not in history or METRIC_NAME not in history:
        return 0, 0, 0, False
    
    valid_history = history.dropna(subset=[METRIC_NAME]).sort_values('_step')
    
    if valid_history.empty:
        return 0, 0, 0, False
    
    max_step = int(valid_history['_step'].max())
    actual_total = valid_history[METRIC_NAME].sum()
    
    if max_step >= target_steps:
        return actual_total, actual_total, max_step, False
    
    cutoff_step = max_step - window
    recent_history = valid_history[valid_history['_step'] > cutoff_step]
    
    if not recent_history.empty:
        avg_time_per_iter = recent_history[METRIC_NAME].mean()
    else:
        avg_time_per_iter = valid_history[METRIC_NAME].mean()
        
    n_logged_points = len(valid_history)
    if n_logged_points > 1:
        steps_per_point = max_step / n_logged_points
        missing_steps = target_steps - max_step
        missing_points = int(missing_steps / steps_per_point)
    else:
        missing_points = 0
    
    estimated_missing_time = avg_time_per_iter * missing_points
    estimated_total = actual_total + estimated_missing_time
    
    return estimated_total, actual_total, max_step, True


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

    history_table = run.history(samples=5000,keys=[METRIC_NAME, "_step"]) #! alaways retuns 500 points
    history_graph = run.history(samples=500, keys=[METRIC_NAME, "_step"]) 
    
    
    # Calculate total time (estimated if crashed)
    estimated_total, actual_total, max_step, was_estimated = estimate_total_time(
        history_table, target_steps=TARGET_STEPS
    )
    
    # if was_estimated:
    #     print(f"  [Estimate] Run {run.name}: crashed at step {max_step}, "
    #           f"actual={actual_total:.2f}s, estimated={estimated_total:.2f}s")

    return {
        "id": run.id,
        "name": run.name,
        "framework": framework,
        "parameter": param_value,
        "history": history_graph,
        "total_time": estimated_total,  # Use estimated total for table
        "actual_time": actual_total,
        "max_step": max_step,
        "was_estimated": was_estimated
    }

def process_experiment_group(group_name, exp_tag, param_prefix, has_failed_experiments=False):
    print(f"Processing group: {group_name} (Tag: {exp_tag}, Param: {param_prefix})")
    api = wandb.Api()
    project_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    
    if has_failed_experiments:
        runs = get_run_partially_failed(api, project_path, exp_tag)
    else:
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

    # ========== TABLE ==========
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
                std = np.std(times)
                row[f"{fw} Mean"] = mean
                row[f"{fw} Std"] = std
            else:
                row[f"{fw} Mean"] = np.nan
                row[f"{fw} Std"] = np.nan
        
        df_rows.append(row)
        
    df_table = pd.DataFrame(df_rows)
    output_csv_path = os.path.join("experiments", group_name, f"results_{filename_suffix}_table.csv")
    df_table.to_csv(output_csv_path, index=False)
    print(f"Saved table to {output_csv_path}")

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

    min_max_step = min(item["max_step"] for item in data)
    
    full_df["time_train_smooth"] = (
        full_df
        .groupby(["Parameter", "Framework"])[METRIC_NAME]
        .transform(lambda s: s.rolling(window=WINDOW, min_periods=1).mean())
    )
    
    # ========== PLOT 1: FULL PLOT (0 to 5000) ==========
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    sns.lineplot(
        data=full_df,
        x="_step",
        y=METRIC_NAME_SMOOTH,
        hue="Parameter",
        style="Framework",
        markers=False,
        palette="viridis",
        errorbar=('ci', 95)
    )
    
    plt.title(f"Training Speed Comparison: {group_name} (Full)")
    plt.xlabel("Iteration")
    plt.ylabel("Time per Iteration (s)")
    plt.yscale("log")
    
    output_plot_path = os.path.join("experiments", group_name, f"training_curve_{filename_suffix}_full.png")
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Saved full plot to {output_plot_path}")

    # ========== PLOT 2: TRUNCATED PLOT (0 to min_max_step) ==========
    truncated_df = full_df[full_df["_step"] <= min_max_step]

    truncated_df["time_train_smooth"] = (
        truncated_df
        .groupby(["Parameter", "Framework"])[METRIC_NAME]
        .transform(lambda s: s.rolling(window=WINDOW, min_periods=1).mean())
    )
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    sns.lineplot(
        data=truncated_df,
        x="_step",
        y=METRIC_NAME_SMOOTH,
        hue="Parameter",
        style="Framework",
        markers=False,
        palette="viridis"
    )
    
    plt.title(f"Training Speed Comparison: {group_name} (Truncated to {min_max_step} steps)")
    plt.xlabel("Iteration")
    plt.ylabel("Time per Iteration (s)")
    plt.yscale("log")
    
    output_plot_path = os.path.join("experiments", group_name, f"training_curve_{filename_suffix}_truncated.png")
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Saved truncated plot to {output_plot_path} (truncated at step {min_max_step})")

if __name__ == "__main__":

    #! Neural Networks Experiments
    #process_experiment_group("neural-nets", "exp_mlp", "nlayers_", has_failed_experiments=True)
    #process_experiment_group("neural-nets", "exp_mlp", "nhid_")
    
    #! Environment Experiments
    process_experiment_group("environment", "exp_env", "length_", has_failed_experiments=True)
    process_experiment_group("environment", "exp_env", "dim_", has_failed_experiments=True)
    process_experiment_group("environment", "exp_env", "env_")
    
    #! Hyperparameters Experiments
    #process_experiment_group("hyperparameters", "exp_hyperparam", "batchsize_")
    #process_experiment_group("hyperparameters", "exp_hyperparam", "trajectory_")
    
    #! Hardware Experiments
    #process_experiment_group("hardware", "exp_hardware", "gpu_")
    
    
