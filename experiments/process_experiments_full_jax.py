import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict


WANDB_ENTITY = "fatty_data" 
WANDB_PROJECT = "GFlowNet-Experiments" 

# Metrics per framework
METRIC_TORCH_TRAIN = "time_train"
METRIC_TORCH_SAMPLE = "time_sample"
METRIC_JAX = "time_epoch"

METRIC_NAME_COMBINED = "time_iteration"  # Combined metric name for plotting
METRIC_NAME_SMOOTH = "time_iteration_smooth"

TARGET_STEPS = 5000 
WINDOW = 20

# Output folder (relative to experiments/)
OUTPUT_FOLDER = "full_jax"

# Framework tags to compare
FRAMEWORK_TORCH = "legacy"
FRAMEWORK_JAX = "jax-full-eval1" #! "jax-full"

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
        "state": {"$in": ["finished", "crashed", "killed"]}
    }
    runs = api.runs(project_path, filters=filters)
    return runs


def estimate_total_time(history, framework, target_steps=5000, window=500, verbose=False):
    """
    Estimate total training time based on logged history.
    If the run has not reached target_steps, estimate the remaining time
    using the average time per iteration from the last `window` logged points.
    
    For PyTorch (legacy): uses time_train + time_sample
    For JAX (jax-full): uses time_epoch
    
    Returns:
        estimated_total_time (float): Estimated total training time.
        actual_total_time (float): Actual logged training time.
        max_step (int): Maximum step reached in the run.
        was_estimated (bool): Whether the total time was estimated due to crash.
    """
    if history.empty or '_step' not in history:
        return 0, 0, 0, False
    
    # Determine which metric(s) to use based on framework
    if framework == FRAMEWORK_TORCH:
        # PyTorch: combine time_train + time_sample
        if METRIC_TORCH_TRAIN not in history or METRIC_TORCH_SAMPLE not in history:
            return 0, 0, 0, False
        history = history.copy()
        history[METRIC_NAME_COMBINED] = history[METRIC_TORCH_TRAIN].fillna(0) + history[METRIC_TORCH_SAMPLE].fillna(0)
        metric_col = METRIC_NAME_COMBINED
    else:
        # JAX: use time_epoch
        if METRIC_JAX not in history:
            return 0, 0, 0, False
        metric_col = METRIC_JAX
    
    valid_history = history.dropna(subset=[metric_col]).sort_values('_step')
    
    if valid_history.empty:
        return 0, 0, 0, False
    
    max_step = int(valid_history['_step'].max())
    n_logged_points = len(valid_history)
    actual_total = valid_history[metric_col].sum()
    
    if max_step >= target_steps:
        return actual_total, actual_total, max_step, False
    
    cutoff_step = max_step - window
    recent_history = valid_history[valid_history['_step'] > cutoff_step]
    
    if not recent_history.empty:
        avg_time_per_iter = recent_history[metric_col].mean()
    else:
        avg_time_per_iter = valid_history[metric_col].mean()
        
    if n_logged_points > 1:
        steps_per_point = max_step / n_logged_points
        missing_steps = target_steps - max_step
        missing_points = int(missing_steps / steps_per_point)
    else:
        missing_points = 0
    
    estimated_missing_time = avg_time_per_iter * missing_points
    estimated_total = actual_total + estimated_missing_time
    
    return estimated_total, actual_total, max_step, True


def parse_run_info(run, param_prefix, verbose=False, use_wallclock=True):
    """
    Extract relevant info from a run based on specific tag prefixes.
    Expected tags:
    - framework_<name>
    - <param_prefix><value>
    
    Args:
        use_wallclock: If True, use WandB's _runtime for total time (recommended for fair comparison).
                      If False, use logged time metrics (only measures computation, not JIT/overhead).
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
    
    # Only process legacy (PyTorch) and jax-full frameworks
    if framework not in [FRAMEWORK_TORCH, FRAMEWORK_JAX]:
        return None

    # Get wall-clock runtime from WandB summary
    wallclock_time = run.summary.get("_runtime", 0)
    
    # Fetch history with appropriate metrics based on framework
    if framework == FRAMEWORK_TORCH:
        history_table = run.history(samples=5000, keys=[METRIC_TORCH_TRAIN, METRIC_TORCH_SAMPLE, "_step"])
        history_graph = run.history(samples=500, keys=[METRIC_TORCH_TRAIN, METRIC_TORCH_SAMPLE, "_step"])
        
        # Combine metrics for PyTorch
        if not history_table.empty:
            history_table[METRIC_NAME_COMBINED] = history_table[METRIC_TORCH_TRAIN].fillna(0) + history_table[METRIC_TORCH_SAMPLE].fillna(0)
        if not history_graph.empty:
            history_graph[METRIC_NAME_COMBINED] = history_graph[METRIC_TORCH_TRAIN].fillna(0) + history_graph[METRIC_TORCH_SAMPLE].fillna(0)
    else:
        history_table = run.history(samples=5000, keys=[METRIC_JAX, "_step"])
        history_graph = run.history(samples=500, keys=[METRIC_JAX, "_step"])
        
        # Rename JAX metric for consistency
        if not history_table.empty:
            history_table[METRIC_NAME_COMBINED] = history_table[METRIC_JAX]
        if not history_graph.empty:
            history_graph[METRIC_NAME_COMBINED] = history_graph[METRIC_JAX]
    
    # Calculate total time (estimated if crashed)
    estimated_total, actual_total, max_step, was_estimated = estimate_total_time(
        history_table, framework, target_steps=TARGET_STEPS, verbose=verbose
    )
    
    # Use wall-clock time if requested (recommended for fair JAX comparison)
    if use_wallclock:
        total_time = wallclock_time
    else:
        total_time = estimated_total

    # Map framework names for display
    display_framework = "PyTorch" if framework == FRAMEWORK_TORCH else "Full JAX"

    return {
        "id": run.id,
        "name": run.name,
        "framework": display_framework,
        "framework_raw": framework,
        "parameter": param_value,
        "history": history_graph,
        "total_time": total_time,
        "wallclock_time": wallclock_time,
        "logged_time": estimated_total,
        "actual_time": actual_total,
        "max_step": max_step,
        "was_estimated": was_estimated,
        "n_logged_points": len(history_table.dropna(subset=[METRIC_NAME_COMBINED])) if METRIC_NAME_COMBINED in history_table else 0,
    }

def process_experiment_group(group_name, exp_tag, param_prefix, has_failed_experiments=False, verbose=False):
    api = wandb.Api()
    project_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    
    if has_failed_experiments:
        runs = get_run_partially_failed(api, project_path, exp_tag)
    else:
        runs = get_runs(api, project_path, exp_tag)
    
    data = []
    for run in runs:
        info = parse_run_info(run, param_prefix, verbose=verbose, use_wallclock=True)
        if info:
            data.append(info)
            
    if not data:
        print(f"No valid runs found for {group_name}")
        return

    # Create output directory: experiments/full_jax/<group_name>/
    output_dir = os.path.join("experiments", OUTPUT_FOLDER, group_name)
    os.makedirs(output_dir, exist_ok=True)
    
    filename_suffix = param_prefix.rstrip("_")

    # ========== COMBINED SUMMARY TABLE ==========
    df_combined_rows = []
    
    try:
        sorted_params = sorted(set(item["parameter"] for item in data), key=lambda x: float(x))
    except ValueError:
        sorted_params = sorted(set(item["parameter"] for item in data))
    
    grouped_wallclock = defaultdict(lambda: defaultdict(list))
    grouped_logged = defaultdict(lambda: defaultdict(list))
    
    for item in data:
        grouped_wallclock[item["parameter"]][item["framework"]].append(item["wallclock_time"])
        grouped_logged[item["parameter"]][item["framework"]].append(item["logged_time"])
    
    for param in sorted_params:
        row = {"Parameter": param}
        
        # Wall-clock and logged times for each framework
        for fw in ["PyTorch", "Full JAX"]:
            wc_times = grouped_wallclock[param].get(fw, [])
            log_times = grouped_logged[param].get(fw, [])
            
            if wc_times:
                wc_mean = np.mean(wc_times)
                wc_std = np.std(wc_times)
                row[f"{fw} Wallclock Mean (s)"] = wc_mean
                row[f"{fw} Wallclock Std (s)"] = wc_std
            else:
                row[f"{fw} Wallclock Mean (s)"] = np.nan
                row[f"{fw} Wallclock Std (s)"] = np.nan
                wc_mean = np.nan
                
            if log_times:
                log_mean = np.mean(log_times)
                log_std = np.std(log_times)
                row[f"{fw} Logged Mean (s)"] = log_mean
                row[f"{fw} Logged Std (s)"] = log_std
            else:
                row[f"{fw} Logged Mean (s)"] = np.nan
                row[f"{fw} Logged Std (s)"] = np.nan
                log_mean = np.nan
            
            # Calculate Python overhead percentage: (wallclock - logged) / wallclock
            if wc_mean and log_mean and not np.isnan(wc_mean) and not np.isnan(log_mean) and wc_mean > 0:
                overhead = (wc_mean - log_mean) / wc_mean
                row[f"{fw} Python Overhead (%)"] = overhead * 100
            else:
                row[f"{fw} Python Overhead (%)"] = np.nan
        
        # Speedup ratios
        pt_wc = grouped_wallclock[param].get("PyTorch", [])
        jax_wc = grouped_wallclock[param].get("Full JAX", [])
        pt_log = grouped_logged[param].get("PyTorch", [])
        jax_log = grouped_logged[param].get("Full JAX", [])
        
        if pt_wc and jax_wc and np.mean(jax_wc) > 0:
            row["Speedup (Wallclock)"] = np.mean(pt_wc) / np.mean(jax_wc)
        else:
            row["Speedup (Wallclock)"] = np.nan
            
        if pt_log and jax_log and np.mean(jax_log) > 0:
            row["Speedup (Logged)"] = np.mean(pt_log) / np.mean(jax_log)
        else:
            row["Speedup (Logged)"] = np.nan
        
        df_combined_rows.append(row)
    
    df_combined = pd.DataFrame(df_combined_rows)
    combined_csv_path = os.path.join(output_dir, f"results_{filename_suffix}.csv")
    df_combined.to_csv(combined_csv_path, index=False)
    print(f"Saved results to {combined_csv_path}")
    
    print(f"\n{group_name} - Summary (Wallclock vs Logged Time with Python Overhead):")
    print(df_combined.to_string(index=False))
    print()

    # Build a lookup for scaling factors: (run_id) -> (wallclock_time / logged_time)
    # This ensures plot data is consistent with the CSV table totals
    scale_factors = {}
    for item in data:
        if item["logged_time"] > 0:
            scale_factors[item["id"]] = item["wallclock_time"] / item["logged_time"]
        else:
            scale_factors[item["id"]] = 1.0

    all_frames = []
    for item in data:
        h = item["history"].copy()
        h["Framework"] = item["framework"]
        h["Parameter"] = item["parameter"]
        h["run_id"] = item["id"]
        
        # Scale the time metric so that sum matches wall-clock time
        scale = scale_factors.get(item["id"], 1.0)
        h[METRIC_NAME_COMBINED] = h[METRIC_NAME_COMBINED] * scale
        
        all_frames.append(h)
        
    if not all_frames:
        return

    full_df = pd.concat(all_frames)
    
    try:
        full_df["Parameter"] = pd.to_numeric(full_df["Parameter"])
    except ValueError:
        pass

    min_max_step = min(item["max_step"] for item in data)
    
    full_df[METRIC_NAME_SMOOTH] = (
        full_df
        .groupby(["Parameter", "Framework", "run_id"])[METRIC_NAME_COMBINED]
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
    
    plt.title(f"Training Speed Comparison: {group_name} (Full)\nPyTorch vs Full JAX (Wall-clock scaled)")
    plt.xlabel("Iteration")
    plt.ylabel("Time per Iteration (s) - Wall-clock scaled")
    plt.yscale("log")
    
    output_plot_path = os.path.join(output_dir, f"training_curve_{filename_suffix}_full.png")
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Saved full plot to {output_plot_path}")

    # ========== PLOT 2: TRUNCATED PLOT (0 to min_max_step) ==========
    truncated_df = full_df[full_df["_step"] <= min_max_step].copy()

    truncated_df[METRIC_NAME_SMOOTH] = (
        truncated_df
        .groupby(["Parameter", "Framework", "run_id"])[METRIC_NAME_COMBINED]
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
    
    plt.title(f"Training Speed Comparison: {group_name} (Truncated to {min_max_step} steps)\nPyTorch vs Full JAX (Wall-clock scaled)")
    plt.xlabel("Iteration")
    plt.ylabel("Time per Iteration (s) - Wall-clock scaled")
    plt.yscale("log")
    
    output_plot_path = os.path.join(output_dir, f"training_curve_{filename_suffix}_truncated.png")
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Saved truncated plot to {output_plot_path} (truncated at step {min_max_step})")

if __name__ == "__main__":

    #! Neural Networks Experiments
    process_experiment_group("neural-nets", "exp_mlp", "nlayers_")
    process_experiment_group("neural-nets", "exp_mlp", "nhid_")
    
    #! Environment Experiments
    process_experiment_group("environment", "exp_env", "length_")
    process_experiment_group("environment", "exp_env", "dim_")
    # process_experiment_group("environment", "exp_env", "env_")
    
    #! Hyperparameters Experiments
    #process_experiment_group("hyperparameters", "exp_hyperparam", "batchsize_")
    #process_experiment_group("hyperparameters", "exp_hyperparam", "trajectory_")
    
    #! Hardware Experiments
    #process_experiment_group("hardware", "exp_hardware", "hardware_", has_failed_experiments=True)
    
    
