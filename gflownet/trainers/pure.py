from typing import Any, Tuple
from dataclasses import dataclass, replace
from tqdm import tqdm  
import time
import torch
import gc

@dataclass
class TrainingState:
    agent: Any  
    iteration: int
    n_train_steps: int
    sttr: Any
    ttsr: Any
    batch_size: int
    buffer: Any
    loss: Any
    optimizer: Any
    lr_scheduler: Any
    evaluator: Any
    logger: Any
    env: Any
    proxy: Any
    device: Any
    float_type: Any
    clip_grad_norm: float
    garbage_collection_period: int
    use_context: bool

def train_step(state: TrainingState, batch: Any) -> Tuple[TrainingState, dict]:
    """
    Pure function to perform one training iteration.
    Returns updated TrainingState and a dict of metrics/side effects.
    """
    metrics = {}
    t0_iter = time.time()
    
    # Test and log (side effects handled by returning metrics)
    if state.evaluator.should_eval(state.iteration):
        eval_metrics = state.evaluator.eval_and_log(state.iteration)
        metrics.update({"eval": eval_metrics})
    if state.evaluator.should_eval_top_k(state.iteration):
        top_k_metrics = state.evaluator.eval_and_log_top_k(state.iteration)
        metrics.update({"top_k": top_k_metrics})
    
    # Sample sub-batches and merge into batch
    updated_batch = batch  # Assume batch is passed in; modify if needed
    for j in range(state.sttr):
        sub_batch, times = state.agent.sample_batch(  # Assuming agent is accessible; adjust as needed
            n_forward=state.batch_size.forward,
            n_train=state.batch_size.backward_dataset,
            n_replay=state.batch_size.backward_replay,
            collect_forwards_masks=True,
            collect_backwards_masks=state.loss.requires_backward_policy(),  # Add to TrainingState if needed
        )
        updated_batch.merge(sub_batch)
        metrics.update({"sample_times": times})
    
    # Compute losses and backprop
    for j in range(state.ttsr):
        losses = state.loss.compute(updated_batch, get_sublosses=True)
        if not all([torch.isfinite(loss) for loss in losses.values()]):
            if state.logger.debug:
                print("Loss is not finite - skipping iteration")
            metrics.update({"skipped": True})
            break
        else:
            losses["all"].backward()
            if state.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(state.optimizer.param_groups, state.clip_grad_norm)  # Adjust for parameters
            state.optimizer.step()
            state.lr_scheduler.step()
            state.optimizer.zero_grad()
            updated_batch.zero_logprobs()
    
    # Log training iteration (return metrics instead of mutating)
    updated_buffer, log_metrics = pure_log_train_iteration(
        losses,
        updated_batch,
        metrics.get("sample_times", {}),
        state.buffer,
        state.evaluator,
        state.logger,
        state.iteration,
        state.use_context,
        pbar=None,  # Pass pbar if needed for progress bar updates
    )
    metrics.update({"log": log_metrics})
    
    updated_state = replace(state, buffer=updated_buffer)

    
    # Update times
    t1_iter = time.time()
    metrics.update({"iter_time": t1_iter - t0_iter})
    state.logger.log_time(metrics, use_context=state.use_context)
    
    # Garbage collection (return flag for caller to handle)
    do_gc = (state.garbage_collection_period > 0 and state.iteration % state.garbage_collection_period == 0)
    if do_gc:
        metrics.update({"gc_triggered": True})
    
    # Early stopping check
    if state.loss.do_early_stopping(losses["all"]):
        metrics.update({"early_stop": True})
    
    # Update iteration and return new state
    new_iteration = state.iteration + 1
    updated_state = replace(state, iteration=new_iteration)
    
    return updated_state, metrics

def build_state_from_agent(agent: Any) -> TrainingState:
    """
    Helper to initialize TrainingState from the agent.
    Adjust fields as needed based on agent's attributes.
    """
    return TrainingState(
        agent=agent,
        iteration=agent.it,
        n_train_steps=agent.n_train_steps,
        sttr=agent.sttr,
        ttsr=agent.ttsr,
        batch_size=agent.batch_size,
        buffer=agent.buffer,
        loss=agent.loss,
        optimizer=agent.opt,
        lr_scheduler=agent.lr_scheduler,
        evaluator=agent.evaluator,
        logger=agent.logger,
        env=agent.env,
        proxy=agent.proxy,
        device=agent.device,
        float_type=agent.float,
        clip_grad_norm=agent.clip_grad_norm,
        garbage_collection_period=agent.garbage_collection_period,
        use_context=agent.use_context,
    )

def create_batch(agent: Any) -> Any:
    """
    Helper to create an initial batch. Mirrors the Batch creation in the original train method.
    """
    from gflownet.utils.batch import Batch  # Import as needed
    return Batch(
        env=agent.env,
        proxy=agent.proxy,
        device=agent.device,
        float_type=agent.float,
    )

def train(agent: Any) -> TrainingState:
    """
    Pure-style training loop that calls train_step iteratively.
    Returns the final TrainingState after training.
    Handles side effects like logging, GC, and saving.
    """
    state = build_state_from_agent(agent)
    
    # Initialize progress bar (mirroring original)
    pbar = tqdm(
        initial=state.iteration - 1,
        total=state.n_train_steps,
        disable=state.logger.progressbar["skip"],
    )
    
    # Training loop
    while state.iteration <= state.n_train_steps:
        # Create batch for this iteration (adjust if batch should persist across iterations)
        batch = create_batch(agent)
        
        # Call the pure train_step
        state, metrics = train_step(state, batch)
        
        # Handle side effects from metrics
        if metrics.get("gc_triggered"):
            del batch
            gc.collect()
            torch.cuda.empty_cache()
        
        if metrics.get("early_stop"):
            print(
                "Ending training after meeting early stopping criteria: "
                f"{state.loss.loss_ema} < {state.loss.early_stopping_th}"
            )
            break
        
        # Update progress bar (assuming logger handles it via metrics)
        # If needed, call state.logger.progressbar_update(pbar, ...)
    
    # Final save (mirroring original)
    state.logger.save_checkpoint(
        forward_policy=agent.forward_policy,  # Access from agent
        backward_policy=agent.backward_policy,
        state_flow=agent.state_flow,
        logZ=agent.logZ,
        optimizer=state.optimizer,
        buffer=state.buffer,
        step=state.iteration,
        final=True,
    )
    # Close logger
    if not state.use_context:
        state.logger.end()
    
    return state

def pure_log_train_iteration(
    losses: dict,
    batch: Any,
    times: dict,
    buffer: Any,
    evaluator: Any,
    logger: Any,
    iteration: int,
    use_context: bool,
    pbar: Any = None,
) -> Tuple[Any, dict]:  # Returns updated buffer and metrics dict
    """
    Pure version of log_train_iteration.
    Computes logging metrics and buffer updates without side effects.
    Returns updated buffer and a dict of metrics for the caller to handle.
    """
    metrics = {}
    t0_buffer = time.time()
    
    states_term = batch.get_terminating_states(sort_by="trajectory")
    proxy_vals = batch.get_terminating_proxy_values(sort_by="trajectory")
    
    # Handle rewards (mirroring original logic)
    if batch.rewards_available(log=False):
        rewards = batch.get_terminating_rewards(sort_by="trajectory")
    if batch.rewards_available(log=True):
        logrewards = batch.get_terminating_rewards(sort_by="trajectory", log=True)
    if not batch.rewards_available(log=False):
        rewards = torch.exp(logrewards)
    if not batch.rewards_available(log=True):
        logrewards = torch.log(rewards)
    
    # Update buffers (return updated buffer instead of mutating)
    actions_trajectories = batch.get_actions_trajectories()
    updated_buffer = buffer  # Assume buffer is immutable or copy it
    if buffer.use_main_buffer:
        updated_buffer = buffer.add(  # Implement add to return new buffer
            states_term, actions_trajectories, rewards, iteration, buffer="main"
        )
    updated_buffer = updated_buffer.add(
        states_term, actions_trajectories, rewards, iteration, buffer="replay"
    )
    
    t1_buffer = time.time()
    times.update({"buffer": t1_buffer - t0_buffer})
    
    # Compute metrics for logging
    t0_log = time.time()
    if evaluator.should_log_train(iteration):
        # logZ, trajectory lengths, etc. (compute without logging)
        logz = None  # Compute if needed
        _, trajectory_lengths = torch.unique(batch.get_trajectory_indices(), return_counts=True)
        traj_length_mean = torch.mean(trajectory_lengths.float())
        # ... other metrics ...
        
        # Collect metrics in dict
        metrics.update({
            "rewards": rewards,
            "logrewards": logrewards,
            "proxy_vals": proxy_vals,
            "traj_length_mean": traj_length_mean,
            # ... other metrics ...
        })
    
    t1_log = time.time()
    times.update({"log": t1_log - t0_log})
    
    # Progress bar update (return info for caller)
    if pbar is not None:
        metrics.update({"progressbar_update": (losses["all"].item(), rewards.tolist())})
    
    # Checkpoint saving (return flag/info for caller)
    if evaluator.should_checkpoint(iteration):
        metrics.update({"save_checkpoint": True})
    
    return updated_buffer, metrics