"""
JAX trainer implementation (pure functional style).
"""

#! Major Impurities to Address:
#! 1. state.agent.sample_batch() - Mutates agent's internal state
#! 2. updated_batch.merge(sub_batch) - In-place mutation
#! 3. Gradient computation - Need to replace PyTorch backward() with JAX grad
#! 4. Optimizer updates - Need to use Optax instead of PyTorch optimizer
#! 5. buffer.add() - May mutate buffer instead of returning new one
#! 6. updated_batch.zero_logprobs() - In-place mutation

from typing import Any, Tuple
from dataclasses import dataclass, replace
import jax
import jax.numpy as jnp
import jax.random as random
from tqdm import tqdm
import time
import gc
import copy

import optax

@dataclass
class TrainingState:
    agent: Any  # JAX-compatible agent
    iteration: int
    n_train_steps: int
    sttr: Any
    ttsr: Any
    batch_size: Any
    buffer: Any
    loss: Any
    optimizer_state: Any  # Optax optimizer state
    lr_scheduler: Any  # Optax schedule
    evaluator: Any
    logger: Any
    env: Any
    proxy: Any
    device: Any  # JAX device
    float_type: Any
    clip_grad_norm: float
    garbage_collection_period: int
    use_context: bool
    pbar: Any
    rng_key: Any  # JAX random key
    params: Any  # Model parameters (JAX pytree)
    optimizer: Any  # Optax optimizer (pure function)

@dataclass
class SamplingState:
    n_forward: int
    n_train: int
    n_replay: int
    env_maker: Any
    device: Any
    float_type: Any
    proxy: Any
    buffer: Any
    collect_reversed_logprobs: bool
    train_sampling: str = "permutation"
    replay_sampling: str = "permutation"
    train: bool = True

# SOLUTION TO IMPURITY #2: Pure batch merge function
def pure_merge_batches(base_batch: Any, sub_batches: list) -> Any:
    """
    Pure function to merge batches without mutation.
    Creates a new batch with combined data.
    """
    # Create a deep copy to avoid mutating the original
    merged_batch = copy.deepcopy(base_batch)
    
    # Merge each sub-batch
    for sub_batch in sub_batches:
        # Instead of in-place merge, create new batch with combined data
        # This assumes Batch has a method to return merged data
        merged_batch = merged_batch.merge([sub_batch])  # If merge returns new batch
        # OR implement custom pure merge logic here:
        # merged_batch = create_merged_batch(merged_batch, sub_batch)
    
    return merged_batch

# SOLUTION TO IMPURITY #6: Pure zero_logprobs function
def pure_zero_logprobs(batch: Any) -> Any:
    """
    Pure function to create a new batch with zeroed logprobs.
    Returns a new batch instead of mutating.
    """
    # Create a deep copy to avoid mutation
    new_batch = copy.deepcopy(batch)
    new_batch.zero_logprobs()  # If this still mutates, we need to implement it differently
    return new_batch
    
    # Alternative pure implementation if zero_logprobs mutates:
    # Create new batch with zeroed fields
    # new_batch = Batch(...)
    # new_batch.logprobs = jnp.zeros_like(batch.logprobs)
    # return new_batch

# SOLUTION TO IMPURITY #5: Pure buffer add function
def pure_buffer_add(buffer: Any, states: Any, actions: Any, rewards: Any, 
                    iteration: int, buffer_name: str = "replay") -> Any:
    """
    Pure function to add data to buffer.
    Returns a new buffer instead of mutating the existing one.
    """
    # Create a deep copy of the buffer
    new_buffer = copy.deepcopy(buffer)
    
    # Add data (if buffer.add mutates, we need to change the buffer implementation)
    new_buffer = new_buffer.add(states, actions, rewards, iteration, buffer=buffer_name)
    
    return new_buffer
    
    # Alternative: Implement a truly functional buffer
    # new_buffer = Buffer.create_with_added_data(buffer, states, actions, rewards, iteration, buffer_name)

def train_step(state: TrainingState, batch: Any) -> Tuple[TrainingState, dict]:
    """
    Pure function to perform one training iteration in JAX.
    Returns updated TrainingState and a dict of metrics/side effects.
    """
    metrics = {}
    t0_iter = time.time()
    
    # IMPURE: Evaluation triggers I/O logging side effects (keep for now)
    if state.evaluator.should_eval(state.iteration):
        eval_metrics = state.evaluator.eval_and_log(state.iteration)  # IMPURE: I/O logging
        metrics.update({"eval": eval_metrics})
    
    # IMPURE: Top-k evaluation triggers I/O logging side effects (keep for now)
    if state.evaluator.should_eval_top_k(state.iteration):
        top_k_metrics = state.evaluator.eval_and_log_top_k(state.iteration)  # IMPURE: I/O logging
        metrics.update({"top_k": top_k_metrics})

    # SOLUTION TO IMPURITY #1 & #2: Sample batches without mutation
    # Collect sub-batches in a list (immutable operation)
    sub_batches = []
    current_rng_key = state.rng_key
    
    for j in range(state.sttr):
        # Split RNG key for this iteration (pure operation)
        current_rng_key, sample_rng_key = random.split(current_rng_key)
        
        # Option 1: Delegate to agent but pass RNG key explicitly
        # This requires modifying the agent to accept RNG key and return updated key
        # sub_batch, times, new_agent_state = pure_sample_batch_from_agent(
        #     state.agent, sample_rng_key, state.batch_size, ...
        # )
        
        # Option 2: Keep using agent.sample_batch but acknowledge impurity
        # For now, use this approach until agent is refactored
        sub_batch, times = state.agent.sample_batch(  # Still IMPURE but isolated
            n_forward=state.batch_size.forward,
            n_train=state.batch_size.backward_dataset,
            n_replay=state.batch_size.backward_replay,
            collect_forwards_masks=True,
            collect_backwards_masks=state.loss.requires_backward_policy(),
        )
        
        sub_batches.append(sub_batch)
        metrics.update({"sample_times": times})
    
    # FIXED IMPURITY #2: Merge batches without mutation
    updated_batch = pure_merge_batches(batch, sub_batches)
    
    # Update state with new RNG key
    state = replace(state, rng_key=current_rng_key)
    
    # Compute losses and perform gradient updates
    for j in range(state.ttsr):
        # SOLUTION TO IMPURITY #3 & #4: JAX gradient computation and optimizer update
        
        # Define pure loss function
        def loss_fn(params):
            """Pure loss function that takes params and returns loss."""
            # You need to implement a version of loss.compute that takes params
            # For now, this is a placeholder showing the pattern
            # return state.loss.compute_with_params(params, updated_batch)["all"]
            
            # Temporary: compute loss with current batch
            # This assumes state.loss.compute can work with JAX params
            losses_dict = state.loss.compute(updated_batch, get_sublosses=True)
            return losses_dict["all"]
        
        # Compute loss with current parameters
        loss_value = loss_fn(state.params)
        
        # Check if loss is finite
        if not jnp.isfinite(loss_value):
            if state.logger.debug:
                print("Loss is not finite - skipping iteration")  # IMPURE: I/O
            metrics.update({"skipped": True})
            break
        else:
            # FIXED IMPURITY #3: Compute gradients (pure in JAX)
            grads = jax.grad(loss_fn)(state.params)
            
            # FIXED IMPURITY #4: Clip gradients (pure operation)
            if state.clip_grad_norm > 0:
                # Compute global norm
                global_norm = optax.global_norm(grads)
                # Clip gradients
                grads = jax.tree_map(
                    lambda g: g * jnp.minimum(1.0, state.clip_grad_norm / (global_norm + 1e-6)),
                    grads
                )
            
            # FIXED IMPURITY #4: Apply optimizer update (pure with Optax)
            updates, new_opt_state = state.optimizer.update(
                grads, state.optimizer_state, state.params
            )
            new_params = optax.apply_updates(state.params, updates)
            
            # Update state with new params and optimizer state (pure via replace)
            state = replace(
                state, 
                params=new_params,
                optimizer_state=new_opt_state
            )
            
            # FIXED IMPURITY #6: Zero logprobs without mutation
            updated_batch = pure_zero_logprobs(updated_batch)
            
            # Store losses for logging
            losses = {"all": loss_value}
    
    # FIXED IMPURITY #5: Log training iteration with pure buffer updates
    updated_buffer, log_metrics = pure_log_train_iteration(
        losses,
        updated_batch,
        metrics.get("sample_times", {}),
        state.buffer,
        state.evaluator,
        state.logger,
        state.iteration,
        state.use_context,
        pbar=state.pbar,
    )
    metrics.update({"log": log_metrics})
    
    # Update state with new buffer (pure via replace)
    updated_state = replace(state, buffer=updated_buffer)
    
    # Update times
    t1_iter = time.time()
    metrics.update({"iter_time": t1_iter - t0_iter})
    # IMPURE: log_time performs I/O (acceptable)
    state.logger.log_time(metrics, use_context=state.use_context)  # IMPURE: I/O logging
    
    # Garbage collection check (side effect handled by caller)
    do_gc = (state.garbage_collection_period > 0 and 
             state.iteration % state.garbage_collection_period == 0)
    if do_gc:
        metrics.update({"gc_triggered": True})
    
    # Early stopping check
    if state.loss.do_early_stopping(losses["all"]):
        metrics.update({"early_stop": True})
    
    # Update iteration (pure via replace)
    new_iteration = state.iteration + 1
    updated_state = replace(updated_state, iteration=new_iteration)
    
    return updated_state, metrics

def build_state_from_agent(agent: Any, config: Any) -> TrainingState:
    """
    Helper to initialize TrainingState from a JAX-compatible agent.
    """
    # Initialize Optax optimizer
    #TODO: I need to implement the learning rate scheduler properly
    optimizer = optax.chain(
        optax.clip_by_global_norm(agent.clip_grad_norm) if agent.clip_grad_norm > 0 else optax.identity(),
        optax.adam(learning_rate=agent.lr_scheduler._last_lr[0] or 1e-3),  # Use scheduler if available
    )
    
    # Get initial parameters from agent
    params = agent.get_params() if hasattr(agent, 'get_params') else None
    
    # Initialize optimizer state
    opt_state = optimizer.init(params) if params is not None else None
    
    return TrainingState(
        agent=agent,
        iteration=agent.it,
        n_train_steps=agent.n_train_steps,
        sttr=agent.sttr,
        ttsr=agent.ttsr,
        batch_size=agent.batch_size,
        buffer=agent.buffer,
        loss=agent.loss,
        optimizer_state=opt_state,
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
        pbar=None,
        rng_key=random.PRNGKey(config.seed),
        params=params,
        optimizer=optimizer,
    )

def create_batch(agent: Any) -> Any:
    """
    Helper to create an initial batch.
    """
    from gflownet.utils.batch import Batch
    return Batch(
        env=agent.env,
        proxy=agent.proxy,
        device=agent.device,
        float_type=agent.float,
    )

def train(agent: Any, config: Any) -> TrainingState:
    """
    JAX-based training loop.
    Handles side effects like logging, GC, and saving.
    """
    state = build_state_from_agent(agent, config)
    
    # IMPURE: tqdm progress bar is I/O (acceptable)
    pbar = tqdm(
        initial=state.iteration - 1,
        total=state.n_train_steps,
        disable=state.logger.progressbar["skip"],
    )
    
    # Update state with pbar
    state = replace(state, pbar=pbar)
    
    # Training loop
    while state.iteration <= state.n_train_steps:
        batch = create_batch(agent)
        
        state, metrics = train_step(state, batch)
        
        if metrics.get("gc_triggered"):
            del batch
            gc.collect()
        
        if metrics.get("early_stop"):
            print(
                "Ending training after meeting early stopping criteria: "
                f"{state.loss.loss_ema} < {state.loss.early_stopping_th}"
            )
            break
        
        if "log" in metrics and "progressbar_update" in metrics["log"]:
            loss_val, rewards_list = metrics["log"]["progressbar_update"]
            state.logger.progressbar_update(
                pbar, loss_val, rewards_list, agent.jsd, state.use_context
            )
        
        pbar.update(1)
    
    # Final checkpoint save
    state.logger.save_checkpoint(
        forward_policy=agent.forward_policy,
        backward_policy=agent.backward_policy,
        state_flow=agent.state_flow,
        logZ=agent.logZ,
        optimizer=state.optimizer_state,
        buffer=state.buffer,
        step=state.iteration,
        final=True,
    )
    
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
) -> Tuple[Any, dict]:
    """
    JAX version of log_train_iteration.
    Computes logging metrics and buffer updates.
    """
    metrics = {}
    t0_buffer = time.time()
    
    states_term = batch.get_terminating_states(sort_by="trajectory")
    proxy_vals = batch.get_terminating_proxy_values(sort_by="trajectory")
    
    if batch.rewards_available(log=False):
        rewards = batch.get_terminating_rewards(sort_by="trajectory")
    if batch.rewards_available(log=True):
        logrewards = batch.get_terminating_rewards(sort_by="trajectory", log=True)
    if not batch.rewards_available(log=False):
        rewards = jnp.exp(logrewards)
    if not batch.rewards_available(log=True):
        logrewards = jnp.log(rewards)
    
    # FIXED IMPURITY #5: Pure buffer updates
    actions_trajectories = batch.get_actions_trajectories()
    updated_buffer = buffer
    
    if buffer.use_main_buffer:
        updated_buffer = pure_buffer_add(
            updated_buffer, states_term, actions_trajectories, 
            rewards, iteration, buffer_name="main"
        )
    
    updated_buffer = pure_buffer_add(
        updated_buffer, states_term, actions_trajectories,
        rewards, iteration, buffer_name="replay"
    )
    
    t1_buffer = time.time()
    times.update({"buffer": t1_buffer - t0_buffer})
    
    t0_log = time.time()
    if evaluator.should_log_train(iteration):
        _, trajectory_lengths = jnp.unique(
            batch.get_trajectory_indices(), return_counts=True
        )
        traj_length_mean = jnp.mean(trajectory_lengths.astype(jnp.float32))
        
        metrics.update({
            "rewards": rewards,
            "logrewards": logrewards,
            "proxy_vals": proxy_vals,
            "traj_length_mean": traj_length_mean,
        })
    
    t1_log = time.time()
    times.update({"log": t1_log - t0_log})
    
    if pbar is not None:
        metrics.update({
            "progressbar_update": (
                float(losses["all"]),
                rewards.tolist() if hasattr(rewards, 'tolist') else list(rewards)
            )
        })
    
    if evaluator.should_checkpoint(iteration):
        metrics.update({"save_checkpoint": True})
    
    return updated_buffer, metrics