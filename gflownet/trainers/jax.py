"""
JAX trainer implementation (pure functional style).
"""

from typing import Any, Tuple
from dataclasses import dataclass, replace
import jax
import jax.numpy as jnp
import jax.random as random
from tqdm import tqdm
import time
import gc

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

def train_step(state: TrainingState, batch: Any) -> Tuple[TrainingState, dict]:
    """
    Pure function to perform one training iteration in JAX.
    Returns updated TrainingState and a dict of metrics/side effects.
    """
    metrics = {}
    t0_iter = time.time()
    
    # IMPURE: Evaluation triggers I/O logging side effects
    if state.evaluator.should_eval(state.iteration):
        eval_metrics = state.evaluator.eval_and_log(state.iteration)  # IMPURE: I/O logging
        metrics.update({"eval": eval_metrics})
    
    # IMPURE: Top-k evaluation triggers I/O logging side effects
    if state.evaluator.should_eval_top_k(state.iteration):
        top_k_metrics = state.evaluator.eval_and_log_top_k(state.iteration)  # IMPURE: I/O logging
        metrics.update({"top_k": top_k_metrics})

    # Sample batches (currently delegates to agent - IMPURE due to agent's internal state)
    updated_batch = batch
    for j in range(state.sttr):
        # IMPURE: sample_batch likely mutates agent's internal state (env_cache, rng, etc.)
        sub_batch, times = state.agent.sample_batch(  # IMPURE: mutation of agent state
            n_forward=state.batch_size.forward,
            n_train=state.batch_size.backward_dataset,
            n_replay=state.batch_size.backward_replay,
            collect_forwards_masks=True,
            collect_backwards_masks=state.loss.requires_backward_policy(),
        )
        # IMPURE: merge likely mutates updated_batch in-place
        updated_batch.merge(sub_batch)  # IMPURE: in-place mutation
        metrics.update({"sample_times": times})
    
    # Compute losses and perform gradient updates
    for j in range(state.ttsr):
        # Compute losses (this part can be pure)
        losses = state.loss.compute(updated_batch, get_sublosses=True)
        
        # Check if losses are finite
        if not jnp.all(jnp.array([jnp.isfinite(loss) for loss in losses.values()])):
            # IMPURE: print is I/O
            if state.logger.debug:
                print("Loss is not finite - skipping iteration")  # IMPURE: I/O
            metrics.update({"skipped": True})
            break
        else:
            # JAX functional gradient computation
            # TODO: Replace PyTorch-style backward() with JAX gradient computation
            # Example pure approach:
            # def loss_fn(params):
            #     return state.loss.compute_with_params(params, updated_batch)["all"]
            # grads = jax.grad(loss_fn)(state.agent.get_params())
            
            # IMPURE: PyTorch-style backward (need to replace with JAX grad)
            # losses["all"].backward()  # IMPURE: mutation of gradients
            
            # TODO: Implement JAX gradient computation here
            # For now, placeholder showing the pure JAX pattern:
            def loss_fn(params):
                # This should compute loss given params
                # IMPURE: Currently uses state.loss.compute which may have side effects
                return losses["all"]  # Placeholder - need to recompute with params
            
            # Get current parameters from agent
            # IMPURE: Assumes agent has get_params() method
            params = state.agent.get_params()  # IMPURE: accessing mutable agent state
            
            # Compute gradients (pure in JAX)
            grads = jax.grad(loss_fn)(params)
            
            # Clip gradients if needed (pure operation)
            if state.clip_grad_norm > 0:
                grads = optax.clip_by_global_norm(state.clip_grad_norm)(grads)[0]
            
            # Apply optimizer update (pure in JAX with Optax)
            # IMPURE: optimizer is PyTorch optimizer, need Optax
            # updates, new_opt_state = state.optimizer.update(grads, state.optimizer_state)
            # new_params = optax.apply_updates(params, updates)
            
            # TODO: Replace with pure Optax updates
            # For now, placeholder:
            # state.optimizer.step()  # IMPURE: mutation
            # state.lr_scheduler.step()  # IMPURE: mutation
            # state.optimizer.zero_grad()  # IMPURE: mutation
            
            # IMPURE: zero_logprobs likely mutates batch
            updated_batch.zero_logprobs()  # IMPURE: in-place mutation
    
    # Log training iteration
    # IMPURE: buffer.add() may mutate buffer
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
    # IMPURE: log_time performs I/O
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
    return TrainingState(
        agent=agent,
        iteration=agent.it,
        n_train_steps=agent.n_train_steps,
        sttr=agent.sttr,
        ttsr=agent.ttsr,
        batch_size=agent.batch_size,
        buffer=agent.buffer,
        loss=agent.loss,
        optimizer_state=agent.opt_state,  # Optax optimizer state
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
        rng_key=random.PRNGKey(config.seed),  # JAX RNG key initialization
    )

def create_batch(agent: Any) -> Any:
    """
    Helper to create an initial batch.
    TODO: Make JAX-compatible (use JAX arrays instead of PyTorch tensors)
    """
    # IMPURE: Assumes Batch is JAX-compatible
    from gflownet.utils.batch import Batch  # IMPURE: may use PyTorch internals
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
    
    # IMPURE: tqdm progress bar is I/O
    pbar = tqdm(  # IMPURE: I/O (terminal output)
        initial=state.iteration - 1,
        total=state.n_train_steps,
        disable=state.logger.progressbar["skip"],
    )
    
    # Update state with pbar
    state = replace(state, pbar=pbar)
    
    # Training loop
    while state.iteration <= state.n_train_steps:
        # Create batch for this iteration
        # IMPURE: create_batch may use PyTorch/have side effects
        batch = create_batch(agent)  # IMPURE: batch creation
        
        # Call the train_step
        state, metrics = train_step(state, batch)
        
        # Handle garbage collection
        # IMPURE: gc.collect() is a system side effect
        if metrics.get("gc_triggered"):
            del batch
            gc.collect()  # IMPURE: system side effect
            # Note: JAX uses XLA memory management, may not need manual GC
        
        # Handle early stopping
        if metrics.get("early_stop"):
            # IMPURE: print is I/O
            print(  # IMPURE: I/O
                "Ending training after meeting early stopping criteria: "
                f"{state.loss.loss_ema} < {state.loss.early_stopping_th}"
            )
            break
        
        # Handle progress bar update
        # IMPURE: progressbar_update performs I/O
        if "log" in metrics and "progressbar_update" in metrics["log"]:
            loss_val, rewards_list = metrics["log"]["progressbar_update"]
            state.logger.progressbar_update(  # IMPURE: I/O (terminal output)
                pbar, loss_val, rewards_list, agent.jsd, state.use_context
            )
        
        # IMPURE: pbar.update is I/O
        pbar.update(1)  # IMPURE: I/O (terminal output)
    
    # Final checkpoint save
    # IMPURE: save_checkpoint performs file I/O
    state.logger.save_checkpoint(  # IMPURE: file I/O
        forward_policy=agent.forward_policy,
        backward_policy=agent.backward_policy,
        state_flow=agent.state_flow,
        logZ=agent.logZ,
        optimizer=state.optimizer_state,
        buffer=state.buffer,
        step=state.iteration,
        final=True,
    )
    
    # Close logger
    # IMPURE: logger.end() performs I/O cleanup
    if not state.use_context:
        state.logger.end()  # IMPURE: I/O cleanup
    
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
    
    # Get terminating states and proxy values
    states_term = batch.get_terminating_states(sort_by="trajectory")
    proxy_vals = batch.get_terminating_proxy_values(sort_by="trajectory")
    
    # Handle rewards (convert to JAX arrays)
    if batch.rewards_available(log=False):
        rewards = batch.get_terminating_rewards(sort_by="trajectory")
    if batch.rewards_available(log=True):
        logrewards = batch.get_terminating_rewards(sort_by="trajectory", log=True)
    if not batch.rewards_available(log=False):
        rewards = jnp.exp(logrewards)  # JAX exp
    if not batch.rewards_available(log=True):
        logrewards = jnp.log(rewards)  # JAX log
    
    # Update buffers
    # IMPURE: buffer.add() may mutate buffer instead of returning new one
    actions_trajectories = batch.get_actions_trajectories()
    updated_buffer = buffer
    if buffer.use_main_buffer:
        updated_buffer = buffer.add(  # IMPURE: may mutate buffer
            states_term, actions_trajectories, rewards, iteration, buffer="main"
        )
    updated_buffer = updated_buffer.add(  # IMPURE: may mutate buffer
        states_term, actions_trajectories, rewards, iteration, buffer="replay"
    )
    
    t1_buffer = time.time()
    times.update({"buffer": t1_buffer - t0_buffer})
    
    # Compute metrics for logging
    t0_log = time.time()
    if evaluator.should_log_train(iteration):
        # Compute trajectory statistics
        _, trajectory_lengths = jnp.unique(
            batch.get_trajectory_indices(), return_counts=True
        )
        traj_length_mean = jnp.mean(trajectory_lengths.astype(jnp.float32))
        
        # Collect metrics
        metrics.update({
            "rewards": rewards,
            "logrewards": logrewards,
            "proxy_vals": proxy_vals,
            "traj_length_mean": traj_length_mean,
        })
    
    t1_log = time.time()
    times.update({"log": t1_log - t0_log})
    
    # Progress bar update info
    if pbar is not None:
        metrics.update({
            "progressbar_update": (
                float(losses["all"]),  # Convert to Python float for compatibility
                rewards.tolist() if hasattr(rewards, 'tolist') else list(rewards)
            )
        })
    
    # Checkpoint saving flag
    if evaluator.should_checkpoint(iteration):
        metrics.update({"save_checkpoint": True})
    
    return updated_buffer, metrics