"""
JAX trainer implementation leveraging JIT compilation and functional control flow.
Uses jax.lax.scan, jax.lax.cond, jax.jit for maximum performance.

This implementation mirrors the logic from pure.py and gflownet.py but uses JAX primitives
for performance. The goal is to JIT-compile as much as possible while maintaining
compatibility with the existing agent/env infrastructure.
"""

from typing import Any, Tuple, NamedTuple, Optional
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, lax
from functools import partial
import optax
from tqdm import tqdm
import time
import gc
from functools import reduce




# ============================================================================
# DATA STRUCTURES (JAX pytree-compatible)
# ============================================================================

class TrainingState(NamedTuple):
    """Use NamedTuple for JAX compatibility (pytrees)"""
    params: Any  # Model parameters (JAX pytree)
    optimizer_state: Any  # Optax optimizer state
    rng_key: Any  # JAX random key
    iteration: int
    loss_ema: float  # For early stopping
    # Non-JIT-able fields (buffer, agent, logger) passed separately


class StepMetrics(NamedTuple):
    """Metrics from a single training step"""
    loss: float
    grad_norm: float
    rewards_mean: float
    skipped: bool
    trajectory_length_mean: float


class BatchArrays(NamedTuple):
    """Fixed structure for batch data (JAX pytree-compatible)"""
    states: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    logprobs: jnp.ndarray
    logprobs_rev: jnp.ndarray
    trajectory_indices: jnp.ndarray
    # TODO: Add masks if needed for invalid action masking
    # masks_forward: jnp.ndarray
    # masks_backward: jnp.ndarray


# ============================================================================
# PURE JAX FUNCTIONS (JIT-compiled)
# ============================================================================

@jit
def pure_merge_batches_jax(base_batch: BatchArrays, sub_batch: BatchArrays) -> BatchArrays:
    """
    Pure function to merge batch arrays (JAX-compatible).
    Concatenates JAX arrays without mutation using JAX tree operations.
    """
    merged = jax.tree.map(
        lambda base, sub: jnp.concatenate([base, sub], axis=0),
        base_batch,
        sub_batch
    )
    return merged
    
    # merged_states = jnp.concatenate([base_batch.states, sub_batch.states], axis=0)
    # merged_actions = jnp.concatenate([base_batch.actions, sub_batch.actions], axis=0)
    # merged_rewards = jnp.concatenate([base_batch.rewards, sub_batch.rewards], axis=0)
    # merged_logprobs = jnp.concatenate([base_batch.logprobs, sub_batch.logprobs], axis=0)
    # merged_logprobs_rev = jnp.concatenate([base_batch.logprobs_rev, sub_batch.logprobs_rev], axis=0)
    # merged_trajectory_indices = jnp.concatenate([base_batch.trajectory_indices, sub_batch.trajectory_indices], axis=0)
    
    # return BatchArrays(
    #     states=merged_states,
    #     actions=merged_actions,
    #     rewards=merged_rewards,
    #     logprobs=merged_logprobs,
    #     logprobs_rev=merged_logprobs_rev,
    #     trajectory_indices=merged_trajectory_indices,
    # )


@jit
def pure_zero_logprobs_jax(batch_arrays: BatchArrays) -> BatchArrays:
    """
    Pure function to zero logprobs in batch.
    Returns new BatchArrays with zeroed logprob arrays.
    """
    zeroed_logprobs = jnp.zeros_like(batch_arrays.logprobs)
    zeroed_logprobs_rev = jnp.zeros_like(batch_arrays.logprobs_rev)
    
    return BatchArrays(
        states=batch_arrays.states,
        actions=batch_arrays.actions,
        rewards=batch_arrays.rewards,
        logprobs=zeroed_logprobs,
        logprobs_rev=zeroed_logprobs_rev,
        trajectory_indices=batch_arrays.trajectory_indices,
    )


def compute_trajectory_stats(trajectory_indices: jnp.ndarray) -> Tuple[float, int, int]:
    """
    Compute trajectory statistics: mean, min, max lengths.
    
    SIMPLIFIED VERSION: Just return placeholders for now to get training running.
    TODO: Implement proper trajectory stats computation outside the training loop.
    
    Args:
        trajectory_indices: Array of trajectory indices for each transition
    
    Returns:
        Tuple of (mean_length, min_length, max_length)
    """
    # PLACEHOLDER: Return simple estimates to avoid concretization errors
    # Mean trajectory length ≈ total transitions / estimated num trajectories
    total_transitions = trajectory_indices.shape[0]
    estimated_traj_length = jnp.float32(total_transitions / 10.0)  # Rough estimate
    
    return estimated_traj_length, jnp.int32(1), jnp.int32(100)


# ============================================================================
# CORE TRAINING STEP (JIT-compiled)
# ============================================================================

@partial(jit, static_argnames=['loss_fn', 'optimizer', 'clip_grad_norm', 'ttsr'])
def jitted_train_step(
    state: TrainingState,
    batch_data: BatchArrays,
    loss_fn: Any,  # Static: loss function
    optimizer: Any,  # Static: Optax optimizer
    clip_grad_norm: float,  # Static arg
    ttsr: int,  # Static: train to sample ratio
) -> Tuple[TrainingState, StepMetrics]:
    """
    JIT-compiled training step.
    Performs gradient computation and optimizer updates using JAX control flow.
    
    This mirrors the logic from pure.py:train_step and gflownet.py:train
    """
    
    def single_grad_step(carry, _):
        """
        Single gradient step (for use with lax.scan).
        Mirrors the inner loop in gflownet.py:train (for j in range(self.ttsr))
        """
        params, opt_state, batch, rng_key = carry
        
        # Compute loss and gradients
        # TODO: Replace placeholder loss_fn with actual JAX-compatible loss computation
        # This requires converting the loss module to JAX (see TODO section below)
        loss_value, grads = jax.value_and_grad(loss_fn)(params, batch, rng_key)
        
        # Compute gradient norm
        grad_norm = optax.global_norm(grads)
        
        # Clip gradients using lax.cond (no if/else)
        def clip_grads(g):
            scale = jnp.minimum(1.0, clip_grad_norm / (grad_norm + 1e-6))
            return jax.tree.map(lambda x: x * scale, g)
        
        def no_clip(g):
            return g
        
        grads = lax.cond(
            clip_grad_norm > 0,
            clip_grads,
            no_clip,
            grads
        )
        
        # Apply optimizer update (pure functional with Optax)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Zero logprobs in batch (pure function)
        new_batch = pure_zero_logprobs_jax(batch)
        
        # Split RNG key for next iteration
        rng_key, _ = random.split(rng_key)
        
        return (new_params, new_opt_state, new_batch, rng_key), (loss_value, grad_norm)
    
    # Use lax.scan for multiple gradient steps (ttsr iterations)
    # This mirrors: for j in range(self.ttsr) in gflownet.py
    init_carry = (state.params, state.optimizer_state, batch_data, state.rng_key)
    final_carry, (losses, grad_norms) = lax.scan(
        single_grad_step,
        init_carry,
        jnp.arange(ttsr)  # Dummy array for ttsr iterations
    )
    
    new_params, new_opt_state, final_batch, new_rng_key = final_carry
    
    # Take last loss/grad_norm (mimics the behavior in gflownet.py)
    final_loss = losses[-1]
    final_grad_norm = grad_norms[-1]
    
    # Check if loss is finite (mirrors: if not all([torch.isfinite(loss) ...]))
    skipped = ~jnp.isfinite(final_loss)
    
    # Compute mean rewards and trajectory stats from batch
    rewards_mean = jnp.mean(batch_data.rewards)
    traj_length_mean = 0 # Placeholder
    
    # Update state (pure via NamedTuple replacement)
    new_state = TrainingState(
        params=new_params,
        optimizer_state=new_opt_state,
        rng_key=new_rng_key,
        iteration=state.iteration + 1,
        loss_ema=state.loss_ema,  # Will be updated outside JIT if needed
    )
    
    metrics = StepMetrics(
        loss=final_loss,
        grad_norm=final_grad_norm,
        rewards_mean=rewards_mean,
        skipped=skipped,
        trajectory_length_mean=traj_length_mean,
    )
    
    return new_state, metrics


# ============================================================================
# SAMPLING (Currently IMPURE - major TODO)
# ============================================================================

def sample_batch_jax(agent, rng_key, batch_size, sttr):
    """
    Sample batches for training.
    
    TODO: MAJOR CONVERSION NEEDED
    This function currently delegates to agent.sample_batch which uses PyTorch
    and mutates state. To make this pure JAX:
    
    1. Convert agent.sample_actions to JAX:
       - Forward/backward policies must be JAX models (Flax/Equinox)
       - Action sampling must use JAX random keys
       - Mask computation must be pure JAX operations
    
    2. Convert env.step to JAX:
       - Environment transitions must be pure functions
       - State updates must return new states (no mutation)
    
    3. Use lax.scan for trajectory sampling:
       - Replace while envs: loops with lax.while_loop or lax.scan
       - Ensure all operations are JAX-traceable
    
    For now, keeping the PyTorch delegation but marking it as IMPURE.
    This is the main bottleneck preventing full JAX acceleration.
    """
    # IMPURE: Uses PyTorch agent
    current_key = rng_key
    
    # Use lax.scan pattern (even though inner function is impure)
    def sample_single_batch(carry, _):
        key = carry
        next_key, sample_key = random.split(key)
        
        # IMPURE: agent.sample_batch not JAX-compatible
        sub_batch, times = agent.sample_batch(
            n_forward=batch_size.forward,
            n_train=batch_size.backward_dataset,
            n_replay=batch_size.backward_replay,
            collect_forwards_masks=True,
            collect_backwards_masks=True,
        )
        
        # CRITICAL FIX: Set proxy on batch before conversion
        # The batch returned by agent.sample_batch() doesn't have proxy set
        # but get_rewards() needs it to compute rewards
        if sub_batch.proxy is None and agent.proxy is not None:
            sub_batch.set_proxy(agent.proxy)
        
        # TODO: Convert sub_batch to BatchArrays (JAX arrays)
        # Currently sub_batch is a PyTorch Batch object
        batch_arrays = convert_batch_to_jax(sub_batch)
                
        return next_key, batch_arrays
    
    # Collect sttr batches
    final_key, stacked_batches = lax.scan(
        sample_single_batch,
        current_key,
        jnp.arange(sttr)
    )
    
    # stacked_batches is a BatchArrays where each field has shape (sttr, batch_size, ...)
    # We need to extract individual BatchArrays from each time step
    
    # Convert stacked BatchArrays to list of BatchArrays
    batch_list = [
        BatchArrays(
            states=stacked_batches.states[i],
            actions=stacked_batches.actions[i],
            rewards=stacked_batches.rewards[i],
            logprobs=stacked_batches.logprobs[i],
            logprobs_rev=stacked_batches.logprobs_rev[i],
            trajectory_indices=stacked_batches.trajectory_indices[i],
        )
        for i in range(sttr)
    ]
    
    # Merge all batches using reduce
    merged_batch = reduce(pure_merge_batches_jax, batch_list)
    
    return final_key, merged_batch


def convert_batch_to_jax(pytorch_batch, proxy=None):
    """
    Convert PyTorch Batch to JAX BatchArrays.
    
    Standardizes shapes for JAX compatibility:
    - States: Always (batch_size, flattened_state_dim) 
    - Actions: Always (batch_size, action_dim) - flatten tuples
    - Other fields: Standard tensor shapes
    """
    
    import torch  # TODO: Will be removed once fully JAX-compatible
    
    if proxy is not None and pytorch_batch.proxy is None:
        pytorch_batch.set_proxy(proxy)
    
    # Trajectory indices (consecutive) - (batch_size,)
    traj_indices_tensor = pytorch_batch.get_trajectory_indices(consecutive=True)
    trajectory_indices = jnp.array(traj_indices_tensor.detach().cpu().numpy())
    
    # States: FORCE to (batch_size, flattened_dim) consistently
    states = pytorch_batch.get_states(policy=False)
    
    # Convert to list if tensor (to handle both formats uniformly)
    if not isinstance(states, list):
        # Convert tensor to list of individual state tensors
        states = [states[i] for i in range(states.shape[0])]
    
    # Now states is always a list - flatten each to 1D
    flattened_states = []
    for state in states:
        if isinstance(state, torch.Tensor):
            flattened_states.append(state.reshape(-1))  # Flatten to 1D
        else:
            # Convert to tensor and flatten
            state_tensor = torch.tensor(state, dtype=pytorch_batch.float, device='cpu')
            flattened_states.append(state_tensor.reshape(-1))
    
    # Stack to (batch_size, flattened_dim) - ALWAYS 2D
    states = torch.stack(flattened_states)
    states = jnp.array(states.detach().cpu().numpy())
    
    # Actions: FORCE to (batch_size, flattened_action_dim) consistently
    actions_list = pytorch_batch.get_actions()
    
    # Flatten each action to 1D, regardless of format
    flattened_actions = []
    for action in actions_list:
        if isinstance(action, torch.Tensor):
            flattened_actions.append(action.reshape(-1))
        elif isinstance(action, (tuple, list)):
            action_tensor = torch.tensor(action, dtype=torch.float32)
            flattened_actions.append(action_tensor.reshape(-1))
        else:
            # Single scalar
            flattened_actions.append(torch.tensor([action], dtype=torch.float32))
    
    # Stack to (batch_size, flattened_action_dim) - ALWAYS 2D
    actions = torch.stack(flattened_actions)
    actions = jnp.array(actions.detach().cpu().numpy())
    
    # Rewards: (batch_size,) - already 1D
    if pytorch_batch.proxy is not None:
        rewards_tensor = pytorch_batch.get_rewards()
        rewards = jnp.array(rewards_tensor.detach().cpu().numpy())
    else:
        # Placeholder: Use zeros if proxy is missing
        rewards = jnp.zeros(len(trajectory_indices))
        print("Warning: Proxy not set on batch, using zero rewards placeholder")
    
    # Logprobs forward: (batch_size,) - already 1D
    logprobs_tensor, _ = pytorch_batch.get_logprobs(backward=False)
    logprobs = jnp.array(logprobs_tensor.detach().cpu().numpy())
    
    # Logprobs backward: (batch_size,) - already 1D
    logprobs_rev_tensor, _ = pytorch_batch.get_logprobs(backward=True)
    logprobs_rev = jnp.array(logprobs_rev_tensor.detach().cpu().numpy())
    
    return BatchArrays(
        states=states,  # (batch_size, flattened_state_dim)
        actions=actions,  # (batch_size, flattened_action_dim)
        rewards=rewards,  # (batch_size,)
        logprobs=logprobs,  # (batch_size,)
        logprobs_rev=logprobs_rev,  # (batch_size,)
        trajectory_indices=trajectory_indices,  # (batch_size,)
    )


# ============================================================================
# OPTIMIZER SETUP
# ============================================================================

def build_optimizer_jax(config, n_train_steps):
    """
    Build Optax optimizer matching make_opt from gflownet.py.
    
    This mirrors the logic in make_opt:
    - Uses Adam optimizer with config.lr, adam_beta1, adam_beta2
    - Implements StepLR schedule with lr_decay_period and lr_decay_gamma
    - Handles clip_grad_norm
    
    Note: logZ parameter group with lr_z_mult is handled separately in params
    """
    # Extract LR schedule parameters (mirrors make_opt StepLR)
    step_size = config.gflownet.optimizer.lr_decay_period
    gamma = config.gflownet.optimizer.lr_decay_gamma
    initial_lr = config.gflownet.optimizer.lr

    # Create step boundaries for schedule
    num_steps = n_train_steps // step_size
    boundaries_and_scales = {
        (i * step_size): gamma 
        for i in range(1, num_steps + 1)
    }
    
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=initial_lr,
        boundaries_and_scales=boundaries_and_scales
    )
    
    # Build optimizer chain (mirrors make_opt)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gflownet.optimizer.clip_grad_norm) if config.gflownet.optimizer.clip_grad_norm > 0 else optax.identity(),
        optax.adam(
            learning_rate=lr_schedule,
            b1=config.gflownet.optimizer.adam_beta1,
            b2=config.gflownet.optimizer.adam_beta2,
        ),
    )
    
    return optimizer


def build_state_from_agent(agent: Any, config: Any) -> Tuple[TrainingState, Any]:
    """
    Initialize TrainingState from agent.
    Mirrors build_state_from_agent from pure.py.
    """
    optimizer = build_optimizer_jax(config, agent.n_train_steps)
    
    # TODO: Get JAX parameters from agent
    # This requires the agent's models to be JAX-compatible (Flax/Equinox)
    # For now, using placeholder
    params = agent.get_params() if hasattr(agent, 'get_params') else {'dummy': jnp.zeros(10)}
    
    # Initialize optimizer state
    opt_state = optimizer.init(params)
    
    return TrainingState(
        params=params,
        optimizer_state=opt_state,
        rng_key=random.PRNGKey(config.seed),
        iteration=agent.it,
        loss_ema=0.0,  # Initialize loss EMA
    ), optimizer


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train(agent: Any, config: Any) -> TrainingState:
    """
    Main training loop using JAX control flow.
    Mirrors the logic from pure.py:train and gflownet.py:train.
    
    Uses lax.fori_loop for the main training iterations to enable JIT compilation
    of the core loop, while keeping side effects (logging, checkpointing) outside.
    """
    # Build initial state
    state, optimizer = build_state_from_agent(agent, config)
    
    # Progress bar (non-JIT-able, acceptable side effect)
    pbar = tqdm(
        initial=state.iteration - 1,
        total=agent.n_train_steps,
        disable=agent.logger.progressbar.get("skip", False),
    )
    
    # TODO: Define JAX-compatible loss function
    # This requires converting the loss module to JAX
    def loss_fn_placeholder(params, batch, rng_key):
        """
        TODO: CRITICAL CONVERSION NEEDED
        
        The loss computation requires:
        1. Converting forward_policy, backward_policy, state_flow to JAX models
        2. Implementing pure JAX versions of:
           - compute_logprobs_trajectories
           - Flow matching loss computation
           - Trajectory balance loss computation
        
        This is a major undertaking and should be done separately.
        For now, this is a placeholder that returns a dummy loss.
        """
        return jnp.sum(params['dummy'])
    
    # Training body for lax.fori_loop
    def training_body(i, carry):
        """
        Single training iteration.
        Mirrors the main loop in gflownet.py:train and pure.py:train.
        
        Note: This cannot be fully JIT-compiled due to:
        - Batch sampling (uses PyTorch agent)
        - Buffer updates (uses mutable buffer)
        - Logging (I/O operations)
        
        However, the core gradient computation (jitted_train_step) IS JIT-compiled.
        """
        state = carry
        
        # Evaluation (IMPURE: I/O, acceptable)
        # Mirrors: if self.evaluator.should_eval(self.it)
        should_eval = agent.evaluator.should_eval(i)
        should_eval_top_k = agent.evaluator.should_eval_top_k(i)
        
        def do_eval(_):
            try:
                agent.evaluator.eval_and_log(i)
            except AttributeError as e:
                print(f"Warning: Evaluation failed at step {i}: {e}")
            return None
        
        def do_eval_top_k(_):
            try:
                agent.evaluator.eval_and_log_top_k(i)
            except AttributeError as e:
                print(f"Warning: Top-k evaluation failed at step {i}: {e}")
            return None
        
        def no_eval(_):
            return None
        
        lax.cond(should_eval, do_eval, no_eval, None)
        lax.cond(should_eval_top_k, do_eval_top_k, no_eval, None)
        
        # lax.cond(should_eval, do_eval, no_eval, None)
        # lax.cond(should_eval_top_k, 
        #         lambda _: agent.evaluator.eval_and_log_top_k(i), 
        #         no_eval, None)
        
        # Sample batches (IMPURE: uses PyTorch agent)
        # Mirrors: for j in range(self.sttr): ... batch.merge(sub_batch)
        rng_key, merged_batch = sample_batch_jax(
            agent, 
            state.rng_key, 
            agent.batch_size, 
            agent.sttr
        )
        
        # JIT-compiled training step (PURE JAX)
        # Mirrors: for j in range(self.ttsr): ... losses.backward() ... opt.step()
        new_state, metrics = jitted_train_step(
            state._replace(rng_key=rng_key),
            merged_batch,
            loss_fn_placeholder,
            optimizer,
            agent.clip_grad_norm,
            agent.ttsr,
        )
        
        traj_length_mean, _, _ = compute_trajectory_stats(merged_batch.trajectory_indices)
        metrics = metrics._replace(trajectory_length_mean=traj_length_mean)

        
        # Buffer updates (IMPURE: mutates buffer)
        # TODO: Make buffer functional (return new buffer instead of mutating)
        # Mirrors: self.buffer.add(..., buffer="main") and buffer="replay"
        
        # DISABLED FOR NOW: Buffer updates cause tracer conversion errors
        # The buffer tries to convert JAX arrays to numpy/pandas inside the traced loop
        # TODO: Either move buffer updates outside lax.fori_loop or convert buffer to JAX
        # states_term = merged_batch.states  # TODO: Extract terminating states
        # actions_traj = merged_batch.actions  # TODO: Extract action trajectories
        # rewards = merged_batch.rewards
        # 
        # def update_main_buffer(_):
        #     agent.buffer.add(states_term, actions_traj, rewards, i, buffer="main")
        #     return None
        # 
        # def no_update(_):
        #     return None
        # 
        # lax.cond(agent.buffer.use_main_buffer, update_main_buffer, no_update, None)
        # agent.buffer.add(states_term, actions_traj, rewards, i, buffer="replay")

        # Logging (IMPURE: I/O, acceptable)
        # DISABLED FOR NOW: Logging requires converting traced values to Python floats
        # TODO: Move logging outside lax.fori_loop or use JAX-compatible logging
        # should_log = agent.evaluator.should_log_train(i)
        # 
        # def do_log(_):
        #     agent.logger.progressbar_update(
        #         pbar,
        #         float(metrics.loss),
        #         [float(metrics.rewards_mean)],
        #         agent.jsd,
        #         agent.use_context,
        #     )
        #     return None
        # 
        # lax.cond(should_log, do_log, no_eval, None)
        
        # Progress bar update (IMPURE: I/O, acceptable)
        # DISABLED FOR NOW: Also causes issues inside traced loop
        # pbar.update(1)
        
        # Garbage collection (IMPURE: system side effect, acceptable)
        # DISABLED FOR NOW: Cannot use inside traced loop
        # should_gc = (agent.garbage_collection_period > 0 and 
        #             i % agent.garbage_collection_period == 0)
        # 
        # def do_gc(_):
        #     gc.collect()
        #     return None
        # 
        # lax.cond(should_gc, do_gc, no_eval, None)
        
        # Checkpointing (IMPURE: file I/O, acceptable)
        # DISABLED FOR NOW: Checkpoint evaluation uses Python boolean logic on traced values
        # TODO: Move checkpointing outside lax.fori_loop
        # should_checkpoint = agent.evaluator.should_checkpoint(i)
        # 
        # def do_checkpoint(_):
        #     agent.logger.save_checkpoint(
        #         forward_policy=agent.forward_policy,
        #         backward_policy=agent.backward_policy,
        #         state_flow=agent.state_flow,
        #         logZ=agent.logZ,
        #         optimizer=new_state.optimizer_state,
        #         buffer=agent.buffer, 
        #         step=i,
        #     )
        #     return None
        # 
        # lax.cond(should_checkpoint, do_checkpoint, no_eval, None)
        
        # Early stopping check (cannot break lax.fori_loop, so just flag it)
        # DISABLED FOR NOW: Early stopping also uses Python control flow
        # TODO: Consider using lax.while_loop for early stopping support
        # should_stop = agent.loss.do_early_stopping(metrics.loss)
        # 
        # def do_stop(_):
        #     # Can't actually stop lax.fori_loop early
        #     # Would need lax.while_loop for this
        #     print(
        #         "Early stopping criteria met: "
        #         f"{agent.loss.loss_ema} < {agent.loss.early_stopping_th}"
        #     )
        #     return None
        # 
        # lax.cond(should_stop, do_stop, no_eval, None)
        
        return new_state
    
    # Main training loop using lax.fori_loop
    # Mirrors: for self.it in range(self.it, self.n_train_steps + 1)
    final_state = lax.fori_loop(
        state.iteration,
        agent.n_train_steps + 1,
        training_body,
        state
    )
    
    print(f"\n✓ JAX training completed! Final iteration: {final_state.iteration}")
    print(f"  Final loss EMA: {final_state.loss_ema:.4f}")
    
    # Final checkpoint save (IMPURE: file I/O, acceptable)
    # Note: JAX uses Optax optimizers (tuples), not PyTorch optimizers
    # Skip checkpoint save for now since it expects PyTorch format
    # TODO: Implement JAX-compatible checkpoint saving
    # agent.logger.save_checkpoint(
    #     forward_policy=agent.forward_policy,
    #     backward_policy=agent.backward_policy,
    #     state_flow=agent.state_flow,
    #     logZ=agent.logZ,
    #     optimizer=final_state.optimizer_state,
    #     buffer=agent.buffer,
    #     step=final_state.iteration,
    #     final=True,
    # )
    print("  (Checkpoint saving disabled for JAX trainer - models are still PyTorch)")
    
    # Close logger (IMPURE: I/O cleanup, acceptable)
    if not agent.use_context:
        agent.logger.end()
    
    pbar.close()
    
    return final_state


# ============================================================================
# BUFFER OPERATIONS (Pure JAX versions)
# ============================================================================

@jit
def pure_buffer_add_jax(buffer_arrays: dict, new_data: dict) -> dict:
    """
    Pure functional buffer update.
    
    TODO: This needs to be adapted to the actual buffer structure.
    The buffer currently uses pandas DataFrames and mutates in-place.
    
    For full JAX compatibility, the buffer should:
    1. Use JAX arrays instead of pandas DataFrames
    2. Return new buffer state instead of mutating
    3. Implement selection logic (permutation, prioritized) in pure JAX
    """
    # Use jax.tree.map to merge arrays
    def merge_fn(existing, new):
        return lax.cond(
            new is not None,
            lambda: jnp.concatenate([existing, new], axis=0),
            lambda: existing
        )
    
    updated = jax.tree.map(merge_fn, buffer_arrays, new_data)
    return updated


# ============================================================================
# MAJOR TODOs FOR FULL JAX CONVERSION
# ============================================================================

def should_eval(self, step):
    """
    Check if testing should be done at the current step. The decision is based on
    the ``self.config.period`` attribute.

    Set ``self.config.first_it`` to ``True`` if testing should be done at the
    first iteration step. Otherwise, testing will be done after
    ``self.config.period`` steps.

    Set ``self.config.period`` to ``None`` or a negative value to disable
    testing.

    Parameters
    ----------
    step : int
        Current iteration step.

    Returns
    -------
    bool
        True if testing should be done at the current step, False otherwise.
    """
    # Mirror the original logic exactly, but use JAX control flow for traced operations
    if self.config.period is None or self.config.period <= 0:
        return False
    else:
        # Condition 1: step is a multiple of period
        cond1 = (step % self.config.period) == 0
        
        # Condition 2: step == 1 AND first_it is True (use lax.cond for traced step)
        cond2 = lax.cond(
            step == 1,
            lambda: self.config.first_it,
            lambda: False
        )
        
        # Return True if either condition is met
        return cond1 | cond2

"""
TODO LIST (ordered by priority):

1. **Convert Models to JAX (CRITICAL)**
   - Forward policy: PyTorch nn.Module → Flax/Equinox model
   - Backward policy: PyTorch nn.Module → Flax/Equinox model
   - State flow: PyTorch nn.Module → Flax/Equinox model (if used)
   - Add agent.get_params() to return JAX pytree of parameters

2. **Convert Loss Computation to JAX (CRITICAL)**
   - Implement compute_logprobs_trajectories in pure JAX
   - Convert FlowMatching loss to JAX
   - Convert TrajectoryBalance loss to JAX
   - Convert other loss variants to JAX
   - Make loss.compute() a pure JAX function

3. **Convert Environment to JAX (CRITICAL)**
   - env.step() → pure JAX function (no mutation)
   - env.step_backwards() → pure JAX function
   - env.get_mask_invalid_actions_*() → pure JAX functions
   - env.states2policy() → pure JAX function
   - Make environment state a JAX pytree

4. **Convert Action Sampling to JAX (CRITICAL)**
   - sample_actions() → pure JAX function using jax.random
   - Policy output processing → pure JAX
   - Mask handling → pure JAX operations

5. **Convert Buffer to JAX (IMPORTANT)**
   - Replace pandas DataFrames with JAX arrays
   - Implement pure functional add/select operations
   - Convert sampling strategies to JAX (permutation, prioritized)

6. **Convert Batch to JAX (IMPORTANT)**
   - Use BatchArrays NamedTuple instead of Batch class
   - Implement pure JAX equivalents of Batch methods
   - Handle trajectory tracking in pure JAX

7. **Handle Early Stopping (MEDIUM)**
   - Currently lax.fori_loop doesn't support early exit
   - Options:
     a) Use lax.while_loop instead (requires refactoring)
     b) Continue full loop and ignore later iterations (wasteful)
     c) Accept that early stopping isn't supported in JAX mode

8. **Optimize Memory (MEDIUM)**
   - Implement JAX-native buffer with fixed size (circular buffer)
   - Use jax.lax.dynamic_slice for efficient batch selection
   - Profile memory usage and identify bottlenecks

EXPLANATION OF DIFFICULTY:

The main challenge is that the entire GFlowNet codebase is built on PyTorch
with heavy use of:
- Mutable objects (environment states, buffers, model parameters)
- Object-oriented design (classes with internal state)
- Python control flow in hot loops

JAX requires:
- Pure functions (no mutations, no side effects)
- Functional data structures (pytrees, NamedTuples)
- JAX control flow (lax.cond, lax.scan, lax.while_loop)

This is essentially a complete rewrite of the core components, which is why
items 1-4 are marked CRITICAL. Without these conversions, we can only
JIT-compile small pieces (like gradient computation), but the main bottlenecks
(sampling, environment steps) remain in PyTorch.

The good news: Once these conversions are done, JAX's JIT compilation,
automatic vectorization (vmap), and XLA optimization should provide
significant speedups, especially for batched operations.

RECOMMENDED APPROACH:

Start with a minimal example:
1. Pick the simplest environment (e.g., HyperGrid)
2. Convert it to pure JAX
3. Convert a simple policy (e.g., uniform random)
4. Get sampling working in JAX
5. Add loss computation
6. Benchmark against PyTorch
7. Gradually expand to more complex environments/policies

This incremental approach will help identify issues early and validate
that JAX actually provides speedups before investing in full conversion.
"""