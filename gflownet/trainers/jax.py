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
    merged = jax.tree_map(
        lambda base, sub: jnp.concatenate([base, sub], axis=0),
        base_batch,
        sub_batch
    )
    return merged


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


@partial(jit, static_argnums=(1,))
def compute_trajectory_stats(trajectory_indices: jnp.ndarray, num_trajectories: int) -> Tuple[float, int, int]:
    """
    Compute trajectory statistics: mean, min, max lengths.
    Uses JAX operations to avoid Python control flow.
    
    Args:
        trajectory_indices: Array of trajectory indices for each transition
        num_trajectories: Total number of trajectories (static arg)
    
    Returns:
        Tuple of (mean_length, min_length, max_length)
    """
    # Count transitions per trajectory using bincount
    trajectory_lengths = jnp.bincount(trajectory_indices, length=num_trajectories)
    
    # Filter out zero-length trajectories (not yet started)
    trajectory_lengths = jnp.where(trajectory_lengths > 0, trajectory_lengths, jnp.inf)
    
    traj_length_mean = jnp.mean(trajectory_lengths)
    traj_length_min = jnp.min(trajectory_lengths)
    traj_length_max = jnp.max(trajectory_lengths)
    
    return traj_length_mean, traj_length_min, traj_length_max


# ============================================================================
# CORE TRAINING STEP (JIT-compiled)
# ============================================================================

@partial(jit, static_argnums=(2, 3, 4, 5))
def jitted_train_step(
    state: TrainingState,
    batch_data: BatchArrays,
    loss_fn: Any,  # Static: loss function
    optimizer: Any,  # Static: Optax optimizer
    clip_grad_norm: float,  # Static arg
    ttsr: int,  # Static: number of gradient steps
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
            return jax.tree_map(lambda x: x * scale, g)
        
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
    num_trajectories = jnp.max(batch_data.trajectory_indices) + 1
    traj_length_mean, _, _ = compute_trajectory_stats(
        batch_data.trajectory_indices, 
        num_trajectories
    )
    
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
    batch_list = []
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
        
        # TODO: Convert sub_batch to BatchArrays (JAX arrays)
        # Currently sub_batch is a PyTorch Batch object
        batch_arrays = convert_batch_to_jax(sub_batch)
        
        return next_key, batch_arrays
    
    # Collect sttr batches
    final_key, batch_arrays = lax.scan(
        sample_single_batch,
        current_key,
        jnp.arange(sttr)
    )
    
    # Merge batches
    merged_batch = batch_arrays[0]
    for i in range(1, len(batch_arrays)):
        merged_batch = pure_merge_batches_jax(merged_batch, batch_arrays[i])
    
    return final_key, merged_batch


def convert_batch_to_jax(pytorch_batch):
    """
    Convert PyTorch Batch to JAX BatchArrays.
    
    TODO: Implement based on Batch structure
    This requires accessing the internal data structures of the Batch class
    and converting PyTorch tensors to JAX arrays.
    """
    # Placeholder implementation
    return BatchArrays(
        states=jnp.array([]),  # TODO: Extract from pytorch_batch
        actions=jnp.array([]),
        rewards=jnp.array([]),
        logprobs=jnp.array([]),
        logprobs_rev=jnp.array([]),
        trajectory_indices=jnp.array([]),
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
    step_size = config.lr_decay_period
    gamma = config.lr_decay_gamma
    initial_lr = config.lr
    
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
        optax.clip_by_global_norm(config.clip_grad_norm) if config.clip_grad_norm > 0 else optax.identity(),
        optax.adam(
            learning_rate=lr_schedule,
            b1=config.adam_beta1,
            b2=config.adam_beta2,
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
        state, buffer = carry
        
        # Evaluation (IMPURE: I/O, acceptable)
        # Mirrors: if self.evaluator.should_eval(self.it)
        should_eval = agent.evaluator.should_eval(i)
        should_eval_top_k = agent.evaluator.should_eval_top_k(i)
        
        # Use lax.cond to avoid Python if (even though it's impure inside)
        def do_eval(_):
            agent.evaluator.eval_and_log(i)
            return None
        
        def no_eval(_):
            return None
        
        lax.cond(should_eval, do_eval, no_eval, None)
        lax.cond(should_eval_top_k, 
                lambda _: agent.evaluator.eval_and_log_top_k(i), 
                no_eval, None)
        
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
        
        # Buffer updates (IMPURE: mutates buffer)
        # TODO: Make buffer functional (return new buffer instead of mutating)
        # Mirrors: self.buffer.add(..., buffer="main") and buffer="replay"
        states_term = merged_batch.states  # TODO: Extract terminating states
        actions_traj = merged_batch.actions  # TODO: Extract action trajectories
        rewards = merged_batch.rewards
        
        # Use lax.cond for conditional buffer update
        def update_main_buffer(_):
            buffer.add(states_term, actions_traj, rewards, i, buffer="main")
            return None
        
        def no_update(_):
            return None
        
        lax.cond(buffer.use_main_buffer, update_main_buffer, no_update, None)
        buffer.add(states_term, actions_traj, rewards, i, buffer="replay")
        
        # Logging (IMPURE: I/O, acceptable)
        # Mirrors: self.log_train_iteration(pbar, losses, batch, times)
        should_log = agent.evaluator.should_log_train(i)
        
        def do_log(_):
            agent.logger.progressbar_update(
                pbar,
                float(metrics.loss),
                [float(metrics.rewards_mean)],
                agent.jsd,
                agent.use_context,
            )
            # TODO: Add more logging (rewards, scores, losses, etc.)
            return None
        
        lax.cond(should_log, do_log, no_eval, None)
        
        # Progress bar update (IMPURE: I/O, acceptable)
        pbar.update(1)
        
        # Garbage collection (IMPURE: system side effect, acceptable)
        # Mirrors: if self.garbage_collection_period > 0 and ...
        should_gc = (agent.garbage_collection_period > 0 and 
                    i % agent.garbage_collection_period == 0)
        
        def do_gc(_):
            gc.collect()
            return None
        
        lax.cond(should_gc, do_gc, no_eval, None)
        
        # Checkpointing (IMPURE: file I/O, acceptable)
        should_checkpoint = agent.evaluator.should_checkpoint(i)
        
        def do_checkpoint(_):
            agent.logger.save_checkpoint(
                forward_policy=agent.forward_policy,
                backward_policy=agent.backward_policy,
                state_flow=agent.state_flow,
                logZ=agent.logZ,
                optimizer=new_state.optimizer_state,
                buffer=buffer,
                step=i,
            )
            return None
        
        lax.cond(should_checkpoint, do_checkpoint, no_eval, None)
        
        # Early stopping check (cannot break lax.fori_loop, so just flag it)
        # TODO: Consider using lax.while_loop for early stopping support
        # For now, mirroring gflownet.py logic but cannot actually break
        should_stop = agent.loss.do_early_stopping(metrics.loss)
        
        def do_stop(_):
            # Can't actually stop lax.fori_loop early
            # Would need lax.while_loop for this
            print(
                "Early stopping criteria met: "
                f"{agent.loss.loss_ema} < {agent.loss.early_stopping_th}"
            )
            return None
        
        lax.cond(should_stop, do_stop, no_eval, None)
        
        return new_state, buffer
    
    # Main training loop using lax.fori_loop
    # Mirrors: for self.it in range(self.it, self.n_train_steps + 1)
    final_state, final_buffer = lax.fori_loop(
        state.iteration,
        agent.n_train_steps + 1,
        training_body,
        (state, agent.buffer)
    )
    
    # Final checkpoint save (IMPURE: file I/O, acceptable)
    agent.logger.save_checkpoint(
        forward_policy=agent.forward_policy,
        backward_policy=agent.backward_policy,
        state_flow=agent.state_flow,
        logZ=agent.logZ,
        optimizer=final_state.optimizer_state,
        buffer=final_buffer,
        step=final_state.iteration,
        final=True,
    )
    
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
    # Use jax.tree_map to merge arrays
    def merge_fn(existing, new):
        return lax.cond(
            new is not None,
            lambda: jnp.concatenate([existing, new], axis=0),
            lambda: existing
        )
    
    updated = jax.tree_map(merge_fn, buffer_arrays, new_data)
    return updated


# ============================================================================
# MAJOR TODOs FOR FULL JAX CONVERSION
# ============================================================================

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