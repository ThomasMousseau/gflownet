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
from functools import partial, reduce
import optax
from tqdm import tqdm
import time
import gc
from hydra.utils import instantiate
from gflownet.utils.policy import parse_policy_config




# # ============================================================================
# # DATA STRUCTURES (JAX pytree-compatible)
# # ============================================================================
# class GFlowNetState(NamedTuple):
#     # Core components (now immutable)
#     env_maker: Any  # Function to create env instances
#     env: Any        # Current env state (must be JAX-compatible, e.g., a pytree)
#     proxy: Any      # Proxy model
#     loss: Any       # Loss function
#     logZ: jnp.ndarray  # Log partition function (JAX array)
#     forward_policy: Any  # JAX-compatible policy (e.g., Equinox model)
#     backward_policy: Any
#     state_flow: Any
#     buffer: Any     # Functional buffer (JAX arrays)
#     params: Any     # Model parameters (JAX pytree)
#     optimizer_state: Any  # Optax optimizer state
#     evaluator: Any  # Evaluator (may need functional refactoring)
#     logger: Any     # Logger (side-effect heavy, handle carefully)
    
#     # Training state
#     it: int
#     rng: jnp.ndarray  # JAX random key
#     device: str
#     float_precision: int
#     # ... include all other attributes from GFlowNetAgent
    
# class TrainingState(NamedTuple):
#     """Use NamedTuple for JAX compatibility (pytrees)"""
#     params: Any  # Model parameters (JAX pytree)
#     optimizer_state: Any  # Optax optimizer state
#     rng_key: Any  # JAX random key
#     iteration: int
#     loss_ema: float  # For early stopping
#     # Non-JIT-able fields (buffer, agent, logger) passed separately


# class StepMetrics(NamedTuple):
#     """Metrics from a single training step"""
#     loss: float
#     grad_norm: float
#     rewards_mean: float
#     skipped: bool
#     trajectory_length_mean: float


# class BatchArrays(NamedTuple):
#     """Fixed structure for batch data (JAX pytree-compatible)"""
#     states: jnp.ndarray
#     actions: jnp.ndarray
#     rewards: jnp.ndarray
#     logprobs: jnp.ndarray
#     logprobs_rev: jnp.ndarray
#     trajectory_indices: jnp.ndarray
#     # TODO: Add masks if needed for invalid action masking
#     # masks_forward: jnp.ndarray
#     # masks_backward: jnp.ndarray


# # ============================================================================
# # PURE JAX FUNCTIONS (JIT-compiled)
# # ============================================================================

# @jit
# def pure_merge_batches_jax(base_batch: BatchArrays, sub_batch: BatchArrays) -> BatchArrays:
#     """
#     Pure function to merge batch arrays (JAX-compatible).
#     Concatenates JAX arrays without mutation using JAX tree operations.
#     """
#     merged = jax.tree.map(
#         lambda base, sub: jnp.concatenate([base, sub], axis=0),
#         base_batch,
#         sub_batch
#     )
#     return merged
    
#     # merged_states = jnp.concatenate([base_batch.states, sub_batch.states], axis=0)
#     # merged_actions = jnp.concatenate([base_batch.actions, sub_batch.actions], axis=0)
#     # merged_rewards = jnp.concatenate([base_batch.rewards, sub_batch.rewards], axis=0)
#     # merged_logprobs = jnp.concatenate([base_batch.logprobs, sub_batch.logprobs], axis=0)
#     # merged_logprobs_rev = jnp.concatenate([base_batch.logprobs_rev, sub_batch.logprobs_rev], axis=0)
#     # merged_trajectory_indices = jnp.concatenate([base_batch.trajectory_indices, sub_batch.trajectory_indices], axis=0)
    
#     # return BatchArrays(
#     #     states=merged_states,
#     #     actions=merged_actions,
#     #     rewards=merged_rewards,
#     #     logprobs=merged_logprobs,
#     #     logprobs_rev=merged_logprobs_rev,
#     #     trajectory_indices=merged_trajectory_indices,
#     # )


# @jit
# def pure_zero_logprobs_jax(batch_arrays: BatchArrays) -> BatchArrays:
#     """
#     Pure function to zero logprobs in batch.
#     Returns new BatchArrays with zeroed logprob arrays.
#     """
#     zeroed_logprobs = jnp.zeros_like(batch_arrays.logprobs)
#     zeroed_logprobs_rev = jnp.zeros_like(batch_arrays.logprobs_rev)
    
#     return BatchArrays(
#         states=batch_arrays.states,
#         actions=batch_arrays.actions,
#         rewards=batch_arrays.rewards,
#         logprobs=zeroed_logprobs,
#         logprobs_rev=zeroed_logprobs_rev,
#         trajectory_indices=batch_arrays.trajectory_indices,
#     )


# def compute_trajectory_stats(trajectory_indices: jnp.ndarray) -> Tuple[float, int, int]:
#     """
#     Compute trajectory statistics: mean, min, max lengths.
    
#     SIMPLIFIED VERSION: Just return placeholders for now to get training running.
#     TODO: Implement proper trajectory stats computation outside the training loop.
    
#     Args:
#         trajectory_indices: Array of trajectory indices for each transition
    
#     Returns:
#         Tuple of (mean_length, min_length, max_length)
#     """
#     # PLACEHOLDER: Return simple estimates to avoid concretization errors
#     # Mean trajectory length ≈ total transitions / estimated num trajectories
#     total_transitions = trajectory_indices.shape[0]
#     estimated_traj_length = jnp.float32(total_transitions / 10.0)  # Rough estimate
    
#     return estimated_traj_length, jnp.int32(1), jnp.int32(100)


# # ============================================================================
# # CORE TRAINING STEP (JIT-compiled)
# # ============================================================================

# @partial(jit, static_argnames=['loss_fn', 'optimizer', 'clip_grad_norm', 'ttsr'])
# def jitted_train_step(
#     state: TrainingState,
#     batch_data: BatchArrays,
#     loss_fn: Any,  # Static: loss function
#     optimizer: Any,  # Static: Optax optimizer
#     clip_grad_norm: float,  # Static arg
#     ttsr: int,  # Static: train to sample ratio
# ) -> Tuple[TrainingState, StepMetrics]:
#     """
#     JIT-compiled training step.
#     Performs gradient computation and optimizer updates using JAX control flow.
    
#     This mirrors the logic from pure.py:train_step and gflownet.py:train
#     """
    
#     def single_grad_step(carry, _):
#         """
#         Single gradient step (for use with lax.scan).
#         Mirrors the inner loop in gflownet.py:train (for j in range(self.ttsr))
#         """
#         params, opt_state, batch, rng_key = carry
        
#         # Compute loss and gradients
#         # TODO: Replace placeholder loss_fn with actual JAX-compatible loss computation
#         # This requires converting the loss module to JAX (see TODO section below)
#         loss_value, grads = jax.value_and_grad(loss_fn)(params, batch, rng_key)
        
#         # Compute gradient norm
#         grad_norm = optax.global_norm(grads)
        
#         # Clip gradients using lax.cond (no if/else)
#         def clip_grads(g):
#             scale = jnp.minimum(1.0, clip_grad_norm / (grad_norm + 1e-6))
#             return jax.tree.map(lambda x: x * scale, g)
        
#         def no_clip(g):
#             return g
        
#         grads = lax.cond(
#             clip_grad_norm > 0,
#             clip_grads,
#             no_clip,
#             grads
#         )
        
#         # Apply optimizer update (pure functional with Optax)
#         updates, new_opt_state = optimizer.update(grads, opt_state, params)
#         new_params = optax.apply_updates(params, updates)
        
#         # Zero logprobs in batch (pure function)
#         new_batch = pure_zero_logprobs_jax(batch)
        
#         # Split RNG key for next iteration
#         rng_key, _ = random.split(rng_key)
        
#         return (new_params, new_opt_state, new_batch, rng_key), (loss_value, grad_norm)
    
#     # Use lax.scan for multiple gradient steps (ttsr iterations)
#     # This mirrors: for j in range(self.ttsr) in gflownet.py
#     init_carry = (state.params, state.optimizer_state, batch_data, state.rng_key)
#     final_carry, (losses, grad_norms) = lax.scan(
#         single_grad_step,
#         init_carry,
#         jnp.arange(ttsr)  # Dummy array for ttsr iterations
#     )
    
#     new_params, new_opt_state, final_batch, new_rng_key = final_carry
    
#     # Take last loss/grad_norm (mimics the behavior in gflownet.py)
#     final_loss = losses[-1]
#     final_grad_norm = grad_norms[-1]
    
#     # Check if loss is finite (mirrors: if not all([torch.isfinite(loss) ...]))
#     skipped = ~jnp.isfinite(final_loss)
    
#     # Compute mean rewards and trajectory stats from batch
#     rewards_mean = jnp.mean(batch_data.rewards)
#     traj_length_mean = 0 # Placeholder
    
#     # Update state (pure via NamedTuple replacement)
#     new_state = TrainingState(
#         params=new_params,
#         optimizer_state=new_opt_state,
#         rng_key=new_rng_key,
#         iteration=state.iteration + 1,
#         loss_ema=state.loss_ema,  # Will be updated outside JIT if needed
#     )
    
#     metrics = StepMetrics(
#         loss=final_loss,
#         grad_norm=final_grad_norm,
#         rewards_mean=rewards_mean,
#         skipped=skipped,
#         trajectory_length_mean=traj_length_mean,
#     )
    
#     return new_state, metrics


# # ============================================================================
# # SAMPLING (Currently IMPURE - major TODO)
# # ============================================================================

# def sample_batch_jax(agent, rng_key, batch_size, sttr):
#     """
#     Sample batches for training.
    
#     TODO: MAJOR CONVERSION NEEDED
#     This function currently delegates to agent.sample_batch which uses PyTorch
#     and mutates state. To make this pure JAX:
    
#     1. Convert agent.sample_actions to JAX:
#        - Forward/backward policies must be JAX models (Flax/Equinox)
#        - Action sampling must use JAX random keys
#        - Mask computation must be pure JAX operations
    
#     2. Convert env.step to JAX:
#        - Environment transitions must be pure functions
#        - State updates must return new states (no mutation)
    
#     3. Use lax.scan for trajectory sampling:
#        - Replace while envs: loops with lax.while_loop or lax.scan
#        - Ensure all operations are JAX-traceable
    
#     For now, keeping the PyTorch delegation but marking it as IMPURE.
#     This is the main bottleneck preventing full JAX acceleration.
#     """
#     # IMPURE: Uses PyTorch agent
#     current_key = rng_key
    
#     # Use lax.scan pattern (even though inner function is impure)
#     def sample_single_batch(carry, _):
#         key = carry
#         next_key, sample_key = random.split(key)
        
#         # IMPURE: agent.sample_batch not JAX-compatible
#         sub_batch, times = agent.sample_batch(
#             n_forward=batch_size.forward,
#             n_train=batch_size.backward_dataset,
#             n_replay=batch_size.backward_replay,
#             collect_forwards_masks=True,
#             collect_backwards_masks=True,
#         )
        
#         # CRITICAL FIX: Set proxy on batch before conversion
#         # The batch returned by agent.sample_batch() doesn't have proxy set
#         # but get_rewards() needs it to compute rewards
#         if sub_batch.proxy is None and agent.proxy is not None:
#             sub_batch.set_proxy(agent.proxy)
        
#         # TODO: Convert sub_batch to BatchArrays (JAX arrays)
#         # Currently sub_batch is a PyTorch Batch object
#         batch_arrays = convert_batch_to_jax(sub_batch)
                
#         return next_key, batch_arrays
    
#     # Collect sttr batches
#     final_key, stacked_batches = lax.scan(
#         sample_single_batch,
#         current_key,
#         jnp.arange(sttr)
#     )
    
#     # stacked_batches is a BatchArrays where each field has shape (sttr, batch_size, ...)
#     # We need to extract individual BatchArrays from each time step
    
#     # Convert stacked BatchArrays to list of BatchArrays
#     batch_list = [
#         BatchArrays(
#             states=stacked_batches.states[i],
#             actions=stacked_batches.actions[i],
#             rewards=stacked_batches.rewards[i],
#             logprobs=stacked_batches.logprobs[i],
#             logprobs_rev=stacked_batches.logprobs_rev[i],
#             trajectory_indices=stacked_batches.trajectory_indices[i],
#         )
#         for i in range(sttr)
#     ]
    
#     # Merge all batches using reduce
#     merged_batch = reduce(pure_merge_batches_jax, batch_list)
    
#     return final_key, merged_batch


# def convert_batch_to_jax(pytorch_batch, proxy=None):
#     """
#     Convert PyTorch Batch to JAX BatchArrays.
    
#     Standardizes shapes for JAX compatibility:
#     - States: Always (batch_size, flattened_state_dim) 
#     - Actions: Always (batch_size, action_dim) - flatten tuples
#     - Other fields: Standard tensor shapes
#     """
    
#     import torch  # TODO: Will be removed once fully JAX-compatible
    
#     if proxy is not None and pytorch_batch.proxy is None:
#         pytorch_batch.set_proxy(proxy)
    
#     # Trajectory indices (consecutive) - (batch_size,)
#     traj_indices_tensor = pytorch_batch.get_trajectory_indices(consecutive=True)
#     trajectory_indices = jnp.array(traj_indices_tensor.detach().cpu().numpy())
    
#     # States: FORCE to (batch_size, flattened_dim) consistently
#     states = pytorch_batch.get_states(policy=False)
    
#     # Convert to list if tensor (to handle both formats uniformly)
#     if not isinstance(states, list):
#         # Convert tensor to list of individual state tensors
#         states = [states[i] for i in range(states.shape[0])]
    
#     # Now states is always a list - flatten each to 1D
#     flattened_states = []
#     for state in states:
#         if isinstance(state, torch.Tensor):
#             flattened_states.append(state.reshape(-1))  # Flatten to 1D
#         else:
#             # Convert to tensor and flatten
#             state_tensor = torch.tensor(state, dtype=pytorch_batch.float, device='cpu')
#             flattened_states.append(state_tensor.reshape(-1))
    
#     # Stack to (batch_size, flattened_dim) - ALWAYS 2D
#     states = torch.stack(flattened_states)
#     states = jnp.array(states.detach().cpu().numpy())
    
#     # Actions: FORCE to (batch_size, flattened_action_dim) consistently
#     actions_list = pytorch_batch.get_actions()
    
#     # Flatten each action to 1D, regardless of format
#     flattened_actions = []
#     for action in actions_list:
#         if isinstance(action, torch.Tensor):
#             flattened_actions.append(action.reshape(-1))
#         elif isinstance(action, (tuple, list)):
#             action_tensor = torch.tensor(action, dtype=torch.float32)
#             flattened_actions.append(action_tensor.reshape(-1))
#         else:
#             # Single scalar
#             flattened_actions.append(torch.tensor([action], dtype=torch.float32))
    
#     # Stack to (batch_size, flattened_action_dim) - ALWAYS 2D
#     actions = torch.stack(flattened_actions)
#     actions = jnp.array(actions.detach().cpu().numpy())
    
#     # Rewards: (batch_size,) - already 1D
#     if pytorch_batch.proxy is not None:
#         rewards_tensor = pytorch_batch.get_rewards()
#         rewards = jnp.array(rewards_tensor.detach().cpu().numpy())
#     else:
#         # Placeholder: Use zeros if proxy is missing
#         rewards = jnp.zeros(len(trajectory_indices))
#         print("Warning: Proxy not set on batch, using zero rewards placeholder")
    
#     # Logprobs forward: (batch_size,) - already 1D
#     logprobs_tensor, _ = pytorch_batch.get_logprobs(backward=False)
#     logprobs = jnp.array(logprobs_tensor.detach().cpu().numpy())
    
#     # Logprobs backward: (batch_size,) - already 1D
#     logprobs_rev_tensor, _ = pytorch_batch.get_logprobs(backward=True)
#     logprobs_rev = jnp.array(logprobs_rev_tensor.detach().cpu().numpy())
    
#     return BatchArrays(
#         states=states,  # (batch_size, flattened_state_dim)
#         actions=actions,  # (batch_size, flattened_action_dim)
#         rewards=rewards,  # (batch_size,)
#         logprobs=logprobs,  # (batch_size,)
#         logprobs_rev=logprobs_rev,  # (batch_size,)
#         trajectory_indices=trajectory_indices,  # (batch_size,)
#     )


# # ============================================================================
# # OPTIMIZER SETUP
# # ============================================================================

# def build_optimizer_jax(config, n_train_steps):
#     """
#     Build Optax optimizer matching make_opt from gflownet.py.
    
#     This mirrors the logic in make_opt:
#     - Uses Adam optimizer with config.lr, adam_beta1, adam_beta2
#     - Implements StepLR schedule with lr_decay_period and lr_decay_gamma
#     - Handles clip_grad_norm
    
#     Note: logZ parameter group with lr_z_mult is handled separately in params
#     """
#     # Extract LR schedule parameters (mirrors make_opt StepLR)
#     step_size = config.gflownet.optimizer.lr_decay_period
#     gamma = config.gflownet.optimizer.lr_decay_gamma
#     initial_lr = config.gflownet.optimizer.lr

#     # Create step boundaries for schedule
#     num_steps = n_train_steps // step_size
#     boundaries_and_scales = {
#         (i * step_size): gamma 
#         for i in range(1, num_steps + 1)
#     }
    
#     lr_schedule = optax.piecewise_constant_schedule(
#         init_value=initial_lr,
#         boundaries_and_scales=boundaries_and_scales
#     )
    
#     # Build optimizer chain (mirrors make_opt)
#     optimizer = optax.chain(
#         optax.clip_by_global_norm(config.gflownet.optimizer.clip_grad_norm) if config.gflownet.optimizer.clip_grad_norm > 0 else optax.identity(),
#         optax.adam(
#             learning_rate=lr_schedule,
#             b1=config.gflownet.optimizer.adam_beta1,
#             b2=config.gflownet.optimizer.adam_beta2,
#         ),
#     )
    
#     return optimizer


# def build_state_from_agent(agent: Any, config: Any) -> Tuple[TrainingState, Any]:
#     """
#     Initialize TrainingState from agent.
#     Mirrors build_state_from_agent from pure.py.
#     """
#     optimizer = build_optimizer_jax(config, agent.n_train_steps)
    
#     # TODO: Get JAX parameters from agent
#     # This requires the agent's models to be JAX-compatible (Flax/Equinox)
#     # For now, using placeholder
#     params = agent.get_params() if hasattr(agent, 'get_params') else {'dummy': jnp.zeros(10)}
    
#     # Initialize optimizer state
#     opt_state = optimizer.init(params)
    
#     return TrainingState(
#         params=params,
#         optimizer_state=opt_state,
#         rng_key=random.PRNGKey(config.seed),
#         iteration=agent.it,
#         loss_ema=0.0,  # Initialize loss EMA
#     ), optimizer


# # ============================================================================
# # MAIN TRAINING LOOP
# # ============================================================================

# def train_old(agent: Any, config: Any) -> TrainingState:
#     """
#     Main training loop using JAX control flow.
#     Mirrors the logic from pure.py:train and gflownet.py:train.
    
#     Uses lax.fori_loop for the main training iterations to enable JIT compilation
#     of the core loop, while keeping side effects (logging, checkpointing) outside.
#     """
#     # Build initial state
#     state, optimizer = build_state_from_agent(agent, config)
    
#     # Progress bar (non-JIT-able, acceptable side effect)
#     pbar = tqdm(
#         initial=state.iteration - 1,
#         total=agent.n_train_steps,
#         disable=agent.logger.progressbar.get("skip", False),
#     )
    
#     # TODO: Define JAX-compatible loss function
#     # This requires converting the loss module to JAX
#     def loss_fn_placeholder(params, batch, rng_key):
#         """
#         TODO: CRITICAL CONVERSION NEEDED
        
#         The loss computation requires:
#         1. Converting forward_policy, backward_policy, state_flow to JAX models
#         2. Implementing pure JAX versions of:
#            - compute_logprobs_trajectories
#            - Flow matching loss computation
#            - Trajectory balance loss computation
        
#         This is a major undertaking and should be done separately.
#         For now, this is a placeholder that returns a dummy loss.
#         """
#         return jnp.sum(params['dummy'])
    
#     # Training body for lax.fori_loop
#     def training_body(i, carry):
#         """
#         Single training iteration.
#         Mirrors the main loop in gflownet.py:train and pure.py:train.
        
#         Note: This cannot be fully JIT-compiled due to:
#         - Batch sampling (uses PyTorch agent)
#         - Buffer updates (uses mutable buffer)
#         - Logging (I/O operations)
        
#         However, the core gradient computation (jitted_train_step) IS JIT-compiled.
#         """
#         state = carry
        
#         # Evaluation (IMPURE: I/O, acceptable)
#         # Mirrors: if self.evaluator.should_eval(self.it)
#         should_eval = agent.evaluator.should_eval(i)
#         should_eval_top_k = agent.evaluator.should_eval_top_k(i)
        
#         def do_eval(_):
#             try:
#                 agent.evaluator.eval_and_log(i)
#             except AttributeError as e:
#                 print(f"Warning: Evaluation failed at step {i}: {e}")
#             return None
        
#         def do_eval_top_k(_):
#             try:
#                 agent.evaluator.eval_and_log_top_k(i)
#             except AttributeError as e:
#                 print(f"Warning: Top-k evaluation failed at step {i}: {e}")
#             return None
        
#         def no_eval(_):
#             return None
        
#         lax.cond(should_eval, do_eval, no_eval, None)
#         lax.cond(should_eval_top_k, do_eval_top_k, no_eval, None)
        
#         # lax.cond(should_eval, do_eval, no_eval, None)
#         # lax.cond(should_eval_top_k, 
#         #         lambda _: agent.evaluator.eval_and_log_top_k(i), 
#         #         no_eval, None)
        
#         # Sample batches (IMPURE: uses PyTorch agent)
#         # Mirrors: for j in range(self.sttr): ... batch.merge(sub_batch)
#         rng_key, merged_batch = sample_batch_jax(
#             agent, 
#             state.rng_key, 
#             agent.batch_size, 
#             agent.sttr
#         )
        
#         # JIT-compiled training step (PURE JAX)
#         # Mirrors: for j in range(self.ttsr): ... losses.backward() ... opt.step()
#         new_state, metrics = jitted_train_step(
#             state._replace(rng_key=rng_key),
#             merged_batch,
#             loss_fn_placeholder,
#             optimizer,
#             agent.clip_grad_norm,
#             agent.ttsr,
#         )
        
#         traj_length_mean, _, _ = compute_trajectory_stats(merged_batch.trajectory_indices)
#         metrics = metrics._replace(trajectory_length_mean=traj_length_mean)

        
#         # Buffer updates (IMPURE: mutates buffer)
#         # TODO: Make buffer functional (return new buffer instead of mutating)
#         # Mirrors: self.buffer.add(..., buffer="main") and buffer="replay"
        
#         # DISABLED FOR NOW: Buffer updates cause tracer conversion errors
#         # The buffer tries to convert JAX arrays to numpy/pandas inside the traced loop
#         # TODO: Either move buffer updates outside lax.fori_loop or convert buffer to JAX
#         # states_term = merged_batch.states  # TODO: Extract terminating states
#         # actions_traj = merged_batch.actions  # TODO: Extract action trajectories
#         # rewards = merged_batch.rewards
#         # 
#         # def update_main_buffer(_):
#         #     agent.buffer.add(states_term, actions_traj, rewards, i, buffer="main")
#         #     return None
#         # 
#         # def no_update(_):
#         #     return None
#         # 
#         # lax.cond(agent.buffer.use_main_buffer, update_main_buffer, no_update, None)
#         # agent.buffer.add(states_term, actions_traj, rewards, i, buffer="replay")

#         # Logging (IMPURE: I/O, acceptable)
#         # DISABLED FOR NOW: Logging requires converting traced values to Python floats
#         # TODO: Move logging outside lax.fori_loop or use JAX-compatible logging
#         # should_log = agent.evaluator.should_log_train(i)
#         # 
#         # def do_log(_):
#         #     agent.logger.progressbar_update(
#         #         pbar,
#         #         float(metrics.loss),
#         #         [float(metrics.rewards_mean)],
#         #         agent.jsd,
#         #         agent.use_context,
#         #     )
#         #     return None
#         # 
#         # lax.cond(should_log, do_log, no_eval, None)
        
#         # Progress bar update (IMPURE: I/O, acceptable)
#         # DISABLED FOR NOW: Also causes issues inside traced loop
#         # pbar.update(1)
        
#         # Garbage collection (IMPURE: system side effect, acceptable)
#         # DISABLED FOR NOW: Cannot use inside traced loop
#         # should_gc = (agent.garbage_collection_period > 0 and 
#         #             i % agent.garbage_collection_period == 0)
#         # 
#         # def do_gc(_):
#         #     gc.collect()
#         #     return None
#         # 
#         # lax.cond(should_gc, do_gc, no_eval, None)
        
#         # Checkpointing (IMPURE: file I/O, acceptable)
#         # DISABLED FOR NOW: Checkpoint evaluation uses Python boolean logic on traced values
#         # TODO: Move checkpointing outside lax.fori_loop
#         # should_checkpoint = agent.evaluator.should_checkpoint(i)
#         # 
#         # def do_checkpoint(_):
#         #     agent.logger.save_checkpoint(
#         #         forward_policy=agent.forward_policy,
#         #         backward_policy=agent.backward_policy,
#         #         state_flow=agent.state_flow,
#         #         logZ=agent.logZ,
#         #         optimizer=new_state.optimizer_state,
#         #         buffer=agent.buffer, 
#         #         step=i,
#         #     )
#         #     return None
#         # 
#         # lax.cond(should_checkpoint, do_checkpoint, no_eval, None)
        
#         # Early stopping check (cannot break lax.fori_loop, so just flag it)
#         # DISABLED FOR NOW: Early stopping also uses Python control flow
#         # TODO: Consider using lax.while_loop for early stopping support
#         # should_stop = agent.loss.do_early_stopping(metrics.loss)
#         # 
#         # def do_stop(_):
#         #     # Can't actually stop lax.fori_loop early
#         #     # Would need lax.while_loop for this
#         #     print(
#         #         "Early stopping criteria met: "
#         #         f"{agent.loss.loss_ema} < {agent.loss.early_stopping_th}"
#         #     )
#         #     return None
#         # 
#         # lax.cond(should_stop, do_stop, no_eval, None)
        
#         return new_state
    
#     # Main training loop using lax.fori_loop
#     # Mirrors: for self.it in range(self.it, self.n_train_steps + 1)
#     final_state = lax.fori_loop(
#         state.iteration,
#         agent.n_train_steps + 1,
#         training_body,
#         state
#     )
    
#     print(f"\n✓ JAX training completed! Final iteration: {final_state.iteration}")
#     print(f"  Final loss EMA: {final_state.loss_ema:.4f}")
    
#     # Final checkpoint save (IMPURE: file I/O, acceptable)
#     # Note: JAX uses Optax optimizers (tuples), not PyTorch optimizers
#     # Skip checkpoint save for now since it expects PyTorch format
#     # TODO: Implement JAX-compatible checkpoint saving
#     # agent.logger.save_checkpoint(
#     #     forward_policy=agent.forward_policy,
#     #     backward_policy=agent.backward_policy,
#     #     state_flow=agent.state_flow,
#     #     logZ=agent.logZ,
#     #     optimizer=final_state.optimizer_state,
#     #     buffer=agent.buffer,
#     #     step=final_state.iteration,
#     #     final=True,
#     # )
#     print("  (Checkpoint saving disabled for JAX trainer - models are still PyTorch)")
    
#     # Close logger (IMPURE: I/O cleanup, acceptable)
#     if not agent.use_context:
#         agent.logger.end()
    
#     pbar.close()
    
#     return final_state


# ============================================================================
# FUNCTIONAL JAX REFACTOR OF GFlowNetAgent
# ============================================================================

from typing import NamedTuple, Any, Tuple, List, Optional
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import optax
from gflownet.utils.batch import Batch  # Assuming this gets converted to JAX too

class GFlowNetState(NamedTuple):
    """Immutable state for GFlowNet agent (JAX pytree-compatible)"""
    # Core components
    env_maker: Any  # Function to create env instances
    env: Any        # Current env state (must be JAX-compatible)
    proxy: Any      # Proxy model
    loss_fn: Any    # Loss computation function (pure)
    logZ: jnp.ndarray  # Log partition function
    forward_policy: Any  # JAX policy model
    backward_policy: Any
    state_flow: Any
    buffer: Any     # Functional buffer (JAX arrays)
    optimizer_state: Any  # Optax optimizer state
    evaluator: Any  # Evaluator (may need functional refactoring)
    logger: Any     # Logger (handle side effects carefully)
    
    # Training state
    it: int
    rng: jnp.ndarray  # JAX random key
    device: str
    float_precision: int
    
    # Configuration
    mask_invalid_actions: bool
    temperature_logits: float
    random_action_prob: float
    n_train_steps: int
    batch_size: Any  # Dict-like structure
    ttsr: int  # train-to-sample ratio
    sttr: int  # sample-to-train ratio
    clip_grad_norm: float
    tau: float
    use_context: bool
    collect_backwards_masks: bool
    collect_reversed_logprobs: bool
    garbage_collection_period: int
    
    # Metrics
    l1: float
    kl: float
    jsd: float
    corr_prob_traj_rewards: float
    var_logrewards_logp: float
    nll_tt: float
    mean_logprobs_std: float
    mean_probs_std: float
    logprobs_std_nll_ratio: float
    
    # Environment cache (list of env instances)
    env_cache: List[Any]


# ============================================================================
# PURE FUNCTIONS (JIT-compatible)
# ============================================================================

@jit
def pure_sample_actions(
    state: GFlowNetState,
    envs: List[Any],  # List of JAX-compatible env states
    batch: Optional[Any] = None,  # JAX-compatible batch
    env_cond: Optional[Any] = None,
    sampling_method: str = "policy",
    backward: bool = False,
    temperature: Optional[float] = None,
    random_action_prob: Optional[float] = None,
    no_random: bool = True,
    compute_reversed_logprobs: bool = False,
) -> Tuple[List[Tuple], jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Pure JAX version of sample_actions.
    All inputs are explicit, no mutation of state.
    
    This requires:
    - envs to be JAX pytrees (immutable states)
    - batch to be JAX-compatible
    - All computations to be pure JAX operations
    """
    # Use state fields explicitly
    forward_policy = state.forward_policy
    backward_policy = state.backward_policy
    mask_invalid_actions = state.mask_invalid_actions
    temperature_logits = state.temperature_logits
    random_action_prob_default = state.random_action_prob
    rng = state.rng
    
    # Handle parameters (mirrors original logic)
    if sampling_method == "random":
        random_action_prob = 1.0
        temperature = 1.0
    elif no_random:
        temperature = 1.0
        random_action_prob = 0.0
    else:
        if temperature is None:
            temperature = temperature_logits
        if random_action_prob is None:
            random_action_prob = random_action_prob_default
    
    # Select policy
    model = backward_policy if backward else forward_policy
    model_rev = forward_policy if backward else backward_policy
    
    # Build states (pure operation)
    states = [env.state for env in envs]  # Assuming env is a pytree with .state
    
    # Get masks (pure function call)
    mask_invalid_actions_result = pure_get_masks(
        state, envs, batch, env_cond, backward, backward
    ) if mask_invalid_actions else None
    
    # Get policy inputs (pure)
    states_policy = jnp.array([state.env.states2policy(s) for s in states])
    
    # Forward pass (pure)
    policy_outputs = model(states_policy)
    
    # Sample actions (pure JAX random)
    # This requires implementing env.sample_actions_batch in pure JAX
    actions, logprobs = pure_sample_actions_batch(
        rng, policy_outputs, mask_invalid_actions_result, 
        states, backward, random_action_prob, temperature
    )
    
    # Handle reversed logprobs if needed
    logprobs_rev = None
    if compute_reversed_logprobs:
        # Pure computation of reversed logprobs
        logprobs_rev = pure_compute_reversed_logprobs(
            state, envs, batch, model_rev, policy_outputs, 
            actions, mask_invalid_actions_result, states, backward
        )
    
    # Return actions, logprobs, logprobs_rev (no state mutation)
    return actions, logprobs, logprobs_rev


def pure_get_masks(
    state: GFlowNetState,
    envs: List[Any],
    batch: Optional[Any] = None,
    env_cond: Optional[Any] = None,
    is_backward_mask: bool = False,
    is_backward_traj: bool = False,
) -> Optional[List[List[bool]]]:
    """
    Pure version of _get_masks.
    """
    if not state.mask_invalid_actions:
        return None
    
    # Pure mask computation (requires JAX-compatible env methods)
    if batch is not None:
        # Get masks from batch (pure)
        if is_backward_mask:
            masks = [batch.get_item("mask_backward", env, backward=is_backward_traj) 
                    for env in envs]
        else:
            masks = [batch.get_item("mask_forward", env, backward=is_backward_traj) 
                    for env in envs]
    else:
        # Compute masks from envs (pure)
        if is_backward_mask:
            masks = [env.get_mask_invalid_actions_backward() for env in envs]
        else:
            masks = [env.get_mask_invalid_actions_forward() for env in envs]
    
    # Apply conditioning if needed (pure)
    if env_cond is not None:
        masks = [env.mask_conditioning(mask, env_cond, is_backward_mask) 
                for env, mask in zip(envs, masks)]
    
    return masks


def pure_sample_actions_batch(
    rng: jnp.ndarray,
    policy_outputs: jnp.ndarray,
    masks: Optional[List[List[bool]]],
    states_from: List[Any],
    is_backward: bool,
    random_action_prob: float,
    temperature: float,
) -> Tuple[List[Tuple], jnp.ndarray]:
    """
    Pure JAX implementation of env.sample_actions_batch.
    This requires the environment to provide pure JAX sampling functions.
    """
    # Placeholder: This needs to be implemented based on your env's sampling logic
    # For now, return dummy values
    n_envs = len(states_from)
    actions = [(0, 0)] * n_envs  # Dummy actions
    logprobs = jnp.zeros(n_envs)  # Dummy logprobs
    return actions, logprobs


def pure_compute_reversed_logprobs(
    state: GFlowNetState,
    envs: List[Any],
    batch: Any,
    model_rev: Any,
    policy_outputs: jnp.ndarray,
    actions: List[Tuple],
    mask_invalid_actions_rev: Optional[List[List[bool]]],
    states: List[Any],
    backward: bool,
) -> jnp.ndarray:
    """
    Pure computation of reversed logprobs.
    """
    # Placeholder implementation
    return jnp.zeros(len(actions))


@jit
def pure_step(
    state: GFlowNetState,
    envs: List[Any],
    actions: List[Tuple],
    backward: bool = False,
) -> Tuple[List[Any], List[Tuple], List[bool]]:
    """
    Pure version of step method.
    Returns new env states instead of mutating.
    """
    if backward:
        # Pure backward stepping
        results = [env.step_backwards(action) for env, action in zip(envs, actions)]
    else:
        # Pure forward stepping  
        results = [env.step(action) for env, action in zip(envs, actions)]
    
    # Unpack results: (new_env, action, valid)
    new_envs, actions_out, valids = zip(*results)
    return list(new_envs), list(actions_out), list(valids)


def pure_get_env_instances(
    state: GFlowNetState,
    nb_env_instances: int,
) -> Tuple[GFlowNetState, List[Any]]:
    """
    Pure version of get_env_instances.
    Returns updated state with new env cache.
    """
    env_maker = state.env_maker
    env_cache = state.env_cache
    
    # Create new instances if needed
    if len(env_cache) < nb_env_instances:
        nb_new = nb_env_instances - len(env_cache)
        new_instances = [env_maker() for _ in range(nb_new)]
        env_cache = env_cache + new_instances
    
    # Return requested instances and updated state
    instances = env_cache[:nb_env_instances]
    new_state = state._replace(env_cache=env_cache)
    
    return new_state, instances


def pure_sample_batch(
    state: GFlowNetState,
    n_forward: int = 0,
    n_train: int = 0,
    n_replay: int = 0,
    env_cond: Optional[Any] = None,
    train: bool = True,
) -> Tuple[GFlowNetState, Any, dict]:  # Returns updated state, batch, times
    """
    Pure JAX version of sample_batch.
    Uses lax.scan for trajectory sampling loops.
    """
    # Get env instances (pure)
    state, env_instances = pure_get_env_instances(state, n_forward + n_train + n_replay)
    
    # Initialize batch (pure)
    batch = Batch(...)  # JAX-compatible batch
    
    times = {"all": 0.0, "forward_actions": 0.0, "train_actions": 0.0, "replay_actions": 0.0}
    
    # Forward trajectories (using lax.scan instead of while loop)
    if n_forward > 0:
        envs = [env.reset(idx) for idx, env in enumerate(env_instances[:n_forward])]
        env_instances = env_instances[n_forward:]
        
        def forward_step(carry, _):
            current_envs, current_batch, current_rng = carry
            
            # Sample actions (pure)
            actions, logprobs, logprobs_rev = pure_sample_actions(
                state, current_envs, current_batch, env_cond, 
                no_random=not train, compute_reversed_logprobs=state.collect_reversed_logprobs
            )
            
            # Step environments (pure)
            new_envs, actions_out, valids = pure_step(state, current_envs, actions)
            
            # Add to batch (pure)
            new_batch = pure_add_to_batch(current_batch, new_envs, actions_out, 
                                         logprobs, logprobs_rev, valids, train=train)
            
            # Filter unfinished trajectories
            active_envs = [env for env in new_envs if not env.done]
            
            return (active_envs, new_batch, current_rng), None
        
        # Use lax.while_loop for the trajectory sampling
        def cond_fun(carry):
            envs, _, _ = carry
            return len(envs) > 0
        
        init_carry = (envs, batch, state.rng)
        final_carry, _ = lax.while_loop(cond_fun, forward_step, init_carry)
        _, batch, _ = final_carry
    
    # Similar logic for train and replay trajectories...
    # (Implementation would mirror forward trajectories with backward=True)
    
    return state, batch, times


def pure_add_to_batch(batch, envs, actions, logprobs, logprobs_rev, valids, 
                     backward=False, train=True):
    """
    Pure version of batch.add_to_batch.
    Returns new batch instead of mutating.
    """
    # Implementation depends on Batch structure
    # This would need to be converted to functional style
    return batch  # Placeholder


@jit
def pure_train_step(
    state: GFlowNetState,
    batch: Any,
) -> Tuple[GFlowNetState, dict]:  # Returns updated state and metrics
    """
    Pure JAX training step.
    Mirrors the inner training loop in original train().
    """
    # Compute loss (pure)
    loss_value = state.loss_fn(state.params, batch, state.rng)
    
    # Compute gradients (pure)
    grads = jax.grad(state.loss_fn)(state.params, batch, state.rng)
    
    # Clip gradients (pure)
    if state.clip_grad_norm > 0:
        grad_norm = optax.global_norm(grads)
        scale = jnp.minimum(1.0, state.clip_grad_norm / (grad_norm + 1e-6))
        grads = jax.tree.map(lambda g: g * scale, grads)
    
    # Update parameters (pure with Optax)
    updates, new_opt_state = state.optimizer.update(grads, state.optimizer_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    
    # Update state
    new_state = state._replace(
        params=new_params,
        optimizer_state=new_opt_state,
        it=state.it + 1,
    )
    
    metrics = {"loss": loss_value}
    return new_state, metrics


def pure_train(
    state: GFlowNetState,
) -> GFlowNetState:
    """
    Pure JAX training loop.
    Uses lax.fori_loop for the main training iterations.
    """
    
    def training_iteration(state, _):
        # Sample batch (pure)
        state, batch, _ = pure_sample_batch(
            state,
            n_forward=state.batch_size['forward'],
            n_train=state.batch_size['backward_dataset'], 
            n_replay=state.batch_size['backward_replay'],
            collect_forwards_masks=True,
            collect_backwards_masks=state.collect_backwards_masks,
        )
        
        # Train step (pure, JIT-compiled)
        state, metrics = pure_train_step(state, batch)
        
        # Handle side effects (logging, evaluation) outside JIT
        # These would be returned as data or handled separately
        
        return state, None
    
    # Main training loop using lax.fori_loop
    final_state, _ = lax.fori_loop(
        state.it,
        state.n_train_steps + 1,
        training_iteration,
        state
    )
    
    return final_state


# ============================================================================
# INITIALIZATION
# ============================================================================

def create_gflownet_state(config, env_maker, proxy, loss_fn, forward_policy, 
                         backward_policy, state_flow, buffer, evaluator, logger) -> GFlowNetState:
    """
    Create initial GFlowNetState from configuration.
    Mirrors the __init__ logic but returns immutable state.
    """
    
    # Initialize components (similar to original __init__)
    env = env_maker()
    proxy.setup(env)
    
    # Handle logZ
    logZ = jnp.ones(config.gflownet.optimizer.z_dim) * 150.0 / 64 if loss_fn.requires_log_z else None
    
    # Initialize optimizer state (Optax)
    # For now, placeholder - need to extract actual params from policies
    params = {}  # TODO: Extract JAX params from policies
    optimizer = optax.adam(learning_rate=config.gflownet.optimizer.lr)
    opt_state = optimizer.init(params)
    
    # Compute batch sizes and ratios
    batch_size = config.gflownet.optimizer.batch_size
    batch_size_total = sum(batch_size.values())
    ttsr = max(int(config.gflownet.optimizer.train_to_sample_ratio), 1)
    sttr = max(int(1 / config.gflownet.optimizer.train_to_sample_ratio), 1)
    
    # Create initial state
    state = GFlowNetState(
        env_maker=env_maker,
        env=env,
        proxy=proxy,
        loss_fn=loss_fn,
        logZ=logZ,
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        state_flow=state_flow,
        buffer=buffer,
        optimizer_state=opt_state,
        evaluator=evaluator,
        logger=logger,
        it=1,
        rng=jax.random.PRNGKey(config.gflownet.seed),
        device=config.device,
        float_precision=config.float_precision,
        mask_invalid_actions=config.gflownet.mask_invalid_actions,
        temperature_logits=config.gflownet.temperature_logits,
        random_action_prob=config.gflownet.random_action_prob,
        n_train_steps=config.gflownet.optimizer.n_train_steps,
        batch_size=batch_size,
        ttsr=ttsr,
        sttr=sttr,
        clip_grad_norm=config.gflownet.optimizer.clip_grad_norm,
        tau=config.gflownet.optimizer.bootstrap_tau,
        use_context=config.gflownet.use_context,
        collect_backwards_masks=loss_fn.requires_backward_policy(),
        collect_reversed_logprobs=config.gflownet.collect_reversed_logprobs,
        garbage_collection_period=config.gflownet.garbage_collection_period,
        l1=-1.0,
        kl=-1.0,
        jsd=-1.0,
        corr_prob_traj_rewards=0.0,
        var_logrewards_logp=-1.0,
        nll_tt=0.0,
        mean_logprobs_std=-1.0,
        mean_probs_std=-1.0,
        logprobs_std_nll_ratio=-1.0,
        env_cache=[],
    )
    
    return state


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def train(agent, config):  # train_gflownet_jax
    """
    Example of how to use the functional JAX GFlowNet.
    """
    
    # Initialize components (PyTorch for now, convert to JAX gradually)
    # Follow the same instantiation logic as gflownet_from_config
    logger = instantiate(config.logger, config, _recursive_=False)
    
    # Proxy
    proxy = instantiate(
        config.proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    
    # Environment maker (partial instantiation)
    env_maker = instantiate(
        config.env,
        device=config.device,
        float_precision=config.float_precision,
        _partial_=True,
    )
    env = env_maker()  # Create actual env instance
    
    # Setup proxy with env
    proxy.setup(env)
    
    # Buffer
    buffer = instantiate(
        config.buffer,
        env=env,
        proxy=proxy,
        datadir=logger.datadir,
    )
    
    # Evaluator
    evaluator = instantiate(config.evaluator)
    
    # Parse policy configs (important for JAX vs PyTorch selection)
    forward_config = parse_policy_config(config, kind="forward")
    backward_config = parse_policy_config(config, kind="backward")
    
    # Policies
    forward_policy = instantiate(
        forward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
    )
    backward_policy = instantiate(
        backward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
        base=forward_policy,
    )
    
    # State flow
    state_flow = None
    if config.gflownet.state_flow is not None:
        # Switch to JAX version if trainer mode is jax
        if config.trainer.mode == "jax":
            config.gflownet.state_flow._target_ = "gflownet.policy.state_flow_jax.StateFlowJAX"
        state_flow = instantiate(
            config.gflownet.state_flow,
            env=env,
            device=config.device,
            float_precision=config.float_precision,
            base=forward_policy,
        )
    
    # Loss
    loss_fn = instantiate(
        config.loss,
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        state_flow=state_flow,
        device=config.device,
        float_precision=config.float_precision,
    )
    
    # Create initial state
    initial_state = create_gflownet_state(
        config, env_maker, proxy, loss_fn, forward_policy, 
        backward_policy, state_flow, buffer, evaluator, logger
    )
    
    # Train (pure function)
    final_state = pure_train(initial_state)
    
    # Handle side effects (logging, saving) with final state
    # ...
    
    return final_state