"""
Minimal JAX trainer - Phase 1: JIT only the backpropagation
Keeps PyTorch GFlowNetAgent but JITs gradient computation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad, grad
import optax
from functools import partial
from tqdm import tqdm

from gflownet.utils.common import instantiate, set_device
from gflownet.utils.batch import Batch


# ============================================================================
# PHASE 1: Minimal JAX - Only JIT the gradient step
# ============================================================================

def convert_batch_to_jax_arrays(pytorch_batch: Batch):
    """
    Extract JAX-compatible arrays from PyTorch Batch.
    This is the bridge between PyTorch sampling and JAX training.
    """
    import torch
    
    # Get trajectory indices (consecutive numbering for grouping)
    traj_indices = pytorch_batch.get_trajectory_indices(consecutive=True)
    trajectory_indices = jnp.array(traj_indices.detach().cpu().numpy())
    
    # Compute n_trajs concretely (outside JIT)
    traj_indices_np = traj_indices.detach().cpu().numpy()
    n_trajs = int(np.max(traj_indices_np)) + 1
    
    # Get states for policy input
    states_policy = pytorch_batch.get_states(policy=True)
    if isinstance(states_policy, torch.Tensor):
        states = jnp.array(states_policy.detach().cpu().numpy())
    else:
        # Handle list of states
        states = jnp.array([s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else s 
                           for s in states_policy])
    
    # Get actions
    actions_list = pytorch_batch.get_actions()
    # Flatten actions to consistent format
    actions_flat = []
    for action in actions_list:
        if isinstance(action, (tuple, list)):
            actions_flat.append(list(action))
        else:
            actions_flat.append([action])
    # Pad to same length and convert
    max_len = max(len(a) for a in actions_flat)
    actions_padded = [a + [0] * (max_len - len(a)) for a in actions_flat]
    actions = jnp.array(actions_padded, dtype=jnp.float32)
    
    # Get rewards - CRITICAL: Check if proxy is set first
    if pytorch_batch.proxy is None:
        # If no proxy, use dummy rewards (all zeros)
        # This shouldn't happen in real training but helps with testing
        print("WARNING: Batch has no proxy, using zero rewards")
        rewards = jnp.zeros(len(trajectory_indices))
    else:
        rewards_tensor = pytorch_batch.get_rewards()
        rewards = jnp.array(rewards_tensor.detach().cpu().numpy())
    
    # Get logprobs (forward and backward)
    logprobs_fwd, _ = pytorch_batch.get_logprobs(backward=False)
    logprobs_bwd, _ = pytorch_batch.get_logprobs(backward=True)
    logprobs = jnp.array(logprobs_fwd.detach().cpu().numpy())
    logprobs_rev = jnp.array(logprobs_bwd.detach().cpu().numpy())
    
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'logprobs': logprobs,
        'logprobs_rev': logprobs_rev,
        'trajectory_indices': trajectory_indices,
        'n_trajs': n_trajs,  # Add this
    }


def convert_params_to_jax(agent):
    """
    Extract trainable parameters from PyTorch models and convert to JAX.
    """
    import torch
    
    params = {}
    
    # Forward policy parameters
    if (hasattr(agent.forward_policy, 'model') and 
        hasattr(agent.forward_policy.model, 'parameters') and
        hasattr(agent.forward_policy, 'is_model') and 
        agent.forward_policy.is_model):
        params['forward_policy'] = {
            name: jnp.array(param.detach().cpu().numpy())
            for name, param in agent.forward_policy.model.named_parameters()
        }
    else:
        params['forward_policy'] = {}
    
    # Backward policy parameters (if exists and not shared)
    if (hasattr(agent.backward_policy, 'model') and 
        hasattr(agent.backward_policy.model, 'parameters') and
        hasattr(agent.backward_policy, 'is_model') and
        agent.backward_policy.is_model and
        not agent.backward_policy.shared_weights):
        params['backward_policy'] = {
            name: jnp.array(param.detach().cpu().numpy())
            for name, param in agent.backward_policy.model.named_parameters()
        }
    else:
        params['backward_policy'] = {}
    
    # LogZ if needed
    if agent.logZ is not None:
        params['logZ'] = jnp.array(agent.logZ.data.detach().cpu().numpy())
    else:
        params['logZ'] = None
    
    return params


def apply_params_to_pytorch(agent, jax_params):
    """
    Copy JAX parameters back to PyTorch models.
    This allows us to continue using PyTorch models for sampling.
    """
    import torch
    import numpy as np
    
    # Update forward policy
    if jax_params['forward_policy'] and hasattr(agent.forward_policy, 'is_model') and agent.forward_policy.is_model:
        for name, param in agent.forward_policy.model.named_parameters():
            if name in jax_params['forward_policy']:
                param.data = torch.from_numpy(
                    np.array(jax_params['forward_policy'][name])
                ).to(param.device, param.dtype)
    
    # Update backward policy
    if (jax_params['backward_policy'] and 
        hasattr(agent.backward_policy, 'is_model') and 
        agent.backward_policy.is_model):
        for name, param in agent.backward_policy.model.named_parameters():
            if name in jax_params['backward_policy']:
                param.data = torch.from_numpy(
                    np.array(jax_params['backward_policy'][name])
                ).to(param.device, param.dtype)
    
    # Update logZ
    if jax_params['logZ'] is not None and agent.logZ is not None:
        agent.logZ.data = torch.from_numpy(
            np.array(jax_params['logZ'])
        ).to(agent.logZ.device, agent.logZ.dtype)


@partial(jit, static_argnames=['loss_type', 'n_trajs'])
def jax_loss_wrapper(params, batch_arrays, loss_type='trajectorybalance', n_trajs=None):
    """
    JAX-compatible loss computation.
    
    Phase 1: Simple wrapper that computes a basic loss from batch data.
    This is a placeholder - you'll need to implement the actual loss logic.
    
    For trajectory balance: loss = (log_pF - log_pB + log_R - logZ)^2
    """
    
    if loss_type == 'trajectorybalance':
        
        # Use the passed n_trajs (static)
        traj_indices = jnp.arange(n_trajs)
        
        # Compute per-trajectory losses
        def compute_traj_loss(traj_idx):
            # Get indices for this trajectory
            mask = (batch_arrays['trajectory_indices'] == traj_idx).astype(jnp.float32)
            
            # Sum logprobs along trajectory
            log_pF = jnp.sum(batch_arrays['logprobs'] * mask)
            log_pB = jnp.sum(batch_arrays['logprobs_rev'] * mask)
            
            # Get terminal reward (last state in trajectory)
            # We take the maximum reward in the trajectory (should be the terminal state)
            rewards_traj = batch_arrays['rewards'] * mask
            terminal_reward = jnp.max(rewards_traj)
            log_R = jnp.log(terminal_reward + 1e-8)  # Safe log
            
            # LogZ (scalar or per-trajectory)
            logZ = params.get('logZ', jnp.array(0.0))
            if logZ is None:
                logZ = jnp.array(0.0)
            if logZ.ndim > 0:
                logZ = jnp.sum(logZ)  # Sum the logZ vector
            
            # Trajectory balance loss
            traj_loss = (log_pF - log_pB + log_R - logZ) ** 2
            return traj_loss
        
        # Compute loss for all trajectories
        losses = jax.vmap(compute_traj_loss)(traj_indices)
        
        # Mean loss
        return jnp.mean(losses)
    
    else:
        # Fallback: simple MSE on logprobs
        return jnp.mean((batch_arrays['logprobs'] - batch_arrays['logprobs_rev']) ** 2)


@partial(jit, static_argnames=['optimizer', 'loss_type', 'n_trajs'])
def jax_grad_step(params, opt_state, batch_arrays, optimizer, loss_type='trajectorybalance', n_trajs=None):
    """
    JIT-compiled gradient step.
    This is the ONLY function that needs to be pure JAX.
    """
    # Compute loss and gradients
    loss_value, grads = value_and_grad(jax_loss_wrapper)(
        params, batch_arrays, loss_type=loss_type, n_trajs=n_trajs
    )
    
    # loss_value = jax_loss_wrapper(params, batch_arrays, loss_type=loss_type, n_trajs=n_trajs, debug=True)
    # grads = grad(lambda p: jax_loss_wrapper(p, batch_arrays, loss_type=loss_type, n_trajs=n_trajs, debug=False))(params)
    
    # Apply optimizer update
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss_value, grads


def train(agent, config):
    """
    Phase 1 JAX trainer: Uses PyTorch agent for sampling, JAX for backprop.
    
    This is the MINIMAL change to get JAX working:
    1. Keep all PyTorch code for sampling, env, buffer
    2. Only convert gradient computation to JAX
    3. Sync parameters between PyTorch and JAX each iteration
    """
    
    print("=" * 80)
    print("PHASE 1 JAX TRAINER: Minimal Integration")
    print("  - PyTorch: Sampling, environment, buffer")
    print("  - JAX: Gradient computation (JIT-compiled)")
    print("=" * 80)
    
    # Setup Optax optimizer
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=config.gflownet.optimizer.lr,
        boundaries_and_scales={
            i * config.gflownet.optimizer.lr_decay_period: config.gflownet.optimizer.lr_decay_gamma
            for i in range(1, config.gflownet.optimizer.n_train_steps // config.gflownet.optimizer.lr_decay_period + 1)
        }
    )
    
    optimizer = optax.adam(
        learning_rate=lr_schedule,
        b1=config.gflownet.optimizer.adam_beta1,
        b2=config.gflownet.optimizer.adam_beta2,
    )
    
    # Initialize JAX parameters from PyTorch agent
    jax_params = convert_params_to_jax(agent)
    
    # Check if we have any trainable parameters
    has_trainable = (
        len(jax_params['forward_policy']) > 0 or 
        len(jax_params['backward_policy']) > 0 or 
        jax_params['logZ'] is not None
    )
    
    if not has_trainable:
        print("WARNING: No trainable parameters found!")
        print("  - Forward policy is not a neural network model")
        print("  - Backward policy is not a neural network model")
        print("  - LogZ is None")
        print("  Training will proceed but parameters won't be updated.")
        # Create a dummy parameter to avoid empty pytree issues
        jax_params['_dummy'] = jnp.array([0.0])
    
    opt_state = optimizer.init(jax_params)
    
    # Determine loss type from config
    loss_type = config.loss.get('_target_', 'trajectorybalance').split('.')[-1].lower()
    
    # Training loop (mostly identical to original)
    total_iterations = agent.n_train_steps - agent.it + 1  # Exact number of iterations
    pbar = tqdm(
        initial=agent.it - 1,
        total=total_iterations,
        disable=agent.logger.progressbar.get("skip", False),
    )
    
    for iteration in range(agent.it, agent.n_train_steps + 1):
        # ========== PYTORCH: Sampling ==========
        # Sample batch using PyTorch (unchanged)
        
        batch = Batch(
            env=agent.env,
            proxy=agent.proxy,
            device=agent.device,
            float_type=agent.float,
        )
        
        for _ in range(agent.sttr):
            sub_batch, _ = agent.sample_batch(
                n_forward=agent.batch_size.forward,
                n_train=agent.batch_size.backward_dataset,
                n_replay=agent.batch_size.backward_replay,
            )
            batch.merge(sub_batch)
        
        # Convert batch to JAX arrays
        batch_arrays = convert_batch_to_jax_arrays(batch)
        
        n_trajs = batch_arrays.pop('n_trajs')  # Extract and remove from dict
        
        # ========== JAX: Training Steps ==========
        # Perform multiple gradient steps (train-to-sample ratio)
        for _ in range(agent.ttsr):
            jax_params, opt_state, loss_value, grads = jax_grad_step(
                jax_params, opt_state, batch_arrays, optimizer, loss_type=loss_type, n_trajs=n_trajs
            )
            
            # Zero out logprobs for next iteration (if needed)
            batch_arrays['logprobs'] = jnp.zeros_like(batch_arrays['logprobs'])
            batch_arrays['logprobs_rev'] = jnp.zeros_like(batch_arrays['logprobs_rev'])
        
        # ========== PYTORCH: Copy parameters back ==========
        # Sync JAX params back to PyTorch (for next sampling iteration)
        apply_params_to_pytorch(agent, jax_params)
        
        # ========== LOGGING & SIDE EFFECTS (unchanged) ==========
        # Update loss EMA
        if agent.loss.loss_ema is None:
            agent.loss.loss_ema = float(loss_value)
        else:
            agent.loss.loss_ema = (
                agent.loss.ema_alpha * float(loss_value) +
                (1 - agent.loss.ema_alpha) * agent.loss.loss_ema
            )
        
        # Logging        
        # if agent.evaluator.should_log_train(iteration):
        #     agent.logger.log_train(
        #         iteration,
        #         {
        #             'loss': float(loss_value),
        #             'loss_ema': agent.loss.loss_ema,
        #             'mean_reward': float(jnp.mean(batch_arrays['rewards'])),
        #         }
        #     )
        
        # Progress bar
        # pbar.update(1)
        # pbar.set_postfix({
        #     'loss': f"{loss_value:.4f}",
        #     'loss_ema': f"{agent.loss.loss_ema:.4f}",
        # })

        agent.logger.progressbar_update(
            pbar, float(loss_value), batch_arrays['rewards'], agent.jsd, agent.use_context
        )
        pbar.update(1)
        
        # Evaluation
        if agent.evaluator.should_eval(iteration):
            agent.evaluator.eval_and_log(iteration)
        
        class DummyOptimizer:
            def state_dict(self):
                return {}  # Return an empty dict as a placeholder
        
        dummy_optimizer = DummyOptimizer()

        # Checkpointing
        if agent.evaluator.should_checkpoint(iteration):
            agent.logger.save_checkpoint(
                forward_policy=agent.forward_policy,
                backward_policy=agent.backward_policy,
                state_flow=agent.state_flow,
                logZ=agent.logZ,
                optimizer=dummy_optimizer,  #! Optax state - skip for now
                buffer=agent.buffer,
                step=iteration,
            )
        
        # Early stopping
        # if agent.loss.do_early_stopping():
        #     print(f"\nEarly stopping at iteration {iteration}")
        #     break
    
    pbar.close()
    
    print("\n" + "=" * 80)
    print("PHASE 1 TRAINING COMPLETE")
    print(f"  Final loss: {loss_value:.4f}")
    print(f"  Final loss EMA: {agent.loss.loss_ema:.4f}")
    print("=" * 80)
    
    return agent
