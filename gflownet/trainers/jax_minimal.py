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
from gflownet.policy.base_jax import PolicyJAX
from gflownet.utils.policy import parse_policy_config


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
    actions = jnp.array([a if isinstance(a, int) else a[0] for a in actions_list], dtype=jnp.int32)
    
    # Get rewards - Use terminating rewards only, as in original PyTorch
    if pytorch_batch.proxy is None:
        # If no proxy, use dummy rewards (all zeros)
        # This shouldn't happen in real training but helps with testing
        print("WARNING: Batch has no proxy, using zero rewards")
        terminating_rewards = jnp.zeros(pytorch_batch.get_n_trajectories())
    else:
        terminating_rewards_tensor = pytorch_batch.get_terminating_rewards(sort_by="trajectory")
        terminating_rewards = jnp.array(terminating_rewards_tensor.detach().cpu().numpy())
    
    # For compatibility with existing code, also get all rewards (but we'll use terminating)
    rewards_tensor = pytorch_batch.get_rewards()
    rewards = jnp.array(rewards_tensor.detach().cpu().numpy())
    
    # Get logprobs (forward and backward)
    logprobs_fwd, _ = pytorch_batch.get_logprobs(backward=False)
    logprobs_bwd, _ = pytorch_batch.get_logprobs(backward=True)
    logprobs = jnp.array(logprobs_fwd.detach().cpu().numpy())
    logprobs_rev = jnp.array(logprobs_bwd.detach().cpu().numpy())
    
    return {
        'states': states,
        'states_policy': states,
        'actions': actions,
        'rewards': rewards,
        'terminating_rewards': terminating_rewards,
        'logprobs': logprobs,
        'logprobs_rev': logprobs_rev,
        'trajectory_indices': trajectory_indices,
        'n_trajs': n_trajs,  # Add this
    }


def convert_params_to_jax(agent, config, key):
    # Map PyTorch dtype to integer precision for JAX
    import torch
    if isinstance(agent.float, torch.dtype):
        if agent.float == torch.float16:
            float_precision = 16
        elif agent.float == torch.float32:
            float_precision = 32
        elif agent.float == torch.float64:
            float_precision = 64
        else:
            float_precision = 32  # Default fallback
    else:
        float_precision = agent.float  # Assume it's already an int
    
    jax_params = {}
    jax_policies = {}
    
    # Parse forward policy config like in common.py
    forward_config = parse_policy_config(config, kind="forward")
    forward_config = forward_config["config"]
    if forward_config is not None:
        # Change target to use JAX policy
        forward_config["_target_"] = "gflownet.policy.base_jax.PolicyJAX"
        # Add key for JAX initialization
        forward_config["key"] = None
        # Don't override state_dim - let it be determined from env.policy_input_dim
        
        jax_policy_f = instantiate(
            forward_config,
            env=agent.env,
            device=agent.device,
            float_precision=float_precision,
        )
        
        jax_policies['forward'] = jax_policy_f
        
        # Sync parameters from PyTorch model
        if agent.forward_policy.is_model:
            pt_params = dict(agent.forward_policy.model.named_parameters())
            # Assuming MLP with 2 Linear layers
            jax_policy_f.model = eqx.tree_at(lambda m: m.layers[0].weight, jax_policy_f.model, jnp.array(pt_params['0.weight'].detach().cpu().numpy()))
            jax_policy_f.model = eqx.tree_at(lambda m: m.layers[0].bias, jax_policy_f.model, jnp.array(pt_params['0.bias'].detach().cpu().numpy()))
            jax_policy_f.model = eqx.tree_at(lambda m: m.layers[2].weight, jax_policy_f.model, jnp.array(pt_params['2.weight'].detach().cpu().numpy()))
            jax_policy_f.model = eqx.tree_at(lambda m: m.layers[2].bias, jax_policy_f.model, jnp.array(pt_params['2.bias'].detach().cpu().numpy()))
            
            # Partition into trainable and static for optax
            trainable, static = eqx.partition(jax_policy_f.model, eqx.is_array)
            jax_params['forward_policy_trainable'] = trainable
            jax_policies['forward_static'] = static
        else:
            jax_params['forward_policy_trainable'] = None
            jax_policies['forward_static'] = None

    # Parse backward policy config
    backward_config = parse_policy_config(config, kind="backward")
    backward_config = backward_config["config"]
    if backward_config is not None:
        # If shared_weights but no base, copy n_hid and n_layers from forward
        if backward_config.get("shared_weights", False) and forward_config is not None:
            if "n_hid" not in backward_config and "n_hid" in forward_config:
                backward_config["n_hid"] = forward_config["n_hid"]
            if "n_layers" not in backward_config and "n_layers" in forward_config:
                backward_config["n_layers"] = forward_config["n_layers"]
        
        # Change target to use JAX policy
        backward_config["_target_"] = "gflownet.policy.base_jax.PolicyJAX"
        # Add key for JAX initialization
        backward_config["key"] = None
        # Don't override state_dim - let it be determined from env.policy_input_dim
        # Don't instantiate yet
        backward_config["instantiate_now"] = False
        
        jax_policy_b = instantiate(
            backward_config,
            env=agent.env,
            device=agent.device,
            float_precision=float_precision,
            # base=None  # Don't pass base to avoid config issues
        )
        
        # Set base after instantiation
        jax_policy_b.base = jax_policies['forward']
        
        # Now instantiate the model
        jax_policy_b.instantiate(jax.random.PRNGKey(0))
        
        jax_policies['backward'] = jax_policy_b
        
        # Sync parameters from PyTorch model
        if agent.backward_policy.is_model:
            pt_params = dict(agent.backward_policy.model.named_parameters())
            jax_policy_b.model = eqx.tree_at(lambda m: m.layers[0].weight, jax_policy_b.model, jnp.array(pt_params['0.0.weight'].detach().cpu().numpy()))
            jax_policy_b.model = eqx.tree_at(lambda m: m.layers[0].bias, jax_policy_b.model, jnp.array(pt_params['0.0.bias'].detach().cpu().numpy()))
            jax_policy_b.model = eqx.tree_at(lambda m: m.layers[2].weight, jax_policy_b.model, jnp.array(pt_params['0.2.weight'].detach().cpu().numpy()))
            jax_policy_b.model = eqx.tree_at(lambda m: m.layers[2].bias, jax_policy_b.model, jnp.array(pt_params['0.2.bias'].detach().cpu().numpy()))
            
            # Partition into trainable and static for optax
            trainable, static = eqx.partition(jax_policy_b.model, eqx.is_array)
            jax_params['backward_policy_trainable'] = trainable
            jax_policies['backward_static'] = static
        else:
            jax_params['backward_policy_trainable'] = None
            jax_policies['backward_static'] = None

    # Handle logZ
    if agent.logZ is not None:
        jax_params['logZ'] = jnp.array(agent.logZ.detach().cpu().numpy())
    else:
        jax_params['logZ'] = None

    return jax_params, jax_policies


import equinox as eqx

class MLP(eqx.Module):
    layers: list

    def __init__(self, in_dim, hidden_dim, out_dim):
        self.layers = [
            eqx.nn.Linear(in_dim, hidden_dim, key=jax.random.PRNGKey(0)),
            eqx.nn.Linear(hidden_dim, out_dim, key=jax.random.PRNGKey(1))
        ]

    def __call__(self, x):
        x = self.layers[0](x)
        x = jax.nn.relu(x)
        x = self.layers[1](x)
        return x

def jax_mlp_forward(params, x):
    """
    JAX implementation using Equinox Linear layers.
    """
    
    dummy_key = jax.random.PRNGKey(0)

    # Create Linear layers from params
    linear1 = eqx.nn.Linear(params['0.weight'].shape[1], params['0.weight'].shape[0], key=dummy_key)
    linear1 = eqx.tree_at(lambda m: m.weight, linear1, params['0.weight'])
    linear1 = eqx.tree_at(lambda m: m.bias, linear1, params['0.bias'])

    linear2 = eqx.nn.Linear(params['2.weight'].shape[1], params['2.weight'].shape[0], key=dummy_key)
    linear2 = eqx.tree_at(lambda m: m.weight, linear2, params['2.weight'])
    linear2 = eqx.tree_at(lambda m: m.bias, linear2, params['2.bias'])
    
    x = linear1(x)
    x = jax.nn.relu(x)
    x = linear2(x)
    return x


def apply_params_to_pytorch(jax_params, agent, jax_policies):
    """
    Copy JAX parameters back to PyTorch models.
    This allows us to continue using PyTorch models for sampling.
    """
    import torch
    import numpy as np
    
    try:
        # Update forward policy
        if 'forward_policy_trainable' in jax_params and jax_params['forward_policy_trainable'] is not None:
            jax_model = eqx.combine(jax_params['forward_policy_trainable'], jax_policies['forward_static'])
            pt_model = agent.forward_policy.model
            pt_model[0].weight.data = torch.tensor(jax_model.layers[0].weight).to(pt_model[0].weight.device)
            pt_model[0].bias.data = torch.tensor(jax_model.layers[0].bias).to(pt_model[0].bias.device)
            pt_model[2].weight.data = torch.tensor(jax_model.layers[2].weight).to(pt_model[2].weight.device)
            pt_model[2].bias.data = torch.tensor(jax_model.layers[2].bias).to(pt_model[2].bias.device)

        # Update backward policy
        if 'backward_policy_trainable' in jax_params and jax_params['backward_policy_trainable'] is not None:
            jax_model = eqx.combine(jax_params['backward_policy_trainable'], jax_policies['backward_static'])
            pt_model = agent.backward_policy.model
            pt_model[0].weight.data = torch.tensor(jax_model.layers[0].weight).to(pt_model[0].weight.device)
            pt_model[0].bias.data = torch.tensor(jax_model.layers[0].bias).to(pt_model[0].bias.device)
            pt_model[2].weight.data = torch.tensor(jax_model.layers[2].weight).to(pt_model[2].weight.device)
            pt_model[2].bias.data = torch.tensor(jax_model.layers[2].bias).to(pt_model[2].bias.device)
        
        # Update logZ
        if jax_params['logZ'] is not None and agent.logZ is not None:
            agent.logZ.data = torch.from_numpy(
                np.array(jax_params['logZ'])
            ).to(agent.logZ.device, agent.logZ.dtype)
    except Exception as e:
        print(f"Warning: Failed to sync JAX parameters back to PyTorch: {e}")
        print("Continuing with JAX-only training (parameters won't be synced back)")


def train(agent, config):
    """
    Phase 1 JAX trainer: Uses PyTorch agent for sampling, JAX for backprop.
    
    This is the MINIMAL change to get JAX working:
    1. Keep all PyTorch code for sampling, env, buffer
    2. Only convert gradient computation to JAX
    3. Sync parameters between PyTorch and JAX each iteration
    """
    
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
    key = jax.random.PRNGKey(42)
    jax_params, jax_policies = convert_params_to_jax(agent, config, key)
    
    # Check if we have any trainable parameters
    has_trainable = (
        'forward_policy_trainable' in jax_params and jax_params['forward_policy_trainable'] is not None or 
        'backward_policy_trainable' in jax_params and jax_params['backward_policy_trainable'] is not None or 
        jax_params.get('logZ') is not None
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
    
    # Define JAX loss wrapper with access to jax_policies
    @partial(jit, static_argnames=['loss_type', 'n_trajs', 'debug'])
    def jax_loss_wrapper(params, batch_arrays, loss_type=loss_type, n_trajs=None, debug=False):
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
                
                # Combine trainable and static parts for forward policy
                if 'forward_policy_trainable' in params and params['forward_policy_trainable'] is not None:
                    model_f = eqx.combine(params['forward_policy_trainable'], jax_policies['forward_static'])
                else:
                    model_f = jax_policies['forward'].model
                
                # Compute logprobs using JAX forward
                # Use vmap to vectorize the model over the batch dimension
                # Equinox models expect single inputs (features,), so we vmap over batch
                logits_f = jax.vmap(model_f)(batch_arrays['states_policy'])
                logprobs_f_all = jax.nn.log_softmax(logits_f, axis=1)
                logprobs_f = logprobs_f_all[jnp.arange(len(batch_arrays['actions'])), batch_arrays['actions']]
                log_pF = jnp.sum(logprobs_f * mask)
                
                # Combine trainable and static parts for backward policy
                if 'backward_policy_trainable' in params and params['backward_policy_trainable'] is not None:
                    model_b = eqx.combine(params['backward_policy_trainable'], jax_policies['backward_static'])
                else:
                    model_b = jax_policies['backward'].model
                
                # Use vmap to vectorize the model over the batch dimension
                logits_b = jax.vmap(model_b)(batch_arrays['states_policy'])
                logprobs_b_all = jax.nn.log_softmax(logits_b, axis=1)
                logprobs_b = logprobs_b_all[jnp.arange(len(batch_arrays['actions'])), batch_arrays['actions']]
                log_pB = jnp.sum(logprobs_b * mask)
                
                # Get terminal reward (from terminating_rewards array)
                log_R = jnp.log(batch_arrays['terminating_rewards'][traj_idx] + 1e-8)  # Safe log
                
                # LogZ (scalar or per-trajectory)
                logZ = params.get('logZ', jnp.array(0.0))
                if logZ is None:
                    logZ = jnp.array(0.0)
                if logZ.ndim > 0:
                    logZ = jnp.sum(logZ)  # Sum the logZ vector
                
                # Trajectory balance loss: log Z + log P_F - log P_B - log R = 0
                traj_loss = (logZ + log_pF - log_pB - log_R) ** 2
                
                # Debug print for first trajectory (will print every iteration)
                # if traj_idx == 0 and debug:
                #     print(f"JAX Debug traj 0: log_pF={log_pF:.4f}, log_pB={log_pB:.4f}, log_R={log_R:.4f}, logZ={logZ:.4f}, reward={batch_arrays['terminating_rewards'][traj_idx]:.4f}")
                #     print(f"JAX Debug loss components: logZ + log_pF - log_pB - log_R = {logZ + log_pF - log_pB - log_R:.4f}, squared = {traj_loss:.4f}")
                
                return traj_loss
            
            # Compute loss for all trajectories
            losses = jax.vmap(compute_traj_loss)(traj_indices)
            #!losses = jnp.array([compute_traj_loss(tidx) for tidx in traj_indices])
            
            
            # Mean loss
            return jnp.mean(losses)
        
        else:
            # Fallback: simple MSE on logprobs
            return jnp.mean((batch_arrays['logprobs'] - batch_arrays['logprobs_rev']) ** 2)
    
    # Define JAX grad step
    @partial(jit, static_argnames=['loss_type', 'n_trajs'])
    def jax_grad_step(params, opt_state, batch_arrays, optimizer, loss_type=loss_type, n_trajs=None):
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
        apply_params_to_pytorch(jax_params, agent, jax_policies)
        
                # Debug: Print parameters for first 5 iterations
        if iteration <= 5:
            print(f"JAX Iteration {iteration}: logZ sum = {jnp.sum(jax_params['logZ']):.4f}")
            if 'forward_policy_trainable' in jax_params and jax_params['forward_policy_trainable'] is not None:
                jax_model = eqx.combine(jax_params['forward_policy_trainable'], jax_policies['forward_static'])
                print(f"JAX forward_policy weight[0,0]: {jax_model.layers[0].weight[0,0]:.4f}")
            if 'backward_policy_trainable' in jax_params and jax_params['backward_policy_trainable'] is not None:
                jax_model = eqx.combine(jax_params['backward_policy_trainable'], jax_policies['backward_static'])
                print(f"JAX backward_policy weight[0,0]: {jax_model.layers[0].weight[0,0]:.4f}")
        
        # ========== LOGGING & SIDE EFFECTS (unchanged) ==========
        # Update loss EMA
        if agent.loss.loss_ema is None:
            agent.loss.loss_ema = float(loss_value)
        else:
            agent.loss.loss_ema = (
                agent.loss.ema_alpha * float(loss_value) +
                (1 - agent.loss.ema_alpha) * agent.loss.loss_ema
            )
        
        # !Logging        
        # if agent.evaluator.should_log_train(iteration):
        #     agent.logger.log_train(
        #         iteration,
        #         {
        #             'loss': float(loss_value),
        #             'loss_ema': agent.loss.loss_ema,
        #             'mean_reward': float(jnp.mean(batch_arrays['rewards'])),
        #         }
        #     )
        

        agent.logger.progressbar_update(
            pbar, float(loss_value), batch_arrays['terminating_rewards'], agent.jsd, agent.use_context
        )
        
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
    
    pbar.close()
    
    print("\n" + "=" * 80)
    print("PHASE 1 TRAINING COMPLETE")
    print(f"  Final loss: {loss_value:.4f}")
    print(f"  Final loss EMA: {agent.loss.loss_ema:.4f}")
    print("=" * 80)
    
    return agent
