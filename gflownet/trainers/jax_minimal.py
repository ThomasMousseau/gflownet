"""
Minimal JAX trainer - Phase 1: JIT only the backpropagation
Keeps PyTorch GFlowNetAgent but JITs gradient computation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad, grad
import equinox as eqx
import torch.nn as nn
import optax
from functools import partial
from tqdm import tqdm

from gflownet.utils.common import instantiate, set_device
from gflownet.utils.batch import Batch
from gflownet.policy.base_jax import PolicyJAX
from gflownet.utils.policy import parse_policy_config

import torch



# ============================================================================
# PHASE 1: Minimal JAX - Only JIT the gradient step
# ============================================================================

def convert_batch_to_jax_arrays(pytorch_batch: Batch):
    """
    Extract JAX-compatible arrays from PyTorch Batch.
    This is the bridge between PyTorch sampling and JAX training.
    """
    
    # Get trajectory indices (consecutive numbering for grouping)
    traj_indices = pytorch_batch.get_trajectory_indices(consecutive=True)
    trajectory_indices = jnp.array(traj_indices.detach().cpu().numpy(), dtype=jnp.int32)
    
    # Compute n_trajs concretely (outside JIT)
    traj_indices_np = traj_indices.detach().cpu().numpy()
    n_trajs = int(np.max(traj_indices_np)) + 1
    
    # Get states for policy input
    states_policy = pytorch_batch.get_states(policy=True)
    if isinstance(states_policy, torch.Tensor):
        states = jnp.array(states_policy.detach().cpu().numpy(), dtype=jnp.float32)
    else:
        # Handle list of states
        states = jnp.array([s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else s 
                           for s in states_policy], dtype=jnp.float32)
    
    #! FOURNIER, le probleme etait ici
    # Get actions
    # actions_list = pytorch_batch.get_actions()       #! [0] of a tuple is not the right logic :(
    # actions = jnp.array([a if isinstance(a, int) else a[0] for a in actions_list], dtype=jnp.int32)
    
    actions = []
    for i in range(len(pytorch_batch)):
        action = pytorch_batch.actions[i]
        traj_idx = pytorch_batch.traj_indices[i]
        env = pytorch_batch.envs[traj_idx]
        
        if isinstance(action, tuple):
            action_idx = env.action2index(action) # Convert (0,0) -> 2
        else:
            action_idx = action
        
        actions.append(action_idx)
    
    actions = jnp.array(actions, dtype=jnp.int32)
    
    # Get rewards - Use terminating rewards only, as in original PyTorch
    if pytorch_batch.proxy is None:
        # If no proxy, use dummy rewards (all zeros)
        # This shouldn't happen in real training but helps with testing
        print("WARNING: Batch has no proxy, using zero rewards")
        terminating_rewards = jnp.zeros(pytorch_batch.get_n_trajectories(), dtype=jnp.float32)
    else:
        terminating_rewards_tensor = pytorch_batch.get_terminating_rewards(sort_by="trajectory")
        terminating_rewards = jnp.array(terminating_rewards_tensor.detach().cpu().numpy(), dtype=jnp.float32)
    
    # For compatibility with existing code, also get all rewards (but we'll use terminating)
    rewards_tensor = pytorch_batch.get_rewards()
    rewards = jnp.array(rewards_tensor.detach().cpu().numpy(), dtype=jnp.float32)
    
    logrewards_tensor = pytorch_batch.get_terminating_rewards(log=True, sort_by="trajectory")
    logrewards = jnp.array(logrewards_tensor.detach().cpu().numpy(), dtype=jnp.float32)
    
    # Get logprobs (forward and backward)
    logprobs_fwd, _ = pytorch_batch.get_logprobs(backward=False)
    logprobs_bwd, _ = pytorch_batch.get_logprobs(backward=True)
    logprobs = jnp.array(logprobs_fwd.detach().cpu().numpy(), dtype=jnp.float32)
    logprobs_rev = jnp.array(logprobs_bwd.detach().cpu().numpy(), dtype=jnp.float32)
    
    parents_policy = pytorch_batch.get_parents(policy=True)
    if isinstance(parents_policy, torch.Tensor):
        parents_policy = jnp.array(parents_policy.detach().cpu().numpy(), dtype=jnp.float32)
    else:
        parents_policy = jnp.array([p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p for p in parents_policy], dtype=jnp.float32)

    masks_forward_pt = pytorch_batch.get_masks_forward(of_parents=True)
    masks_backward_pt = pytorch_batch.get_masks_backward()
    
    masks_forward = jnp.array(masks_forward_pt.detach().cpu().numpy())
    masks_backward = jnp.array(masks_backward_pt.detach().cpu().numpy())
    
    # for i in range(min(5, len(pytorch_batch))):
    #     action = pytorch_batch.actions[i]
    #     traj_idx = pytorch_batch.traj_indices[i]
    #     env = pytorch_batch.envs[traj_idx]
        
    #     if isinstance(action, tuple):
    #         action_idx = env.action2index(action)
    #     else:
    #         action_idx = action
        
    #     mask_f = masks_forward_pt[i]
    #     mask_b = masks_backward_pt[i]
    #     print(f"Sample {i}: action={action}, action_idx={action_idx}, mask_f[action]={mask_f[action_idx]}, mask_b[action]={mask_b[action_idx]}")
    
    return {
        'states': states,
        'states_policy': states,
        'actions': actions,
        'rewards': rewards,
        'terminating_rewards': terminating_rewards,
        'logprobs': logprobs,
        'logprobs_rev': logprobs_rev,
        'trajectory_indices': trajectory_indices,
        'n_trajs': n_trajs, 
        'logrewards': logrewards,
        'parents_policy': parents_policy,
        'masks_forward': masks_forward,
        'masks_backward': masks_backward,
    }


def convert_params_to_jax(agent, config, key):

    # ----- float precision -----
    if isinstance(agent.float, torch.dtype):
        if agent.float == torch.float16:
            float_precision = 16
        elif agent.float == torch.float32:
            float_precision = 32
        elif agent.float == torch.float64:
            float_precision = 64
        else:
            float_precision = 32
    else:
        float_precision = agent.float

    jax_params = {}
    jax_policies = {}

    # ------------------------------------------------------------------
    # FORWARD POLICY
    # ------------------------------------------------------------------
    forward_config = parse_policy_config(config, kind="forward")
    forward_config = forward_config["config"]
    if forward_config is not None:
        forward_config["_target_"] = "gflownet.policy.base_jax.PolicyJAX"
        forward_config["key"] = None  # will be set internally

        jax_policy_f = instantiate(
            forward_config,
            env=agent.env,
            device=agent.device,
            float_precision=float_precision,
        )
        jax_policies["forward"] = jax_policy_f

        if agent.forward_policy.is_model:
            pt_model_f = agent.forward_policy.model          # nn.Sequential
            jax_model_f = jax_policy_f.model                # Equinox Sequential

            # Sync 3 Linear layers: PT[0,2,4] <-> JAX.layers[0,2,4]
            jax_model_f = eqx.tree_at(
                lambda m: m.layers[0].weight,
                jax_model_f,
                jnp.array(pt_model_f[0].weight.detach().cpu().numpy(), dtype=jnp.float32),
            )
            jax_model_f = eqx.tree_at(
                lambda m: m.layers[0].bias,
                jax_model_f,
                jnp.array(pt_model_f[0].bias.detach().cpu().numpy(), dtype=jnp.float32),
            )

            jax_model_f = eqx.tree_at(
                lambda m: m.layers[2].weight,
                jax_model_f,
                jnp.array(pt_model_f[2].weight.detach().cpu().numpy(), dtype=jnp.float32),
            )
            jax_model_f = eqx.tree_at(
                lambda m: m.layers[2].bias,
                jax_model_f,
                jnp.array(pt_model_f[2].bias.detach().cpu().numpy(), dtype=jnp.float32),
            )

            jax_model_f = eqx.tree_at(
                lambda m: m.layers[4].weight,
                jax_model_f,
                jnp.array(pt_model_f[4].weight.detach().cpu().numpy(), dtype=jnp.float32),
            )
            jax_model_f = eqx.tree_at(
                lambda m: m.layers[4].bias,
                jax_model_f,
                jnp.array(pt_model_f[4].bias.detach().cpu().numpy(), dtype=jnp.float32),
            )

            jax_policy_f.model = jax_model_f

            # Partition into trainable/static
            trainable_f, static_f = eqx.partition(jax_model_f, eqx.is_array)
            jax_params["forward_policy_trainable"] = trainable_f
            jax_policies["forward_static"] = static_f
        else:
            jax_params["forward_policy_trainable"] = None
            jax_policies["forward_static"] = None

    # ------------------------------------------------------------------
    # BACKWARD POLICY
    # ------------------------------------------------------------------
    backward_config = parse_policy_config(config, kind="backward")
    backward_config = backward_config["config"]
    if backward_config is not None:
        # copy n_hid / n_layers if shared_weights
        if backward_config.get("shared_weights", False) and forward_config is not None:
            if "n_hid" in forward_config and "n_hid" not in backward_config:
                backward_config["n_hid"] = forward_config["n_hid"]
            if "n_layers" in forward_config and "n_layers" not in backward_config:
                backward_config["n_layers"] = forward_config["n_layers"]

        backward_config["_target_"] = "gflownet.policy.base_jax.PolicyJAX"
        backward_config["key"] = None
        backward_config["instantiate_now"] = False

        jax_policy_b = instantiate(
            backward_config,
            env=agent.env,
            device=agent.device,
            float_precision=float_precision,
        )

        # if PolicyJAX uses base for shared weights, keep this:
        jax_policy_b.base = jax_policies.get("forward", None)
        jax_policy_b.instantiate(jax.random.PRNGKey(0))

        jax_policies["backward"] = jax_policy_b

        if agent.backward_policy.is_model:
            pt_model_b = agent.backward_policy.model        # now flat nn.Sequential
            jax_model_b = jax_policy_b.model                # Equinox Sequential

            # Same mapping: PT[0,2,4] <-> JAX.layers[0,2,4]
            jax_model_b = eqx.tree_at(
                lambda m: m.layers[0].weight,
                jax_model_b,
                jnp.array(pt_model_b[0].weight.detach().cpu().numpy(), dtype=jnp.float32),
            )
            jax_model_b = eqx.tree_at(
                lambda m: m.layers[0].bias,
                jax_model_b,
                jnp.array(pt_model_b[0].bias.detach().cpu().numpy(), dtype=jnp.float32),
            )

            jax_model_b = eqx.tree_at(
                lambda m: m.layers[2].weight,
                jax_model_b,
                jnp.array(pt_model_b[2].weight.detach().cpu().numpy(), dtype=jnp.float32),
            )
            jax_model_b = eqx.tree_at(
                lambda m: m.layers[2].bias,
                jax_model_b,
                jnp.array(pt_model_b[2].bias.detach().cpu().numpy(), dtype=jnp.float32),
            )

            jax_model_b = eqx.tree_at(
                lambda m: m.layers[4].weight,
                jax_model_b,
                jnp.array(pt_model_b[4].weight.detach().cpu().numpy(), dtype=jnp.float32),
            )
            jax_model_b = eqx.tree_at(
                lambda m: m.layers[4].bias,
                jax_model_b,
                jnp.array(pt_model_b[4].bias.detach().cpu().numpy(), dtype=jnp.float32),
            )

            jax_policy_b.model = jax_model_b

            trainable_b, static_b = eqx.partition(jax_model_b, eqx.is_array)
            jax_params["backward_policy_trainable"] = trainable_b
            jax_policies["backward_static"] = static_b
        else:
            jax_params["backward_policy_trainable"] = None
            jax_policies["backward_static"] = None

    # ------------------------------------------------------------------
    # logZ
    # ------------------------------------------------------------------
    if agent.logZ is not None:
        jax_params["logZ"] = jnp.array(agent.logZ.detach().cpu().numpy(), dtype=jnp.float32)
    else:
        jax_params["logZ"] = None

    return jax_params, jax_policies

def apply_params_to_pytorch(jax_params, agent, jax_policies):
    """
    Copy JAX parameters back to PyTorch models.
    Assumes forward and backward policies both have a flat MLP:
    [Linear, Act, Linear, Act, Linear].
    """
    try:
        # ---------- FORWARD POLICY ----------
        if (
            "forward_policy_trainable" in jax_params
            and jax_params["forward_policy_trainable"] is not None
        ):
            jax_model_f = eqx.combine(
                jax_params["forward_policy_trainable"],
                jax_policies["forward_static"],
            )
            pt_model_f = agent.forward_policy.model  # nn.Sequential

            forward_pairs = [
                (pt_model_f[0], jax_model_f.layers[0]),
                (pt_model_f[2], jax_model_f.layers[2]),
                (pt_model_f[4], jax_model_f.layers[4]),
            ]

            for pt_lin, jax_lin in forward_pairs:
                w_np = np.asarray(jax_lin.weight, dtype=np.float32)
                w_t = torch.from_numpy(w_np).to(pt_lin.weight.device, pt_lin.weight.dtype)
                pt_lin.weight.data.copy_(w_t)

                b_np = np.asarray(jax_lin.bias, dtype=np.float32)
                b_t = torch.from_numpy(b_np).to(pt_lin.bias.device, pt_lin.bias.dtype)
                pt_lin.bias.data.copy_(b_t)

        # ---------- BACKWARD POLICY ----------
        if (
            "backward_policy_trainable" in jax_params
            and jax_params["backward_policy_trainable"] is not None
        ):
            jax_model_b = eqx.combine(
                jax_params["backward_policy_trainable"],
                jax_policies["backward_static"],
            )
            pt_model_b = agent.backward_policy.model  # nn.Sequential (flat now!)

            backward_pairs = [
                (pt_model_b[0], jax_model_b.layers[0]),
                (pt_model_b[2], jax_model_b.layers[2]),
                (pt_model_b[4], jax_model_b.layers[4]),
            ]

            for pt_lin, jax_lin in backward_pairs:
                w_np = np.asarray(jax_lin.weight, dtype=np.float32)
                w_t = torch.from_numpy(w_np).to(pt_lin.weight.device, pt_lin.weight.dtype)
                pt_lin.weight.data.copy_(w_t)

                b_np = np.asarray(jax_lin.bias, dtype=np.float32)
                b_t = torch.from_numpy(b_np).to(pt_lin.bias.device, pt_lin.bias.dtype)
                pt_lin.bias.data.copy_(b_t)

        # ---------- logZ ----------
        if jax_params.get("logZ", None) is not None and agent.logZ is not None:
            logZ_np = np.asarray(jax_params["logZ"])
            logZ_t = torch.from_numpy(logZ_np).to(agent.logZ.device, agent.logZ.dtype)
            agent.logZ.data.copy_(logZ_t)

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
    # !Setup Optax optimizer with separate LR for logZ
    lr_schedule_main = optax.exponential_decay(
        init_value=config.gflownet.optimizer.lr,
        transition_steps=config.gflownet.optimizer.lr_decay_period,
        decay_rate=config.gflownet.optimizer.lr_decay_gamma,
        staircase=True  # ← Makes it step-wise like PyTorch StepLR
    )

    lr_schedule_logz = optax.exponential_decay(
        init_value=config.gflownet.optimizer.lr * config.gflownet.optimizer.lr_z_mult,
        transition_steps=config.gflownet.optimizer.lr_decay_period,
        decay_rate=config.gflownet.optimizer.lr_decay_gamma,
        staircase=True
    )

    optimizer = optax.multi_transform(
        {
            'main': optax.adam(
                learning_rate=lr_schedule_main,
                b1=config.gflownet.optimizer.adam_beta1,
                b2=config.gflownet.optimizer.adam_beta2,
                eps=1e-8, # like in torch.optim.Adam
                eps_root=0.0,
                
            ),
            'logz': optax.adam(
                learning_rate=lr_schedule_logz,
                b1=config.gflownet.optimizer.adam_beta1,
                b2=config.gflownet.optimizer.adam_beta2,
                eps=1e-8, # like in torch.optim.Adam
                eps_root=0.0,
            ),
        },
        {
            'forward_policy_trainable': 'main',
            'backward_policy_trainable': 'main',
            'logZ': 'logz'
        }
    )
    
    # lr_schedule_main = 0.0001  # Hard-coded, matching PyTorch
    # lr_schedule_logz = 0.001   # Hard-coded, 10x for logZ (matching PyTorch)
    
    # optimizer = optax.multi_transform(
    #     {
    #         'main': optax.sgd(learning_rate=lr_schedule_main),  # Simple SGD, no momentum
    #         'logz': optax.sgd(learning_rate=lr_schedule_logz),
    #     },
    #     {
    #         'forward_policy_trainable': 'main',
    #         'backward_policy_trainable': 'main',
    #         'logZ': 'logz'
    #     }
    # )
    
    key = jax.random.PRNGKey(config.seed)
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
    
    opt_state = optimizer.init(jax_params)
    loss_type = config.loss.get('_target_', 'trajectorybalance').split('.')[-1].lower()
        
    @partial(jit, static_argnames=['loss_type', 'n_trajs', 'debug'])
    def jax_loss_wrapper(params, batch_arrays, loss_type=loss_type, n_trajs=None, debug=False):
        if loss_type == 'trajectorybalance':
            # Reconstruct models from params
            if 'forward_policy_trainable' in params and params['forward_policy_trainable'] is not None:
                model_f = eqx.combine(params['forward_policy_trainable'], jax_policies['forward_static'])
            else:
                model_f = jax_policies['forward'].model
            
            if 'backward_policy_trainable' in params and params['backward_policy_trainable'] is not None:
                model_b = eqx.combine(params['backward_policy_trainable'], jax_policies['backward_static'])
            else:
                model_b = jax_policies['backward'].model
                                
            # logits_f = jax.vmap(model_f)(batch_arrays['parents_policy'])
            # logits_f_masked = jnp.where(batch_arrays['masks_forward'], -jnp.inf, logits_f)
            # logprobs_f_all = jax.nn.log_softmax(logits_f_masked, axis=1)
            # logprobs_f = logprobs_f_all[jnp.arange(len(batch_arrays['actions'])), batch_arrays['actions']]

            # logits_b = jax.vmap(model_b)(batch_arrays['states_policy'])
            # logits_b_masked = jnp.where(batch_arrays['masks_backward'], -jnp.inf, logits_b)
            # logprobs_b_all = jax.nn.log_softmax(logits_b_masked, axis=1)
            # logprobs_b = logprobs_b_all[jnp.arange(len(batch_arrays['actions'])), batch_arrays['actions']]
            
            #! TO COMMENT OUT FOR DEBUGGING
            logprobs_f = batch_arrays['logprobs']       # Collected during sampling
            logprobs_b = batch_arrays['logprobs_rev']
        
            # Get logZ
            logZ = params.get('logZ', jnp.array(0.0))
            if logZ is not None and logZ.ndim > 0:
                logZ = jnp.sum(logZ)
            
            traj_indices = batch_arrays['trajectory_indices']
            
            # def compute_traj_logprob_ratio(traj_idx):
            #     mask = (traj_indices == traj_idx).astype(jnp.float32)
            #     #TODO: LET'S MINE!
            #     log_pF = jnp.sum(logprobs_f * mask)
            #     log_pB = jnp.sum(logprobs_b * mask)
            #     return log_pF - log_pB
            # logprob_ratios = jax.vmap(compute_traj_logprob_ratio)(jnp.arange(n_trajs))
            # logprob_ratios = jnp.array([compute_traj_logprob_ratio(tidx) for tidx in jnp.arange(n_trajs)])
            
            traj_one_hot = jax.nn.one_hot(traj_indices, n_trajs, dtype=jnp.float32)  # shape: (batch_size, n_trajs)
            log_pF_per_traj = jnp.sum(logprobs_f[:, None] * traj_one_hot, axis=0)  # shape: (n_trajs,)
            log_pB_per_traj = jnp.sum(logprobs_b[:, None] * traj_one_hot, axis=0)  # shape: (n_trajs,)

            logprob_ratios = log_pF_per_traj - log_pB_per_traj
            
            jax.debug.print("Using vmap (not list comp): logprob_ratios shape: {}", logprob_ratios.shape)
            log_rewards = batch_arrays['logrewards']
            
            losses = (logZ + logprob_ratios - log_rewards) ** 2
            
            jax.debug.print("=== JAX LOSS COMPONENTS ===")
            jax.debug.print("logZ: {}", logZ)
            jax.debug.print("logprob_ratios (first 3): {}", logprob_ratios[:3])
            jax.debug.print("log_rewards (first 3): {}", log_rewards[:3])
            jax.debug.print("logprob_ratios - log_rewards (first 3): {}", (logprob_ratios - log_rewards)[:3])
            jax.debug.print("logZ + logprob_ratios - log_rewards (first 3): {}", (logZ + logprob_ratios - log_rewards)[:3])

            # Individual logprobs
            jax.debug.print("logprobs_f (first 5): {}", logprobs_f[:5])
            jax.debug.print("logprobs_b (first 5): {}", logprobs_b[:5])

            # Trajectory grouping
            jax.debug.print("n_trajs: {}", n_trajs)
            jax.debug.print("traj_indices (first 10): {}", traj_indices[:10])
            
            return jnp.mean(losses)
        
        # Fallback
        return jnp.mean((batch_arrays['logprobs'] - batch_arrays['logprobs_rev']) ** 2)
    
    # Define JAX grad step
    @partial(jit, static_argnames=['optimizer','loss_type', 'n_trajs'])
    def jax_grad_step(params, opt_state, batch_arrays, optimizer, loss_type=loss_type, n_trajs=None):
        """
        JIT-compiled gradient step.
        This is the ONLY function that needs to be pure JAX.
        """
        
        loss_value, grads = value_and_grad(jax_loss_wrapper)(
            params, batch_arrays, loss_type=loss_type, n_trajs=n_trajs
        )
                
        # loss_value = jax_loss_wrapper(params, batch_arrays, loss_type=loss_type, n_trajs=n_trajs, debug=True)
        # grads = grad(lambda p: jax_loss_wrapper(p, batch_arrays, loss_type=loss_type, n_trajs=n_trajs, debug=False))(params)
                
        #! Apply optimizer update
        #! COMMENTED OUT FOR DEBUGGING
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        # new_opt_state = opt_state
        # new_params = params

        first_layer_grad = grads['forward_policy_trainable'].layers[0].weight
        jax.debug.print("JAX grad_forward_layer0_norm: {}", jnp.linalg.norm(first_layer_grad))
        
        return new_params, new_opt_state, loss_value, grads
    
    total_iterations = agent.n_train_steps - agent.it + 1
    pbar = tqdm(
        initial=agent.it - 1,
        total=total_iterations,
        disable=agent.logger.progressbar.get("skip", False),
    )
    
    for iteration in range(agent.it, agent.n_train_steps + 1):
        
        jax.debug.print("=== TRAINING ITERATION {} ===", iteration)
        
        batch = Batch(
            env=agent.env,
            proxy=agent.proxy,
            device=agent.device,
            float_type=agent.float,
        )
        
        # Sample batch (PyTorch)
        for _ in range(agent.sttr):
            sub_batch, _ = agent.sample_batch(
                n_forward=agent.batch_size.forward,
                n_train=agent.batch_size.backward_dataset,
                n_replay=agent.batch_size.backward_replay,
                collect_forwards_masks=True,
                collect_backwards_masks=agent.collect_backwards_masks,
            )
            batch.merge(sub_batch)
        
        # Convert batch to JAX arrays
        batch_arrays = convert_batch_to_jax_arrays(batch)
        
        #! TO COMMENT AFTER RNG DEBUGGING
        if iteration == 1:
            saved_batch_arrays = batch_arrays.copy()
        if iteration <= 10:
            batch_arrays = saved_batch_arrays
        
        n_trajs = batch_arrays.get('n_trajs') 
        
        # ========== JAX: Training Steps ==========
        # Perform multiple gradient steps (train-to-sample ratio)
        for _ in range(agent.ttsr):
            jax_params, opt_state, loss_value, grads = jax_grad_step(
                jax_params, opt_state, batch_arrays, optimizer, loss_type=loss_type, n_trajs=n_trajs
            )
        
        # ========== PYTORCH: Copy parameters back ==========
        # Sync JAX params back to PyTorch (for next sampling iteration)
        apply_params_to_pytorch(jax_params, agent, jax_policies)
        
        # ========== BUFFER UPDATE (mirroring PyTorch log_train_iteration) ==========
        # Update the replay buffer with the current batch data
        # states_term = batch.get_terminating_states(sort_by="trajectory")
        # proxy_vals = batch.get_terminating_proxy_values(sort_by="trajectory")
        # rewards = batch.get_terminating_rewards(sort_by="trajectory")
        # actions_trajectories = batch.get_actions_trajectories()
        
        # # Update main buffer (if enabled)
        # if agent.buffer.use_main_buffer:
        #     agent.buffer.add(states_term, actions_trajectories, rewards, iteration, buffer="main")
        
        # # Update replay buffer
        # agent.buffer.add(states_term, actions_trajectories, rewards, iteration, buffer="replay")
    
        
        # ========== LOGGING & SIDE EFFECTS (unchanged) ==========
        # Update loss EMA
        if agent.loss.loss_ema is None:
            agent.loss.loss_ema = float(loss_value)
        else:
            agent.loss.loss_ema = (
                agent.loss.ema_alpha * float(loss_value) +
                (1 - agent.loss.ema_alpha) * agent.loss.loss_ema
            )
        
        # Logging Logic
        if agent.evaluator.should_log_train(iteration):

            # ---------- Rewards & log-rewards ----------
            rewards = jnp.asarray(batch_arrays["terminating_rewards"])
            logrewards = jnp.log(rewards + 1e-8)

            rewards_t    = torch.from_numpy(np.asarray(rewards, dtype=np.float32))
            logrewards_t = torch.from_numpy(np.asarray(logrewards, dtype=np.float32))

            # If you have proxy scores (like original `proxy_vals`), convert them too
            proxy_vals = batch_arrays.get("terminating_scores", None)
            if proxy_vals is not None:
                proxy_vals_t = torch.from_numpy(np.asarray(proxy_vals, dtype=np.float32))
            else:
                proxy_vals_t = None

            agent.logger.log_rewards_and_scores(
                rewards=rewards_t,
                logrewards=logrewards_t,
                scores=proxy_vals_t,
                step=iteration,
                prefix="Train batch -",
                use_context=agent.use_context,
            )

            # ---------- Trajectory lengths ----------
            # This part assumes you have an array similar to `get_trajectory_indices()`
            # If you don't, you can comment this block out.
            traj_indices = batch_arrays.get("trajectory_indices", None)
            if traj_indices is not None:
                traj_indices = jnp.asarray(traj_indices)
                _, counts = jnp.unique(traj_indices, return_counts=True)
                counts = counts.astype(jnp.float32)

                traj_length_mean = float(jnp.mean(counts))
                traj_length_min  = float(jnp.min(counts))
                traj_length_max  = float(jnp.max(counts))
            else:
                traj_length_mean = traj_length_min = traj_length_max = None

            # ---------- logZ ----------
            if "logZ" in jax_params:
                # jax_params['logZ'] is a JAX array, no detach needed
                logz = float(jnp.sum(jax_params["logZ"]))
            else:
                logz = None

            # ---------- Learning rates ----------
            # Adapt this depending on how you store LR in your JAX setup.

            lr_main = float(lr_schedule_main(iteration))
            lr_logz = float(lr_schedule_logz(iteration))
            # lr_main = float(lr_schedule_main)
            # lr_logz = float(lr_schedule_logz)
            grad_logZ = np.asarray(grads["logZ"], dtype=np.float32)

            # ---------- Scalar metrics exactly like original ----------
            metrics = {
                "step": iteration,
                "Trajectory lengths mean": traj_length_mean,
                "Trajectory lengths min": traj_length_min,
                "Trajectory lengths max": traj_length_max,
                "Batch size": len(batch),
                "Batch_arrays size": batch_arrays['states'].shape[0],
                "logZ": logz,
                "Learning rate": lr_main,
                "Learning rate logZ": lr_logz,
                "grad_logZ_mean": float(grad_logZ.mean()),
                "first_layer_grad_norm": float(jnp.linalg.norm(grads['forward_policy_trainable'].layers[0].weight)),
            }

            agent.logger.log_metrics(
                metrics=metrics,
                step=iteration,
                use_context=agent.use_context,
            )

            # ---------- Loss dict mimicking original ----------
            # Original does: losses["Loss"] = losses["all"]; logger.log_metrics(losses, ...)
            # Here we reconstruct a similar dict.
            losses = {
                "all": float(loss_value),                 # same as original "all"
                "Loss": float(loss_value),               # explicit "Loss" key
            }

            agent.logger.log_metrics(
                metrics=losses,
                step=iteration,
                use_context=agent.use_context,
            )
            # End logging block
        
        agent.logger.progressbar_update(
            pbar,
            float(loss_value),
            np.asarray(batch_arrays["terminating_rewards"], dtype=np.float32),
            agent.jsd,
            agent.use_context,
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

#! DEBUGGING PURPOSES
#! DEBUGGING PURPOSES
def sync_params_from_pytorch_to_jax(agent, jax_params, jax_policies):
    """
    Sync parameters from PyTorch agent back to JAX params dict.
    This is the reverse of apply_params_to_pytorch.
    """
    import equinox as eqx
    
    # Sync logZ
    if agent.logZ is not None:
        jax_params["logZ"] = jnp.array(
            agent.logZ.detach().cpu().numpy(), dtype=jnp.float32
        )
    
    # Sync forward policy
    if agent.forward_policy.is_model:
        pt_model_f = agent.forward_policy.model
        # Get the current JAX model (combine trainable + static)
        jax_model_f = eqx.combine(
            jax_params["forward_policy_trainable"],
            jax_policies["forward_static"]
        )
        
        # Update each LINEAR layer (indices 0, 2, 4 in both PyTorch and JAX)
        for pt_layer_idx in [0, 2, 4]:
            pt_layer = pt_model_f[pt_layer_idx]
            weight = jnp.array(pt_layer.weight.detach().cpu().numpy(), dtype=jnp.float32)
            bias = jnp.array(pt_layer.bias.detach().cpu().numpy(), dtype=jnp.float32)
            
            # Update the layer - use pt_layer_idx for JAX too (not i)
            jax_model_f = eqx.tree_at(
                lambda m: (m.layers[pt_layer_idx].weight, m.layers[pt_layer_idx].bias),
                jax_model_f,
                (weight, bias),
                is_leaf=lambda x: x is None  # Treat None as leaf
            )
        
        # Re-partition into trainable and static
        trainable_f, static_f = eqx.partition(jax_model_f, eqx.is_array)
        jax_params["forward_policy_trainable"] = trainable_f
        jax_policies["forward_static"] = static_f
    
    # Sync backward policy (same fix)
    if agent.backward_policy.is_model:
        pt_model_b = agent.backward_policy.model
        # Get the current JAX model
        jax_model_b = eqx.combine(
            jax_params["backward_policy_trainable"],
            jax_policies["backward_static"]
        )
        
        # Update each LINEAR layer (indices 0, 2, 4)
        for pt_layer_idx in [0, 2, 4]:
            pt_layer = pt_model_b[pt_layer_idx]
            weight = jnp.array(pt_layer.weight.detach().cpu().numpy(), dtype=jnp.float32)
            bias = jnp.array(pt_layer.bias.detach().cpu().numpy(), dtype=jnp.float32)
            
            # Update the layer - use pt_layer_idx for JAX too
            jax_model_b = eqx.tree_at(
                lambda m: (m.layers[pt_layer_idx].weight, m.layers[pt_layer_idx].bias),
                jax_model_b,
                (weight, bias),
                is_leaf=lambda x: x is None  # Treat None as leaf
            )
        
        # Re-partition
        trainable_b, static_b = eqx.partition(jax_model_b, eqx.is_array)
        jax_params["backward_policy_trainable"] = trainable_b
        jax_policies["backward_static"] = static_b
    
    return jax_params, jax_policies