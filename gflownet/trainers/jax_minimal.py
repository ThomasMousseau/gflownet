import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad, grad
import equinox as eqx
import optax
import gc
from collections import defaultdict
from tqdm import tqdm
from gflownet.utils.common import instantiate
from gflownet.utils.policy import parse_policy_config
from gflownet.envs.grid_jax import (
    sample_trajectories_jax,
    states2proxy_jax,
    states2policy_jax,
)


def init_agent_jax(config, env):
    """
    Initialize JAX policies and parameters from configuration.
    """
    float_precision = 32
    device = "cpu"
    
    jax_params = {}
    jax_policies = {}
    
    # Forward Policy
    forward_config = parse_policy_config(config, kind="forward")
    forward_config = forward_config["config"]
    if forward_config is not None:
        forward_config["_target_"] = "gflownet.policy.base_jax.PolicyJAX"
        forward_config["key"] = None

        jax_policy_f = instantiate(
            forward_config,
            env=env,
            device=device,
            float_precision=float_precision,
        )
        jax_policies["forward"] = jax_policy_f

        if hasattr(jax_policy_f, 'model'):
            trainable_f, static_f = eqx.partition(jax_policy_f.model, eqx.is_array)
            jax_params["forward_policy_trainable"] = trainable_f
            jax_policies["forward_static"] = static_f
        else:
            jax_params["forward_policy_trainable"] = None
            jax_policies["forward_static"] = None
            
    # Backward Policy
    backward_config = parse_policy_config(config, kind="backward")
    backward_config = backward_config["config"]
    if backward_config is not None:
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
            env=env,
            device=device,
            float_precision=float_precision,
            base=jax_policy_f
        )

        jax_policy_b.base = jax_policies.get("forward", None)
        jax_policy_b.instantiate(jax.random.PRNGKey(0))

        jax_policies["backward"] = jax_policy_b

        if hasattr(jax_policy_b, 'model'):
            trainable_b, static_b = eqx.partition(jax_policy_b.model, eqx.is_array)
            jax_params["backward_policy_trainable"] = trainable_b
            jax_policies["backward_static"] = static_b
        else:
            jax_params["backward_policy_trainable"] = None
            jax_policies["backward_static"] = None

    # LogZ
    z_dim = config.gflownet.optimizer.z_dim
    jax_params["logZ"] = jnp.ones(z_dim, dtype=jnp.float32) * 150.0 / 64.0
    
    return jax_params, jax_policies


def train(agent, config):
    """
    Pure JAX trainer: Uses JAX for both sampling and training (end-to-end JIT).
    
    This mode requires a JAX-native environment (e.g., GridJAX).
    """
    if agent is None:
        # Instantiate minimal components needed for JAX training
        print("Initializing JAX components from config...")
        
        # Logger
        logger = instantiate(config.logger, config, _recursive_=False)
        
        # Proxy
        proxy = instantiate(config.proxy, device=config.device, float_precision=config.float_precision)
        
        # Env
        env = instantiate(config.env, device=config.device, float_precision=config.float_precision)
        proxy.setup(env) # Setup proxy with env
        
        # Evaluator
        evaluator = instantiate(config.evaluator)
        
        # Config values
        n_train_steps = config.gflownet.optimizer.n_train_steps
        batch_size = config.gflownet.optimizer.batch_size
        
        # Calculate train/sample ratios
        train_to_sample_ratio = config.gflownet.optimizer.train_to_sample_ratio
        ttsr = max(int(train_to_sample_ratio), 1)
        sttr = max(int(1 / train_to_sample_ratio), 1)
        
        start_it = 0
        use_context = False
        jsd = False
        device = config.device
        
    else:
        env = agent.env
        logger = agent.logger
        proxy = agent.proxy
        evaluator = agent.evaluator
        n_train_steps = agent.n_train_steps
        batch_size = agent.batch_size
        sttr = agent.sttr
        ttsr = agent.ttsr
        start_it = agent.it
        use_context = agent.use_context
        jsd = agent.jsd
        device = agent.device

    # Setup Optax optimizer with separate LR for logZ
    lr_schedule_main = optax.exponential_decay(
        init_value=config.gflownet.optimizer.lr,
        transition_steps=config.gflownet.optimizer.lr_decay_period,
        decay_rate=config.gflownet.optimizer.lr_decay_gamma,
        staircase=True
    )

    lr_schedule_logz = optax.exponential_decay(
        init_value=config.gflownet.optimizer.lr * config.gflownet.optimizer.lr_z_mult,
        transition_steps=config.gflownet.optimizer.lr_decay_period,
        decay_rate=config.gflownet.optimizer.lr_decay_gamma,
        staircase=True
    )
    
    max_grad_norm = 1.0

    optimizer = optax.multi_transform(
        {
            'main': optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(
                    learning_rate=lr_schedule_main,
                    b1=config.gflownet.optimizer.adam_beta1,
                    b2=config.gflownet.optimizer.adam_beta2,
                    eps=1e-8
                ),
            ),
            'logz': optax.adam(
                learning_rate=lr_schedule_logz,
                b1=config.gflownet.optimizer.adam_beta1,
                b2=config.gflownet.optimizer.adam_beta2,
                eps=1e-8,
            ),
        },
        {
            'forward_policy_trainable': 'main',
            'backward_policy_trainable': 'main',
            'logZ': 'logz'
        }
    )
    
    key = jax.random.PRNGKey(config.seed)
    jax_params, jax_policies = init_agent_jax(config, env)
    opt_state = optimizer.init(jax_params)
    loss_type = config.loss.get('_target_', 'trajectorybalance').split('.')[-1].lower()
    
    # Environment config
    grid_config = env.config
    
    # Batch size
    n_trajs_per_sample = batch_size.forward + batch_size.backward_dataset + batch_size.backward_replay
    n_trajs_total = int(n_trajs_per_sample * sttr)
    
    print(f"Pure JAX Config: n_trajs_per_sample={n_trajs_per_sample}, sttr={sttr}, total={n_trajs_total}")
    
    # Loss function (will be JIT-compiled by jax_grad_step)
    def jax_loss_and_grad(params, states, parents, actions, action_indices,
                         masks_forward, masks_backward, trajectory_indices,
                         actual_n_states, actual_n_trajs, actual_lengths, log_rewards, loss_type=loss_type):
        """Compute loss and gradients for pure JAX batch."""
        if loss_type == 'trajectorybalance':
            # Combine params into models
            if 'forward_policy_trainable' in params and params['forward_policy_trainable'] is not None:
                model_f = eqx.combine(params['forward_policy_trainable'], jax_policies['forward_static'])
            else:
                model_f = jax_policies['forward'].model
            
            if 'backward_policy_trainable' in params and params['backward_policy_trainable'] is not None:
                model_b = eqx.combine(params['backward_policy_trainable'], jax_policies['backward_static'])
            else:
                model_b = jax_policies['backward'].model
            
            logZ = params['logZ']
            if logZ.ndim > 0:
                logZ = jnp.sum(logZ)
            
            # Forward logits and logprobs
            from gflownet.envs.grid_jax import states2policy_jax
            states_policy = states2policy_jax(states, grid_config)
            parents_policy = states2policy_jax(parents, grid_config)
            
            logits_f = jax.vmap(model_f)(parents_policy)
            logits_f_masked = jnp.where(masks_forward, -jnp.inf, logits_f)
            logprobs_f_all = jax.nn.log_softmax(logits_f_masked, axis=1)
            
            # Get logprobs for taken actions
            batch_size = states.shape[0]
            logprobs_f = logprobs_f_all[jnp.arange(batch_size), action_indices]
            
            # Backward logits and logprobs
            logits_b = jax.vmap(model_b)(states_policy)
            logits_b_masked = jnp.where(masks_backward, -jnp.inf, logits_b)
            logprobs_b_all = jax.nn.log_softmax(logits_b_masked, axis=1)
            logprobs_b = logprobs_b_all[jnp.arange(batch_size), action_indices]
            
            # Create state mask (valid states only)
            # Note: states are flattened [traj0, traj1, ...] with padding at end of each traj
            max_len = batch_size // actual_n_trajs
            step_indices = jnp.arange(max_len)
            mask_2d = step_indices[None, :] < actual_lengths[:, None]
            state_mask = mask_2d.reshape(-1)
            
            # Use where to avoid 0 * -inf = nan
            logprobs_f = jnp.where(state_mask, logprobs_f, 0.0)
            logprobs_b = jnp.where(state_mask, logprobs_b, 0.0)
            
            # Sum logprobs per trajectory
            traj_indices = trajectory_indices
            num_trajs = actual_n_trajs
            
            # Use segment_sum for trajectory-level aggregation
            segment_ids = traj_indices + 1  # Offset by 1
            num_segments = num_trajs + 1
            
            sum_logprobs_f = jax.ops.segment_sum(logprobs_f, segment_ids, num_segments=num_segments)
            sum_logprobs_b = jax.ops.segment_sum(logprobs_b, segment_ids, num_segments=num_segments)
            
            # Remove offset
            sum_logprobs_f = sum_logprobs_f[1:]
            sum_logprobs_b = sum_logprobs_b[1:]
            
            # Compute trajectory balance loss
            logprob_ratios = sum_logprobs_f - sum_logprobs_b
            losses = (logZ + logprob_ratios - log_rewards) ** 2
            
            # Average over trajectories
            traj_mask = jnp.arange(num_trajs) < actual_n_trajs
            loss_sum = jnp.sum(losses * traj_mask)
            loss = loss_sum / (jnp.sum(traj_mask) + 1e-8)
            
            return loss
        
        return jnp.array(0.0)
    
    def jax_grad_step(params, opt_state, states, parents, actions, action_indices,
                     masks_forward, masks_backward, trajectory_indices,
                     actual_n_states, actual_n_trajs, actual_lengths, log_rewards,
                     optimizer, loss_type=loss_type):
        """Gradient step for pure JAX (called inside scan)."""
        
        loss_value, grads = value_and_grad(jax_loss_and_grad)(
            params, states, parents, actions, action_indices,
            masks_forward, masks_backward, trajectory_indices,
            actual_n_states, actual_n_trajs, actual_lengths, log_rewards, loss_type=loss_type
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_value, grads

    # Extract proxy config
    proxy_mu = config.proxy.get("mu", 0.75)
    proxy_sigma = config.proxy.get("sigma", 0.05)
    proxy_n_dim = config.env.n_dim

    def corners_reward_jax(states, mu, sigma, n_dim):
        """
        Pure JAX implementation of the Corners proxy reward.
        states: (batch_size, n_dim) - normalized states in [-1, 1]
        """
        # PyTorch implementation uses abs(states) to map to positive quadrant
        # and computes distance to mu * ones(n_dim).
        # This is equivalent to finding the closest corner.
        
        mu_vec = mu * jnp.ones(n_dim)
        
        # Calculate negative squared Euclidean distance / (2 * sigma^2)
        # states: (B, D)
        # mu_vec: (D)
        
        # Use abs(states) to fold into positive quadrant
        diff = jnp.abs(states) - mu_vec[None, :]
        dist_sq = jnp.sum(diff**2, axis=-1)
        
        # Log probability
        # PyTorch implementation uses sigma as the diagonal of the covariance matrix (variance)
        # NOT as the standard deviation.
        # cov = sigma * I
        # inv(cov) = (1/sigma) * I
        # term = -0.5 * (x-mu)^T * inv(cov) * (x-mu) = -0.5 * dist_sq / sigma
        log_prob = -0.5 * dist_sq / sigma
        
        # Normalization constant
        # log(1 / sqrt((2*pi)^D * det(cov)))
        # det(cov) = sigma^D
        # log_norm = -0.5 * (D * log(2*pi) + D * log(sigma))
        log_norm = -0.5 * n_dim * jnp.log(2 * jnp.pi) - 0.5 * n_dim * jnp.log(sigma)
        
        return log_prob + log_norm

    def compute_log_rewards_jax(batch, mu, sigma, n_dim):
        """
        Pure JAX reward computation.
        """
        states = batch.states
        actual_lengths = batch.actual_lengths
        
        num_trajs = actual_lengths.shape[0]
        total_states = states.shape[0]
        max_len = total_states // num_trajs
        
        # Reshape states to (n_trajs, max_len, n_dim)
        states_reshaped = states.reshape(num_trajs, max_len, -1)

        # Extract terminal states
        # Clip indices to be within [0, max_len-1]
        last_indices = jnp.clip(actual_lengths - 1, 0, max_len - 1)
        
        traj_ids = jnp.arange(num_trajs)
        terminal_states = states_reshaped[traj_ids, last_indices]

        # Convert to proxy input (normalize to [-1, 1])
        grid_size = grid_config.length
        proxy_inputs = (terminal_states / (grid_size - 1)) * 2.0 - 1.0

        # Compute rewards
        log_rewards = corners_reward_jax(proxy_inputs, mu, sigma, n_dim)
        
        # Clip rewards
        log_rewards = jnp.maximum(log_rewards, -100.0)
        
        return log_rewards

    # Host-side reward computation removed (Pure JAX now)

    @jit
    def train_step(jax_params, opt_state, key):
        """Single training step (sample + reward + train)."""
        key, subkey = jax.random.split(key)
        
        # Reconstruct model for sampling
        if 'forward_policy_trainable' in jax_params and jax_params['forward_policy_trainable'] is not None:
            model_f = eqx.combine(jax_params['forward_policy_trainable'], jax_policies['forward_static'])
        else:
            model_f = jax_policies['forward'].model
            
        # Sample trajectories
        batch, _ = sample_trajectories_jax(
            key=subkey,
            config=grid_config,
            policy_model=model_f,
            n_trajectories=n_trajs_total,
            temperature=1.0,
            return_logprobs=False
        )
        
        # Compute rewards in JAX
        log_rewards = compute_log_rewards_jax(batch, proxy_mu, proxy_sigma, proxy_n_dim)
        
        # Clip rewards
        log_rewards = jnp.maximum(log_rewards, -100.0)
        
        # Training updates (ttsr loop)
        def update_body(carry_inner, _):
            p, o = carry_inner
            new_p, new_o, l, g = jax_grad_step(
                p, o, 
                batch.states, batch.parents, batch.actions, batch.action_indices,
                batch.masks_forward, batch.masks_backward, batch.trajectory_indices,
                batch.actual_n_states, batch.actual_n_trajs, batch.actual_lengths, 
                log_rewards,
                optimizer, loss_type
            )
            return (new_p, new_o), (l, g)
            
        if ttsr == 1:
            (jax_params, opt_state), (loss, grads) = update_body((jax_params, opt_state), None)
            step_losses = jnp.reshape(loss, (1,))
        else:
            (jax_params, opt_state), (step_losses, grads) = jax.lax.scan(
                update_body, (jax_params, opt_state), None, length=ttsr
            )
        
        # Compute gradient metrics
        grad_logZ_mean = jnp.mean(grads['logZ']) if grads['logZ'] is not None else 0.0
        
        # Compute first layer gradient norm for forward policy
        first_layer_grad_norm = 0.0
        if grads['forward_policy_trainable'] is not None:
            # Get first layer weights from the pytree
            leaves = jax.tree_util.tree_leaves(grads['forward_policy_trainable'])
            if len(leaves) > 0:
                first_layer_grad_norm = jnp.sqrt(jnp.sum(leaves[0] ** 2))
        
        # Trajectory length stats
        traj_lengths = batch.actual_lengths
        traj_len_mean = jnp.mean(traj_lengths)
        traj_len_min = jnp.min(traj_lengths)
        traj_len_max = jnp.max(traj_lengths)
        
        # Pack metrics
        metrics = {
            'loss': jnp.mean(step_losses),
            'traj_len_mean': traj_len_mean,
            'traj_len_min': traj_len_min,
            'traj_len_max': traj_len_max,
            'grad_logZ_mean': grad_logZ_mean,
            'first_layer_grad_norm': first_layer_grad_norm,
        }
        
        return jax_params, opt_state, key, metrics, log_rewards

    @jit
    def train_epoch(jax_params, opt_state, key):
        """Run a chunk of training steps fully JIT-compiled using scan."""
        def body_fn(carry, _):
            params, opt, k = carry
            new_params, new_opt, new_k, metrics, log_rewards = train_step(params, opt, k)
            return (new_params, new_opt, new_k), (metrics, log_rewards)

        (jax_params, opt_state, key), (all_metrics, all_log_rewards) = jax.lax.scan(
            body_fn, (jax_params, opt_state, key), None, length=steps_per_epoch
        )
        return jax_params, opt_state, key, all_metrics, all_log_rewards

    # Training loop
    steps_per_epoch = 10
    n_epochs = (n_train_steps - start_it + 1) // steps_per_epoch
    
    pbar = tqdm(
        initial=start_it,
        total=n_train_steps,
        disable=logger.progressbar.get("skip", False),
    )
    
    for epoch in range(n_epochs):
        t0 = time.time()
        
        # Run JIT epoch
        jax_params, opt_state, key, all_metrics, all_log_rewards = train_epoch(jax_params, opt_state, key)
        
        t1 = time.time()
        
        # Metrics (aggregate over steps in epoch)
        avg_loss = float(jnp.mean(all_metrics['loss']))
        current_step = start_it + (epoch + 1) * steps_per_epoch
        
        # Compute current learning rates from schedules
        lr_main = float(lr_schedule_main(current_step))
        lr_logz = float(lr_schedule_logz(current_step))
        
        # Get current logZ value
        logZ_value = float(jnp.sum(jax_params['logZ']))
        
        # Logging
        if evaluator.should_log_train(current_step):
            metrics = {
                "step": current_step,
                "time_epoch": t1 - t0,
                "steps_per_sec": steps_per_epoch / (t1 - t0),
                # Loss
                "Loss": avg_loss,
                "all": avg_loss,
                # Trajectory stats (mean over epoch)
                "Trajectory lengths mean": float(jnp.mean(all_metrics['traj_len_mean'])),
                "Trajectory lengths min": float(jnp.min(all_metrics['traj_len_min'])),
                "Trajectory lengths max": float(jnp.max(all_metrics['traj_len_max'])),
                # Batch info
                "Batch size": n_trajs_total,
                # LogZ and learning rates
                "logZ": logZ_value,
                "Learning rate": lr_main,
                "Learning rate logZ": lr_logz,
                # Gradient metrics (mean over epoch)
                "grad_logZ_mean": float(jnp.mean(all_metrics['grad_logZ_mean'])),
                "first_layer_grad_norm": float(jnp.mean(all_metrics['first_layer_grad_norm'])),
            }
            logger.log_metrics(metrics, step=current_step, use_context=use_context)
            
        # Progress bar
        # Flatten rewards from all steps in epoch
        flat_rewards = jnp.exp(all_log_rewards).reshape(-1)
        
        logger.progressbar_update(
            pbar,
            avg_loss,
            flat_rewards,
            jsd,
            use_context,
        )
        pbar.update(steps_per_epoch - 1)
        
        # Evaluation (if needed)
        if evaluator.should_eval(current_step):
            # Sample for evaluation
            key, eval_key = jax.random.split(key)
            
            # Reconstruct model for sampling
            if 'forward_policy_trainable' in jax_params and jax_params['forward_policy_trainable'] is not None:
                model_f_eval = eqx.combine(jax_params['forward_policy_trainable'], jax_policies['forward_static'])
            else:
                model_f_eval = jax_policies['forward'].model
            
            # Sample trajectories for evaluation (with logprobs for additional metrics)
            n_eval_samples = config.evaluator.get('n', 1000)
            eval_batch, _ = sample_trajectories_jax(
                key=eval_key,
                config=grid_config,
                policy_model=model_f_eval,
                n_trajectories=n_eval_samples,
                temperature=1.0,
                return_logprobs=True  # Need logprobs for metrics
            )
            
            # Extract terminal states
            eval_lengths = eval_batch.actual_lengths
            num_eval_trajs = eval_lengths.shape[0]
            max_eval_len = eval_batch.states.shape[0] // num_eval_trajs
            eval_states_reshaped = eval_batch.states.reshape(num_eval_trajs, max_eval_len, -1)
            last_eval_indices = jnp.clip(eval_lengths - 1, 0, max_eval_len - 1)
            terminal_states_eval = eval_states_reshaped[jnp.arange(num_eval_trajs), last_eval_indices]
            
            # Convert to numpy for plotting
            x_sampled = np.array(terminal_states_eval)
            
            # Compute density metrics if we have all states test buffer
            eval_metrics = {}
            if hasattr(env, 'get_all_terminating_states'):
                # Get all terminating states
                all_states = env.get_all_terminating_states()
                x_tt = np.array(all_states)
                
                # Compute true rewards (density)
                x_tt_proxy = (x_tt / (grid_config.length - 1)) * 2.0 - 1.0
                
                # Compute rewards using JAX corners function
                rewards = np.exp(np.array(corners_reward_jax(
                    jnp.array(x_tt_proxy), proxy_mu, proxy_sigma, proxy_n_dim
                )))
                z_true = rewards.sum()
                density_true = rewards / z_true
                
                # Compute predicted density from samples histogram
                hist = defaultdict(int)
                for x in x_sampled:
                    hist[tuple(x.tolist())] += 1
                z_pred = sum([hist[tuple(x.tolist())] for x in x_tt]) + 1e-9
                density_pred = np.array([hist[tuple(x.tolist())] / z_pred for x in x_tt])
                
                log_density_true = np.log(density_true + 1e-8)
                log_density_pred = np.log(density_pred + 1e-8)
                
                # ==================== DENSITY METRICS ====================
                # L1 error
                l1 = np.abs(density_pred - density_true).mean()
                # KL divergence
                kl = (density_true * (log_density_true - log_density_pred)).mean()
                # Jensen-Shannon divergence
                log_mean_dens = np.logaddexp(log_density_true, log_density_pred) + np.log(0.5)
                jsd_val = 0.5 * np.sum(density_true * (log_density_true - log_mean_dens))
                jsd_val += 0.5 * np.sum(density_pred * (log_density_pred - log_mean_dens))
                
                eval_metrics["L1 error"] = l1
                eval_metrics["KL Div."] = kl
                eval_metrics["Jensen Shannon Div."] = jsd_val
                
                # ==================== LOG-PROB METRICS ====================
                # Compute log probabilities for sampled trajectories
                # We need to recompute log probs using the policy on the sampled states
                
                # Get sampled terminal states rewards
                x_sampled_proxy = (x_sampled / (grid_config.length - 1)) * 2.0 - 1.0
                sampled_rewards = np.exp(np.array(corners_reward_jax(
                    jnp.array(x_sampled_proxy), proxy_mu, proxy_sigma, proxy_n_dim
                )))
                
                # Compute log probs for each trajectory
                # Vectorized approach: compute all log probs at once
                eval_states_flat = eval_batch.states  # [n_trajs * max_len, n_dim]
                eval_actions_flat = eval_batch.action_indices  # [n_trajs * max_len]
                eval_masks_flat = eval_batch.masks_forward  # [n_trajs * max_len, n_actions]
                
                # Convert all states to policy format
                all_states_policy = states2policy_jax(eval_states_flat, grid_config)
                
                # Get logits for all states using vmap
                all_logits = jax.vmap(model_f_eval)(all_states_policy)
                
                # Apply masks and compute log probs
                all_logits_masked = jnp.where(eval_masks_flat, -1e10, all_logits)
                all_log_probs = jax.nn.log_softmax(all_logits_masked, axis=1)
                
                # Get log prob of taken action for each state
                n_states_total = eval_states_flat.shape[0]
                action_log_probs = all_log_probs[jnp.arange(n_states_total), eval_actions_flat]
                
                # Reshape to [n_trajs, max_len]
                action_log_probs_2d = action_log_probs.reshape(num_eval_trajs, max_eval_len)
                
                # Create mask for valid steps
                step_indices = np.arange(max_eval_len)[None, :]
                valid_mask = step_indices < np.array(eval_lengths)[:, None]
                
                # Zero out padding and sum per trajectory
                action_log_probs_2d = np.where(valid_mask, np.array(action_log_probs_2d), 0.0)
                traj_log_probs = action_log_probs_2d.sum(axis=1)
                
                # NLL of test data (negative log likelihood)
                nll_tt = -traj_log_probs.mean()
                eval_metrics["NLL of test data"] = float(nll_tt)
                
                # Bootstrap std of log probs (simplified - using trajectory variance)
                logprobs_std = np.std(traj_log_probs)
                eval_metrics["Mean BS Std(logp)"] = float(logprobs_std)
                
                # Std of probs
                probs = np.exp(np.clip(traj_log_probs, -50, 0))  # Clip to avoid overflow
                probs_std = np.std(probs)
                eval_metrics["Mean BS Std(p)"] = float(probs_std)
                
                # Correlation between trajectory probs and rewards
                if len(sampled_rewards) > 1 and np.std(sampled_rewards) > 1e-10:
                    corr = np.corrcoef(probs, sampled_rewards)[0, 1]
                    if not np.isnan(corr):
                        eval_metrics["Corr. (test probs., rewards)"] = float(corr)
                
                # Variance of (log rewards - log probs)
                log_rewards = np.log(sampled_rewards + 1e-8)
                var_logrewards_logp = np.var(log_rewards - traj_log_probs)
                eval_metrics["Var(logR - logp) test"] = float(var_logrewards_logp)
                
                # Ratio of logprobs std to NLL
                if abs(nll_tt) > 1e-10:
                    logprobs_std_nll_ratio = logprobs_std / abs(nll_tt)
                    eval_metrics["BS Std(logp) / NLL"] = float(logprobs_std_nll_ratio)
                
                logger.log_metrics(eval_metrics, step=current_step, use_context=use_context)
                
                # Plot reward samples
                if hasattr(env, 'plot_reward_samples'):
                    fig = env.plot_reward_samples(
                        x_sampled_proxy,
                        x_tt_proxy,
                        rewards,
                    )
                    if fig is not None:
                        logger.log_plots(
                            {"True reward and GFlowNet samples": fig},
                            step=current_step,
                            use_context=use_context
                        )
                
                print(f"\n[Eval @ step {current_step}] L1: {l1:.4f}, KL: {kl:.4f}, JSD: {jsd_val:.4f}, NLL: {nll_tt:.4f}")
            
        # Periodic GC
        if epoch % 10 == 0:
            gc.collect()
    
    pbar.close()
    
    print("\n" + "=" * 80)
    print("PURE JAX TRAINING COMPLETE")
    print("=" * 80)
    
    jax.clear_caches()
    return agent



