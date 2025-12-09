import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad, grad
import equinox as eqx
import optax
import gc
from tqdm import tqdm
from gflownet.utils.common import instantiate
from gflownet.utils.policy import parse_policy_config
from gflownet.envs.grid_jax import (
    sample_trajectories_jax,
    states2proxy_jax,
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
            return (new_p, new_o), l
            
        if ttsr == 1:
            (jax_params, opt_state), loss = update_body((jax_params, opt_state), None)
            step_losses = jnp.reshape(loss, (1,))
        else:
            (jax_params, opt_state), step_losses = jax.lax.scan(
                update_body, (jax_params, opt_state), None, length=ttsr
            )
        
        return jax_params, opt_state, key, jnp.mean(step_losses), log_rewards

    @jit
    def train_epoch(jax_params, opt_state, key):
        """Run a chunk of training steps fully JIT-compiled using scan."""
        def body_fn(carry, _):
            params, opt, k = carry
            new_params, new_opt, new_k, loss, log_rewards = train_step(params, opt, k)
            return (new_params, new_opt, new_k), (loss, log_rewards)

        (jax_params, opt_state, key), (losses, all_log_rewards) = jax.lax.scan(
            body_fn, (jax_params, opt_state, key), None, length=steps_per_epoch
        )
        return jax_params, opt_state, key, losses, all_log_rewards

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
        jax_params, opt_state, key, losses, all_log_rewards = train_epoch(jax_params, opt_state, key)
        
        t1 = time.time()
        
        # Metrics
        avg_loss = float(jnp.mean(losses))
        current_step = start_it + (epoch + 1) * steps_per_epoch
        
        # Logging
        if evaluator.should_log_train(current_step):
            metrics = {
                "step": current_step,
                "time_epoch": t1 - t0,
                "steps_per_sec": steps_per_epoch / (t1 - t0),
                "all": avg_loss
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
        if agent is not None and evaluator.should_eval(current_step):
            # Note: Evaluation might still be slow/PyTorch-based
            evaluator.eval_and_log(current_step)
            
        # Periodic GC
        if epoch % 10 == 0:
            gc.collect()
    
    pbar.close()
    
    print("\n" + "=" * 80)
    print("PURE JAX TRAINING COMPLETE")
    print("=" * 80)
    
    jax.clear_caches()
    return agent



