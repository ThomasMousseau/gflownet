# import os
# # Configure JAX to use standard malloc and disable preallocation to prevent OOM on CPU
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad, grad
import equinox as eqx
import torch.nn as nn
import torch
import optax
import gc
from functools import partial
from tqdm import tqdm

from gflownet.utils.common import instantiate
from gflownet.policy.base_jax import PolicyJAX
from gflownet.utils.policy import parse_policy_config
from gflownet.envs.grid_jax import (
    sample_trajectories_jax,
    TrajectoryBatch,
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

    def compute_log_rewards(batch: TrajectoryBatch) -> jnp.ndarray:
        num_trajs = int(batch.actual_n_trajs)
        if num_trajs == 0 or proxy is None:
            return jnp.zeros(num_trajs, dtype=jnp.float32)

        states = batch.states.reshape(
            num_trajs, grid_config.max_traj_length, grid_config.n_dim
        )
        traj_lengths = jnp.asarray(batch.actual_lengths, dtype=jnp.int32)
        last_indices = jnp.clip(traj_lengths - 1, 0)
        traj_ids = jnp.arange(num_trajs, dtype=jnp.int32)
        terminal_states = states[traj_ids, last_indices]

        proxy_inputs = states2proxy_jax(terminal_states, grid_config)
        proxy_inputs_np = np.asarray(proxy_inputs, dtype=np.float32)

        proxy_device = getattr(proxy, "device", device)
        proxy_dtype = getattr(proxy, "float", torch.float32)
        states_proxy_t = torch.from_numpy(proxy_inputs_np).to(
            device=proxy_device, dtype=proxy_dtype
        )

        with torch.no_grad():
            logrewards_t = proxy.rewards(states_proxy_t, log=True)

        log_rewards_np = logrewards_t.detach().to("cpu").numpy().reshape(num_trajs)
        # Clamp to avoid -inf which causes NaN loss
        log_rewards_np = np.maximum(log_rewards_np, -100.0)
        return jnp.asarray(log_rewards_np, dtype=jnp.float32)
    
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
    
    @partial(jit, static_argnames=['optimizer', 'loss_type', 'actual_n_states', 'actual_n_trajs'])
    def jax_grad_step(params, opt_state, states, parents, actions, action_indices,
                     masks_forward, masks_backward, trajectory_indices,
                     actual_n_states, actual_n_trajs, actual_lengths, log_rewards,
                     optimizer, loss_type=loss_type):
        """JIT-compiled gradient step for pure JAX."""
        
        loss_value, grads = value_and_grad(jax_loss_and_grad)(
            params, states, parents, actions, action_indices,
            masks_forward, masks_backward, trajectory_indices,
            actual_n_states, actual_n_trajs, actual_lengths, log_rewards, loss_type=loss_type
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_value, grads
    
    # Training loop
    total_iterations = n_train_steps - start_it + 1
    pbar = tqdm(
        initial=start_it - 1,
        total=total_iterations,
        disable=logger.progressbar.get("skip", False),
    )
    
    for iteration in range(start_it, n_train_steps + 1):
        t0 = time.time()
        
        # Sample trajectories using pure JAX
        batch, key = sample_trajectories_jax(
            key=key,
            config=grid_config,
            policy_model=jax_policies['forward'].model,
            n_trajectories=n_trajs_total,
            temperature=1.0,
            return_logprobs=False
        )
        
        t1 = time.time()
        log_rewards = compute_log_rewards(batch)
        
        # No conversion needed - already JAX!
        t2 = time.time()
        
        # Train with JAX
        for _ in range(ttsr):
            jax_params, opt_state, loss_value, grads = jax_grad_step(
                jax_params, opt_state, 
                batch.states, batch.parents, batch.actions, batch.action_indices,
                batch.masks_forward, batch.masks_backward, batch.trajectory_indices,
                batch.actual_n_states, batch.actual_n_trajs, batch.actual_lengths, log_rewards,
                optimizer, loss_type=loss_type
            )
        
        loss_value.block_until_ready()
        
        t3 = time.time()
        
        # Sync back to PyTorch only when needed
        # if agent is not None and (iteration % 10 == 0 or evaluator.should_eval(iteration)):
        #     apply_params_to_pytorch(jax_params, agent, jax_policies)
        
        t4 = time.time()
        
        time_sample = t1 - t0
        time_convert = t2 - t1  # Should be ~0
        time_train = t3 - t2
        time_sync = t4 - t3
        
        if iteration % 10 == 0:
            print(f"Iter {iteration}: Sample={time_sample:.4f}s, Convert={time_convert:.6f}s, Train={time_train:.4f}s, Sync={time_sync:.4f}s")
        
        # Logging
        if evaluator.should_log_train(iteration):
            metrics = {
                "step": iteration,
                "time_sample_jax": time_sample,
                "time_convert": time_convert,
                "time_train": time_train,
                "time_sync": time_sync,
            }
            
            logger.log_metrics(
                metrics=metrics,
                step=iteration,
                use_context=use_context,
            )
            
            losses = {"Loss": float(loss_value)}
            logger.log_metrics(
                metrics=losses,
                step=iteration,
                use_context=use_context,
            )
        
        # Progress bar
        logger.progressbar_update(
            pbar,
            float(loss_value),
            jnp.zeros(n_trajs_total),  # Placeholder
            jsd,
            use_context,
        )
        
        # Evaluation
        if agent is not None and hasattr(agent, 'sample_batch') and evaluator.should_eval(iteration):
            evaluator.eval_and_log(iteration)
        
        # Periodic GC
        if iteration % 100 == 0:
            gc.collect()
    
    pbar.close()
    
    print("\\n" + "=" * 80)
    print("PURE JAX TRAINING COMPLETE")
    print(f"  Final loss: {loss_value:.4f}")
    print("=" * 80)
    
    jax.clear_caches()
    return agent



