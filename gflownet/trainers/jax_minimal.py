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
from torch.utils.dlpack import to_dlpack as torch_to_dlpack
from torch.utils.dlpack import from_dlpack as torch_from_dlpack
from jax.dlpack import from_dlpack as jax_from_dlpack

from gflownet.utils.common import instantiate, set_device
from gflownet.utils.batch import Batch
from gflownet.policy.base_jax import PolicyJAX
from gflownet.utils.policy import parse_policy_config
from gflownet.envs.grid_jax import (
    sample_trajectories_jax,
    TrajectoryBatch,
    states2proxy_jax,
)

def t2j(tensor):
    """
    Convert PyTorch tensor to JAX array.
    
    NOTE: We use a CPU fallback (numpy) here to avoid a known memory leak 
    with JAX/PyTorch DLPack interop. While slightly slower due to host 
    roundtrip, it ensures stable memory usage.
    """
    if isinstance(tensor, torch.Tensor):
        return jnp.array(tensor.detach().cpu().numpy())
    return jnp.array(tensor)

def j2t(array):
    """
    Convert JAX array to PyTorch tensor.
    
    NOTE: Using CPU fallback to avoid DLPack memory leaks.
    """
    return torch.from_numpy(np.array(array))

class JAXBatchConverter:
    def __init__(self, max_states, max_trajs, device):
        self.max_states = max_states
        self.max_trajs = max_trajs
        self.device = device
        self.buffers = {}

    def pad_and_convert(self, tensor, size, fill_value=0, name=""):
        curr_size = tensor.shape[0]
        
        if curr_size > size:
            tensor = tensor[:size]
            curr_size = size

        key = name
        if key not in self.buffers:
            shape = list(tensor.shape)
            shape[0] = size
            self.buffers[key] = torch.full(shape, fill_value, dtype=tensor.dtype, device=self.device)
        
        buf = self.buffers[key]
        
        # Reallocate if shape mismatch (e.g. embedding dim changed? unlikely)
        if buf.shape[1:] != tensor.shape[1:] or buf.dtype != tensor.dtype:
             shape = list(tensor.shape)
             shape[0] = size
             buf = torch.full(shape, fill_value, dtype=tensor.dtype, device=self.device)
             self.buffers[key] = buf
        else:
             buf.fill_(fill_value)

        buf[:curr_size] = tensor
        return t2j(buf)

    def __call__(self, pytorch_batch: Batch):
        """
        Extract JAX-compatible arrays from PyTorch Batch using pre-allocated buffers.
        """
        
        traj_indices = pytorch_batch.get_trajectory_indices(consecutive=True)
        actual_n_trajs = int(traj_indices.max().item()) + 1
        actual_n_states = len(pytorch_batch)
        
        traj_indices = self.pad_and_convert(traj_indices, self.max_states, fill_value=self.max_trajs, name="traj_indices") #! fill_value=self.max_trajs
        
        # Get states for policy input
        states_policy = pytorch_batch.get_states(policy=True)
        if isinstance(states_policy, list):
            if len(states_policy) > 0 and isinstance(states_policy[0], torch.Tensor):
                states_policy = torch.stack(states_policy)
            else:
                states_policy = torch.tensor(np.array(states_policy), device=pytorch_batch.device)
        
        # Ensure states is 2D (N, state_dim)
        # if states_policy.ndim == 1:
        #     states_policy = states_policy.reshape(-1, self.state_dim)
                
        states = self.pad_and_convert(states_policy, self.max_states, name="states")
        
        actions = []
        for i in range(len(pytorch_batch)):
            action = pytorch_batch.actions[i]
            traj_idx = pytorch_batch.traj_indices[i]
            env = pytorch_batch.envs[traj_idx]
            
            if isinstance(action, tuple):
                action_idx = env.action2index(action)
            else:
                action_idx = action
            
            actions.append(action_idx)
        
        actions = torch.tensor(actions, dtype=torch.int32, device=pytorch_batch.device)
        actions = self.pad_and_convert(actions, self.max_states, name="actions")
        
        if pytorch_batch.proxy is None:
            print("WARNING: Batch has no proxy, using zero rewards")
            terminating_rewards = torch.zeros(pytorch_batch.get_n_trajectories(), dtype=torch.float32, device=pytorch_batch.device)
        else:
            terminating_rewards = pytorch_batch.get_terminating_rewards(sort_by="trajectory")
        
        terminating_rewards = self.pad_and_convert(terminating_rewards, self.max_trajs, name="terminating_rewards")
        
        rewards = pytorch_batch.get_rewards()
        rewards = self.pad_and_convert(rewards, self.max_states, name="rewards")
        
        logrewards = pytorch_batch.get_terminating_rewards(log=True, sort_by="trajectory")
        # Clamp to avoid -inf which causes NaN loss
        logrewards = torch.clamp(logrewards, min=-100.0)
        logrewards = self.pad_and_convert(logrewards, self.max_trajs, name="logrewards")
        
        logprobs_fwd, _ = pytorch_batch.get_logprobs(backward=False)
        logprobs_bwd, _ = pytorch_batch.get_logprobs(backward=True)
        
        logprobs = self.pad_and_convert(logprobs_fwd, self.max_states, name="logprobs")
        logprobs_rev = self.pad_and_convert(logprobs_bwd, self.max_states, name="logprobs_rev")
        
        parents_policy = pytorch_batch.get_parents(policy=True)
        if isinstance(parents_policy, list):
            if len(parents_policy) > 0 and isinstance(parents_policy[0], torch.Tensor):
                parents_policy = torch.stack(parents_policy)
            else:
                parents_policy = torch.tensor(np.array(parents_policy), device=pytorch_batch.device)
        
        # Ensure parents_policy is 2D (N, state_dim)
        # if parents_policy.ndim == 1:
        #     parents_policy = parents_policy.reshape(-1, self.state_dim)

        parents_policy = self.pad_and_convert(parents_policy, self.max_states, name="parents_policy")

        masks_forward_pt = pytorch_batch.get_masks_forward(of_parents=True)
        masks_backward_pt = pytorch_batch.get_masks_backward()
        
        masks_forward = self.pad_and_convert(masks_forward_pt, self.max_states, fill_value=1, name="masks_forward")
        masks_backward = self.pad_and_convert(masks_backward_pt, self.max_states, fill_value=1, name="masks_backward")
        
        return {
            'states': states,
            'parents_policy': parents_policy,
            'actions': actions,
            'rewards': rewards,
            'logprobs': logprobs,
            'logprobs_rev': logprobs_rev,
            'trajectory_indices': traj_indices,
            'masks_forward': masks_forward,
            'masks_backward': masks_backward,
            
            'terminating_rewards': terminating_rewards,
            'logrewards': logrewards,
            
            'actual_n_states': jnp.array(actual_n_states, dtype=jnp.int32),
            'actual_n_trajs': jnp.array(actual_n_trajs, dtype=jnp.int32),
        }


def _match_linear_layers(pt_model, jax_model, context):
    """
    Return PyTorch Linear modules and matching JAX Linear layers (with indices).
    Raises if the number of Linear layers does not match.
    """
    pt_linears = [layer for layer in pt_model if isinstance(layer, nn.Linear)]
    jax_linears = [
        (idx, layer)
        for idx, layer in enumerate(jax_model.layers)
        if isinstance(layer, eqx.nn.Linear)
    ]
    if len(pt_linears) != len(jax_linears):
        raise ValueError(
            f"{context}: mismatch in Linear layer count between PyTorch "
            f"({len(pt_linears)}) and JAX ({len(jax_linears)}). Ensure MLP configs align."
        )
    return pt_linears, jax_linears


def convert_params_to_jax(agent, config, key, env=None):

    if agent is not None:
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
        device = agent.device
        env_instance = agent.env
    else:
        # Defaults for pure JAX mode
        float_precision = 32
        device = "cpu"
        if env is None:
            raise ValueError("env must be provided if agent is None")
        env_instance = env

    jax_params = {}
    jax_policies = {}
    
    forward_config = parse_policy_config(config, kind="forward")
    forward_config = forward_config["config"]
    if forward_config is not None:
        forward_config["_target_"] = "gflownet.policy.base_jax.PolicyJAX"
        forward_config["key"] = None  # will be set internally

        jax_policy_f = instantiate(
            forward_config,
            env=env_instance,
            device=device,
            float_precision=float_precision,
        )
        jax_policies["forward"] = jax_policy_f

        if agent is not None and agent.forward_policy.is_model:
            pt_model_f = agent.forward_policy.model          # nn.Sequential
            jax_model_f = jax_policy_f.model                # Equinox Sequential

            pt_linears_f, jax_linears_f = _match_linear_layers(
                pt_model_f, jax_model_f, "Forward policy"
            )

            for pt_lin, (jax_idx, _) in zip(pt_linears_f, jax_linears_f):
                jax_model_f = eqx.tree_at(
                    lambda m, idx=jax_idx: m.layers[idx].weight,
                    jax_model_f,
                    jnp.array(pt_lin.weight.detach().cpu().numpy(), dtype=jnp.float32),
                )
                jax_model_f = eqx.tree_at(
                    lambda m, idx=jax_idx: m.layers[idx].bias,
                    jax_model_f,
                    jnp.array(pt_lin.bias.detach().cpu().numpy(), dtype=jnp.float32),
                )

            jax_policy_f.model = jax_model_f

        # Partition into trainable/static
        # In pure JAX mode, we assume we always have a model if policy was instantiated
        if hasattr(jax_policy_f, 'model'):
            trainable_f, static_f = eqx.partition(jax_policy_f.model, eqx.is_array)
            jax_params["forward_policy_trainable"] = trainable_f
            jax_policies["forward_static"] = static_f
        else:
            jax_params["forward_policy_trainable"] = None
            jax_policies["forward_static"] = None
    
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
            env=env_instance,
            device=device,
            float_precision=float_precision,
            base=jax_policy_f
        )

        jax_policy_b.base = jax_policies.get("forward", None)
        jax_policy_b.instantiate(jax.random.PRNGKey(0))

        jax_policies["backward"] = jax_policy_b

        if agent is not None and agent.backward_policy.is_model:
            pt_model_b = agent.backward_policy.model        # now flat nn.Sequential
            jax_model_b = jax_policy_b.model                # Equinox Sequential

            pt_linears_b, jax_linears_b = _match_linear_layers(
                pt_model_b, jax_model_b, "Backward policy"
            )

            for pt_lin, (jax_idx, _) in zip(pt_linears_b, jax_linears_b):
                jax_model_b = eqx.tree_at(
                    lambda m, idx=jax_idx: m.layers[idx].weight,
                    jax_model_b,
                    jnp.array(pt_lin.weight.detach().cpu().numpy(), dtype=jnp.float32),
                )
                jax_model_b = eqx.tree_at(
                    lambda m, idx=jax_idx: m.layers[idx].bias,
                    jax_model_b,
                    jnp.array(pt_lin.bias.detach().cpu().numpy(), dtype=jnp.float32),
                )

            jax_policy_b.model = jax_model_b

        if hasattr(jax_policy_b, 'model'):
            trainable_b, static_b = eqx.partition(jax_policy_b.model, eqx.is_array)
            jax_params["backward_policy_trainable"] = trainable_b
            jax_policies["backward_static"] = static_b
        else:
            jax_params["backward_policy_trainable"] = None
            jax_policies["backward_static"] = None
    
    if agent is not None and agent.logZ is not None:
        jax_params["logZ"] = jnp.array(agent.logZ.detach().cpu().numpy(), dtype=jnp.float32)
    else:
        # Match PyTorch initialization: ones(z_dim) * 150.0 / 64
        z_dim = config.gflownet.optimizer.z_dim
        jax_params["logZ"] = jnp.ones(z_dim, dtype=jnp.float32) * 150.0 / 64.0

    return jax_params, jax_policies

def apply_params_to_pytorch(jax_params, agent, jax_policies):
    """
    Copy JAX parameters back to PyTorch models.
    Works with arbitrary-depth MLPs (Linear + activation repeats).
    """
    # Track updated layers to avoid overwriting shared weights
    updated_modules = set()
    
    def _copy_model(pt_model, jax_model, context):
        pt_linears, jax_linears = _match_linear_layers(pt_model, jax_model, context)

        for pt_lin, (_, jax_lin) in zip(pt_linears, jax_linears):
            if id(pt_lin) in updated_modules:
                continue

            # JAX -> PyTorch (Zero Copy on GPU)
            w_t = j2t(jax_lin.weight)
            with torch.no_grad():
                pt_lin.weight.copy_(w_t)

            b_t = j2t(jax_lin.bias)
            with torch.no_grad():
                pt_lin.bias.copy_(b_t)
            
            updated_modules.add(id(pt_lin))

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
            _copy_model(pt_model_f, jax_model_f, "Forward policy")

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

            _copy_model(pt_model_b, jax_model_b, "Backward policy")

        # ---------- logZ ----------
        if jax_params.get("logZ", None) is not None and agent.logZ is not None:
            logZ_t = j2t(jax_params["logZ"])
            agent.logZ.data.copy_(logZ_t)

    except Exception as e:
        print(f"Warning: Failed to sync JAX parameters back to PyTorch: {e}")
        print("Continuing with JAX-only training (parameters won't be synced back)")


def train_pure_jax(agent, config):
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
    jax_params, jax_policies = convert_params_to_jax(agent, config, key, env=env)
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
        if agent is not None and (iteration % 10 == 0 or evaluator.should_eval(iteration)):
            apply_params_to_pytorch(jax_params, agent, jax_policies)
        
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


def train(agent, config):
    """
    JAX trainer: Supports both hybrid (PyTorch sampling + JAX training)
    and pure JAX (JAX sampling + JAX training) modes.
    """
    
    # Detect if using JAX-native environment
    if agent is None:
        # If agent is None, we assume we are in pure JAX mode and need to instantiate components
        # This handles the case where train.py passes None for "jax" mode
        use_jax_env = True
    else:
        use_jax_env = hasattr(agent.env, 'is_jax_env') and agent.env.is_jax_env
    
    if use_jax_env:
        print("\n" + "="*80)
        print("PURE JAX MODE: Using JAX-native environment and sampling")
        print("="*80 + "\n")
        return train_pure_jax(agent, config)
    else:
        print("\n" + "="*80)
        print("HYBRID MODE: PyTorch sampling + JAX training")
        print("="*80 + "\n")
    
    # !Setup Optax optimizer with separate LR for logZ
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
    
    max_grad_norm = 1.0  # Hard-coded, matching PyTorch

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
    jax_params, jax_policies = convert_params_to_jax(agent, config, key)
    opt_state = optimizer.init(jax_params)
    loss_type = config.loss.get('_target_', 'trajectorybalance').split('.')[-1].lower()
    
    #TODO: find the right sizes for states and trajectories
    n_trajs_per_sample = agent.batch_size.forward + agent.batch_size.backward_dataset + agent.batch_size.backward_replay
    MAX_TRAJS = int(n_trajs_per_sample * agent.sttr) 
    
    max_traj_len = 0
    if hasattr(config.env, 'max_step'):
        max_traj_len = config.env.max_step
    elif hasattr(config.env, 'length') and hasattr(config.env, 'n_dim'):
        max_traj_len = config.env.length * config.env.n_dim
    else:
        max_traj_len = 100 # Fallback
        
    MAX_STATES = int(MAX_TRAJS * max_traj_len * 1.2) #! used to add a 20% buffer
    
    # MAX_STATES = config.env.length ** config.env.n_dim
    
    print(f"JAX Compilation Config: MAX_TRAJS={MAX_TRAJS}, MAX_STATES={MAX_STATES}")

    @partial(jit, static_argnames=['loss_type', 'debug'])
    def jax_loss_wrapper(params, batch_arrays, loss_type=loss_type, debug=False):
        if loss_type == 'trajectorybalance': #TODO: implement different losses
            if 'forward_policy_trainable' in params and params['forward_policy_trainable'] is not None:
                model_f = eqx.combine(params['forward_policy_trainable'], jax_policies['forward_static'])
            else:
                model_f = jax_policies['forward'].model
            
            if 'backward_policy_trainable' in params and params['backward_policy_trainable'] is not None:
                model_b = eqx.combine(params['backward_policy_trainable'], jax_policies['backward_static'])
            else:
                model_b = jax_policies['backward'].model
            
            logZ = params['logZ']
            if logZ.ndim > 0: logZ = jnp.sum(logZ)
                                
            logits_f = jax.vmap(model_f)(batch_arrays['parents_policy'])
            logits_f_masked = jnp.where(batch_arrays['masks_forward'], -jnp.inf, logits_f)
            logprobs_f_all = jax.nn.log_softmax(logits_f_masked, axis=1)
            logprobs_f = logprobs_f_all[jnp.arange(MAX_STATES), batch_arrays['actions']]

            state_mask = jnp.arange(MAX_STATES) < batch_arrays['actual_n_states']
            logprobs_f = logprobs_f * state_mask

            logits_b = jax.vmap(model_b)(batch_arrays['states']) 
            logits_b_masked = jnp.where(batch_arrays['masks_backward'], -jnp.inf, logits_b)
            logprobs_b_all = jax.nn.log_softmax(logits_b_masked, axis=1)
            logprobs_b = logprobs_b_all[jnp.arange(MAX_STATES), batch_arrays['actions']]
            logprobs_b = logprobs_b * state_mask
            
            traj_indices = batch_arrays['trajectory_indices']
            
            segment_ids = traj_indices + 1  
            num_segments = MAX_TRAJS + 1    
            
            sum_logprobs_f = jax.ops.segment_sum(logprobs_f, segment_ids, num_segments=num_segments)
            sum_logprobs_b = jax.ops.segment_sum(logprobs_b, segment_ids, num_segments=num_segments)
            
            sum_logprobs_f = sum_logprobs_f[1:]
            sum_logprobs_b = sum_logprobs_b[1:]
            
            logprob_ratios = sum_logprobs_f - sum_logprobs_b
            
            log_rewards = batch_arrays['logrewards']
            
            losses = (logZ + logprob_ratios - log_rewards) ** 2
            
            #! Average over actual trajectories and avoid division by zero
            traj_mask = jnp.arange(MAX_TRAJS) < batch_arrays['actual_n_trajs']
            loss_sum = jnp.sum(losses * traj_mask)
            n_valid = jnp.sum(traj_mask)
            return loss_sum / (n_valid + 1e-8)
            
        return jnp.array(0.0)
    
    @partial(jit, static_argnames=['optimizer','loss_type'])
    def jax_grad_step(params, opt_state, batch_arrays, optimizer, loss_type=loss_type):
        """
        JIT-compiled gradient step.
        This is the ONLY function that needs to be pure JAX.
        """
        
        loss_value, grads = value_and_grad(jax_loss_wrapper)(
            params, batch_arrays, loss_type=loss_type
        )
                
        # loss_value = jax_loss_wrapper(params, batch_arrays, loss_type=loss_type, debug=True)
        # grads = grad(lambda p: jax_loss_wrapper(p, batch_arrays, loss_type=loss_type, debug=False))(params)
            
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss_value, grads
    
    total_iterations = agent.n_train_steps - agent.it + 1
    pbar = tqdm(
        initial=agent.it - 1,
        total=total_iterations,
        disable=agent.logger.progressbar.get("skip", False),
    )
    
    batch_converter = JAXBatchConverter(MAX_STATES, MAX_TRAJS, agent.device)

    for iteration in range(agent.it, agent.n_train_steps + 1):
        
        t0 = time.time()
        
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
        
        t1 = time.time()
        
        batch_arrays = batch_converter(batch)

        t2 = time.time()

        for _ in range(agent.ttsr):
            jax_params, opt_state, loss_value, grads = jax_grad_step(
                jax_params, opt_state, batch_arrays, optimizer, loss_type=loss_type
            )
        
        loss_value.block_until_ready()

            
        t3 = time.time()
        
        apply_params_to_pytorch(jax_params, agent, jax_policies)
        
        t4 = time.time()
        
        time_sample = t1 - t0
        time_convert = t2 - t1
        time_train = t3 - t2
        time_sync = t4 - t3
        
        if iteration % 10 == 0:
            print(f"Iter {iteration}: Sample={time_sample:.4f}s, Convert={time_convert:.4f}s, Train={time_train:.4f}s, Sync={time_sync:.4f}s")
        
        #! Buffer
        # states_term = batch.get_terminating_states(sort_by="trajectory")
        # proxy_vals = batch.get_terminating_proxy_values(sort_by="trajectory")
        # rewards = batch.get_terminating_rewards(sort_by="trajectory")
        # actions_trajectories = batch.get_actions_trajectories()
        
        # # Update main buffer (if enabled)
        # if agent.buffer.use_main_buffer:
        #     agent.buffer.add(states_term, actions_trajectories, rewards, iteration, buffer="main")
        
        # # Update replay buffer
        # agent.buffer.add(states_term, actions_trajectories, rewards, iteration, buffer="replay")
        
        #! Logging WandB 
        if agent.evaluator.should_log_train(iteration):

            # Extract raw numpy arrays directly to minimize conversions
            rewards_np = np.asarray(batch_arrays["terminating_rewards"], dtype=np.float32)
            logrewards_np = np.log(rewards_np + 1e-8)

            rewards_t = torch.from_numpy(rewards_np)
            logrewards_t = torch.from_numpy(logrewards_np)

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
            
            #! Explicitly delete intermediate tensors
            del rewards_t, logrewards_t, proxy_vals_t, rewards_np, logrewards_np

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

            if "logZ" in jax_params:
                logz = float(jnp.sum(jax_params["logZ"]))
            else:
                logz = None


            lr_main = float(lr_schedule_main(iteration))
            lr_logz = float(lr_schedule_logz(iteration))
            grad_logZ = np.asarray(grads["logZ"], dtype=np.float32)

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
                "time_sample": time_sample,
                "time_convert": time_convert,
                "time_train": time_train,
                "time_sync": time_sync,
            }

            agent.logger.log_metrics(
                metrics=metrics,
                step=iteration,
                use_context=agent.use_context,
            )

            losses = {
                "all": float(loss_value),                 
                "Loss": float(loss_value),               
            }

            agent.logger.log_metrics(
                metrics=losses,
                step=iteration,
                use_context=agent.use_context,
            )
            
            del metrics, losses, grad_logZ

        term_rewards_for_pbar = np.asarray(batch_arrays["terminating_rewards"], dtype=np.float32)
        agent.logger.progressbar_update(
            pbar,
            float(loss_value),
            term_rewards_for_pbar,
            agent.jsd,
            agent.use_context,
        )
        del term_rewards_for_pbar
        
        if agent.evaluator.should_eval(iteration):
            agent.evaluator.eval_and_log(iteration)
        
        if hasattr(batch, 'envs'):
            batch.envs = []
        
        del batch
        del batch_arrays
        
        if iteration % 100 == 0:
            batch_converter.buffers.clear()
            gc.collect()

    pbar.close()
    
    print("\n" + "=" * 80)
    print("PHASE 1 TRAINING COMPLETE")
    print(f"  Final loss: {loss_value:.4f}")
    print("=" * 80)
    
    jax.clear_caches()
    return agent
