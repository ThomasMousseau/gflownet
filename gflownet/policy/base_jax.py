# jax_model_base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx
from omegaconf import OmegaConf
# ---- Base -------------------------------------------------------------------
class ModelBaseJAX(ABC):
    def __init__(self, config, env, device, float_precision, base=None):
        # device is implicit in JAX; keep for API parity
        self.dtype: jnp.dtype = to_jnp_dtype(float_precision)

        # Env-provided dims / fixed distributions
        self.state_dim: int = config.get("state_dim", int(env.policy_input_dim))
        self.fixed_output = jnp.asarray(env.fixed_policy_output, dtype=self.dtype)
        self.random_output = jnp.asarray(env.random_policy_output, dtype=self.dtype)
        self.output_dim: int = int(self.fixed_output.shape[0])

        self.base = base
        self.parse_config(config)

    def parse_config(self, config):
        if config is None:
            config = OmegaConf.create({"type": "uniform"})

        self.checkpoint = config.get("checkpoint", None)
        self.shared_weights = bool(config.get("shared_weights", False))
        self.n_hid = config.get("n_hid", None)
        self.n_layers = config.get("n_layers", None)
        self.tail: Sequence = config.get("tail", [])  # ignored here unless you map tails

        if "type" in config:
            self.type = config.type
        elif self.shared_weights and self.base is not None:
            self.type = self.base.type
        else:
            # Default to mlp for backward policies with shared_weights but no base
            self.type = "mlp"

    @abstractmethod
    def instantiate(self, key: jax.Array):
        pass

    # keep a nice entrypoint that casts inputs
    def __call__(self, states: jnp.ndarray) -> jnp.ndarray:
        states = states.astype(self.dtype)
        return self.model(states)

    # handy if you want params/state splits later (e.g., for optimizers)
    def get_params(self):
        # With Equinox you can do:
        # trainable, static = eqx.partition(self, eqx.is_array)
        # return trainable
        return self

    # ----------------- Equinox MLP builder -----------------
    def make_mlp(self, activation: Callable, key: jax.Array) -> eqx.Module:
        """
        Build an MLP ending in a Linear to output_dim.
        If shared_weights=True and base is present, reuse base trunk (all but last Linear)
        and create a fresh last Linear with matching shapes.
        """
        # If shared_weights but no base, disable sharing
        if self.shared_weights and self.base is None:
            self.shared_weights = False
        
        if self.shared_weights:
            if self.base is None or not isinstance(self.base.model, eqx.nn.Sequential):
                raise ValueError(
                    "For shared_weights=True, base.model must be an eqx.nn.Sequential."
                )

            base_layers = list(self.base.model.layers)
            if not base_layers or not isinstance(base_layers[-1], eqx.nn.Linear):
                raise ValueError("Base model must end with eqx.nn.Linear for sharing.")

            last: eqx.nn.Linear = base_layers[-1]
            trunk_layers = base_layers[:-1]  # reuse by reference

            # fresh last layer (same in/out as base's last)
            new_last = eqx.nn.Linear(
                in_features=last.in_features,
                out_features=last.out_features,
                use_bias=True,
                key=key,
            )
            return eqx.nn.Sequential(tuple(trunk_layers + [new_last]))

        # Build new MLP
        if (self.n_hid is None) or (self.n_layers is None):
            raise ValueError("n_hid and n_layers must be set when shared_weights=False.")

        dims = [self.state_dim] + [int(self.n_hid)] * int(self.n_layers) + [self.output_dim]
        keys = jax.random.split(key, num=len(dims) - 1)

        layers = []
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(eqx.nn.Linear(int(din), int(dout), use_bias=True, key=keys[i]))
            if i < len(dims) - 2:
                # Hidden layer activation
                layers.append(Activation(lambda x, f=activation: f(x)))

        return eqx.nn.Sequential(tuple(layers))


# ---- Policy -----------------------------------------------------------------
class PolicyJAX(ModelBaseJAX):
    def __init__(self, env, device, float_precision, base=None, key: Optional[jax.Array] = None, instantiate_now: bool = True, **config):
        # Convert dict config to OmegaConf for consistency with PyTorch
        config = OmegaConf.create(config)
        super().__init__(config, env, device, float_precision, base)
        instantiate_now = config.get("instantiate_now", True)
        if instantiate_now:
            if key is None:
                key = jax.random.PRNGKey(0)
            self.instantiate(key)

    def instantiate(self, key: jax.Array):
        if self.type == "fixed":
            self.model = self.fixed_distribution
            self.is_model = False
        elif self.type == "uniform":
            self.model = self.uniform_distribution
            self.is_model = False
        elif self.type == "mlp":
            # You can use jax.nn.leaky_relu here
            self.model = self.make_mlp(lambda x: jax.nn.leaky_relu(x, negative_slope=0.01), key)
            self.is_model = True
        else:
            raise RuntimeError("Policy model type not defined")

    # The distribution "models" below are callables matching eqx.Module signature
    def fixed_distribution(self, states: jnp.ndarray) -> jnp.ndarray:
        batch = states.shape[0]
        out = jnp.broadcast_to(self.fixed_output[None, :], (batch, self.output_dim))
        return out.astype(self.dtype)

    def random_distribution(self, states: jnp.ndarray) -> jnp.ndarray:
        batch = states.shape[0]
        out = jnp.broadcast_to(self.random_output[None, :], (batch, self.output_dim))
        return out.astype(self.dtype)

    def uniform_distribution(self, states: jnp.ndarray) -> jnp.ndarray:
        batch = states.shape[0]
        return jnp.ones((batch, self.output_dim), dtype=self.dtype)
    
    def parameters(self):
        """
        Returns an empty generator for compatibility with PyTorch parameter API.
        JAX/Equinox models don't use the same parameter structure as PyTorch.
        
        For actual JAX training, use equinox.partition to get trainable/static parts.
        """
        # Return empty generator for compatibility
        return iter([])
    
# ---- Utilities --------------------------------------------------------------
def to_jnp_dtype(precision) -> jnp.dtype:
    if isinstance(precision, jnp.dtype):
        return precision
    if precision in (16, "16", "float16"):
        return jnp.float16
    if precision in (32, "32", "float32"):
        return jnp.float32
    if precision in (64, "64", "float64"):
        return jnp.float64
    raise ValueError("precision must be one of {16,32,64} or a jnp.dtype")

class Activation(eqx.Module):
    fn: Callable

    def __call__(self, x):
        return self.fn(x)


