# state_flow_jax.py
from typing import Optional
import jax
import jax.numpy as jnp
import equinox as eqx

from gflownet.policy.base_jax import ModelBaseJAX  # your base we fixed earlier
# If you wrote a tiny wrapper:
# from .activations import leaky_relu   # or use jax.nn.leaky_relu directly

class StateFlowJAX(ModelBaseJAX):
    """
    Takes state in the policy format and predicts its flow (a scalar).
    JAX/Equinox version of the PyTorch StateFlow.
    """

    def __init__(self, config, env, device, float_precision, base=None, key: Optional[jax.Array]=None):
        super().__init__(config, env, device, float_precision, base)
        # Override output dimension to 1 (scalar flow)
        self.output_dim = 1
        # PRNG key (create one if not provided)
        if key is None:
            key = jax.random.PRNGKey(0)
        self.instantiate(key)

    def instantiate(self, key: jax.Array):
        if self.type == "mlp":
            # Use JAX's leaky_relu
            self.model = self.make_mlp(lambda x: jax.nn.leaky_relu(x, negative_slope=0.01), key)
            self.is_model = True
        else:
            raise RuntimeError("StateFlow model type not defined")

    def __call__(self, states: jnp.ndarray) -> jnp.ndarray:
        """
        Returns a 1D array of state flows with shape (batch_size,).
        """
        # Route through base’s call to keep dtype behavior consistent
        out = super().__call__(states)
        return jnp.squeeze(out, axis=-1)

    # Optional: return the trainable pytree (useful for Optax)
    def get_params(self):
        # For simple cases you can just return `self`
        # If you introduce non-trainable state later, use:
        # trainable, static = eqx.partition(self, eqx.is_array)
        # return trainable
        return self
