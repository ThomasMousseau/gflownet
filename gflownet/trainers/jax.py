"""
JAX trainer implementation (pure functional style).
"""

from typing import Any, Tuple
from dataclasses import dataclass, replace
import jax
import jax.numpy as jnp #! missing pip install jax
import jax.random as random
from tqdm import tqdm
import time
import gc

import optax #! missing pip install jax
import equinox as eqx  #! missing pip install equinox

@dataclass
class TrainingState:
    agent: Any  # JAX-compatible agent
    iteration: int
    n_train_steps: int
    sttr: Any
    ttsr: Any
    batch_size: Any
    buffer: Any
    loss: Any
    optimizer: Any  # Optax optimizer state
    lr_scheduler: Any  # Optax schedule
    evaluator: Any
    logger: Any
    env: Any
    proxy: Any
    device: Any  # JAX device
    float_type: Any
    clip_grad_norm: float
    garbage_collection_period: int
    use_context: bool
    pbar: Any
    rng_key: Any  # JAX random key

def train(agent: Any, config: Any) -> None:
    pass