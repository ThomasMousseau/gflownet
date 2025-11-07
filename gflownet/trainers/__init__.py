"""
Registry of training backends.

This keeps the legacy PyTorch loop untouched while exposing hooks for the
upcoming pure-PyTorch and JAX implementations.
"""

from typing import Any, Callable, Dict, TYPE_CHECKING

from . import jax as jax_trainer
from . import legacy as legacy_trainer
from . import pure as pure_trainer

if TYPE_CHECKING:  # pragma: no cover
    from gflownet.gflownet import GFlowNetAgent

TrainerFn = Callable[["GFlowNetAgent", Any], None]

_TRAINERS: Dict[str, TrainerFn] = {
    "legacy": legacy_trainer.run,
    "pure": pure_trainer.run,
    "jax": jax_trainer.run,
}


def get_trainer(mode: str) -> TrainerFn:
    """
    Return the trainer entry point matching `mode`.
    """
    mode_key = mode.lower()
    if mode_key not in _TRAINERS:
        available = ", ".join(sorted(_TRAINERS.keys()))
        raise ValueError(f"Unknown trainer.mode='{mode}'. Available: {available}")
    return _TRAINERS[mode_key]
