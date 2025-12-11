from typing import Any, Callable, Dict, TYPE_CHECKING

from. import jax as jax_trainer
from . import jax_minimal as jax_minimal_trainer
from . import legacy as legacy_trainer

if TYPE_CHECKING:  # pragma: no cover
    from gflownet.gflownet import GFlowNetAgent

TrainerFn = Callable[["GFlowNetAgent", Any], Any]

_TRAINERS: Dict[str, TrainerFn] = {
    "legacy": legacy_trainer.train,
    "jax_minimal": jax_minimal_trainer.train,
    "jax": jax_trainer.train,
    
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
