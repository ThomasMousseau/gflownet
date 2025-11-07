"""
Wrapper around the existing GFlowNetAgent.train() loop.
"""

from typing import Any


def run(agent, config: Any) -> None:
    """
    Run the legacy PyTorch training loop.

    Parameters
    ----------
    agent : gflownet.gflownet.GFlowNetAgent
        Trainer-ready agent built from the Hydra config.
    config : omegaconf.DictConfig
        Full Hydra config. Unused for now but kept for API parity with future
        trainer implementations.
    """

    _ = config  # Avoid unused-argument warnings
    agent.train()
