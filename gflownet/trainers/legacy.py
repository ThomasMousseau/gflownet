"""
Wrapper around the existing GFlowNetAgent.train() loop.
"""

from typing import Any

def train(agent, config: Any) -> None:
    config = config  # Unused in this wrapper
    agent.train(config) #! Calling train method of GFlowNetAgent in gflownet/gflownet.py