"""
Wrapper around the existing GFlowNetAgent.train() loop.
"""

from typing import Any

def train(agent, config: Any) -> None:
    _ = config  # Unused in this wrapper
    agent.train() #! Calling train method of GFlowNetAgent in gflownet/gflownet.py
