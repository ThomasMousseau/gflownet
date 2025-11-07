"""
Wrapper around the existing GFlowNetAgent.train() loop.
"""

from typing import Any

def train(agent) -> None:
    agent.train() #! Calling train method of GFlowNetAgent in gflownet/gflownet.py
