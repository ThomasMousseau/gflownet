import torch
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from hydra import compose, initialize
from omegaconf import OmegaConf

from gflownet.utils.common import gflownet_from_config
from gflownet.trainers.jax_minimal import convert_params_to_jax, apply_params_to_pytorch

def test_sync():
    with initialize(version_base=None, config_path="config"):
        config = compose(config_name="train", overrides=[
            "env=grid", 
            "trainer.mode=jax",
            "gflownet.optimizer.batch_size.forward=1"
        ])

    agent = gflownet_from_config(config)
    
    key = jax.random.PRNGKey(0)
    jax_params, jax_policies = convert_params_to_jax(agent, config, key)
    
    print("JAX params keys:", jax_params.keys())
    
    if "forward_policy_trainable" not in jax_params:
        print("Forward policy not trainable in JAX!")
        return

    # Modify JAX params
    print("\nModifying JAX params...")
    # Add 1.0 to the first layer weight
    old_weight = jax_params["forward_policy_trainable"].layers[0].weight
    new_weight = old_weight + 1.0
    
    jax_params["forward_policy_trainable"] = eqx.tree_at(
        lambda m: m.layers[0].weight,
        jax_params["forward_policy_trainable"],
        new_weight
    )
    
    print("Syncing back to PyTorch...")
    print(f"Agent forward policy model[0] ID: {id(agent.forward_policy.model[0])}")
    print(f"Agent forward policy model[0] weight ID: {id(agent.forward_policy.model[0].weight)}")
    
    if hasattr(agent.backward_policy, 'model'):
        print(f"Agent backward policy model[0] ID: {id(agent.backward_policy.model[0])}")
        print(f"Agent backward policy model[0] weight ID: {id(agent.backward_policy.model[0].weight)}")
        
    apply_params_to_pytorch(jax_params, agent, jax_policies)
    print(f"Agent forward policy model[0] ID after: {id(agent.forward_policy.model[0])}")
    print(f"Agent forward policy model[0] weight ID after: {id(agent.forward_policy.model[0].weight)}")
    print(f"Agent forward policy model[0] weight[0,0] after: {agent.forward_policy.model[0].weight[0,0].item()}")
    
    pt_weight = agent.forward_policy.model[0].weight.detach().cpu().numpy()
    jax_weight = np.array(new_weight)
    
    diff = np.abs(pt_weight - jax_weight).max()
    print(f"Max difference: {diff}")
    
    if diff < 1e-5:
        print("SUCCESS: PyTorch params updated correctly.")
    else:
        print("FAILURE: PyTorch params NOT updated correctly.")
        print("PyTorch weight sample:", pt_weight[0, :5])
        print("JAX weight sample:    ", jax_weight[0, :5])

if __name__ == "__main__":
    test_sync()
