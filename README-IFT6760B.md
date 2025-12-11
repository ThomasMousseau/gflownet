# GFlowNet JAX Implementation

This project implements GFlowNet using JAX and benchmarks it against the legacy PyTorch implementation.

## Main Changes
- `gflownet/trainers/jax_minimal.py`: Minimal JAX trainer implementation.
- `gflownet/trainers/jax.py`: Full JAX trainer implementation.
- `gflownet/envs/grid_jax.py`: JAX-optimized Grid environment.
- `gflownet/policy/base_jax.py`: Base class for JAX policies.

## Requirements
- **Python**: 3.12.0
- **Dependencies**: `requirements.txt` (includes `jax`, `equinox`, `optax`).

## Experiments
- **CPU Experiments**: Run using `run_experiments.ipynb`.
- **Hardware Experiments**: Copied from Google Colab.

## Usage

### Complete JAX
Runs the full JAX implementation with the JAX-optimized Grid environment.
```bash
python train.py \
    trainer.mode=jax \
    buffer.test.type=all \
    env=grid_jax \
    env.n_dim=2 \
    env.length=16 \
    proxy=box/corners \
    gflownet.random_action_prob=0.0 \
    gflownet.optimizer.batch_size.forward=100 \
    gflownet.optimizer.lr=0.0001 \
    gflownet.optimizer.z_dim=16 \
    gflownet.optimizer.lr_z_mult=100 \
    gflownet.optimizer.n_train_steps=5000 \
    policy.forward.n_hid=128 \
    policy.forward.n_layers=2 \
    evaluator.first_it=true \
    evaluator.period=500 \
    evaluator.n=1000 \
    evaluator.checkpoints_period=500 \
    seed=100
```

### Minimal JAX
Runs the minimal JAX trainer with the standard Grid environment.
```bash
python train.py \
    trainer.mode=jax_minimal \
    buffer.test.type=all \
    env=grid \
    env.n_dim=2 \
    env.length=16 \
    proxy=box/corners \
    gflownet.random_action_prob=0.0 \
    gflownet.optimizer.batch_size.forward=100 \
    gflownet.optimizer.lr=0.0001 \
    gflownet.optimizer.z_dim=16 \
    gflownet.optimizer.lr_z_mult=100 \
    gflownet.optimizer.n_train_steps=5000 \
    policy.forward.n_hid=128 \
    policy.forward.n_layers=2 \
    evaluator.first_it=true \
    evaluator.period=500 \
    evaluator.n=1000 \
    evaluator.checkpoints_period=500 \
    seed=100
```

### Legacy PyTorch
Runs the minimal JAX trainer with the standard Grid environment.
```bash
python train.py \
    trainer.mode=legacy \
    buffer.test.type=all \
    env=grid \
    env.n_dim=2 \
    env.length=16 \
    proxy=box/corners \
    gflownet.random_action_prob=0.0 \
    gflownet.optimizer.batch_size.forward=100 \
    gflownet.optimizer.lr=0.0001 \
    gflownet.optimizer.z_dim=16 \
    gflownet.optimizer.lr_z_mult=100 \
    gflownet.optimizer.n_train_steps=5000 \
    policy.forward.n_hid=128 \
    policy.forward.n_layers=2 \
    evaluator.first_it=true \
    evaluator.period=500 \
    evaluator.n=1000 \
    evaluator.checkpoints_period=500 \
    seed=100
```
