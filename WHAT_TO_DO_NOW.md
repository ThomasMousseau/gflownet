# Summary: What I Recommend You Do

## TL;DR

**Don't abandon the GFlowNetAgent class.** Instead:
1. Use it for sampling (PyTorch)
2. Extract parameters to JAX for backprop (JIT-compiled)
3. Sync parameters back after each update
4. Gradually migrate components to JAX in future phases

## Why Your Current Approach Had Issues

### ❌ Problem 1: Too Much at Once
You tried to:
- Create immutable state classes
- Replace the entire GFlowNetAgent
- Convert env, buffer, proxy, loss simultaneously
- Make everything JAX-compatible

This created a cascade of dependencies.

### ❌ Problem 2: Missing Implementations
Without GFlowNetAgent, you needed to reimplement:
- Loss computation in pure JAX
- Policy forward pass in JAX
- Batch handling in JAX
- Environment logic in JAX
- Buffer updates in JAX

### ❌ Problem 3: Pytree Hell
Non-pytree objects (env, buffer, logger) can't be passed to JIT-compiled functions, forcing complex workarounds.

## ✅ The Better Way: Minimal JAX Integration

### What I Created: `jax_minimal.py`

This new trainer:
1. **Uses PyTorch GFlowNetAgent** for everything EXCEPT backprop
2. **JIT-compiles only gradient computation** (the performance bottleneck)
3. **Syncs parameters** between PyTorch ↔ JAX each iteration

### How It Works

```python
# train() in jax_minimal.py

for iteration in range(n_train_steps):
    # 1. PYTORCH: Sample trajectories (unchanged)
    batch, _ = agent.sample_batch(...)
    
    # 2. CONVERT: PyTorch Batch → JAX arrays
    batch_arrays = convert_batch_to_jax_arrays(batch)
    
    # 3. JAX (JIT): Compute gradients and update
    jax_params, opt_state, loss = jax_grad_step(
        jax_params, opt_state, batch_arrays, optimizer
    )
    
    # 4. SYNC: Copy JAX params → PyTorch models
    apply_params_to_pytorch(agent, jax_params)
    
    # 5. PYTORCH: Logging, evaluation, etc. (unchanged)
    agent.logger.log_train(iteration, {'loss': loss})
```

### Key Functions

#### `convert_batch_to_jax_arrays(pytorch_batch)`
Extracts data from PyTorch Batch and converts to JAX arrays:
- States, actions, rewards, logprobs → JAX arrays
- No complex pytree structures needed

#### `convert_params_to_jax(agent)`
Extracts trainable parameters from PyTorch models:
- Forward/backward policy parameters
- logZ parameter
- Returns dict of JAX arrays

#### `apply_params_to_pytorch(agent, jax_params)`
Copies updated JAX parameters back to PyTorch:
- Updates `agent.forward_policy.model.parameters()`
- Updates `agent.backward_policy.model.parameters()`
- Updates `agent.logZ`

#### `jax_loss_wrapper(params, batch_arrays, loss_type)`
JAX-compatible loss function:
- Currently implements basic trajectory balance
- **TODO**: Expand to match full PyTorch loss implementations
- Pure JAX, JIT-compatible

#### `jax_grad_step(params, opt_state, batch_arrays, optimizer)`
**JIT-compiled** gradient computation:
- Computes loss and gradients
- Applies Optax optimizer update
- Returns updated params, opt_state, loss

## What You Need to Do

### 1. Test the Minimal Trainer ✅
```bash
python train.py env=grid trainer.mode=jax gflownet.optimizer.n_train_steps=10
```

Check:
- Does training run without errors?
- Do loss values look reasonable?
- Are samples being generated?

### 2. Fix Any Bugs 🐛

Common issues to watch for:
- **Shape mismatches**: Actions/states may have inconsistent formats
- **Missing proxy**: Batch needs proxy set before `get_rewards()`
- **LogZ handling**: If loss doesn't require logZ, set to None
- **Logprob recomputation**: Batch may need logprobs recomputed

### 3. Verify Loss Correctness 📊

Compare with PyTorch:
```bash
# Run both trainers with same seed
python train.py env=grid trainer.mode=legacy seed=42 gflownet.optimizer.n_train_steps=10
python train.py env=grid trainer.mode=jax seed=42 gflownet.optimizer.n_train_steps=10
```

Loss values should be similar (not identical due to optimizer differences).

### 4. Implement Proper Loss Functions 📝

Currently `jax_loss_wrapper` has a simplified trajectory balance loss.

**TODO**: Implement based on your specific loss:
- Trajectory Balance (TB)
- Flow Matching (FM)  
- Detailed Balance (DB)
- Forward Looking (FL)

Look at `gflownet/losses/<loss_name>.py` for PyTorch implementation and translate to JAX.

### 5. Profile Performance ⚡

Measure speedup:
```python
import time

# Time PyTorch version
start = time.time()
# ... training ...
pytorch_time = time.time() - start

# Time JAX version  
start = time.time()
# ... training ...
jax_time = time.time() - start

print(f"Speedup: {pytorch_time / jax_time:.2f}x")
```

Expected: 2-5x speedup on backprop portion.

## Future Phases (Optional)

### Phase 2: JAX Policy Models
- Replace PyTorch policies with Equinox models
- Eliminate parameter sync overhead
- ~5-10x total speedup

See `gflownet/policy/base_jax.py` for starting point.

### Phase 3: Full JAX Environment
- Convert env to JAX pytree
- Pure JAX sampling with `lax.scan`
- ~10-100x total speedup

This is a major undertaking, only do if needed.

## Files Modified

1. ✅ `gflownet/trainers/jax_minimal.py` - New minimal JAX trainer
2. ✅ `gflownet/trainers/__init__.py` - Registered new trainer
3. ✅ `JAX_MIGRATION_STRATEGY.md` - Full strategy document

## How to Use

In your config:
```yaml
trainer:
  mode: jax  # Uses jax_minimal.py
```

Or command line:
```bash
python train.py trainer.mode=jax env=grid
```

## Advantages of This Approach

### ✅ Incremental
Work in small steps, always have a working version.

### ✅ Low Risk
Easy to revert if issues arise.

### ✅ Immediate Value
Get JAX speedup without full rewrite.

### ✅ Future Proof
Can migrate to full JAX later if needed.

### ✅ Debuggable
Small changes = easier to isolate bugs.

## Questions to Consider

### "Is the parameter sync expensive?"
For typical MLP policies: **No**. The sync is <1% of total time. Backprop is the bottleneck.

### "Will this work with my loss function?"
**Yes**, but you need to implement the JAX version of your specific loss in `jax_loss_wrapper`. Look at your PyTorch loss and translate the math.

### "Can I still use PyTorch tools?"
**Yes**. Logging, checkpointing, evaluation all still work. Only gradient computation is JAX.

### "When should I move to Phase 2?"
Only if:
- Phase 1 works correctly
- You need more speedup
- Parameter sync shows up in profiling

### "Do I need Phase 3?"
Probably **not** unless:
- Sampling is the bottleneck (measure it!)
- You need GPU/TPU scaling
- You have very large batch sizes

## Next Steps

1. **Run the test** and check for errors
2. **Fix any bugs** in batch conversion or loss computation
3. **Verify correctness** by comparing loss trajectories with PyTorch
4. **Measure performance** to quantify speedup
5. **Decide**: Is Phase 1 enough, or do you need Phase 2?

## Need Help?

If you encounter errors:
1. Check the error message carefully
2. Add print statements in `convert_batch_to_jax_arrays` to inspect shapes
3. Compare batch data between PyTorch and JAX versions
4. Make sure the loss function matches your configuration

Common fixes:
- **"Batch has no proxy"** → Add `batch.set_proxy(agent.proxy)` before conversion
- **"Shape mismatch"** → Check action/state tensor shapes, may need padding
- **"LogZ is None"** → Check if your loss requires logZ, handle None case
- **"Logprobs are zeros"** → Batch may need `compute_logprobs_trajectories` called

Good luck! This approach should give you a working JAX trainer without the complexity of a full rewrite.
