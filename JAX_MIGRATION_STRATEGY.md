# JAX Migration Strategy: Incremental Approach

## Problem Statement

You're trying to port a PyTorch GFlowNet training loop to JAX, but hitting issues with:
- Non-pytree compatible objects (env, buffer, proxy, logger)
- Complex interdependencies (policies, loss, batch)
- Trying to do too much at once (full functional rewrite)

## Recommended Solution: 3-Phase Incremental Migration

### **Phase 1: Minimal JAX Integration** ✅ (CURRENT)

**File**: `gflownet/trainers/jax_minimal.py`

**Goal**: Get backpropagation working with JAX while keeping PyTorch for everything else.

**What Changes**:
- ✅ Gradient computation → JAX (JIT-compiled)
- ✅ Optimizer → Optax
- ✅ Loss computation → JAX wrapper
- ❌ Sampling → Still PyTorch
- ❌ Environment → Still PyTorch
- ❌ Buffer → Still PyTorch
- ❌ Policies (models) → Still PyTorch (params synced to/from JAX)

**Benefits**:
- **Minimal changes** to existing codebase
- **Working version at each step**
- **Immediate JAX speedup** on gradient computation
- Can still use all PyTorch infrastructure (logging, evaluation, checkpointing)

**How It Works**:
```python
# 1. Sample using PyTorch (unchanged)
batch, _ = agent.sample_batch(...)  # Uses PyTorch GFlowNetAgent

# 2. Convert batch to JAX arrays
batch_arrays = convert_batch_to_jax_arrays(batch)

# 3. JIT-compiled gradient step (PURE JAX)
new_params, new_opt_state, loss = jax_grad_step(
    jax_params, opt_state, batch_arrays, optimizer
)

# 4. Sync parameters back to PyTorch
apply_params_to_pytorch(agent, new_params)
```

**Implementation Details**:
1. **Parameter Sync**: 
   - Extract PyTorch params → JAX arrays before training
   - Update PyTorch params ← JAX arrays after each iteration
   - This allows PyTorch policies to be used for sampling

2. **Loss Wrapper**:
   - Simple JAX function wrapping loss computation
   - Takes JAX arrays as input (batch data)
   - Returns scalar loss
   - Start with trajectory balance, expand later

3. **Batch Conversion**:
   - Extract tensors from PyTorch Batch
   - Convert to JAX arrays (`.detach().cpu().numpy()`)
   - No complex pytree requirements

**Current Limitations**:
- Parameter sync overhead (small for most models)
- Loss implementation is simplified
- Not fully JIT-compiled (sampling still Python)

**Next Steps**:
1. ✅ Test with simple environment (grid)
2. Verify gradient correctness (compare with PyTorch)
3. Add proper loss implementations
4. Profile to measure speedup

---

### **Phase 2: JAX Policy Models** (FUTURE)

**Goal**: Convert policies to native JAX (Equinox), eliminate parameter sync.

**What Changes**:
- ✅ Forward policy → Equinox/Flax model
- ✅ Backward policy → Equinox/Flax model
- ✅ Policy forward pass → JAX
- ✅ Full backprop loop → JIT-compiled
- ❌ Sampling → Still PyTorch (but uses JAX policies)
- ❌ Environment → Still PyTorch
- ❌ Buffer → Still PyTorch

**Benefits**:
- No parameter sync overhead
- Faster policy evaluation
- Better JAX integration

**Implementation**:
1. Create `PolicyJAX` using Equinox (already exists in `base_jax.py`)
2. Update `parse_policy_config` to switch based on `trainer.mode`
3. Implement JAX-compatible policy forward pass
4. Update loss to use JAX policies directly

**Challenges**:
- Need to reimplement MLP architectures in Equinox
- Shared weights handling
- Checkpoint compatibility

---

### **Phase 3: Full JAX Environment** (FUTURE)

**Goal**: Full JAX training loop, fully JIT-compiled.

**What Changes**:
- ✅ Environment → JAX pytree
- ✅ Sampling → Pure JAX functions
- ✅ Trajectory generation → lax.scan/while_loop
- ✅ Full training loop → JIT-compiled
- ❌ Buffer → May stay Python (I/O bound)
- ❌ Logger → May stay Python (I/O bound)

**Benefits**:
- Maximum speedup
- GPU/TPU friendly
- Vectorized sampling

**Implementation**:
1. Define environment as JAX pytree
2. Implement pure `step()` function
3. Use `lax.scan` for trajectory sampling
4. Batch environment vectorization

**Challenges**:
- Major environment refactor
- Complex action masking in JAX
- Different envs have different state structures

---

## Why This Approach Works

### ✅ Incremental Progress
Each phase delivers a working system. You can stop at any phase.

### ✅ Risk Mitigation
Smaller changes = easier debugging. Can always revert to previous phase.

### ✅ Performance Gains at Each Step
- Phase 1: ~2-5x speedup on backprop
- Phase 2: ~5-10x speedup (no sync overhead)
- Phase 3: ~10-100x speedup (full JIT + vectorization)

### ✅ Learning Curve
Master JAX concepts incrementally:
- Phase 1: Basic JAX, value_and_grad, JIT
- Phase 2: Equinox models, pytrees
- Phase 3: lax control flow, scan, vectorization

---

## Common Pitfalls to Avoid

### ❌ Don't try to JIT everything at once
JAX's `jit` has strict requirements (pure functions, no side effects). Start small.

### ❌ Don't abandon PyTorch infrastructure prematurely
Logging, evaluation, checkpointing work fine in PyTorch. No need to convert.

### ❌ Don't worry about buffer/logger JAX compatibility
These are I/O-bound and inherently side-effectful. Keep them in Python.

### ❌ Don't mix tracing with Python control flow
Use `lax.cond`, `lax.scan`, `lax.while_loop` inside JIT-compiled functions.

---

## Testing Strategy

### Phase 1 Tests:
```bash
# 1. Verify training runs
python train.py env=grid trainer.mode=jax gflownet.optimizer.n_train_steps=10

# 2. Compare loss values with legacy
python train.py env=grid trainer.mode=legacy gflownet.optimizer.n_train_steps=10
python train.py env=grid trainer.mode=jax gflownet.optimizer.n_train_steps=10

# 3. Check gradient correctness
# (Implement gradient comparison test)

# 4. Profile performance
# (Measure time per iteration)
```

### Phase 2 Tests:
- Model output comparison (JAX vs PyTorch)
- Checkpoint save/load
- Shared weights verification

### Phase 3 Tests:
- Trajectory correctness
- Action masking
- Reward computation

---

## Decision Tree: When to Move to Next Phase

### Move to Phase 2 if:
- ✅ Phase 1 training converges correctly
- ✅ Loss values match PyTorch implementation
- ✅ No parameter sync bugs
- ✅ You need better performance

### Move to Phase 3 if:
- ✅ Phase 2 models work correctly
- ✅ Sampling is the bottleneck
- ✅ You need GPU/TPU scaling
- ✅ You're comfortable with advanced JAX

### Stay in Current Phase if:
- ❌ Current phase has bugs
- ❌ Performance is already acceptable
- ❌ Time constraints

---

## File Structure

```
gflownet/trainers/
├── __init__.py              # Trainer registry
├── legacy.py                # Original PyTorch trainer
├── pure.py                  # Refactored PyTorch trainer  
├── jax_minimal.py          # Phase 1: Minimal JAX ✅
├── jax_policy.py           # Phase 2: JAX policies (future)
└── jax_full.py             # Phase 3: Full JAX (future)
```

---

## Current Status

- ✅ Phase 1 implementation complete (`jax_minimal.py`)
- 🔄 Testing in progress
- ⏳ Phase 2/3 design documented

## Next Actions

1. **Fix any Phase 1 bugs** from testing
2. **Implement proper trajectory balance loss** in JAX
3. **Add gradient correctness tests**
4. **Measure performance vs PyTorch**
5. **Document findings** for Phase 2 planning
