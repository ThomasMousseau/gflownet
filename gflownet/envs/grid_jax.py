"""
JAX-native Grid Environment for GFlowNet

Pure functional implementation of the Grid environment using JAX primitives.
All operations are stateless, vectorized, and JIT-compilable for maximum performance.
"""

import itertools
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from jax import jit, vmap


@dataclass(frozen=True)
class GridConfig:
    """
    Immutable configuration for JAX Grid environment.
    
    Attributes
    ----------
    n_dim : int
        Dimensionality of the grid
    length : int
        Size of the grid (cells per dimension)
    max_increment : int
        Maximum increment of each dimension by actions
    max_dim_per_action : int
        Maximum number of dimensions to increment per action
    cell_min : float
        Lower bound of the cells range
    cell_max : float
        Upper bound of the cells range
    """
    n_dim: int
    length: int
    max_increment: int
    max_dim_per_action: int
    cell_min: float
    cell_max: float
    action_space: Tuple[Tuple[int, ...], ...] = None
    n_actions: int = None
    eos: Tuple[int, ...] = None
    source: Tuple[int, ...] = None
    cells: Tuple[float, ...] = None
    policy_input_dim: int = None
    state_dim: int = None
    max_traj_length: int = None
    
    def __post_init__(self):
        """Compute derived attributes after initialization."""
        # Use object.__setattr__ for frozen dataclass
        if self.action_space is None:
            action_space_list = self._compute_action_space()
            object.__setattr__(self, 'action_space', tuple(action_space_list))
            object.__setattr__(self, 'n_actions', len(self.action_space))
            object.__setattr__(self, 'eos', tuple([0 for _ in range(self.n_dim)]))
            object.__setattr__(self, 'source', tuple([0 for _ in range(self.n_dim)]))
            cells_array = np.linspace(self.cell_min, self.cell_max, self.length)
            object.__setattr__(self, 'cells', tuple(cells_array.tolist()))
            object.__setattr__(self, 'policy_input_dim', self.length * self.n_dim)
            object.__setattr__(self, 'state_dim', self.n_dim)
            object.__setattr__(self, 'max_traj_length', self.n_dim * self.length + 1)
    
    def _compute_action_space(self) -> List[Tuple[int, ...]]:
        """Constructs list with all possible actions, including EOS."""
        increments = [el for el in range(self.max_increment + 1)]
        actions = []
        for action in itertools.product(increments, repeat=self.n_dim):
            if (
                sum(action) != 0
                and len([el for el in action if el > 0]) <= self.max_dim_per_action
            ):
                actions.append(tuple(action))
        actions.append(tuple([0 for _ in range(self.n_dim)]))  # EOS
        return actions


@jax.tree_util.register_pytree_node_class
@dataclass
class BatchState:
    """
    Immutable batch state for vectorized Grid environment operations.
    
    Represents multiple environment states that can be processed in parallel.
    """
    positions: jnp.ndarray  # [batch_size, n_dim], dtype=jnp.int32
    done: jnp.ndarray       # [batch_size], dtype=bool
    n_actions: jnp.ndarray  # [batch_size], dtype=jnp.int32
    
    def tree_flatten(self):
        """Flatten for JAX pytree."""
        return ((self.positions, self.done, self.n_actions), None)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten for JAX pytree."""
        return cls(*children)


def create_initial_batch_state(batch_size: int, n_dim: int) -> BatchState:
    """
    Create initial batch state with all environments at source.
    
    Parameters
    ----------
    batch_size : int
        Number of parallel environments
    n_dim : int
        Dimensionality of the grid
    
    Returns
    -------
    BatchState
        Initial state with all positions at [0, 0, ..., 0]
    """
    return BatchState(
        positions=jnp.zeros((batch_size, n_dim), dtype=jnp.int32),
        done=jnp.zeros(batch_size, dtype=bool),
        n_actions=jnp.zeros(batch_size, dtype=jnp.int32)
    )


@partial(jit, static_argnames=['config'])
def states2policy_jax(states: jnp.ndarray, config: GridConfig) -> jnp.ndarray:
    """
    Convert states to one-hot encoded policy inputs (vectorized).
    
    The output is a 2D tensor where each row represents a state as a concatenation
    of one-hot encodings for each dimension.
    
    Example (n_dim=3, length=4, state=[0, 3, 1]):
      [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
       |     0    |      3    |      1    |
    
    Parameters
    ----------
    states : jnp.ndarray
        Batch of states, shape [batch_size, n_dim], dtype int32
    config : GridConfig
        Grid configuration
    
    Returns
    -------
    jnp.ndarray
        One-hot encoded states, shape [batch_size, length * n_dim], dtype float32
    """
    batch_size = states.shape[0]
    
    # Compute column indices: offset each dimension by length
    # states: [batch_size, n_dim]
    # offsets: [n_dim]
    offsets = jnp.arange(config.n_dim) * config.length
    cols = states + offsets[None, :]  # [batch_size, n_dim]
    
    # Compute row indices: repeat each batch index n_dim times
    rows = jnp.repeat(jnp.arange(batch_size), config.n_dim)  # [batch_size * n_dim]
    
    # Flatten column indices
    cols_flat = cols.flatten()  # [batch_size * n_dim]
    
    # Create one-hot encoded tensor
    states_policy = jnp.zeros((batch_size, config.policy_input_dim), dtype=jnp.float32)
    states_policy = states_policy.at[rows, cols_flat].set(1.0)
    
    return states_policy


@partial(jit, static_argnames=['config'])
def states2proxy_jax(states: jnp.ndarray, config: GridConfig) -> jnp.ndarray:
    """
    Convert states to continuous proxy format (vectorized).
    
    Maps discrete grid positions to continuous coordinates in [cell_min, cell_max].
    
    Parameters
    ----------
    states : jnp.ndarray
        Batch of states, shape [batch_size, n_dim], dtype int32
    config : GridConfig
        Grid configuration
    
    Returns
    -------
    jnp.ndarray
        Continuous coordinates, shape [batch_size, n_dim], dtype float32
    """
    # Convert to one-hot, reshape, and map to continuous values
    states_onehot = states2policy_jax(states, config)  # [batch_size, length * n_dim]
    states_onehot = states_onehot.reshape((states.shape[0], config.n_dim, config.length))
    
    # Map to continuous space using cells
    cells = jnp.array(config.cells, dtype=jnp.float32)  # [length]
    states_proxy = jnp.sum(states_onehot * cells[None, None, :], axis=2)  # [batch_size, n_dim]
    
    return states_proxy


@partial(jit, static_argnames=['config'])
def get_mask_invalid_actions_forward_batch(
    states: jnp.ndarray, 
    done: jnp.ndarray, 
    config: GridConfig
) -> jnp.ndarray:
    """
    Compute masks of invalid forward actions for a batch of states (vectorized).
    
    Parameters
    ----------
    states : jnp.ndarray
        Batch of states, shape [batch_size, n_dim]
    done : jnp.ndarray
        Done flags, shape [batch_size]
    config : GridConfig
        Grid configuration
    
    Returns
    -------
    jnp.ndarray
        Boolean mask where True indicates invalid action
        Shape: [batch_size, n_actions]
    """
    batch_size = states.shape[0]
    
    # Convert action space to array (exclude EOS for now)
    actions_array = jnp.array(config.action_space[:-1], dtype=jnp.int32)  # [n_actions-1, n_dim]
    
    # Broadcast and compute next states
    # states: [batch_size, 1, n_dim]
    # actions_array: [1, n_actions-1, n_dim]
    # children: [batch_size, n_actions-1, n_dim]
    children = states[:, None, :] + actions_array[None, :, :]
    
    # Check if any dimension exceeds boundary
    invalid = jnp.any(children >= config.length, axis=-1)  # [batch_size, n_actions-1]
    
    # EOS is always valid (unless done), so add False column
    eos_mask = jnp.zeros((batch_size, 1), dtype=bool)
    mask = jnp.concatenate([invalid, eos_mask], axis=-1)  # [batch_size, n_actions]
    
    # If done, all actions are invalid
    mask = jnp.where(done[:, None], True, mask)
    
    return mask


@partial(jit, static_argnames=['config'])
def get_mask_invalid_actions_backward_batch(
    states: jnp.ndarray,
    done: jnp.ndarray,
    config: GridConfig
) -> jnp.ndarray:
    """
    Compute masks of invalid backward actions for a batch of states (vectorized).
    
    Parameters
    ----------
    states : jnp.ndarray
        Batch of states, shape [batch_size, n_dim]
    done : jnp.ndarray
        Done flags, shape [batch_size]
    config : GridConfig
        Grid configuration
    
    Returns
    -------
    jnp.ndarray
        Boolean mask where True indicates invalid action
        Shape: [batch_size, n_actions]
    """
    batch_size = states.shape[0]
    
    # Convert action space to array (exclude EOS)
    actions_array = jnp.array(config.action_space[:-1], dtype=jnp.int32)
    
    # Compute parent states (reverse the action)
    parents = states[:, None, :] - actions_array[None, :, :]
    
    # Check if any dimension goes negative
    invalid = jnp.any(parents < 0, axis=-1)  # [batch_size, n_actions-1]
    
    # EOS handling: if done, only EOS is valid; if not done, EOS is invalid
    eos_mask = jnp.where(done, False, True)[:, None]  # [batch_size, 1]
    mask = jnp.concatenate([invalid, eos_mask], axis=-1)
    
    return mask


@partial(jit, static_argnames=['config'])
def step_batch(
    states: jnp.ndarray,
    actions: jnp.ndarray,
    done: jnp.ndarray,
    n_actions: jnp.ndarray,
    config: GridConfig
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Execute actions on a batch of states (vectorized, pure functional).
    
    Parameters
    ----------
    states : jnp.ndarray
        Current states, shape [batch_size, n_dim]
    actions : jnp.ndarray
        Actions to execute, shape [batch_size, n_dim] (as increments)
    done : jnp.ndarray
        Done flags, shape [batch_size]
    n_actions : jnp.ndarray
        Action counts, shape [batch_size]
    config : GridConfig
        Grid configuration
    
    Returns
    -------
    new_states : jnp.ndarray
        Updated states, shape [batch_size, n_dim]
    new_done : jnp.ndarray
        Updated done flags, shape [batch_size]
    new_n_actions : jnp.ndarray
        Updated action counts, shape [batch_size]
    valid : jnp.ndarray
        Validity of each action, shape [batch_size]
    """
    # Compute next states
    states_next = states + actions
    
    # Check if EOS action (all zeros)
    is_eos = jnp.all(actions == 0, axis=-1)  # [batch_size]
    
    # Check if at maximum (forced EOS)
    at_max = jnp.all(states == config.length - 1, axis=-1)  # [batch_size]
    
    # Check validity: all dimensions must be in [0, length)
    in_bounds = jnp.all((states_next >= 0) & (states_next < config.length), axis=-1)
    
    # Action is valid if:
    # - Not already done AND
    # - (Is EOS OR (In bounds AND not forced EOS))
    valid = ~done & (is_eos | (in_bounds & ~at_max))
    
    # Update done: if EOS or forced EOS or already done
    new_done = done | is_eos | (at_max & ~done)
    
    # Update states: only if valid and not EOS
    should_update = valid & ~is_eos & ~done
    new_states = jnp.where(should_update[:, None], states_next, states)
    
    # Increment action count if valid
    new_n_actions = jnp.where(valid, n_actions + 1, n_actions)
    
    return new_states, new_done, new_n_actions, valid


@partial(jit, static_argnames=['config'])
def get_parents_batch(
    states: jnp.ndarray,
    done: jnp.ndarray,
    config: GridConfig
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Get all valid parents for a batch of states (vectorized).
    
    For each state, computes all possible parent states by reversing each action.
    
    Parameters
    ----------
    states : jnp.ndarray
        Batch of states, shape [batch_size, n_dim]
    done : jnp.ndarray
        Done flags, shape [batch_size]
    config : GridConfig
        Grid configuration
    
    Returns
    -------
    parents : jnp.ndarray
        All potential parent states, shape [batch_size, n_actions-1, n_dim]
    actions : jnp.ndarray
        Corresponding actions, shape [batch_size, n_actions-1, n_dim]
    valid : jnp.ndarray
        Validity mask for each parent, shape [batch_size, n_actions-1]
    """
    batch_size = states.shape[0]
    
    # Convert action space to array (exclude EOS)
    actions_array = jnp.array(config.action_space[:-1], dtype=jnp.int32)
    n_actions_excl_eos = len(config.action_space) - 1
    
    # Compute all potential parents by reversing actions
    # states: [batch_size, 1, n_dim]
    # actions_array: [1, n_actions-1, n_dim]
    parents = states[:, None, :] - actions_array[None, :, :]  # [batch_size, n_actions-1, n_dim]
    
    # Check validity: all dimensions must be >= 0
    valid = jnp.all(parents >= 0, axis=-1)  # [batch_size, n_actions-1]
    
    # If done, the only parent is itself with EOS action
    # Set all parents invalid if done
    valid = jnp.where(done[:, None], False, valid)
    
    # Broadcast actions to match parents shape
    actions = jnp.broadcast_to(actions_array[None, :, :], parents.shape)
    
    return parents, actions, valid


class GridJAX:
    """
    JAX-native Grid Environment for GFlowNet.
    
    This is a stateless wrapper that provides a similar interface to the PyTorch
    Grid environment but uses pure JAX functions internally.
    
    For actual trajectory sampling, use the functional API (sample_trajectories_jax).
    This class is primarily for configuration and integration with existing code.
    """
    
    def __init__(
        self,
        n_dim: int = 2,
        length: int = 3,
        max_increment: int = 1,
        max_dim_per_action: int = 1,
        cell_min: float = -1.0,
        cell_max: float = 1.0,
        device: str = "cpu",
        float_precision: int = 32,
        **kwargs
    ):
        """
        Initialize JAX Grid environment.
        
        Parameters
        ----------
        n_dim : int
            Dimensionality of the grid
        length : int
            Size of the grid (cells per dimension)
        max_increment : int
            Maximum increment of each dimension by actions
        max_dim_per_action : int
            Maximum number of dimensions to increment per action. If -1, set to n_dim
        cell_min : float
            Lower bound of the cells range
        cell_max : float
            Upper bound of the cells range
        device : str
            Ignored (JAX manages devices internally)
        float_precision : int
            Float precision (32 or 64)
        """
        if max_dim_per_action == -1:
            max_dim_per_action = n_dim
        
        self.config = GridConfig(
            n_dim=n_dim,
            length=length,
            max_increment=max_increment,
            max_dim_per_action=max_dim_per_action,
            cell_min=cell_min,
            cell_max=cell_max
        )
        
        # Expose config attributes for compatibility
        self.n_dim = n_dim
        self.length = length
        self.max_increment = max_increment
        self.max_dim_per_action = max_dim_per_action
        self.cell_min = cell_min
        self.cell_max = cell_max
        self.cells = np.array(self.config.cells)  # Convert tuple back to array for compatibility
        self.action_space = list(self.config.action_space)  # Convert tuple back to list for compatibility
        self.n_actions = self.config.n_actions
        self.eos = self.config.eos
        self.source = self.config.source
        self.policy_output_dim = self.config.n_actions
        self.policy_input_dim = self.config.policy_input_dim
        self.state_dim = self.config.state_dim
        
        # JAX-specific attributes
        self.is_jax_env = True
        self.device = device
        self.float = jnp.float32 if float_precision == 32 else jnp.float64
        
        # For compatibility with existing code
        self.state = [0] * n_dim
        self.done = False
        self.n_actions_taken = 0
        self.conditional = kwargs.get('conditional', False)
        self.continuous = False
        self.skip_mask_check = kwargs.get('skip_mask_check', False)
        self.id = kwargs.get('env_id', 'env')  # Environment ID for batch tracking
        
        # Policy outputs (uniform distribution over actions)
        self.fixed_distr_params = kwargs.get('fixed_distr_params', None)
        self.random_distr_params = kwargs.get('random_distr_params', None)
        self.fixed_policy_output = jnp.ones(self.n_actions, dtype=self.float)
        self.random_policy_output = jnp.ones(self.n_actions, dtype=self.float)
        self.random_policy_output = jnp.ones(self.n_actions, dtype=self.float)
        
        # Create action lookup for index conversion
        self._action_to_index = {action: idx for idx, action in enumerate(self.action_space)}
    
    def action2index(self, action):
        """Convert action tuple to index."""
        return self._action_to_index[tuple(action)]
    
    def index2action(self, index):
        """Convert action index to tuple."""
        return self.action_space[index]
    
    def states2policy(self, states):
        """Convert states to policy format (wrapper for JAX function)."""
        if isinstance(states, list):
            states = jnp.array(states, dtype=jnp.int32)
        elif not isinstance(states, jnp.ndarray):
            states = jnp.array(states, dtype=jnp.int32)
        
        if states.ndim == 1:
            states = states[None, :]
        
        # Convert to numpy for compatibility with PyTorch
        return np.array(states2policy_jax(states, self.config))
    
    def states2proxy(self, states):
        """Convert states to proxy format (wrapper for JAX function)."""
        if isinstance(states, list):
            states = jnp.array(states, dtype=jnp.int32)
        elif not isinstance(states, jnp.ndarray):
            states = jnp.array(states, dtype=jnp.int32)
            
        if states.ndim == 1:
            states = states[None, :]
            
        # Convert to numpy for compatibility with PyTorch
        return np.array(states2proxy_jax(states, self.config))

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        """Get forward mask for single state (wrapper for JAX function)."""
        if state is None:
            state = self.state
        if done is None:
            done = self.done
        
        state_arr = jnp.array([state], dtype=jnp.int32)
        done_arr = jnp.array([done], dtype=bool)
        
        mask = get_mask_invalid_actions_forward_batch(state_arr, done_arr, self.config)
        return mask[0].tolist()
    
    def get_mask_invalid_actions_backward(self, state=None, done=None):
        """Get backward mask for single state (wrapper for JAX function)."""
        if state is None:
            state = self.state
        if done is None:
            done = self.done
        
        state_arr = jnp.array([state], dtype=jnp.int32)
        done_arr = jnp.array([done], dtype=bool)
        
        mask = get_mask_invalid_actions_backward_batch(state_arr, done_arr, self.config)
        return mask[0].tolist()
    
    def reset(self, env_id=None):
        """Reset environment to source state."""
        self.state = list(self.source)
        self.done = False
        self.n_actions_taken = 0
        return self
    
    def set_state(self, state, done=False):
        """Set environment to specific state."""
        self.state = list(state)
        self.done = done
        return self

    def get_all_terminating_states(self):
        """
        Returns all possible terminating states (all grid positions).
        
        For a grid of dimensions n_dim and length L, this returns all L^n_dim
        possible states as a list of lists.
        
        Returns
        -------
        List[List[int]]
            All possible terminating states on the grid.
        """
        import itertools
        all_states = list(itertools.product(range(self.length), repeat=self.n_dim))
        return [list(s) for s in all_states]

    def plot_reward_samples(
        self,
        samples,
        samples_reward,
        rewards,
        dpi: int = 150,
        n_ticks_max: int = 50,
        reward_norm: bool = True,
        **kwargs,
    ):
        """
        Plots the reward density as a 2D histogram on the grid, alongside a histogram
        representing the samples density.

        It is assumed that the rewards correspond to entire domain of the grid and are
        sorted from left to right (first) and top to bottom of the grid of samples.

        Parameters
        ----------
        samples : array-like
            A batch of samples from the GFlowNet policy in proxy format. These samples
            will be plotted on top of the reward density.
        samples_reward : array-like
            A batch of samples containing a grid over the sample space, from which the
            reward has been obtained. Ignored by this method.
        rewards : array-like
            The rewards of samples_reward. It should be a vector of dimensionality
            length ** 2 and be sorted such that the each block at rewards[i *
            length:i * length + length] correspond to the rewards at the i-th
            row of the grid of samples, from top to bottom.
        dpi : int
            Dots per inch, indicating the resolution of the plot.
        n_ticks_max : int
            Maximum of number of ticks to include in the axes.
        reward_norm : bool
            Whether to normalize the histogram. True by default.
        """
        # Only available for 2D grids
        if self.n_dim != 2:
            return None
        
        # Convert to numpy if needed
        if isinstance(samples, jnp.ndarray):
            samples = np.array(samples)
        elif hasattr(samples, 'numpy'):  # torch tensor
            samples = samples.numpy()
        else:
            samples = np.array(samples)
            
        if isinstance(rewards, jnp.ndarray):
            rewards = np.array(rewards)
        elif hasattr(rewards, 'numpy'):  # torch tensor
            rewards = rewards.numpy()
        else:
            rewards = np.array(rewards)
            
        assert rewards.shape[0] == self.length**2
        
        # Init figure
        fig, axes = plt.subplots(ncols=2, dpi=dpi)
        step_ticks = np.ceil(self.length / n_ticks_max).astype(int)
        
        # 2D histogram of samples
        samples_hist, xedges, yedges = np.histogram2d(
            samples[:, 0], samples[:, 1], bins=(self.length, self.length), density=True
        )
        # Transpose and reverse rows so that [0, 0] is at bottom left
        samples_hist = samples_hist.T[::-1, :]
        
        # Normalize and reshape reward into a grid with [0, 0] at the bottom left
        if reward_norm:
            rewards = rewards / rewards.sum()
        rewards_2d = rewards.reshape(self.length, self.length).T[::-1, :]
        
        # Plot reward
        self._plot_grid_2d(rewards_2d, axes[0], step_ticks, title="True reward")
        # Plot samples histogram
        self._plot_grid_2d(samples_hist, axes[1], step_ticks, title="Samples density")
        fig.tight_layout()
        return fig

    @staticmethod
    def _plot_grid_2d(img: np.ndarray, ax: Axes, step_ticks: int, title: str):
        """
        Plots a 2D histogram of a grid environment as an image.

        Parameters
        ----------
        img : np.ndarray
            An array containing a 2D histogram over a grid.
        ax : Axes
            A matplotlib Axes object on which the image will be plotted.
        step_ticks : int
            The step value to add ticks to the axes. For example, if it is 2, the ticks
            will be at 0, 2, 4, ...
        title : str
            Title for the axes.
        """
        ax_img = ax.imshow(img)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        ax.set_xticks(np.arange(start=0, stop=img.shape[0], step=step_ticks))
        ax.set_yticks(np.arange(start=0, stop=img.shape[1], step=step_ticks)[::-1])
        cax.set_title(title)
        plt.colorbar(ax_img, cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position("top")


# ==============================================================================
# Trajectory Sampling (Pure JAX)
# ==============================================================================

@jax.tree_util.register_pytree_node_class
@dataclass
class TrajectoryBatch:
    """
    Container for a batch of trajectories sampled from the Grid environment.
    
    All arrays are padded to max_length for JIT compatibility.
    Use actual_lengths to mask padding.
    """
    states: jnp.ndarray          # [batch_size * max_len, n_dim]
    parents: jnp.ndarray         # [batch_size * max_len, n_dim]
    actions: jnp.ndarray         # [batch_size * max_len, n_dim]
    action_indices: jnp.ndarray  # [batch_size * max_len]
    masks_forward: jnp.ndarray   # [batch_size * max_len, n_actions]
    masks_backward: jnp.ndarray  # [batch_size * max_len, n_actions]
    trajectory_indices: jnp.ndarray  # [batch_size * max_len]
    actual_lengths: jnp.ndarray  # [batch_size]
    actual_n_states: int
    actual_n_trajs: int
    
    def tree_flatten(self):
        """Flatten to (children, aux_data) for JAX pytree."""
        children = (
            self.states,
            self.parents,
            self.actions,
            self.action_indices,
            self.masks_forward,
            self.masks_backward,
            self.trajectory_indices,
            self.actual_lengths,
            self.actual_n_states,  # Moved to children because it's dynamic in JIT
        )
        aux_data = (self.actual_n_trajs,)
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten from (children, aux_data) to reconstruct TrajectoryBatch."""
        (actual_n_trajs,) = aux_data
        return cls(
            states=children[0],
            parents=children[1],
            actions=children[2],
            action_indices=children[3],
            masks_forward=children[4],
            masks_backward=children[5],
            trajectory_indices=children[6],
            actual_lengths=children[7],
            actual_n_states=children[8],
            actual_n_trajs=actual_n_trajs,
        )


def sample_trajectories_jax(
    key: jax.Array,
    config: GridConfig,
    policy_model,
    n_trajectories: int,
    temperature: float = 1.0,
    return_logprobs: bool = True
) -> Tuple[TrajectoryBatch, jax.Array]:
    """
    Sample a batch of trajectories from the Grid environment using JAX.
    
    This is a pure functional implementation that can be JIT-compiled.
    
    Parameters
    ----------
    key : jax.Array
        JAX PRNGKey for random sampling
    config : GridConfig
        Grid environment configuration
    policy_model : eqx.Module
        JAX policy model (forward function)
    n_trajectories : int
        Number of trajectories to sample
    temperature : float
        Temperature for sampling (default 1.0)
    return_logprobs : bool
        Whether to compute and return log probabilities
    
    Returns
    -------
    batch : TrajectoryBatch
        Batch of sampled trajectories
    key : jax.Array
        Updated PRNGKey
    """
    max_len = config.max_traj_length
    
    # Initialize state
    state = create_initial_batch_state(n_trajectories, config.n_dim)
    
    # Preallocate trajectory storage
    all_states = jnp.zeros((n_trajectories, max_len, config.n_dim), dtype=jnp.int32)
    all_parents = jnp.zeros((n_trajectories, max_len, config.n_dim), dtype=jnp.int32)
    all_actions = jnp.zeros((n_trajectories, max_len, config.n_dim), dtype=jnp.int32)
    all_action_indices = jnp.zeros((n_trajectories, max_len), dtype=jnp.int32)
    all_masks_forward = jnp.zeros((n_trajectories, max_len, config.n_actions), dtype=bool)
    all_masks_backward = jnp.zeros((n_trajectories, max_len, config.n_actions), dtype=bool)
    trajectory_lengths = jnp.zeros(n_trajectories, dtype=jnp.int32)
    
    if return_logprobs:
        all_logprobs = jnp.zeros((n_trajectories, max_len), dtype=jnp.float32)
    
    def step_fn(carry, step_idx):
        """Single step of trajectory sampling."""
        state, all_states, all_parents, all_actions, all_action_indices, \
            all_masks_forward, all_masks_backward, trajectory_lengths, key_local = carry
        
        # Store parent state
        all_parents = all_parents.at[:, step_idx, :].set(state.positions)
        
        # Get masks
        masks_forward = get_mask_invalid_actions_forward_batch(
            state.positions, state.done, config
        )
        
        all_masks_forward = all_masks_forward.at[:, step_idx, :].set(masks_forward)
        
        # Get policy inputs
        states_policy = states2policy_jax(state.positions, config)
        
        # Get logits from policy
        logits = vmap(policy_model)(states_policy)  # [batch_size, n_actions]
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply masks
        logits_masked = jnp.where(masks_forward, -jnp.inf, logits)
        
        # Sample actions
        key_local, *subkeys = jax.random.split(key_local, n_trajectories + 1)
        subkeys = jnp.array(subkeys)
        action_indices = vmap(jax.random.categorical)(subkeys, logits_masked)
        
        # Convert action indices to action tuples
        actions_array = jnp.array(config.action_space, dtype=jnp.int32)
        actions = actions_array[action_indices]
        
        # Store actions
        all_action_indices = all_action_indices.at[:, step_idx].set(action_indices)
        all_actions = all_actions.at[:, step_idx, :].set(actions)
        
        # Execute step
        new_positions, new_done, new_n_actions, valid = step_batch(
            state.positions, actions, state.done, state.n_actions, config
        )
        
        # Store state (after action)
        all_states = all_states.at[:, step_idx, :].set(new_positions)
        
        # Compute backward mask for NEW state (s_{t+1})
        masks_backward = get_mask_invalid_actions_backward_batch(
            new_positions, new_done, config
        )
        all_masks_backward = all_masks_backward.at[:, step_idx, :].set(masks_backward)
        
        # Update trajectory lengths (only for non-done trajectories)
        trajectory_lengths = jnp.where(~state.done, trajectory_lengths + 1, trajectory_lengths)
        
        # Update state
        new_state = BatchState(new_positions, new_done, new_n_actions)
        
        new_carry = (
            new_state, all_states, all_parents, all_actions, all_action_indices,
            all_masks_forward, all_masks_backward, trajectory_lengths, key_local
        )
        
        return new_carry, None
    
    # Run trajectory sampling loop
    initial_carry = (
        state, all_states, all_parents, all_actions, all_action_indices,
        all_masks_forward, all_masks_backward, trajectory_lengths, key
    )
    
    # Use scan for efficiency
    final_carry, _ = jax.lax.scan(step_fn, initial_carry, jnp.arange(max_len))
    
    state, all_states, all_parents, all_actions, all_action_indices, \
        all_masks_forward, all_masks_backward, trajectory_lengths, key = final_carry
    
    # Flatten trajectory data: [batch_size, max_len, ...] -> [batch_size * max_len, ...]
    total_states = n_trajectories * max_len
    
    states_flat = all_states.reshape(total_states, config.n_dim)
    parents_flat = all_parents.reshape(total_states, config.n_dim)
    actions_flat = all_actions.reshape(total_states, config.n_dim)
    action_indices_flat = all_action_indices.reshape(total_states)
    masks_forward_flat = all_masks_forward.reshape(total_states, config.n_actions)
    masks_backward_flat = all_masks_backward.reshape(total_states, config.n_actions)
    
    # Create trajectory indices
    traj_indices = jnp.repeat(jnp.arange(n_trajectories), max_len)
    
    # Compute actual number of states (sum of trajectory lengths)
    actual_n_states = jnp.sum(trajectory_lengths)
    
    batch = TrajectoryBatch(
        states=states_flat,
        parents=parents_flat,
        actions=actions_flat,
        action_indices=action_indices_flat,
        masks_forward=masks_forward_flat,
        masks_backward=masks_backward_flat,
        trajectory_indices=traj_indices,
        actual_lengths=trajectory_lengths,
        actual_n_states=actual_n_states,
        actual_n_trajs=n_trajectories
    )
    
    return batch, key
