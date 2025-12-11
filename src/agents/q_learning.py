"""Tabular Q-Learning agent implementation."""
from typing import Any, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict

from .base import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning agent.

    Uses a dictionary to store Q-values for state-action pairs.
    States are converted to hashable tuples for lookup.
    """

    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_actions: Number of available actions.
            learning_rate: Learning rate (alpha).
            discount_factor: Discount factor (gamma).
            epsilon: Initial exploration rate.
            epsilon_decay: Decay rate for epsilon after each episode.
            epsilon_min: Minimum epsilon value.
            seed: Random seed.
        """
        super().__init__(
            n_actions=n_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            seed=seed,
        )
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: maps state -> action values
        self.q_table: Dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

        # Track seen states for feature detection
        self._seen_states: set = set()

    def _state_to_key(self, observation: Any) -> tuple:
        """Convert observation to hashable key."""
        if isinstance(observation, dict):
            # For TextureGrid observations, use direction as simple state
            # In practice, we'd use a more sophisticated state representation
            if 'direction' in observation:
                # Simplified: use agent direction
                return (observation['direction'],)
            return (0,)  # Fallback
        elif isinstance(observation, np.ndarray):
            # Flatten and discretize
            return tuple(observation.flatten().astype(int))
        else:
            return (hash(str(observation)),)

    def select_action(self, observation: Any) -> int:
        """Select action using epsilon-greedy policy with random tie-breaking."""
        self.total_steps += 1

        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)

        state_key = self._state_to_key(observation)
        q_values = self.q_table[state_key]
        # Break ties randomly to avoid always selecting action 0
        max_q = np.max(q_values)
        max_actions = np.where(q_values == max_q)[0]
        return int(self.rng.choice(max_actions))

    def update(
        self,
        observation: Any,
        action: int,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> Dict[str, float]:
        """Update Q-values using Q-learning update rule."""
        self.total_updates += 1

        state_key = self._state_to_key(observation)
        next_state_key = self._state_to_key(next_observation)

        # Check for new state (feature detection)
        if state_key not in self._seen_states:
            self._seen_states.add(state_key)
            # Notify about new feature
            self.on_new_feature(hash(state_key), observation)

        # Q-learning update: Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        current_q = self.q_table[state_key][action]

        if done:
            target = reward
        else:
            next_max_q = np.max(self.q_table[next_state_key])
            target = reward + self.discount_factor * next_max_q

        td_error = target - current_q
        self.q_table[state_key][action] += self.learning_rate * td_error

        return {'td_error': td_error, 'q_value': current_q}

    def get_weights(self, feature_idx: int) -> np.ndarray:
        """Get Q-values for a feature (state)."""
        # For tabular Q-learning, we can't directly map feature_idx to state
        # This is a simplified implementation
        for state_key, q_values in self.q_table.items():
            if hash(state_key) == feature_idx:
                return q_values.copy()
        return np.zeros(self.n_actions)

    def set_weights(self, feature_idx: int, values: np.ndarray) -> None:
        """Set Q-values for a feature (state)."""
        # For tabular Q-learning, this sets all actions to the same initial value
        for state_key in self.q_table:
            if hash(state_key) == feature_idx:
                self.q_table[state_key] = values.copy()
                return

    def get_q_values(self, observation: Any) -> np.ndarray:
        """Get Q-values for all actions."""
        state_key = self._state_to_key(observation)
        return self.q_table[state_key].copy()

    def end_episode(self) -> None:
        """Called at end of episode. Decay epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = super().get_stats()
        stats['n_states'] = len(self.q_table)
        return stats

    def get_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing."""
        state = super().get_state()
        state['q_table'] = {str(k): v.tolist() for k, v in self.q_table.items()}
        state['seen_states'] = [str(s) for s in self._seen_states]
        state['epsilon'] = self.epsilon
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore agent state from checkpoint."""
        super().set_state(state)
        if 'q_table' in state:
            self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
            for k, v in state['q_table'].items():
                self.q_table[eval(k)] = np.array(v)
        if 'seen_states' in state:
            self._seen_states = {eval(s) for s in state['seen_states']}
        if 'epsilon' in state:
            self.epsilon = state['epsilon']


class TileCodingQLearning(BaseAgent):
    """
    Q-Learning agent with tile coding for function approximation.

    This is the MVP approach for the NeuroSwift experiments:
    - Use tile coding to extract features from observations
    - New tile activations trigger the on_new_feature hook
    - Weights can be initialized based on VLM sentiment
    """

    def __init__(
        self,
        n_actions: int,
        n_tiles: int = 8,
        n_tilings: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_actions: Number of available actions.
            n_tiles: Number of tiles per dimension per tiling.
            n_tilings: Number of overlapping tilings.
            learning_rate: Learning rate (divided by n_tilings).
            discount_factor: Discount factor.
            epsilon: Exploration rate.
            seed: Random seed.
        """
        # Adjust learning rate for tile coding
        adjusted_lr = learning_rate / n_tilings

        super().__init__(
            n_actions=n_actions,
            learning_rate=adjusted_lr,
            discount_factor=discount_factor,
            epsilon=epsilon,
            seed=seed,
        )

        self.n_tiles = n_tiles
        self.n_tilings = n_tilings

        # Weight vector for each action
        # Total features = n_tilings * n_tiles^2 (for 2D state)
        self.n_features = n_tilings * n_tiles * n_tiles
        self.weights = np.zeros((self.n_features, n_actions))

        # Track which features have been seen
        self._seen_features: set = set()

        # Tiling offsets for each tiling
        self.offsets = [
            (i / n_tilings, i / n_tilings) for i in range(n_tilings)
        ]

    def _get_tile_indices(self, x: float, y: float) -> list:
        """Get active tile indices for a 2D position."""
        indices = []
        for tiling_idx, (ox, oy) in enumerate(self.offsets):
            # Offset and discretize
            tx = int((x + ox) * self.n_tiles) % self.n_tiles
            ty = int((y + oy) * self.n_tiles) % self.n_tiles

            # Compute feature index
            feature_idx = (
                tiling_idx * self.n_tiles * self.n_tiles +
                ty * self.n_tiles +
                tx
            )
            indices.append(feature_idx)

        return indices

    def _extract_position(self, observation: Any) -> Tuple[float, float]:
        """Extract normalized position from observation."""
        if isinstance(observation, dict):
            # For TextureGrid, we might have position info
            if 'image' in observation:
                # Simplified: use image mean as proxy for position
                img = observation['image']
                # Normalize position based on agent view
                x = np.mean(img[:, :, 0]) / 255.0
                y = np.mean(img[:, :, 1]) / 255.0
                return (x, y)
        return (0.5, 0.5)

    def _get_features(self, observation: Any) -> list:
        """Get active feature indices for an observation."""
        x, y = self._extract_position(observation)
        return self._get_tile_indices(x, y)

    def select_action(self, observation: Any) -> int:
        """Select action using epsilon-greedy."""
        self.total_steps += 1

        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)

        q_values = self.get_q_values(observation)
        return int(np.argmax(q_values))

    def update(
        self,
        observation: Any,
        action: int,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> Dict[str, float]:
        """Update weights using semi-gradient TD."""
        self.total_updates += 1

        features = self._get_features(observation)
        next_features = self._get_features(next_observation)

        # Check for new features
        for f_idx in features:
            if f_idx not in self._seen_features:
                self._seen_features.add(f_idx)
                self.on_new_feature(f_idx, observation)

        # Compute Q-values
        current_q = sum(self.weights[f, action] for f in features)

        if done:
            target = reward
        else:
            next_q_values = np.array([
                sum(self.weights[f, a] for f in next_features)
                for a in range(self.n_actions)
            ])
            target = reward + self.discount_factor * np.max(next_q_values)

        # TD error
        td_error = target - current_q

        # Update weights for active features
        for f in features:
            self.weights[f, action] += self.learning_rate * td_error

        return {'td_error': td_error, 'q_value': current_q}

    def get_weights(self, feature_idx: int) -> np.ndarray:
        """Get weight vector for a feature."""
        if 0 <= feature_idx < self.n_features:
            return self.weights[feature_idx].copy()
        return np.zeros(self.n_actions)

    def set_weights(self, feature_idx: int, values: np.ndarray) -> None:
        """Set weight vector for a feature."""
        if 0 <= feature_idx < self.n_features:
            self.weights[feature_idx] = values.copy()

    def initialize_feature_weight(self, feature_idx: int, value: float) -> None:
        """
        Initialize all action weights for a feature to a single value.

        This is the key method for VLM-based weight initialization:
        weight = VLM_sentiment * alpha_prior

        Args:
            feature_idx: Feature to initialize.
            value: Initial weight value for all actions.
        """
        if 0 <= feature_idx < self.n_features:
            self.weights[feature_idx, :] = value

    def get_q_values(self, observation: Any) -> np.ndarray:
        """Get Q-values for all actions."""
        features = self._get_features(observation)
        q_values = np.zeros(self.n_actions)
        for f in features:
            q_values += self.weights[f]
        return q_values

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = super().get_stats()
        stats['n_features'] = self.n_features
        stats['n_seen_features'] = len(self._seen_features)
        stats['weight_norm'] = float(np.linalg.norm(self.weights))
        return stats
