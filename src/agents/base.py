"""Base agent interface for NeuroSwift experiments."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all NeuroSwift agents.

    Provides common interface for:
    - Action selection
    - Learning from experience
    - Weight access for VLM integration
    - Hooks for new feature detection
    """

    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_actions: Number of available actions.
            learning_rate: Learning rate (alpha).
            discount_factor: Discount factor (gamma).
            epsilon: Exploration rate for epsilon-greedy.
            seed: Random seed for reproducibility.
        """
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.rng = np.random.default_rng(seed)

        # Track metrics
        self.total_steps = 0
        self.total_updates = 0

        # Callback for new feature detection (VLM integration)
        self._on_new_feature_callback: Optional[callable] = None

    @abstractmethod
    def select_action(self, observation: Any) -> int:
        """
        Select an action given an observation.

        Args:
            observation: Current environment observation.

        Returns:
            Action index to take.
        """
        pass

    @abstractmethod
    def update(
        self,
        observation: Any,
        action: int,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> Dict[str, float]:
        """
        Update agent from a transition.

        Args:
            observation: State before action.
            action: Action taken.
            reward: Reward received.
            next_observation: State after action.
            done: Whether episode ended.

        Returns:
            Dict with update metrics (e.g., td_error, loss).
        """
        pass

    @abstractmethod
    def get_weights(self, feature_idx: int) -> np.ndarray:
        """
        Get the weight vector for a specific feature.

        Args:
            feature_idx: Index of the feature.

        Returns:
            Weight vector for the feature.
        """
        pass

    @abstractmethod
    def set_weights(self, feature_idx: int, values: np.ndarray) -> None:
        """
        Set the weight vector for a specific feature.

        Args:
            feature_idx: Index of the feature.
            values: New weight values.
        """
        pass

    def on_new_feature(self, feature_idx: int, observation: Any) -> None:
        """
        Hook called when a new feature is detected.

        This is the key integration point for VLM initialization.
        When a new feature is encountered, the VLM can be queried
        and the weights can be initialized based on semantic content.

        Args:
            feature_idx: Index of the newly detected feature.
            observation: The observation containing the new feature.
        """
        if self._on_new_feature_callback is not None:
            self._on_new_feature_callback(feature_idx, observation)

    def set_new_feature_callback(self, callback: callable) -> None:
        """
        Set callback for new feature detection.

        Args:
            callback: Function(feature_idx, observation) to call on new feature.
        """
        self._on_new_feature_callback = callback

    def get_value(self, observation: Any) -> float:
        """
        Get the estimated value of an observation.

        Args:
            observation: Environment observation.

        Returns:
            Estimated value.
        """
        # Default: max Q-value
        q_values = self.get_q_values(observation)
        return float(np.max(q_values))

    @abstractmethod
    def get_q_values(self, observation: Any) -> np.ndarray:
        """
        Get Q-values for all actions given an observation.

        Args:
            observation: Environment observation.

        Returns:
            Q-values for each action.
        """
        pass

    def reset(self) -> None:
        """Reset agent state for new episode (optional override)."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'total_steps': self.total_steps,
            'total_updates': self.total_updates,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing."""
        return {
            'total_steps': self.total_steps,
            'total_updates': self.total_updates,
            'rng_state': self.rng.bit_generator.state,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore agent state from checkpoint."""
        self.total_steps = state.get('total_steps', 0)
        self.total_updates = state.get('total_updates', 0)
        if 'rng_state' in state:
            self.rng.bit_generator.state = state['rng_state']
