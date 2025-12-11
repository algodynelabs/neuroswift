"""SwiftTD Agent Implementation.

Based on SwiftTD from Javed's reference implementation:
https://github.com/kjaved0/swifttd

Key features:
- Adaptive step sizes per feature (beta vector with meta-step-size)
- Multiple eligibility traces (z, z_bar for IDBD-style adaptation)
- Overshoot bounding based on rate of learning
- Trace pruning for efficiency
- VLM-based weight initialization for new features
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .base import BaseAgent


class SwiftTDAgent(BaseAgent):
    """
    SwiftTD agent for fast real-time learning.

    Based on Javed's implementation with:
    - True Online TD(λ) with dutch traces
    - Adaptive step sizes (IDBD-style)
    - Overshoot bound based on rate of learning
    - New feature hook for VLM initialization
    """

    def __init__(
        self,
        n_actions: int,
        n_features: int = 512,
        learning_rate: float = 0.1,  # Initial alpha
        discount_factor: float = 0.99,  # gamma
        lambda_: float = 0.9,
        epsilon: float = 0.1,  # Exploration rate
        eta: float = 1.0,  # Maximum step size bound (overshoot)
        eta_min: float = 1e-6,  # Minimum step size
        meta_step_size: float = 0.01,  # For step size adaptation
        trace_threshold: float = 1e-4,  # For trace pruning
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_actions: Number of available actions.
            n_features: Number of features in the representation.
            learning_rate: Initial step size (alpha).
            discount_factor: Discount factor (gamma).
            lambda_: Eligibility trace decay parameter.
            epsilon: Exploration rate.
            eta: Maximum step size (overshoot bound).
            eta_min: Minimum step size.
            meta_step_size: Step size for adapting alpha (0 to disable).
            trace_threshold: Threshold for pruning traces.
            seed: Random seed.
        """
        super().__init__(
            n_actions=n_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            seed=seed,
        )

        self.n_features = n_features
        self.lambda_ = lambda_
        self.eta = eta
        self.eta_min = eta_min
        self.meta_step_size = meta_step_size
        self.trace_threshold = trace_threshold

        # For backward compatibility
        self.overshoot_bound = eta

        # Weight vector for each action (linear function approximation)
        self.weights = np.zeros((n_features, n_actions))

        # Adaptive step sizes (log scale for numerical stability)
        # beta[i] = log(alpha[i]), so alpha[i] = exp(beta[i])
        self.beta = np.full((n_features, n_actions), np.log(learning_rate))

        # Eligibility traces
        self.traces = np.zeros((n_features, n_actions))  # Main traces (z)
        self.z_bar = np.zeros((n_features, n_actions))  # Secondary traces for IDBD

        # For step size adaptation (IDBD)
        self.h = np.zeros((n_features, n_actions))

        # Previous values for True Online TD
        self.v_old = 0.0
        self.v_delta = 0.0

        # Track active traces for efficiency (sparse computation)
        self._active_traces: Dict[int, set] = {a: set() for a in range(n_actions)}

        # Track seen features for VLM initialization
        self._seen_features: set = set()

        # Feature extractor (can be set externally)
        self._feature_extractor: Optional[callable] = None

    def set_feature_extractor(self, extractor: callable) -> None:
        """Set the feature extraction function."""
        self._feature_extractor = extractor

    def _get_features(self, observation: Any) -> np.ndarray:
        """Get feature vector from observation."""
        if self._feature_extractor is not None:
            return self._feature_extractor(observation)

        # Default: hash-based sparse features
        features = np.zeros(self.n_features)
        if isinstance(observation, dict) and 'direction' in observation:
            features[observation['direction'] % self.n_features] = 1.0
        else:
            features[self.rng.integers(0, self.n_features)] = 1.0
        return features

    def _get_active_features(self, features: np.ndarray) -> List[int]:
        """Get indices of active (non-zero) features."""
        return list(np.where(features > 0)[0])

    def _get_step_size(self, feature_idx: int, action: int) -> float:
        """Get current step size for a feature-action pair."""
        return np.clip(np.exp(self.beta[feature_idx, action]), self.eta_min, self.eta)

    def select_action(self, observation: Any) -> int:
        """Select action using epsilon-greedy policy with random tie-breaking."""
        self.total_steps += 1

        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)

        q_values = self.get_q_values(observation)
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
        """
        SwiftTD update with adaptive step sizes and overshoot bounding.

        Implements True Online TD(λ) with:
        - Dutch trace correction
        - IDBD-style step size adaptation
        - Overshoot bounding based on rate of learning
        """
        self.total_updates += 1

        features = self._get_features(observation)
        next_features = self._get_features(next_observation)

        # Check for new features and trigger VLM initialization
        active_features = self._get_active_features(features)
        for f_idx in active_features:
            if f_idx not in self._seen_features:
                self._seen_features.add(f_idx)
                self.on_new_feature(f_idx, observation)

        # Compute current Q-value
        q_current = float(np.dot(features, self.weights[:, action]))

        # Compute next Q-value (Q-learning: max over actions)
        if done:
            q_next = 0.0
        else:
            next_q_values = np.array([
                float(np.dot(next_features, self.weights[:, a]))
                for a in range(self.n_actions)
            ])
            q_next = np.max(next_q_values)

        # TD error
        delta = reward + self.discount_factor * q_next - self.v_old

        # Trace decay factor
        trace_decay = self.discount_factor * self.lambda_

        # Get active trace set for this action
        active_set = self._active_traces[action]

        # Decay all active traces and prune small ones
        for f_idx in list(active_set):
            self.traces[f_idx, action] *= trace_decay
            self.z_bar[f_idx, action] *= trace_decay

            # Prune traces below threshold
            alpha = self._get_step_size(f_idx, action)
            if abs(self.traces[f_idx, action]) < alpha * self.trace_threshold:
                active_set.discard(f_idx)

        # Update traces for active features (dutch trace style)
        for f_idx in active_features:
            active_set.add(f_idx)
            alpha = self._get_step_size(f_idx, action)

            # Dutch trace correction: 1 - alpha * gamma * lambda * z'_{t-1} * x_t
            dutch_correction = 1.0 - alpha * trace_decay * self.traces[f_idx, action]
            self.traces[f_idx, action] += dutch_correction * features[f_idx]
            self.z_bar[f_idx, action] += features[f_idx]

        # Compute rate of learning for overshoot bounding
        rate_of_learning = 0.0
        for f_idx in active_set:
            alpha = self._get_step_size(f_idx, action)
            rate_of_learning += alpha * abs(self.traces[f_idx, action])

        # Overshoot bound multiplier (scale down if rate too high)
        if rate_of_learning > self.eta:
            multiplier = self.eta / rate_of_learning
        else:
            multiplier = 1.0

        # Compute v_delta for step size adaptation
        v_delta_new = 0.0
        for f_idx in active_features:
            alpha = self._get_step_size(f_idx, action)
            v_delta_new += alpha * self.traces[f_idx, action] * features[f_idx]

        # Update weights and adapt step sizes
        for f_idx in active_set:
            alpha = self._get_step_size(f_idx, action)

            # Weight update with overshoot bound
            delta_w = multiplier * alpha * delta * self.traces[f_idx, action]
            self.weights[f_idx, action] += delta_w

            # Step size adaptation (IDBD-style)
            if self.meta_step_size > 0:
                step_size_update = (
                    self.meta_step_size / alpha *
                    (delta - self.v_delta) *
                    self.h[f_idx, action]
                )
                self.beta[f_idx, action] += step_size_update

                # Bound beta to [log(eta_min), log(eta)]
                self.beta[f_idx, action] = np.clip(
                    self.beta[f_idx, action],
                    np.log(self.eta_min),
                    np.log(self.eta)
                )

            # Update h for next step size adaptation
            self.h[f_idx, action] = (
                self.h[f_idx, action] * trace_decay *
                max(0, 1 - alpha * self.z_bar[f_idx, action])
                + alpha * self.traces[f_idx, action]
            )

        # Store for next iteration
        self.v_old = q_next
        self.v_delta = v_delta_new

        return {
            'td_error': delta,
            'q_value': q_current,
            'rate_of_learning': rate_of_learning,
            'multiplier': multiplier,
            'n_active_traces': len(active_set),
        }

    def reset(self) -> None:
        """Reset traces at episode start."""
        self.traces.fill(0.0)
        self.z_bar.fill(0.0)
        self.h.fill(0.0)
        self.v_old = 0.0
        self.v_delta = 0.0
        self._active_traces = {a: set() for a in range(self.n_actions)}

    def get_weights(self, feature_idx: int) -> np.ndarray:
        """Get weight vector for a specific feature."""
        if 0 <= feature_idx < self.n_features:
            return self.weights[feature_idx].copy()
        return np.zeros(self.n_actions)

    def set_weights(self, feature_idx: int, values: np.ndarray) -> None:
        """Set weight vector for a specific feature."""
        if 0 <= feature_idx < self.n_features:
            self.weights[feature_idx] = values.copy()

    def initialize_feature_weight(
        self,
        feature_idx: int,
        value: float,
        action: Optional[int] = None,
    ) -> None:
        """
        Initialize weights for a feature based on VLM sentiment.

        The key NeuroSwift mechanism: w_new = VLM_sentiment * alpha_prior

        Args:
            feature_idx: Feature to initialize.
            value: Initial weight (typically VLM_sentiment * alpha_prior).
            action: If specified, only initialize for this action.
        """
        if 0 <= feature_idx < self.n_features:
            if action is not None:
                self.weights[feature_idx, action] = value
            else:
                self.weights[feature_idx, :] = value

    def get_q_values(self, observation: Any) -> np.ndarray:
        """Get Q-values for all actions."""
        features = self._get_features(observation)
        return np.array([
            float(np.dot(features, self.weights[:, a]))
            for a in range(self.n_actions)
        ])

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = super().get_stats()

        # Average step size across all features
        avg_alpha = np.mean(np.exp(self.beta))

        stats.update({
            'n_features': self.n_features,
            'n_seen_features': len(self._seen_features),
            'weight_norm': float(np.linalg.norm(self.weights)),
            'avg_step_size': float(avg_alpha),
            'lambda': self.lambda_,
            'eta': self.eta,
            'meta_step_size': self.meta_step_size,
            'overshoot_bound': self.overshoot_bound,
        })
        return stats

    def get_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing."""
        state = super().get_state()
        state['weights'] = self.weights.tolist()
        state['beta'] = self.beta.tolist()
        state['traces'] = self.traces.tolist()
        state['z_bar'] = self.z_bar.tolist()
        state['h'] = self.h.tolist()
        state['v_old'] = self.v_old
        state['v_delta'] = self.v_delta
        state['seen_features'] = list(self._seen_features)
        state['epsilon'] = self.epsilon
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore agent state from checkpoint."""
        super().set_state(state)
        if 'weights' in state:
            self.weights = np.array(state['weights'])
        if 'beta' in state:
            self.beta = np.array(state['beta'])
        if 'traces' in state:
            self.traces = np.array(state['traces'])
        if 'z_bar' in state:
            self.z_bar = np.array(state['z_bar'])
        if 'h' in state:
            self.h = np.array(state['h'])
        if 'v_old' in state:
            self.v_old = state['v_old']
        if 'v_delta' in state:
            self.v_delta = state['v_delta']
        if 'seen_features' in state:
            self._seen_features = set(state['seen_features'])
        if 'epsilon' in state:
            self.epsilon = state['epsilon']


class SwiftTDWithVLM(SwiftTDAgent):
    """
    SwiftTD agent with integrated VLM initialization.

    When a new feature is detected, this agent:
    1. Queries the VLM for semantic understanding
    2. Gets a sentiment score [-1, 1]
    3. Initializes the feature weight as: w = sentiment * alpha_prior
    """

    def __init__(
        self,
        n_actions: int,
        n_features: int = 512,
        alpha_prior: float = 1.0,
        vlm_oracle: Optional[Any] = None,
        **kwargs,
    ):
        """
        Args:
            n_actions: Number of actions.
            n_features: Number of features.
            alpha_prior: Scaling factor for VLM sentiment in weight init.
            vlm_oracle: VLM oracle for semantic queries.
            **kwargs: Additional arguments for SwiftTDAgent.
        """
        super().__init__(n_actions=n_actions, n_features=n_features, **kwargs)

        self.alpha_prior = alpha_prior
        self.vlm_oracle = vlm_oracle

        # Track VLM queries
        self.vlm_queries = 0
        self.vlm_cache: Dict[int, float] = {}

        # Set up the new feature callback
        self.set_new_feature_callback(self._vlm_initialize_feature)

    def _vlm_initialize_feature(self, feature_idx: int, observation: Any) -> None:
        """
        Initialize feature weight using VLM sentiment.

        This is the core mechanism of NeuroSwift:
        w_new = VLM_sentiment * alpha_prior
        """
        if self.vlm_oracle is None:
            return

        # Check cache first
        if feature_idx in self.vlm_cache:
            sentiment = self.vlm_cache[feature_idx]
        else:
            # Query VLM
            self.vlm_queries += 1
            sentiment = self.vlm_oracle.query(observation)
            self.vlm_cache[feature_idx] = sentiment

        # Initialize weight
        initial_weight = sentiment * self.alpha_prior
        self.initialize_feature_weight(feature_idx, initial_weight)

    def set_vlm_oracle(self, oracle: Any) -> None:
        """Set the VLM oracle for semantic queries."""
        self.vlm_oracle = oracle

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics including VLM query count."""
        stats = super().get_stats()
        stats.update({
            'vlm_queries': self.vlm_queries,
            'vlm_cache_size': len(self.vlm_cache),
            'alpha_prior': self.alpha_prior,
        })
        return stats

    def get_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing."""
        state = super().get_state()
        state['vlm_queries'] = self.vlm_queries
        state['vlm_cache'] = dict(self.vlm_cache)
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore agent state from checkpoint."""
        super().set_state(state)
        if 'vlm_queries' in state:
            self.vlm_queries = state['vlm_queries']
        if 'vlm_cache' in state:
            self.vlm_cache = dict(state['vlm_cache'])
