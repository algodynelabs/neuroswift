"""Trigger mechanisms for VLM queries.

Different strategies for when to query the VLM:
- Imprint-Trigger: Query only when new feature is detected
- Random-Trigger: Query every N steps randomly
- Frame-Trigger: Query every frame (expensive baseline)
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, Future

from .oracle import VLMOracle


class TriggerMechanism(ABC):
    """
    Abstract base class for VLM trigger mechanisms.

    Controls when and how the VLM is queried.
    """

    def __init__(self, vlm_oracle: VLMOracle):
        """
        Args:
            vlm_oracle: The VLM oracle to use for queries.
        """
        self.vlm_oracle = vlm_oracle
        self.query_count = 0
        self.total_steps = 0

    @abstractmethod
    def should_query(
        self,
        observation: Any,
        is_new_feature: bool = False,
        feature_idx: Optional[int] = None,
    ) -> bool:
        """
        Determine if VLM should be queried for this observation.

        Args:
            observation: Current environment observation.
            is_new_feature: Whether this observation contains a new feature.
            feature_idx: Index of the new feature (if any).

        Returns:
            True if VLM should be queried.
        """
        pass

    def query(self, observation: Any) -> float:
        """Query the VLM and track statistics."""
        self.query_count += 1
        return self.vlm_oracle.query(observation)

    def step(self) -> None:
        """Called each environment step."""
        self.total_steps += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get trigger statistics."""
        return {
            'query_count': self.query_count,
            'total_steps': self.total_steps,
            'queries_per_step': self.query_count / max(1, self.total_steps),
        }


class ImprintTrigger(TriggerMechanism):
    """
    Imprint-Trigger: Query VLM only when new feature is detected.

    This is the efficient mechanism that achieves near-optimal
    performance with minimal VLM queries.
    """

    def __init__(self, vlm_oracle: VLMOracle):
        super().__init__(vlm_oracle)
        self._queried_features: set = set()

    def should_query(
        self,
        observation: Any,
        is_new_feature: bool = False,
        feature_idx: Optional[int] = None,
    ) -> bool:
        """Query only on new features."""
        if not is_new_feature or feature_idx is None:
            return False

        # Only query if we haven't queried this feature before
        if feature_idx in self._queried_features:
            return False

        self._queried_features.add(feature_idx)
        return True


class RandomTrigger(TriggerMechanism):
    """
    Random-Trigger: Query VLM with probability p each step.

    Used as a baseline to compare against Imprint-Trigger.
    """

    def __init__(
        self,
        vlm_oracle: VLMOracle,
        query_probability: float = 0.01,
        seed: Optional[int] = None,
    ):
        """
        Args:
            vlm_oracle: VLM oracle.
            query_probability: Probability of querying each step.
            seed: Random seed.
        """
        super().__init__(vlm_oracle)
        self.query_probability = query_probability
        self.rng = np.random.default_rng(seed)

    def should_query(
        self,
        observation: Any,
        is_new_feature: bool = False,
        feature_idx: Optional[int] = None,
    ) -> bool:
        """Query with fixed probability."""
        return self.rng.random() < self.query_probability


class FrameTrigger(TriggerMechanism):
    """
    Frame-Trigger: Query VLM every frame.

    Expensive but provides maximum information. Used as
    performance upper bound in experiments.
    """

    def should_query(
        self,
        observation: Any,
        is_new_feature: bool = False,
        feature_idx: Optional[int] = None,
    ) -> bool:
        """Always query."""
        return True


class NoTrigger(TriggerMechanism):
    """
    No-Trigger: Never query VLM.

    Used as baseline (tabula rasa learning without VLM).
    """

    def should_query(
        self,
        observation: Any,
        is_new_feature: bool = False,
        feature_idx: Optional[int] = None,
    ) -> bool:
        """Never query."""
        return False


class AsyncVLMWrapper:
    """
    Asynchronous VLM query wrapper.

    Allows VLM queries to run in background without blocking
    the main agent loop. Essential for real-time operation.
    """

    def __init__(
        self,
        vlm_oracle: VLMOracle,
        max_workers: int = 2,
        timeout: float = 5.0,
        default_sentiment: float = 0.0,
    ):
        """
        Args:
            vlm_oracle: The VLM oracle to wrap.
            max_workers: Maximum concurrent queries.
            timeout: Timeout for getting results.
            default_sentiment: Default value if query not ready.
        """
        self.vlm_oracle = vlm_oracle
        self.timeout = timeout
        self.default_sentiment = default_sentiment

        # Thread pool for async queries
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Pending queries: feature_idx -> Future
        self._pending: Dict[int, Future] = {}

        # Completed results: feature_idx -> sentiment
        self._results: Dict[int, float] = {}

        # Stats
        self.queries_submitted = 0
        self.queries_completed = 0
        self.queries_timeout = 0

    def submit_query(
        self,
        feature_idx: int,
        observation: Any,
    ) -> None:
        """
        Submit async VLM query for a feature.

        Args:
            feature_idx: Feature index to associate with result.
            observation: Observation to query.
        """
        if feature_idx in self._pending or feature_idx in self._results:
            return

        future = self.executor.submit(self.vlm_oracle.query, observation)
        self._pending[feature_idx] = future
        self.queries_submitted += 1

    def get_result(
        self,
        feature_idx: int,
        block: bool = False,
    ) -> Optional[float]:
        """
        Get result for a feature query.

        Args:
            feature_idx: Feature index.
            block: If True, wait for result.

        Returns:
            Sentiment score or None if not ready.
        """
        # Check completed results first
        if feature_idx in self._results:
            return self._results[feature_idx]

        # Check pending
        if feature_idx not in self._pending:
            return None

        future = self._pending[feature_idx]

        if block:
            try:
                result = future.result(timeout=self.timeout)
                self._results[feature_idx] = result
                del self._pending[feature_idx]
                self.queries_completed += 1
                return result
            except Exception:
                self.queries_timeout += 1
                del self._pending[feature_idx]
                return self.default_sentiment

        # Non-blocking check
        if future.done():
            try:
                result = future.result(timeout=0)
                self._results[feature_idx] = result
                del self._pending[feature_idx]
                self.queries_completed += 1
                return result
            except Exception:
                del self._pending[feature_idx]
                return self.default_sentiment

        return None

    def get_or_default(self, feature_idx: int) -> float:
        """Get result or default if not ready."""
        result = self.get_result(feature_idx, block=False)
        return result if result is not None else self.default_sentiment

    def poll_pending(self) -> List[Tuple[int, float]]:
        """
        Poll all pending queries and return completed ones.

        Returns:
            List of (feature_idx, sentiment) for completed queries.
        """
        completed = []

        for feature_idx in list(self._pending.keys()):
            result = self.get_result(feature_idx, block=False)
            if result is not None:
                completed.append((feature_idx, result))

        return completed

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.executor.shutdown(wait=False)

    def get_stats(self) -> Dict[str, Any]:
        """Get wrapper statistics."""
        return {
            'queries_submitted': self.queries_submitted,
            'queries_completed': self.queries_completed,
            'queries_timeout': self.queries_timeout,
            'pending_count': len(self._pending),
            'cached_results': len(self._results),
        }


class VLMIntegration:
    """
    High-level VLM integration for agents.

    Combines trigger mechanism with async querying and
    provides weight initialization interface.
    """

    def __init__(
        self,
        vlm_oracle: VLMOracle,
        trigger: TriggerMechanism,
        alpha_prior: float = 1.0,
        async_enabled: bool = True,
    ):
        """
        Args:
            vlm_oracle: VLM oracle.
            trigger: Trigger mechanism.
            alpha_prior: Weight initialization scaling.
            async_enabled: Use async queries.
        """
        self.vlm_oracle = vlm_oracle
        self.trigger = trigger
        self.alpha_prior = alpha_prior

        if async_enabled:
            self.async_wrapper = AsyncVLMWrapper(vlm_oracle)
        else:
            self.async_wrapper = None

        # Track feature initializations
        self._initialized_features: Dict[int, float] = {}

    def on_new_feature(
        self,
        feature_idx: int,
        observation: Any,
    ) -> Optional[float]:
        """
        Handle new feature detection.

        Returns initial weight if available, None otherwise.
        """
        if not self.trigger.should_query(
            observation,
            is_new_feature=True,
            feature_idx=feature_idx,
        ):
            return None

        if self.async_wrapper:
            self.async_wrapper.submit_query(feature_idx, observation)
            return None  # Will be ready later
        else:
            sentiment = self.trigger.query(observation)
            initial_weight = sentiment * self.alpha_prior
            self._initialized_features[feature_idx] = initial_weight
            return initial_weight

    def get_feature_weight(self, feature_idx: int) -> Optional[float]:
        """Get initialized weight for a feature."""
        if feature_idx in self._initialized_features:
            return self._initialized_features[feature_idx]

        if self.async_wrapper:
            sentiment = self.async_wrapper.get_result(feature_idx, block=False)
            if sentiment is not None:
                weight = sentiment * self.alpha_prior
                self._initialized_features[feature_idx] = weight
                return weight

        return None

    def step(self) -> None:
        """Called each environment step."""
        self.trigger.step()

        # Poll async results
        if self.async_wrapper:
            for feature_idx, sentiment in self.async_wrapper.poll_pending():
                weight = sentiment * self.alpha_prior
                self._initialized_features[feature_idx] = weight

    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        stats = {
            'trigger_stats': self.trigger.get_stats(),
            'initialized_features': len(self._initialized_features),
            'alpha_prior': self.alpha_prior,
        }
        if self.async_wrapper:
            stats['async_stats'] = self.async_wrapper.get_stats()
        return stats
