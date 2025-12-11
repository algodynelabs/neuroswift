"""VLM Oracle interface for semantic grounding.

The VLM Oracle provides sentiment scores for visual observations,
enabling semantic weight initialization in RL agents.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import base64
import io
import re
import requests
import time
from PIL import Image


class VLMOracle(ABC):
    """
    Abstract base class for VLM oracles.

    A VLM oracle takes an observation (typically an image) and returns
    a sentiment score indicating whether the content is dangerous,
    neutral, or beneficial.
    """

    @abstractmethod
    def query(self, observation: Any) -> float:
        """
        Query the VLM for a sentiment score.

        Args:
            observation: Environment observation (dict with 'image' key).

        Returns:
            Sentiment score in range [-1, 1]:
                -1: Dangerous/harmful
                 0: Neutral
                +1: Beneficial/good
        """
        pass

    @abstractmethod
    def query_with_label(self, observation: Any) -> Tuple[str, float]:
        """
        Query the VLM and get both a label and sentiment.

        Args:
            observation: Environment observation.

        Returns:
            Tuple of (label, sentiment_score).
        """
        pass


class OllamaVLM(VLMOracle):
    """
    VLM Oracle using Ollama with LLaVA model.

    Queries the local Ollama API for visual understanding.
    """

    def __init__(
        self,
        model: str = "llava:7b",
        api_url: str = "http://localhost:11434/api/generate",
        timeout: float = 60.0,
        max_retries: int = 3,
        cache_enabled: bool = True,
    ):
        """
        Args:
            model: Ollama model name (default: llava:7b).
            api_url: Ollama API endpoint.
            timeout: Request timeout in seconds.
            max_retries: Number of retries on failure.
            cache_enabled: Whether to cache query results.
        """
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_enabled = cache_enabled

        # Query cache (image hash -> sentiment)
        self._cache: Dict[int, float] = {}
        self._label_cache: Dict[int, Tuple[str, float]] = {}

        # Stats
        self.total_queries = 0
        self.cache_hits = 0
        self.total_time = 0.0

        # Prompt for identification (simpler approach - identify then map)
        self.sentiment_prompt = """What element is shown in this game texture image?
Just name it in one word: fire, water, grass, lava, floor, goal, or wall."""

        # Mapping from identified elements to sentiment scores
        self.element_to_sentiment = {
            'fire': -1.0,
            'lava': -1.0,
            'spikes': -1.0,
            'danger': -1.0,
            'water': 0.5,
            'grass': 0.5,
            'floor': 0.0,
            'wall': 0.0,
            'goal': 1.0,
            'reward': 1.0,
            'target': 1.0,
        }

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image array to base64 string."""
        img = Image.fromarray(image)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()

    def _compute_image_hash(self, image: np.ndarray) -> int:
        """Compute hash of image for caching."""
        # Use downsampled image for faster hashing
        small = image[::8, ::8, :] if image.shape[0] >= 8 else image
        return hash(small.tobytes())

    def _parse_sentiment(self, response: str) -> float:
        """Parse sentiment from VLM response using element identification."""
        response_lower = response.lower().strip()

        # Get the first word (the identified element)
        first_word = response_lower.split()[0] if response_lower else ""

        # Check element mapping first
        if first_word in self.element_to_sentiment:
            return self.element_to_sentiment[first_word]

        # Check if any known element appears in the response
        for element, sentiment in self.element_to_sentiment.items():
            if element in response_lower:
                return sentiment

        # Legacy fallback: explicit danger/safe labels
        response_upper = response.upper()
        if "DANGEROUS" in response_upper or "DANGER" in response_upper:
            return -1.0
        elif "SAFE" in response_upper:
            return 1.0
        elif "NEUTRAL" in response_upper:
            return 0.0

        return 0.0

    def _parse_label(self, response: str) -> str:
        """Extract object label from response."""
        # Try to find object mentions
        common_objects = ['fire', 'lava', 'water', 'grass', 'goal', 'wall', 'floor']
        for obj in common_objects:
            if obj in response.lower():
                return obj
        return "unknown"

    def query(self, observation: Any) -> float:
        """Query VLM for sentiment score."""
        label, sentiment = self.query_with_label(observation)
        return sentiment

    def query_with_label(self, observation: Any) -> Tuple[str, float]:
        """Query VLM and get both label and sentiment."""
        # Extract image from observation
        if isinstance(observation, dict) and 'image' in observation:
            image = observation['image']
        elif isinstance(observation, np.ndarray):
            image = observation
        else:
            return ("unknown", 0.0)

        # Check cache
        img_hash = self._compute_image_hash(image)
        if self.cache_enabled and img_hash in self._label_cache:
            self.cache_hits += 1
            return self._label_cache[img_hash]

        # Prepare request
        img_base64 = self._image_to_base64(image)

        # Query with retries
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                response = requests.post(
                    self.api_url,
                    json={
                        'model': self.model,
                        'prompt': self.sentiment_prompt,
                        'images': [img_base64],
                        'stream': False,
                    },
                    timeout=self.timeout,
                )

                elapsed = time.time() - start_time
                self.total_time += elapsed
                self.total_queries += 1

                if response.status_code == 200:
                    result = response.json()
                    text = result.get('response', '')

                    label = self._parse_label(text)
                    sentiment = self._parse_sentiment(text)

                    # Cache result
                    if self.cache_enabled:
                        self._label_cache[img_hash] = (label, sentiment)
                        self._cache[img_hash] = sentiment

                    return (label, sentiment)

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(0.5)
                continue
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(0.5)
                continue

        # Failed all retries
        return ("unknown", 0.0)

    def get_stats(self) -> Dict[str, Any]:
        """Get query statistics."""
        return {
            'total_queries': self.total_queries,
            'cache_hits': self.cache_hits,
            'cache_size': len(self._cache),
            'avg_query_time': self.total_time / max(1, self.total_queries),
            'cache_hit_rate': self.cache_hits / max(1, self.total_queries + self.cache_hits),
        }


class MockVLM(VLMOracle):
    """
    Mock VLM for testing and development.

    Returns deterministic sentiment scores based on image properties.
    """

    def __init__(
        self,
        default_sentiment: float = 0.0,
        color_sentiments: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            default_sentiment: Default sentiment for unknown images.
            color_sentiments: Map of dominant color to sentiment.
        """
        self.default_sentiment = default_sentiment
        self.color_sentiments = color_sentiments or {
            'red': -1.0,      # Fire/danger
            'orange': -0.5,   # Warning
            'blue': 0.5,      # Water/safe
            'green': 0.5,     # Grass/safe
            'yellow': 1.0,    # Goal/reward
            'gray': 0.0,      # Wall/neutral
        }
        self.query_count = 0

    def _get_dominant_color(self, image: np.ndarray) -> str:
        """Determine dominant color of image."""
        if image.size == 0:
            return 'gray'

        r_mean = np.mean(image[:, :, 0])
        g_mean = np.mean(image[:, :, 1])
        b_mean = np.mean(image[:, :, 2])

        # Classify based on RGB values (order matters - check most specific first)
        if r_mean > 200 and g_mean > 200 and b_mean < 150:
            return 'yellow'  # Both red and green high, blue low
        elif r_mean > 150 and g_mean < 100 and b_mean < 100:
            return 'red'  # Red dominant, others low
        elif r_mean > 200 and g_mean > 100 and g_mean < 200 and b_mean < 100:
            return 'orange'  # Red high, green medium, blue low
        elif g_mean > 100 and r_mean < 100 and b_mean < 100:
            return 'green'  # Green dominant
        elif b_mean > 150 and r_mean < 100 and g_mean < 150:
            return 'blue'  # Blue dominant
        else:
            return 'gray'

    def query(self, observation: Any) -> float:
        """Return sentiment based on dominant color."""
        label, sentiment = self.query_with_label(observation)
        return sentiment

    def query_with_label(self, observation: Any) -> Tuple[str, float]:
        """Return label and sentiment."""
        self.query_count += 1

        if isinstance(observation, dict) and 'image' in observation:
            image = observation['image']
        elif isinstance(observation, np.ndarray):
            image = observation
        else:
            return ('unknown', self.default_sentiment)

        color = self._get_dominant_color(image)
        sentiment = self.color_sentiments.get(color, self.default_sentiment)

        # Map color to object label
        color_to_label = {
            'red': 'fire',
            'orange': 'fire',
            'blue': 'water',
            'green': 'grass',
            'yellow': 'goal',
            'gray': 'floor',
        }
        label = color_to_label.get(color, 'unknown')

        return (label, sentiment)


class HallucinatingVLM(MockVLM):
    """
    Mock VLM that hallucinates on specific colors.

    Used for Experiment B (Fake Lava) to test hallucination correction.
    The VLM incorrectly classifies red water as dangerous.
    """

    def __init__(self):
        super().__init__()
        # Override: red is seen as dangerous even when it's safe (red water)
        self.color_sentiments = {
            'red': -1.0,      # Hallucinates: red water is "dangerous"
            'orange': -0.5,
            'blue': 0.5,
            'green': 0.5,
            'yellow': 1.0,
            'gray': 0.0,
        }


class GroundTruthVLM(VLMOracle):
    """
    Oracle VLM that returns ground truth based on actual tile danger.

    Used as an upper bound baseline in experiments.
    """

    def __init__(self, danger_mapping: Optional[Dict[str, float]] = None):
        """
        Args:
            danger_mapping: Map of tile type to danger score.
        """
        self.danger_mapping = danger_mapping or {
            'fire': -1.0,
            'lava': -1.0,
            'water': 0.5,
            'red_water': 0.5,  # Ground truth: red water is safe!
            'grass': 0.3,
            'goal': 1.0,
            'wall': 0.0,
            'floor': 0.0,
        }
        self.query_count = 0

    def query(self, observation: Any, tile_type: Optional[str] = None) -> float:
        """Return ground truth sentiment for tile type."""
        self.query_count += 1

        if tile_type:
            return self.danger_mapping.get(tile_type, 0.0)

        # Can't determine without tile type info
        return 0.0

    def query_with_label(
        self,
        observation: Any,
        tile_type: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Return label and ground truth sentiment."""
        if tile_type:
            return (tile_type, self.danger_mapping.get(tile_type, 0.0))
        return ('unknown', 0.0)
