"""Feature extraction for tile-based observations.

The key MVP simplification: When the agent enters a new tile type,
treat it as "recruiting a new feature" - this is the trigger for
VLM-based weight initialization.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from collections import defaultdict


class TileFeatureExtractor:
    """
    Feature extractor that maps tile observations to feature vectors.

    Each unique tile type gets its own feature index.
    New tile types trigger novelty detection.
    """

    def __init__(
        self,
        n_features: int = 512,
        n_tiles_x: int = 8,
        n_tiles_y: int = 8,
        n_tilings: int = 4,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_features: Total number of features.
            n_tiles_x: Number of tiles in x dimension per tiling.
            n_tiles_y: Number of tiles in y dimension per tiling.
            n_tilings: Number of overlapping tilings.
            seed: Random seed.
        """
        self.n_features = n_features
        self.n_tiles_x = n_tiles_x
        self.n_tiles_y = n_tiles_y
        self.n_tilings = n_tilings
        self.rng = np.random.default_rng(seed)

        # Tiling offsets
        self.offsets = [
            (i / n_tilings, i / n_tilings)
            for i in range(n_tilings)
        ]

        # Map tile signatures to feature indices
        self._tile_to_feature: Dict[tuple, int] = {}
        self._next_feature_idx = 0

        # Track seen features
        self._seen_features: Set[int] = set()

        # Store tile observations for VLM queries
        self._feature_observations: Dict[int, Any] = {}

    def _compute_tile_signature(self, image: np.ndarray) -> tuple:
        """
        Compute a signature for a tile image.

        Uses color histogram binning for simple texture classification.
        """
        if image.size == 0:
            return (0,)

        # Compute color histogram (simplified)
        r_mean = int(np.mean(image[:, :, 0]) / 32)  # 8 bins
        g_mean = int(np.mean(image[:, :, 1]) / 32)
        b_mean = int(np.mean(image[:, :, 2]) / 32)

        # Also capture variance for texture info
        r_std = int(np.std(image[:, :, 0]) / 32)
        g_std = int(np.std(image[:, :, 1]) / 32)

        return (r_mean, g_mean, b_mean, r_std, g_std)

    def _get_tile_feature(self, signature: tuple) -> int:
        """Get or create feature index for a tile signature."""
        if signature not in self._tile_to_feature:
            # New tile type detected
            self._tile_to_feature[signature] = self._next_feature_idx
            self._next_feature_idx += 1

        return self._tile_to_feature[signature]

    def extract_features(self, observation: Any) -> np.ndarray:
        """
        Extract feature vector from observation.

        Args:
            observation: Dict with 'image' key containing RGB array.

        Returns:
            Binary feature vector.
        """
        features = np.zeros(self.n_features)

        if not isinstance(observation, dict) or 'image' not in observation:
            return features

        image = observation['image']
        height, width = image.shape[:2]

        # Determine tile size based on image dimensions
        tile_h = height // self.n_tiles_y
        tile_w = width // self.n_tiles_x

        # Extract features for each tile position
        for tiling_idx, (ox, oy) in enumerate(self.offsets):
            # Apply tiling offset
            offset_x = int(ox * tile_w)
            offset_y = int(oy * tile_h)

            for ty in range(self.n_tiles_y):
                for tx in range(self.n_tiles_x):
                    # Calculate tile boundaries
                    y_start = ty * tile_h + offset_y
                    x_start = tx * tile_w + offset_x

                    y_end = min(y_start + tile_h, height)
                    x_end = min(x_start + tile_w, width)

                    if y_start >= height or x_start >= width:
                        continue

                    # Extract tile
                    tile = image[y_start:y_end, x_start:x_end]

                    # Compute signature and get feature
                    signature = self._compute_tile_signature(tile)
                    base_feature = self._get_tile_feature(signature)

                    # Compute final feature index with position encoding
                    position_hash = (
                        tiling_idx * self.n_tiles_x * self.n_tiles_y +
                        ty * self.n_tiles_x + tx
                    )
                    feature_idx = (base_feature + position_hash) % self.n_features

                    features[feature_idx] = 1.0

        return features

    def is_novel_feature(self, observation: Any) -> Tuple[bool, List[int]]:
        """
        Check if observation contains novel features.

        Args:
            observation: Environment observation.

        Returns:
            Tuple of (has_novel, list_of_novel_feature_indices).
        """
        features = self.extract_features(observation)
        active = np.where(features > 0)[0]

        novel_features = []
        for f_idx in active:
            if f_idx not in self._seen_features:
                novel_features.append(f_idx)

        return (len(novel_features) > 0, novel_features)

    def mark_features_seen(self, feature_indices: List[int]) -> None:
        """Mark features as seen."""
        self._seen_features.update(feature_indices)

    def store_observation(self, feature_idx: int, observation: Any) -> None:
        """Store observation associated with a feature for later VLM query."""
        self._feature_observations[feature_idx] = observation

    def get_observation_for_feature(self, feature_idx: int) -> Optional[Any]:
        """Get stored observation for a feature."""
        return self._feature_observations.get(feature_idx)

    def get_stats(self) -> Dict[str, Any]:
        """Get extractor statistics."""
        return {
            'n_tile_types': len(self._tile_to_feature),
            'n_seen_features': len(self._seen_features),
            'n_stored_observations': len(self._feature_observations),
        }


class SimpleTileExtractor:
    """
    Simplified tile feature extractor for MVP.

    Maps the agent's current tile position to a feature.
    New tile visits trigger novelty detection.
    """

    def __init__(self, n_features: int = 64, seed: Optional[int] = None):
        """
        Args:
            n_features: Number of features.
            seed: Random seed.
        """
        self.n_features = n_features
        self.rng = np.random.default_rng(seed)

        # Track tile visits
        self._visited_tiles: Set[tuple] = set()

        # Map tile types to features
        self._tile_type_features: Dict[str, int] = {}
        self._next_feature = 0

    def get_tile_feature(self, tile_type: str) -> int:
        """Get feature index for a tile type."""
        if tile_type not in self._tile_type_features:
            self._tile_type_features[tile_type] = self._next_feature
            self._next_feature += 1
        return self._tile_type_features[tile_type]

    def extract_features(
        self,
        observation: Any,
        tile_type: Optional[str] = None,
        position: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Extract features from observation.

        Args:
            observation: Environment observation.
            tile_type: Optional tile type string (from info dict).
            position: Optional (x, y) position.

        Returns:
            Feature vector.
        """
        features = np.zeros(self.n_features)

        # Feature 1: Tile type
        if tile_type:
            f_idx = self.get_tile_feature(tile_type)
            if f_idx < self.n_features:
                features[f_idx] = 1.0

        # Feature 2: Direction (if available)
        if isinstance(observation, dict) and 'direction' in observation:
            dir_feature = self.n_features // 2 + observation['direction']
            if dir_feature < self.n_features:
                features[dir_feature] = 1.0

        return features

    def is_novel(self, tile_type: str, position: Tuple[int, int]) -> bool:
        """Check if this is a novel tile visit."""
        key = (tile_type, position)
        if key not in self._visited_tiles:
            self._visited_tiles.add(key)
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            'n_tile_types': len(self._tile_type_features),
            'n_visited_tiles': len(self._visited_tiles),
        }
