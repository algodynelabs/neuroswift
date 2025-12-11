"""Tests for NeuroSwift agents."""
import pytest
import numpy as np

from src.agents import (
    BaseAgent,
    QLearningAgent,
    TileCodingQLearning,
    SwiftTDAgent,
    SwiftTDWithVLM,
    TileFeatureExtractor,
    SimpleTileExtractor,
)
from src.environments import make_fire_env


class TestQLearningAgent:
    """Tests for Q-Learning agent."""

    def test_creation(self):
        """Test agent can be created."""
        agent = QLearningAgent(n_actions=4)
        assert agent is not None
        assert agent.n_actions == 4

    def test_select_action(self):
        """Test action selection."""
        agent = QLearningAgent(n_actions=4, epsilon=0.0, seed=42)
        obs = {'direction': 0}
        action = agent.select_action(obs)
        assert 0 <= action < 4

    def test_epsilon_greedy(self):
        """Test epsilon-greedy exploration."""
        agent = QLearningAgent(n_actions=4, epsilon=1.0, seed=42)
        obs = {'direction': 0}

        # With epsilon=1.0, all actions should be random
        actions = [agent.select_action(obs) for _ in range(100)]
        unique_actions = set(actions)
        assert len(unique_actions) > 1  # Should explore multiple actions

    def test_update(self):
        """Test Q-value update."""
        agent = QLearningAgent(n_actions=4, learning_rate=0.1, seed=42)
        obs = {'direction': 0}
        next_obs = {'direction': 1}

        # Update with positive reward
        metrics = agent.update(obs, action=0, reward=1.0, next_observation=next_obs, done=False)

        assert 'td_error' in metrics
        assert 'q_value' in metrics

    def test_learning(self):
        """Test that agent learns from rewards."""
        agent = QLearningAgent(n_actions=4, learning_rate=0.5, epsilon=0.0, seed=42)
        obs = {'direction': 0}

        # Give consistent positive reward for action 0
        for _ in range(10):
            agent.update(obs, action=0, reward=1.0, next_observation=obs, done=False)

        # Action 0 should now have highest Q-value
        q_values = agent.get_q_values(obs)
        assert np.argmax(q_values) == 0


class TestSwiftTDAgent:
    """Tests for SwiftTD agent."""

    def test_creation(self):
        """Test agent can be created."""
        agent = SwiftTDAgent(n_actions=4, n_features=64)
        assert agent is not None
        assert agent.n_features == 64

    def test_select_action(self):
        """Test action selection."""
        agent = SwiftTDAgent(n_actions=4, n_features=64, epsilon=0.0, seed=42)
        obs = {'direction': 0}
        action = agent.select_action(obs)
        assert 0 <= action < 4

    def test_update(self):
        """Test TD update."""
        agent = SwiftTDAgent(n_actions=4, n_features=64, seed=42)
        obs = {'direction': 0}
        next_obs = {'direction': 1}

        metrics = agent.update(obs, action=0, reward=1.0, next_observation=next_obs, done=False)

        assert 'td_error' in metrics
        assert 'rate_of_learning' in metrics
        assert 'n_active_traces' in metrics

    def test_overshoot_bound(self):
        """Test overshoot bound limits rate of learning."""
        agent = SwiftTDAgent(
            n_actions=4,
            n_features=64,
            eta=0.1,  # Maximum step size bound
            learning_rate=10.0,  # Large learning rate (will be clamped to eta)
            seed=42,
        )
        obs = {'direction': 0}

        metrics = agent.update(obs, action=0, reward=100.0, next_observation=obs, done=False)

        # Rate of learning should be bounded by eta (or multiplier applied)
        assert metrics['multiplier'] <= 1.0

    def test_reset_clears_traces(self):
        """Test reset clears eligibility traces."""
        agent = SwiftTDAgent(n_actions=4, n_features=64, seed=42)
        obs = {'direction': 0}

        # Do some updates to accumulate traces
        for _ in range(5):
            agent.update(obs, action=0, reward=1.0, next_observation=obs, done=False)

        assert np.sum(np.abs(agent.traces)) > 0

        # Reset should clear traces
        agent.reset()
        assert np.sum(np.abs(agent.traces)) == 0

    def test_weight_initialization(self):
        """Test manual weight initialization."""
        agent = SwiftTDAgent(n_actions=4, n_features=64, seed=42)

        # Initialize feature 0 with value 0.5
        agent.initialize_feature_weight(0, 0.5)

        weights = agent.get_weights(0)
        assert np.allclose(weights, 0.5)


class TestSwiftTDWithVLM:
    """Tests for SwiftTD with VLM integration."""

    def test_creation(self):
        """Test agent can be created."""
        agent = SwiftTDWithVLM(n_actions=4, n_features=64, alpha_prior=1.0)
        assert agent is not None
        assert agent.alpha_prior == 1.0

    def test_vlm_callback(self):
        """Test VLM callback is triggered on new feature."""
        callback_calls = []

        class MockVLM:
            def query(self, obs):
                callback_calls.append(obs)
                return -0.5  # Negative sentiment

        agent = SwiftTDWithVLM(
            n_actions=4,
            n_features=64,
            alpha_prior=1.0,
            vlm_oracle=MockVLM(),
            seed=42,
        )

        obs = {'direction': 0}
        agent.update(obs, action=0, reward=0, next_observation=obs, done=False)

        # VLM should have been queried
        assert len(callback_calls) > 0
        assert agent.vlm_queries > 0

    def test_weight_init_from_vlm(self):
        """Test weights are initialized from VLM sentiment."""
        class MockVLM:
            def query(self, obs):
                return -1.0  # Fire is dangerous

        agent = SwiftTDWithVLM(
            n_actions=4,
            n_features=64,
            alpha_prior=2.0,
            vlm_oracle=MockVLM(),
            seed=42,
        )

        obs = {'direction': 0}
        agent.update(obs, action=0, reward=0, next_observation=obs, done=False)

        # Check that some weights were initialized to -2.0 (sentiment * alpha_prior)
        # Note: exact feature depends on hash, so we check any weight is -2.0
        assert np.any(agent.weights == -2.0)


class TestFeatureExtractor:
    """Tests for feature extractors."""

    def test_tile_extractor_creation(self):
        """Test tile feature extractor creation."""
        extractor = TileFeatureExtractor(n_features=256)
        assert extractor is not None

    def test_extract_features(self):
        """Test feature extraction from image."""
        extractor = TileFeatureExtractor(n_features=256, n_tilings=2)

        # Create dummy observation
        obs = {'image': np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)}

        features = extractor.extract_features(obs)
        assert features.shape == (256,)
        assert np.sum(features) > 0  # Should have active features

    def test_novelty_detection(self):
        """Test novelty detection."""
        extractor = TileFeatureExtractor(n_features=256)

        obs = {'image': np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)}

        # First observation should have novel features
        is_novel, novel_features = extractor.is_novel_feature(obs)
        assert is_novel
        assert len(novel_features) > 0

        # Mark as seen
        extractor.mark_features_seen(novel_features)

        # Same observation should not be novel
        is_novel, novel_features = extractor.is_novel_feature(obs)
        assert len(novel_features) == 0

    def test_simple_extractor(self):
        """Test simple tile extractor."""
        extractor = SimpleTileExtractor(n_features=32)

        features = extractor.extract_features(
            observation={'direction': 2},
            tile_type='fire',
            position=(1, 1),
        )

        assert features.shape == (32,)
        assert np.sum(features) > 0


class TestAgentEnvironmentIntegration:
    """Integration tests with environment."""

    def test_qlearning_in_environment(self):
        """Test Q-learning agent in TextureGrid environment."""
        env = make_fire_env(size=6)
        agent = QLearningAgent(n_actions=env.action_space.n, seed=42)

        obs, info = env.reset(seed=42)

        total_reward = 0
        for _ in range(50):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            total_reward += reward

            if terminated or truncated:
                obs, info = env.reset()

        env.close()
        # Just verify it runs without error

    def test_swifttd_in_environment(self):
        """Test SwiftTD agent in TextureGrid environment."""
        env = make_fire_env(size=6)
        agent = SwiftTDAgent(n_actions=env.action_space.n, n_features=128, seed=42)

        obs, info = env.reset(seed=42)

        for _ in range(50):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs

            if terminated or truncated:
                agent.reset()
                obs, info = env.reset()

        env.close()
        # Verify it collected stats
        stats = agent.get_stats()
        assert stats['total_steps'] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
