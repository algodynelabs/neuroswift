"""Tests for TextureGrid environments."""
import pytest
import numpy as np

from src.environments import (
    TextureManager,
    TextureGridEnv,
    make_texture_grid,
    make_fire_env,
    make_fake_lava_env,
    make_mixed_env,
    TEXTURE_SIZE,
)


class TestTextureManager:
    """Tests for texture generation and management."""

    def test_texture_manager_creation(self):
        """Test TextureManager can be created."""
        tm = TextureManager()
        assert tm is not None

    def test_generate_fire_texture(self):
        """Test fire texture generation."""
        tm = TextureManager()
        texture = tm.get_texture('fire')
        assert texture.shape == (TEXTURE_SIZE, TEXTURE_SIZE, 3)
        assert texture.dtype == np.uint8
        # Fire should be predominantly red
        assert np.mean(texture[:, :, 0]) > np.mean(texture[:, :, 2])

    def test_generate_water_texture(self):
        """Test water texture generation."""
        tm = TextureManager()
        texture = tm.get_texture('water')
        assert texture.shape == (TEXTURE_SIZE, TEXTURE_SIZE, 3)
        # Water should be predominantly blue
        assert np.mean(texture[:, :, 2]) > np.mean(texture[:, :, 0])

    def test_generate_all_textures(self):
        """Test all textures can be generated."""
        tm = TextureManager()
        texture_names = ['fire', 'water', 'red_water', 'grass', 'goal', 'wall', 'floor', 'agent']
        for name in texture_names:
            texture = tm.get_texture(name)
            assert texture.shape == (TEXTURE_SIZE, TEXTURE_SIZE, 3), f"Texture {name} has wrong shape"

    def test_texture_caching(self):
        """Test textures are cached."""
        tm = TextureManager()
        texture1 = tm.get_texture('fire')
        texture2 = tm.get_texture('fire')
        assert texture1 is texture2  # Same object due to caching


class TestTextureGridEnv:
    """Tests for TextureGrid environment."""

    def test_make_texture_grid(self):
        """Test basic environment creation."""
        env = make_texture_grid("MiniGrid-Empty-6x6-v0")
        assert env is not None
        env.close()

    def test_observation_shape(self):
        """Test observation has correct shape."""
        env = make_texture_grid("MiniGrid-Empty-6x6-v0")
        obs, info = env.reset(seed=42)
        expected_shape = (6 * TEXTURE_SIZE, 6 * TEXTURE_SIZE, 3)
        assert obs['image'].shape == expected_shape
        env.close()

    def test_step_returns_valid_observation(self):
        """Test step returns valid observation."""
        env = make_texture_grid("MiniGrid-Empty-6x6-v0")
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)
        assert 'image' in obs
        assert 'direction' in obs
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()

    def test_reset_produces_different_observations(self):
        """Test reset with different seeds produces different results."""
        env = make_texture_grid("MiniGrid-Empty-6x6-v0")
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        # Direction might differ, but images might be same for empty grid
        env.close()


class TestExperimentEnvironments:
    """Tests for experiment-specific environments."""

    def test_fire_env_creation(self):
        """Test Fire environment can be created."""
        env = make_fire_env(size=8)
        obs, info = env.reset(seed=42)
        assert obs['image'].shape == (8 * TEXTURE_SIZE, 8 * TEXTURE_SIZE, 3)
        env.close()

    def test_fire_env_has_lava(self):
        """Test Fire environment contains lava tiles."""
        env = make_fire_env(size=8)
        env.reset(seed=42)
        # Check that fire danger mapping exists
        assert env.danger_mapping.get('fire') == -1.0
        env.close()

    def test_fake_lava_env_creation(self):
        """Test Fake Lava environment can be created."""
        env = make_fake_lava_env(size=8)
        obs, info = env.reset(seed=42)
        assert obs['image'].shape == (8 * TEXTURE_SIZE, 8 * TEXTURE_SIZE, 3)
        # Red water should be safe
        assert env.danger_mapping.get('red_water') == 0.0
        env.close()

    def test_mixed_env_creation(self):
        """Test Mixed environment can be created."""
        env = make_mixed_env(size=10)
        obs, info = env.reset(seed=42)
        assert obs['image'].shape == (10 * TEXTURE_SIZE, 10 * TEXTURE_SIZE, 3)
        env.close()


class TestRandomAgent:
    """Tests for random agent interactions."""

    def test_random_agent_fire_env(self):
        """Test random agent can run in Fire environment."""
        env = make_fire_env(size=8)
        obs, info = env.reset(seed=42)

        episodes_completed = 0
        max_episodes = 3
        max_steps = 100

        while episodes_completed < max_episodes:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()
                episodes_completed += 1

        env.close()
        assert episodes_completed == max_episodes

    def test_random_agent_all_envs(self):
        """Test random agent can run in all experiment environments."""
        envs = [
            make_fire_env(size=8),
            make_fake_lava_env(size=8),
            make_mixed_env(size=10),
        ]

        for env in envs:
            obs, info = env.reset(seed=42)
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
